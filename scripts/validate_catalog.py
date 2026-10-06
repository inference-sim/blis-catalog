#!/usr/bin/env python3
"""blis-catalog structural CI gate (R2H3 — inference-sim/blis-catalog#8).

Validate every committed catalog entry so a malformed or incomplete file fails
CI *in this repository*, at the point of authorship, rather than surfacing later
in a downstream ``blis run`` in the wrong repo.

Scope split with inference-sim#1750 (the simulator-side gate) — the two are
complementary and neither re-implements the other's checks:

  * THIS gate (catalog-side): **structural / schema** validation. Every file
    parses; required fields are present; ``hardware/`` fields are physical
    quantities that carry a datasheet **unit** (not a dimensionless factor);
    ``networks/`` fabric classes are shaped and have a positive bandwidth;
    ``Provenance`` is a value drawn from a known enum.
  * inference-sim#1750 (sim-side): loads every entry through the **real
    simulator loader** (the ``blis run`` code path) to catch anything the
    loader's *semantics* reject — model-config field types the estimator reads,
    cross-field consistency the simulator relies on, etc.

Structural facts live here (in the repo that owns the data); loader-semantics
live in inference-sim. This gate deliberately does not load the simulator.

Usage:
    python3 scripts/validate_catalog.py [ROOT]

ROOT defaults to the repository root (the parent of ``scripts/``). The command
prints one ``path: key: message`` line per problem to **stderr** and exits
non-zero if any entry is malformed; on success it prints a one-line summary to
**stdout** and exits 0.

This task adds no simulation number; it is a CI gate (R2 is value-preserving).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import yaml

# --------------------------------------------------------------------------- #
# Shared vocabulary                                                           #
# --------------------------------------------------------------------------- #

# The OPEN set of allowed `Provenance` values. Every value here is a *declared*
# (non-measured, non-fitted) provenance — measured/fitted numbers never enter
# the catalog; they live in blis-registry.
#
#   vendor_spec : a nominal figure taken from a vendor datasheet, including the
#                 datasheet's own unit normalisations (a NIC line rate / 8, an
#                 NVLink bidirectional figure / 2, a dense/sparse selection).
#                 Every catalog entry today is `vendor_spec`.
#   derived     : reserved for a figure *computed* by the catalog from spec(s)
#                 via an arithmetic step the datasheet does not itself state —
#                 still declared, still non-measured. (No entry uses it yet; it
#                 is validated and ready so it can be adopted without a code
#                 change, per blis-catalog#8.)
#
# The set is intentionally open: extend it here (never a per-file exception).
PROVENANCE_ENUM = ("vendor_spec", "derived")

# Physical-unit vocabulary for the `hardware/` units check. A hardware field is
# "dimensioned" iff its NAME carries one of these unit tokens. This tests the
# field's *units*, not a fixed list of field names: a new field with a real
# datasheet unit (e.g. `TFlopsSparse`, `NvlinkBwGBps`, `TdpWatts`) passes
# automatically, while a dimensionless factor (`mfu`, an achieved-utilisation, a
# ratio, an efficiency) carries no unit token and is rejected — which is the
# whole point. It keeps the catalog README's "nothing here is learned, fitted,
# measured, or inferred" load-bearing: a learned utilisation cannot drift back
# in beside the specifications.
#
# Compute throughput is matched anywhere in the name because it is written
# scale-prefixed and mid-name (`TFlopsPeak`, `TFlopsFP8`, …). The token must
# carry a scale prefix: bare `Flops` is NOT a unit, or a dimensionless
# `FlopsUtilization` / `PeakFlopsRatio` / `mfuFlops` would slip through the very
# guarantee this check defends. Every other unit is matched as a trailing
# segment (`BwPeakTBs`, `MemoryGiB`, `IntraNodeBwGBps`) — the catalog's
# PascalCase-unit-suffix convention — which avoids mid-word collisions (a count
# field is not mistaken for a physical quantity).
_COMPUTE_TOKENS = ("KFlops", "MFlops", "GFlops", "TFlops", "PFlops", "EFlops")
_SUFFIX_UNITS = (
    # bandwidth (bytes / second)
    "TBps", "GBps", "MBps", "KBps", "TBs", "GBs", "MBs",
    # memory / size (bytes)
    "TiB", "GiB", "MiB", "KiB", "TB", "GB", "MB", "KB",
    # power
    "MilliWatts", "Watts",
    # frequency
    "THz", "GHz", "MHz", "Hz",
)

# A name carrying one of these is dimensionless by construction (a ratio, a
# fraction, an efficiency, an achieved utilisation) and is rejected even when it
# also carries a unit token — so a `TFlopsUtilization` / `BwPeakTBsRatio` cannot
# ride a real unit past the check. This closes the compute-token hole at its
# root (blis-catalog#11 review).
_DIMENSIONLESS_DESCRIPTORS = (
    "Ratio", "Efficiency", "Utilisation", "Utilization", "Fraction", "Percent",
)

# A dimensioned COUNT: its unit is one of the things counted. `SMCount` is 132
# SMs on a GH100 die and `GPUsPerRack` is 72 GPUs in an NVL72 domain — both are
# declared integers in the same sense a bandwidth figure is, and no more learned
# than one. So a count belongs in the catalog, and the units check must admit one.
#
# The vocabulary is a CLOSED list of whole names rather than a suffix rule, which
# is what keeps it from becoming the hole the rest of this check exists to close.
# A suffix rule on `Count` would admit `FooGBCount` and any other name ending in
# the word, and `Count` says nothing about what was counted — it is the generic
# shape a fitted quantity would borrow. Naming each field outright means a new
# count is a deliberate edit here, reviewed like the physical fields are.
#
# A count is additionally required to be a positive whole number of parts, so a
# fitted 0.91 cannot enter under one of these names either.
_COUNT_FIELDS = frozenset({"SMCount", "GPUsPerNode", "GPUsPerRack"})


def field_is_count(name: str) -> bool:
    """Return True iff the field NAME is one of the declared part counts."""
    return name in _COUNT_FIELDS


def field_has_unit(name: str) -> bool:
    """Return True iff the field NAME carries a recognised physical-unit token.

    A dimensionless descriptor (`…Ratio`, `…Efficiency`, `…Utilisation`, …)
    disqualifies the name outright, even scale-prefixed compute names, so a
    learned factor cannot ride a unit token in.
    """
    if any(desc in name for desc in _DIMENSIONLESS_DESCRIPTORS):
        return False
    if any(tok in name for tok in _COMPUTE_TOKENS):
        return True
    if any(name.endswith(tok) for tok in _SUFFIX_UNITS):
        return True
    return field_is_count(name)


def _coerce_number(value: Any) -> float | None:
    """Return value as a float, or None if it is not numeric.

    Accepts a real int/float, or a numeric string. The string case matters
    because PyYAML's 1.1 resolver does **not** treat an unsigned-exponent float
    such as ``7.0e3`` as a number (it needs ``7.0e+3`` or ``7000.0``), yet the
    Go loader that actually consumes these files reads it as a float. Coercing
    here keeps the gate faithful to that data without editing the verbatim
    values (R2 is value-preserving).

    Non-finite values are rejected: YAML booleans are int subclasses, and NaN /
    infinity (``.nan``, ``.inf``, or their string spellings) are not physical
    quantities — treating them as "not a number" stops them slipping past a
    ``>= 0`` bound, which is true for neither NaN nor a comparison it poisons.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return None
    else:
        return None
    return number if math.isfinite(number) else None


def _is_number(value: Any) -> bool:
    """True iff the value is numeric (see :func:`_coerce_number`)."""
    return _coerce_number(value) is not None


# --------------------------------------------------------------------------- #
# Parse helpers                                                               #
# --------------------------------------------------------------------------- #


# Returned by the loaders when a file could not be read/parsed (the error is
# already recorded). It is distinct from a successfully parsed ``None`` — an
# empty or comment-only document — which the callers must still flag rather than
# silently skip.
_PARSE_ERROR: Any = object()


class _UniqueKeyLoader(yaml.SafeLoader):
    """A SafeLoader that rejects duplicate mapping keys instead of silently
    keeping the last (PyYAML's default). A file with `Provenance: vendor-spec`
    then `Provenance: vendor_spec` is malformed and must fail the gate, not pass
    on whichever value happened to win (blis-catalog#11 re-review)."""

    def construct_mapping(self, node, deep=False):
        seen: set = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping", node.start_mark,
                    f"found duplicate key {key!r}", key_node.start_mark,
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def _rel(path: Path, root: Path) -> str:
    try:
        # POSIX-style so a diagnostic reads identically on every platform.
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _load_yaml(path: Path, root: Path, errors: list[str]) -> Any:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return yaml.load(fh, Loader=_UniqueKeyLoader)
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        errors.append(f"{_rel(path, root)}: not valid YAML: {exc}")
        return _PARSE_ERROR


def _load_json(path: Path, root: Path, errors: list[str]) -> Any:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        errors.append(f"{_rel(path, root)}: not valid JSON: {exc}")
        return _PARSE_ERROR


def _as_mapping(data: Any, path: Path, root: Path, errors: list[str]) -> dict | None:
    """Return ``data`` as a non-empty mapping, else None (recording an error).

    A parse failure was already reported by the loader (the sentinel), so it is
    skipped here. A document that parsed to ``None`` (empty / comment-only /
    explicit null) or to a non-mapping is a malformed entry and is flagged —
    never silently skipped.
    """
    if data is _PARSE_ERROR:
        return None
    if not isinstance(data, dict) or not data:
        errors.append(f"{_rel(path, root)}: must be a non-empty YAML mapping")
        return None
    return data


# --------------------------------------------------------------------------- #
# Per-namespace validators                                                    #
# --------------------------------------------------------------------------- #


def validate_models(models_dir: Path, root: Path) -> list[str]:
    """models/<name>/ pairs a verbatim vendor config.json with an identity model.yaml."""
    errors: list[str] = []
    if not models_dir.is_dir():
        return errors
    for entry in sorted(p for p in models_dir.iterdir() if p.is_dir()):
        name = entry.name
        cfg = entry / "config.json"
        myaml = entry / "model.yaml"

        # config.json: present, parseable, a non-empty object (verbatim vendor
        # file — structural check only; the loader owns which keys it reads).
        if not cfg.is_file():
            errors.append(f"{_rel(cfg, root)}: missing (a model entry needs config.json)")
        else:
            data = _load_json(cfg, root, errors)
            if data is not _PARSE_ERROR and (not isinstance(data, dict) or not data):
                errors.append(f"{_rel(cfg, root)}: must be a non-empty JSON object")

        # model.yaml: identity + provenance.
        if not myaml.is_file():
            errors.append(f"{_rel(myaml, root)}: missing (a model entry needs model.yaml)")
            continue
        meta = _as_mapping(_load_yaml(myaml, root, errors), myaml, root, errors)
        if meta is None:
            continue

        got_name = meta.get("name")
        if not isinstance(got_name, str) or not got_name:
            errors.append(f"{_rel(myaml, root)}: name: missing or not a non-empty string")
        elif got_name != name:
            errors.append(
                f"{_rel(myaml, root)}: name: '{got_name}' does not match directory '{name}'"
            )

        source = meta.get("source")
        if not isinstance(source, dict):
            errors.append(f"{_rel(myaml, root)}: source: missing or not a mapping")
        else:
            for key in ("provider", "repo", "revision"):
                val = source.get(key)
                if not isinstance(val, str) or not val:
                    errors.append(
                        f"{_rel(myaml, root)}: source.{key}: missing or not a non-empty string"
                    )
    return errors


def validate_hardware(hardware_dir: Path, root: Path) -> list[str]:
    """hardware/*.yaml: vendor chip specs — every value field carries a unit."""
    errors: list[str] = []
    if not hardware_dir.is_dir():
        return errors
    for path in sorted(hardware_dir.glob("*.yaml")):
        data = _as_mapping(_load_yaml(path, root, errors), path, root, errors)
        if data is None:
            continue

        if "Provenance" not in data:
            errors.append(f"{_rel(path, root)}: Provenance: required field is missing")
        for key, val in data.items():
            if key.startswith("_comment"):
                # `_comment*` is the catalog's documented free-form PROSE
                # convention (`_comment`, `_comment_interconnect`, …); it must be a
                # string. EVERY other key — including any other underscore-prefixed
                # key — is treated as a data field and validated below, so a
                # dimensionless value cannot hide under an underscore (a `_mfu`,
                # numeric or string, still fails the units check).
                if not isinstance(val, str):
                    errors.append(
                        f"{_rel(path, root)}: {key}: a _comment* key is a prose comment and "
                        f"must have a string value"
                    )
                continue
            if key == "Provenance":
                if val not in PROVENANCE_ENUM:
                    errors.append(
                        f"{_rel(path, root)}: Provenance: '{val}' is not one of "
                        f"{list(PROVENANCE_ENUM)}"
                    )
                continue
            # Every other key must be a physical quantity with a datasheet unit.
            if not field_has_unit(key):
                errors.append(
                    f"{_rel(path, root)}: {key}: dimensionless field — hardware entries hold "
                    f"only physical quantities that carry a datasheet unit (TFlops, TB/s, GiB, "
                    f"GB/s, …); a factor/ratio/utilisation/MFU belongs in blis-registry"
                )
            if not _is_number(val):
                errors.append(f"{_rel(path, root)}: {key}: physical field must be a number")
            elif field_is_count(key):
                # A count names a whole number of physical parts. Requiring that
                # closes the hole the count vocabulary would otherwise open: a
                # fitted 0.91 cannot enter as an `SMCount`.
                n = _coerce_number(val)
                if n is None or n <= 0 or n != int(n):
                    errors.append(
                        f"{_rel(path, root)}: {key}: a count must be a positive whole "
                        f"number of parts, not {val!r}"
                    )
    return errors


def validate_networks(networks_dir: Path, root: Path) -> list[str]:
    """networks/*.yaml: reusable inter-node fabric classes (blis-catalog#7/#10).

    Unlike ``hardware/`` (open by unit vocabulary), a fabric class is a **closed**
    schema: exactly the two fields below, plus ``_comment*`` prose. Any other
    key is rejected, so a fitted/measured field (a `DerateFactor`) cannot hide
    here either — making the README's "nothing measured or fitted" line
    load-bearing for fabrics as well as chips (blis-catalog#11 review).

    ``PDTransferBaseLatencyMs`` used to be required here, always ``0`` (the
    catalog states no inherent fabric base latency); the 0.05 ms
    ``--pd-transfer-base-latency`` modeling placeholder is a registry number
    (blis-registry#10), so the field was dropped from the catalog entirely
    (blis-catalog#12) and now reads as a rejected unknown key.
    """
    errors: list[str] = []
    if not networks_dir.is_dir():
        return errors
    required = ("Provenance", "InterNodeBwGBps")
    for path in sorted(networks_dir.glob("*.yaml")):
        data = _as_mapping(_load_yaml(path, root, errors), path, root, errors)
        if data is None:
            continue

        for key in required:
            if key not in data:
                errors.append(f"{_rel(path, root)}: {key}: required field is missing")

        for key, val in data.items():
            if key in required:
                continue
            if key.startswith("_comment"):  # prose; must be a string (as in hardware/)
                if not isinstance(val, str):
                    errors.append(
                        f"{_rel(path, root)}: {key}: a _comment* key is a prose comment and "
                        f"must have a string value"
                    )
                continue
            errors.append(
                f"{_rel(path, root)}: {key}: unknown field — a fabric class is a closed schema "
                f"of {list(required)} (+ _comment* prose); a fitted/measured value belongs in "
                f"blis-registry"
            )

        prov = data.get("Provenance")
        if "Provenance" in data and prov not in PROVENANCE_ENUM:
            errors.append(
                f"{_rel(path, root)}: Provenance: '{prov}' is not one of {list(PROVENANCE_ENUM)}"
            )

        if "InterNodeBwGBps" in data:
            bw = _coerce_number(data["InterNodeBwGBps"])
            if bw is None:
                errors.append(f"{_rel(path, root)}: InterNodeBwGBps: must be a number")
            elif bw <= 0:
                errors.append(
                    f"{_rel(path, root)}: InterNodeBwGBps: must be positive "
                    f"(got {data['InterNodeBwGBps']})"
                )
    return errors


def validate_workloads(workloads_dir: Path, root: Path) -> list[str]:
    """workloads/*.yaml: a nested token-count distribution for a traffic preset.

    Each preset is a ``workload.Shape`` (blis-catalog#16): an optional top-level
    ``prefix_tokens`` and two nested distributions, ``prompt`` and ``output``,
    each a mapping with a required ``tokens`` (the mean) and optional
    ``tokens_stdev`` / ``tokens_min`` / ``tokens_max``. The mean is positive; the
    other counts are non-negative; the bounds are consistent
    (``min <= mean <= max``); and a shared ``prefix_tokens`` may not exceed the
    mean prompt length. These rules mirror blis-schemas' ``Shape.Validate`` /
    ``Distribution.validate`` so this gate and the Go loader agree. The file
    carries no name — identity is the filename — so no name field is read.
    """
    errors: list[str] = []
    if not workloads_dir.is_dir():
        return errors
    for path in sorted(workloads_dir.glob("*.yaml")):
        data = _as_mapping(_load_yaml(path, root, errors), path, root, errors)
        if data is None:
            continue

        # prefix_tokens is optional (multidoc omits it); non-negative when present.
        if "prefix_tokens" in data:
            num = _coerce_number(data["prefix_tokens"])
            if num is None:
                errors.append(f"{_rel(path, root)}: prefix_tokens: must be a number")
            elif num < 0:
                errors.append(
                    f"{_rel(path, root)}: prefix_tokens: must be non-negative "
                    f"(got {data['prefix_tokens']})"
                )

        # prompt and output are nested Distribution sub-maps. Each is required and
        # must be a mapping; within it `tokens` (the mean) is required and must be
        # POSITIVE, while tokens_stdev/tokens_min/tokens_max are optional and
        # non-negative, with min <= mean <= max.
        for kind in ("prompt", "output"):
            if kind not in data:
                errors.append(f"{_rel(path, root)}: {kind}: required field is missing")
                continue
            dist = data[kind]
            if not isinstance(dist, dict):
                errors.append(f"{_rel(path, root)}: {kind}: must be a mapping")
                continue

            if "tokens" not in dist:
                errors.append(
                    f"{_rel(path, root)}: {kind}.tokens: required field is missing"
                )

            for key in ("tokens", "tokens_stdev", "tokens_min", "tokens_max"):
                if key in dist:
                    num = _coerce_number(dist[key])
                    if num is None:
                        errors.append(f"{_rel(path, root)}: {kind}.{key}: must be a number")
                    elif key == "tokens":
                        # The mean must be positive, mirroring Go
                        # Distribution.validate (`if d.Mean < 1`): a zero-token
                        # distribution describes no request.
                        if num <= 0:
                            errors.append(
                                f"{_rel(path, root)}: {kind}.tokens: must be positive "
                                f"(got {dist[key]})"
                            )
                    elif num < 0:
                        errors.append(
                            f"{_rel(path, root)}: {kind}.{key}: must be non-negative "
                            f"(got {dist[key]})"
                        )

            # Bounds must be consistent: min <= max when both are given, and the
            # mean lies within whichever bounds are present. Each check is
            # one-sided, so a distribution that gives only one bound is still
            # checked against the mean — matching blis-schemas' Distribution.validate,
            # where mean-vs-max and mean-vs-min are independent guards.
            lo = _coerce_number(dist.get("tokens_min"))
            hi = _coerce_number(dist.get("tokens_max"))
            mean = _coerce_number(dist.get("tokens"))
            if lo is not None and hi is not None and lo > hi:
                errors.append(
                    f"{_rel(path, root)}: {kind}.tokens_min: {lo:g} exceeds "
                    f"{kind}.tokens_max {hi:g}"
                )
            if mean is not None and hi is not None and mean > hi:
                errors.append(
                    f"{_rel(path, root)}: {kind}.tokens: mean {mean:g} exceeds "
                    f"{kind}.tokens_max {hi:g}"
                )
            if mean is not None and mean > 0 and lo is not None and mean < lo:
                errors.append(
                    f"{_rel(path, root)}: {kind}.tokens: mean {mean:g} is below "
                    f"{kind}.tokens_min {lo:g}"
                )

        # Shape-level cross-check (mirrors Go Shape.Validate): a shared prefix
        # longer than the mean prompt describes no request. Gated on a positive
        # prompt mean, exactly as Go gates on `s.Prompt.Mean > 0`.
        prefix = _coerce_number(data.get("prefix_tokens"))
        prompt = data.get("prompt")
        prompt_mean = (
            _coerce_number(prompt.get("tokens")) if isinstance(prompt, dict) else None
        )
        if (
            prefix is not None
            and prompt_mean is not None
            and prompt_mean > 0
            and prefix > prompt_mean
        ):
            errors.append(
                f"{_rel(path, root)}: prefix_tokens: {prefix:g} exceeds the mean "
                f"prompt length {prompt_mean:g}, so no prompt contains the prefix"
            )
    return errors


def validate_devices(devices_dir: Path, root: Path) -> list[str]:
    """devices/storage.yaml: storage tiers for KV-cache offload."""
    errors: list[str] = []
    if not devices_dir.is_dir():
        return errors
    for path in sorted(devices_dir.glob("*.yaml")):
        data = _as_mapping(_load_yaml(path, root, errors), path, root, errors)
        if data is None:
            continue
        for tier, spec in data.items():
            if not isinstance(spec, dict):
                errors.append(f"{_rel(path, root)}: {tier}: must be a mapping")
                continue
            for key in ("read_bandwidth", "write_bandwidth"):
                if key not in spec:
                    errors.append(f"{_rel(path, root)}: {tier}.{key}: required field is missing")
                    continue
                val = _coerce_number(spec[key])
                if val is None:
                    errors.append(f"{_rel(path, root)}: {tier}.{key}: must be a number")
                elif val <= 0:
                    errors.append(
                        f"{_rel(path, root)}: {tier}.{key}: must be positive (got {spec[key]})"
                    )
            if "base_latency" not in spec:
                errors.append(f"{_rel(path, root)}: {tier}.base_latency: required field is missing")
            else:
                lat = _coerce_number(spec["base_latency"])
                if lat is None:
                    errors.append(f"{_rel(path, root)}: {tier}.base_latency: must be a number")
                elif lat < 0:
                    errors.append(
                        f"{_rel(path, root)}: {tier}.base_latency: must be non-negative "
                        f"(got {spec['base_latency']})"
                    )
    return errors


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #


NAMESPACES = ("models", "hardware", "networks", "workloads", "devices")


def validate_catalog(root: Path) -> list[str]:
    """Validate every committed entry across all namespaces; return error lines."""
    root = Path(root)
    # A missing root, or a directory that is no catalog at all (none of the
    # namespaces present), must fail loudly — otherwise a mistyped root would
    # report "every entry is well-formed" and exit 0 (blis-catalog#11 re-review).
    if not root.is_dir():
        return [f"{root.as_posix()}: catalog root does not exist or is not a directory"]
    errors: list[str] = []
    if not any((root / ns).is_dir() for ns in NAMESPACES):
        errors.append(
            f"{root.as_posix()}: not a catalog root — none of {list(NAMESPACES)} found"
        )
    errors += validate_models(root / "models", root)
    errors += validate_hardware(root / "hardware", root)
    errors += validate_networks(root / "networks", root)
    errors += validate_workloads(root / "workloads", root)
    errors += validate_devices(root / "devices", root)
    return errors


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    root = Path(argv[0]) if argv else Path(__file__).resolve().parent.parent
    errors = validate_catalog(root)
    if errors:
        for err in errors:
            print(err, file=sys.stderr)
        print(
            f"catalog validation FAILED: {len(errors)} problem(s) in {root}",
            file=sys.stderr,
        )
        return 1
    print(f"catalog validation OK: every entry under {root} is well-formed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
