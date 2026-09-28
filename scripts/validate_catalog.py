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
# `Flops` is matched anywhere in the name (compute throughput is written
# `TFlopsPeak`, `TFlopsFP8`, …); every other unit is matched as a trailing
# segment (`BwPeakTBs`, `MemoryGiB`, `IntraNodeBwGBps`), which is the catalog's
# PascalCase-unit-suffix convention and avoids mid-word collisions (e.g. a
# hypothetical count field would not be mistaken for a physical quantity).
_COMPUTE_TOKENS = ("Flops",)
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


def field_has_unit(name: str) -> bool:
    """Return True iff the field NAME carries a recognised physical-unit token."""
    if any(tok in name for tok in _COMPUTE_TOKENS):
        return True
    return any(name.endswith(tok) for tok in _SUFFIX_UNITS)


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


def _rel(path: Path, root: Path) -> str:
    try:
        # POSIX-style so a diagnostic reads identically on every platform.
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _load_yaml(path: Path, root: Path, errors: list[str]) -> Any:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
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
            if key.startswith("_"):
                # `_comment*` keys are free-form PROSE and exempt — but only when
                # the value is a string. A numeric or structured value under an
                # underscore is not a comment; it must not slip past the units
                # check (a hidden `_mfu: 0.85` would otherwise pass).
                if not isinstance(val, str):
                    errors.append(
                        f"{_rel(path, root)}: {key}: an underscore-prefixed key is a prose "
                        f"comment and must have a string value; a non-string here bypasses "
                        f"the units check"
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
    return errors


def validate_networks(networks_dir: Path, root: Path) -> list[str]:
    """networks/*.yaml: reusable inter-node fabric classes (blis-catalog#7/#10)."""
    errors: list[str] = []
    if not networks_dir.is_dir():
        return errors
    required = ("Provenance", "InterNodeBwGBps", "PDTransferBaseLatencyMs")
    for path in sorted(networks_dir.glob("*.yaml")):
        data = _as_mapping(_load_yaml(path, root, errors), path, root, errors)
        if data is None:
            continue

        for key in required:
            if key not in data:
                errors.append(f"{_rel(path, root)}: {key}: required field is missing")

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

        if "PDTransferBaseLatencyMs" in data:
            base = _coerce_number(data["PDTransferBaseLatencyMs"])
            if base is None:
                errors.append(f"{_rel(path, root)}: PDTransferBaseLatencyMs: must be a number")
            elif base < 0:
                errors.append(
                    f"{_rel(path, root)}: PDTransferBaseLatencyMs: must be non-negative "
                    f"(got {data['PDTransferBaseLatencyMs']})"
                )
    return errors


def validate_workloads(workloads_dir: Path, root: Path) -> list[str]:
    """workloads/*.yaml: token-count distributions for a traffic preset."""
    errors: list[str] = []
    if not workloads_dir.is_dir():
        return errors
    # Common required fields across every preset. `prefix_tokens` is optional
    # (multidoc omits it); when present it is validated as a non-negative count.
    required = (
        "prompt_tokens", "prompt_tokens_stdev", "prompt_tokens_min", "prompt_tokens_max",
        "output_tokens", "output_tokens_stdev", "output_tokens_min", "output_tokens_max",
    )
    for path in sorted(workloads_dir.glob("*.yaml")):
        data = _as_mapping(_load_yaml(path, root, errors), path, root, errors)
        if data is None:
            continue

        for key in required:
            if key not in data:
                errors.append(f"{_rel(path, root)}: {key}: required field is missing")

        for key in (*required, "prefix_tokens"):
            if key in data:
                num = _coerce_number(data[key])
                if num is None:
                    errors.append(f"{_rel(path, root)}: {key}: must be a number")
                elif num < 0:
                    errors.append(
                        f"{_rel(path, root)}: {key}: must be non-negative (got {data[key]})"
                    )

        for kind in ("prompt", "output"):
            lo = _coerce_number(data.get(f"{kind}_tokens_min"))
            hi = _coerce_number(data.get(f"{kind}_tokens_max"))
            mean = _coerce_number(data.get(f"{kind}_tokens"))
            if lo is not None and hi is not None and lo > hi:
                errors.append(
                    f"{_rel(path, root)}: {kind}_tokens_min: {lo:g} exceeds "
                    f"{kind}_tokens_max {hi:g}"
                )
            if None not in (lo, mean, hi) and not (lo <= mean <= hi):
                errors.append(
                    f"{_rel(path, root)}: {kind}_tokens: mean {mean:g} is outside "
                    f"[{kind}_tokens_min {lo:g}, {kind}_tokens_max {hi:g}]"
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


def validate_catalog(root: Path) -> list[str]:
    """Validate every committed entry across all namespaces; return error lines."""
    root = Path(root)
    errors: list[str] = []
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
