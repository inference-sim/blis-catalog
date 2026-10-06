"""Behavioural tests for the blis-catalog structural CI gate (blis-catalog#8).

Each test encodes an acceptance bullet: a malformed or incomplete entry must be
rejected, naming the file and the key; and the real committed catalog must pass.
Bad fixtures are built in tmp_path so no malformed file is ever committed to the
catalog the gate scans.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import validate_catalog as vc  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Fixture: a minimal, valid catalog on disk                                   #
# --------------------------------------------------------------------------- #


def _write_yaml(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def good_catalog(tmp_path: Path) -> Path:
    root = tmp_path
    # models/qwen3-14b
    _write_json(root / "models/qwen3-14b/config.json", {"model_type": "qwen3"})
    _write_yaml(
        root / "models/qwen3-14b/model.yaml",
        {
            "name": "qwen3-14b",
            "source": {"provider": "huggingface", "repo": "Qwen/Qwen3-14B", "revision": "abc123"},
        },
    )
    # hardware/h100
    _write_yaml(
        root / "hardware/h100.yaml",
        {
            "Provenance": "vendor_spec",
            "TFlopsPeak": 989.5,
            "TFlopsFP8": 1979.0,
            "BwPeakTBs": 3.35,
            "MemoryGiB": 80.0,
            "IntraNodeBwGBps": 450,
            "_comment_interconnect": "prose is exempt from the units check",
        },
    )
    # networks/ib-400g
    _write_yaml(
        root / "networks/ib-400g.yaml",
        {"Provenance": "vendor_spec", "InterNodeBwGBps": 50},
    )
    # workloads/chatbot — nested workload.Shape (blis-catalog#16)
    _write_yaml(
        root / "workloads/chatbot.yaml",
        {
            "prefix_tokens": 0,
            "prompt": {"tokens": 256, "tokens_stdev": 100,
                       "tokens_min": 2, "tokens_max": 800},
            "output": {"tokens": 256, "tokens_stdev": 100,
                       "tokens_min": 1, "tokens_max": 1024},
        },
    )
    # workloads/multidoc — the preset that legitimately omits prefix_tokens
    _write_yaml(
        root / "workloads/multidoc.yaml",
        {
            "prompt": {"tokens": 10240, "tokens_stdev": 1200,
                       "tokens_min": 500, "tokens_max": 20480},
            "output": {"tokens": 1536, "tokens_stdev": 300,
                       "tokens_min": 50, "tokens_max": 4096},
        },
    )
    # devices/storage
    _write_yaml(
        root / "devices/storage.yaml",
        {
            "nvme_gen4": {"read_bandwidth": 7.0e3, "write_bandwidth": 5.0e3, "base_latency": 80.0},
            "cpu_dram": {"read_bandwidth": 2.0e4, "write_bandwidth": 2.0e4, "base_latency": 1.0},
        },
    )
    assert vc.validate_catalog(root) == [], "fixture catalog should be valid"
    return root


def _assert_flags(root: Path, *, substrings: list[str]) -> str:
    """Assert validation fails and the joined error text contains each substring."""
    errors = vc.validate_catalog(root)
    assert errors, "expected at least one validation error"
    joined = "\n".join(errors)
    for sub in substrings:
        assert sub in joined, f"expected '{sub}' in errors:\n{joined}"
    return joined


# --------------------------------------------------------------------------- #
# The real catalog must pass (green-on-committed-data guard)                  #
# --------------------------------------------------------------------------- #


def test_real_catalog_passes():
    errors = vc.validate_catalog(REPO_ROOT)
    assert errors == [], "committed catalog must be well-formed:\n" + "\n".join(errors)


# --------------------------------------------------------------------------- #
# hardware/: the units check                                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "name",
    ["TFlopsPeak", "TFlopsFP8", "BwPeakTBs", "MemoryGiB", "IntraNodeBwGBps",
     "TdpWatts", "ClockGHz", "TFlopsSparse", "NvlinkBwGBps"],
)
def test_field_has_unit_accepts_dimensioned(name):
    assert vc.field_has_unit(name)


@pytest.mark.parametrize(
    "name",
    ["mfu", "mfuPrefill", "mfuDecode", "AchievedUtilisation", "SparsityRatio",
     "DerateFactor", "PeakEfficiency", "TensorScale", "Utilization"],
)
def test_field_has_unit_rejects_dimensionless(name):
    assert not vc.field_has_unit(name)


def test_hardware_dimensionless_field_fails_naming_file_and_key(good_catalog):
    hw = good_catalog / "hardware/h100.yaml"
    data = yaml.safe_load(hw.read_text())
    data["mfuPrefill"] = 0.85  # a learned utilisation sneaking in beside the specs
    _write_yaml(hw, data)
    _assert_flags(good_catalog, substrings=["hardware/h100.yaml", "mfuPrefill", "dimensionless"])


def test_hardware_bad_provenance_enum_fails(good_catalog):
    hw = good_catalog / "hardware/h100.yaml"
    data = yaml.safe_load(hw.read_text())
    data["Provenance"] = "measured"  # measured numbers belong in blis-registry
    _write_yaml(hw, data)
    _assert_flags(good_catalog, substrings=["hardware/h100.yaml", "Provenance", "measured"])


def test_hardware_missing_provenance_fails(good_catalog):
    hw = good_catalog / "hardware/h100.yaml"
    data = yaml.safe_load(hw.read_text())
    del data["Provenance"]
    _write_yaml(hw, data)
    _assert_flags(good_catalog, substrings=["hardware/h100.yaml", "Provenance"])


def test_hardware_non_numeric_physical_field_fails(good_catalog):
    hw = good_catalog / "hardware/h100.yaml"
    data = yaml.safe_load(hw.read_text())
    data["MemoryGiB"] = "eighty"
    _write_yaml(hw, data)
    _assert_flags(good_catalog, substrings=["hardware/h100.yaml", "MemoryGiB", "number"])


def test_hardware_derived_provenance_is_accepted(good_catalog):
    hw = good_catalog / "hardware/h100.yaml"
    data = yaml.safe_load(hw.read_text())
    data["Provenance"] = "derived"  # the second, open enum member
    _write_yaml(hw, data)
    assert vc.validate_catalog(good_catalog) == []


# --------------------------------------------------------------------------- #
# networks/: shape / positivity / enum                                        #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad_bw", [0, -50, -0.1])
def test_networks_non_positive_bandwidth_fails(good_catalog, bad_bw):
    net = good_catalog / "networks/ib-400g.yaml"
    data = yaml.safe_load(net.read_text())
    data["InterNodeBwGBps"] = bad_bw
    _write_yaml(net, data)
    _assert_flags(good_catalog, substrings=["networks/ib-400g.yaml", "InterNodeBwGBps", "positive"])


@pytest.mark.parametrize("field", ["InterNodeBwGBps", "Provenance"])
def test_networks_missing_required_field_fails(good_catalog, field):
    net = good_catalog / "networks/ib-400g.yaml"
    data = yaml.safe_load(net.read_text())
    del data[field]
    _write_yaml(net, data)
    _assert_flags(good_catalog, substrings=["networks/ib-400g.yaml", field])


@pytest.mark.parametrize("bad_prov", ["vendor-spec", "vendorspec", "measured", "fitted"])
def test_networks_unknown_provenance_fails(good_catalog, bad_prov):
    net = good_catalog / "networks/ib-400g.yaml"
    data = yaml.safe_load(net.read_text())
    data["Provenance"] = bad_prov
    _write_yaml(net, data)
    _assert_flags(good_catalog, substrings=["networks/ib-400g.yaml", "Provenance", bad_prov])


# --------------------------------------------------------------------------- #
# models/: pairing, parse validity, identity                                  #
# --------------------------------------------------------------------------- #


def test_models_missing_config_json_fails(good_catalog):
    (good_catalog / "models/qwen3-14b/config.json").unlink()
    _assert_flags(good_catalog, substrings=["models/qwen3-14b/config.json", "missing"])


def test_models_missing_model_yaml_fails(good_catalog):
    (good_catalog / "models/qwen3-14b/model.yaml").unlink()
    _assert_flags(good_catalog, substrings=["models/qwen3-14b/model.yaml", "missing"])


def test_models_invalid_json_fails(good_catalog):
    (good_catalog / "models/qwen3-14b/config.json").write_text("{ not json", encoding="utf-8")
    _assert_flags(good_catalog, substrings=["models/qwen3-14b/config.json", "JSON"])


def test_models_empty_config_object_fails(good_catalog):
    (good_catalog / "models/qwen3-14b/config.json").write_text("{}", encoding="utf-8")
    _assert_flags(good_catalog, substrings=["models/qwen3-14b/config.json", "non-empty"])


def test_models_name_mismatch_fails(good_catalog):
    my = good_catalog / "models/qwen3-14b/model.yaml"
    data = yaml.safe_load(my.read_text())
    data["name"] = "qwen3-15b"
    _write_yaml(my, data)
    _assert_flags(good_catalog, substrings=["models/qwen3-14b/model.yaml", "name", "qwen3-14b"])


def test_models_missing_source_revision_fails(good_catalog):
    my = good_catalog / "models/qwen3-14b/model.yaml"
    data = yaml.safe_load(my.read_text())
    del data["source"]["revision"]
    _write_yaml(my, data)
    _assert_flags(good_catalog, substrings=["models/qwen3-14b/model.yaml", "source.revision"])


# --------------------------------------------------------------------------- #
# workloads/: required fields and bound sanity                                #
# --------------------------------------------------------------------------- #


def test_workloads_missing_required_tokens_fails(good_catalog):
    wl = good_catalog / "workloads/chatbot.yaml"
    data = yaml.safe_load(wl.read_text())
    del data["output"]["tokens"]  # the required mean of a distribution
    _write_yaml(wl, data)
    _assert_flags(good_catalog, substrings=["workloads/chatbot.yaml", "output.tokens"])


def test_workloads_missing_distribution_fails(good_catalog):
    wl = good_catalog / "workloads/chatbot.yaml"
    data = yaml.safe_load(wl.read_text())
    del data["output"]  # a whole distribution sub-map is absent
    _write_yaml(wl, data)
    _assert_flags(good_catalog, substrings=["workloads/chatbot.yaml", "output"])


def test_workloads_distribution_not_a_mapping_fails(good_catalog):
    wl = good_catalog / "workloads/chatbot.yaml"
    data = yaml.safe_load(wl.read_text())
    data["prompt"] = 256  # a scalar where a Distribution sub-map is expected
    _write_yaml(wl, data)
    _assert_flags(good_catalog, substrings=["workloads/chatbot.yaml", "prompt"])


def test_workloads_min_exceeds_max_fails(good_catalog):
    wl = good_catalog / "workloads/chatbot.yaml"
    data = yaml.safe_load(wl.read_text())
    data["prompt"]["tokens_min"] = 900  # > prompt.tokens_max 800
    _write_yaml(wl, data)
    _assert_flags(good_catalog, substrings=["workloads/chatbot.yaml", "prompt.tokens_min"])


def test_workloads_negative_value_fails(good_catalog):
    wl = good_catalog / "workloads/chatbot.yaml"
    data = yaml.safe_load(wl.read_text())
    data["output"]["tokens_min"] = -1
    _write_yaml(wl, data)
    _assert_flags(good_catalog, substrings=["workloads/chatbot.yaml", "output.tokens_min"])


def test_workloads_mean_below_min_without_max_fails(good_catalog):
    # One-sided bound: tokens_max is optional, but a mean below tokens_min must
    # still be flagged (min <= mean holds independently of max).
    wl = good_catalog / "workloads/chatbot.yaml"
    data = yaml.safe_load(wl.read_text())
    del data["prompt"]["tokens_max"]
    data["prompt"]["tokens"] = 1  # below tokens_min 2
    _write_yaml(wl, data)
    _assert_flags(good_catalog, substrings=["workloads/chatbot.yaml", "prompt.tokens"])


def test_workloads_mean_above_max_without_min_fails(good_catalog):
    # The mirror one-sided bound: tokens_min is optional, but a mean above
    # tokens_max must still be flagged (mean <= max holds independently of min).
    wl = good_catalog / "workloads/chatbot.yaml"
    data = yaml.safe_load(wl.read_text())
    del data["output"]["tokens_min"]
    data["output"]["tokens"] = 2048  # above tokens_max 1024
    _write_yaml(wl, data)
    _assert_flags(good_catalog, substrings=["workloads/chatbot.yaml", "output.tokens"])


@pytest.mark.parametrize("value", [-1, "oops"])
def test_workloads_invalid_prefix_tokens_fails(good_catalog, value):
    # prefix_tokens is optional, but when present it must be a non-negative number.
    wl = good_catalog / "workloads/chatbot.yaml"
    data = yaml.safe_load(wl.read_text())
    data["prefix_tokens"] = value
    _write_yaml(wl, data)
    _assert_flags(good_catalog, substrings=["workloads/chatbot.yaml", "prefix_tokens"])


# --------------------------------------------------------------------------- #
# devices/: tier shape and positivity                                         #
# --------------------------------------------------------------------------- #


def test_devices_non_positive_bandwidth_fails(good_catalog):
    dev = good_catalog / "devices/storage.yaml"
    data = yaml.safe_load(dev.read_text())
    data["nvme_gen4"]["read_bandwidth"] = 0
    _write_yaml(dev, data)
    _assert_flags(good_catalog, substrings=["devices/storage.yaml", "nvme_gen4.read_bandwidth"])


def test_devices_missing_field_fails(good_catalog):
    dev = good_catalog / "devices/storage.yaml"
    data = yaml.safe_load(dev.read_text())
    del data["cpu_dram"]["base_latency"]
    _write_yaml(dev, data)
    _assert_flags(good_catalog, substrings=["devices/storage.yaml", "cpu_dram.base_latency"])


# --------------------------------------------------------------------------- #
# Robustness: empty documents, non-finite numbers, hidden fields, bad bytes    #
# (qa-review PR #11: F1 / G2 / G5 / G6 / G16)                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "rel",
    ["hardware/h100.yaml", "networks/ib-400g.yaml", "workloads/chatbot.yaml",
     "devices/storage.yaml", "models/qwen3-14b/model.yaml"],
)
@pytest.mark.parametrize("content", ["", "# only a comment\n", "null\n"])
def test_empty_or_comment_only_yaml_is_flagged_not_skipped(good_catalog, rel, content):
    (good_catalog / rel).write_text(content, encoding="utf-8")
    _assert_flags(good_catalog, substrings=[rel])


@pytest.mark.parametrize("bad", [".nan", ".inf", "-.inf", '"nan"', '"inf"'])
def test_workloads_non_finite_value_fails(good_catalog, bad):
    wl = good_catalog / "workloads/chatbot.yaml"
    # Write the nested shape raw so YAML resolves `.nan`/`.inf` as the float the Go
    # loader would read (safe_dump would quote a Python float); the bad value sits in
    # the nested prompt.tokens_stdev.
    wl.write_text(
        "prefix_tokens: 0\n"
        "prompt:\n"
        "  tokens: 256\n"
        f"  tokens_stdev: {bad}\n"
        "  tokens_min: 2\n"
        "  tokens_max: 800\n"
        "output:\n"
        "  tokens: 256\n"
        "  tokens_stdev: 100\n"
        "  tokens_min: 1\n"
        "  tokens_max: 1024\n",
        encoding="utf-8",
    )
    _assert_flags(good_catalog, substrings=["workloads/chatbot.yaml", "prompt.tokens_stdev"])


def test_networks_non_finite_bandwidth_fails(good_catalog):
    net = good_catalog / "networks/ib-400g.yaml"
    lines = [ln for ln in net.read_text().splitlines() if not ln.startswith("InterNodeBwGBps")]
    lines.append("InterNodeBwGBps: .nan")
    net.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _assert_flags(good_catalog, substrings=["networks/ib-400g.yaml", "InterNodeBwGBps"])


def test_coerce_number_rejects_non_finite():
    assert vc._coerce_number(float("nan")) is None
    assert vc._coerce_number(float("inf")) is None
    assert vc._coerce_number(".nan") is None
    assert vc._coerce_number("inf") is None
    assert vc._coerce_number("7.0e3") == 7000.0  # the real-data coercion still works


@pytest.mark.parametrize("value", [0.85, "0.85"])
def test_hardware_non_comment_underscore_key_is_flagged(good_catalog, value):
    # a dimensionless factor hiding under an underscore — numeric OR string-numeric.
    # Only `_comment*` is exempt; every other underscore key is a data field.
    hw = good_catalog / "hardware/h100.yaml"
    data = yaml.safe_load(hw.read_text())
    data["_mfu"] = value
    _write_yaml(hw, data)
    _assert_flags(good_catalog, substrings=["hardware/h100.yaml", "_mfu"])


def test_hardware_non_string_comment_value_is_flagged(good_catalog):
    hw = good_catalog / "hardware/h100.yaml"
    data = yaml.safe_load(hw.read_text())
    data["_comment_bad"] = 5  # a _comment* key must be prose (a string)
    _write_yaml(hw, data)
    _assert_flags(good_catalog, substrings=["hardware/h100.yaml", "_comment_bad"])


def test_hardware_structured_underscore_key_is_flagged(good_catalog):
    hw = good_catalog / "hardware/h100.yaml"
    data = yaml.safe_load(hw.read_text())
    data["_payload"] = {"factor": 0.85}
    _write_yaml(hw, data)
    _assert_flags(good_catalog, substrings=["hardware/h100.yaml", "_payload"])


def test_hardware_string_comment_key_still_exempt(good_catalog):
    hw = good_catalog / "hardware/h100.yaml"
    data = yaml.safe_load(hw.read_text())
    data["_comment_extra"] = "prose stays exempt"
    _write_yaml(hw, data)
    assert vc.validate_catalog(good_catalog) == []


def test_non_utf8_config_is_reported_not_crashed(good_catalog):
    (good_catalog / "models/qwen3-14b/config.json").write_bytes(b"\xff\xfe not utf-8")
    # must not raise; must name the file
    errors = vc.validate_catalog(good_catalog)
    joined = "\n".join(errors)
    assert "models/qwen3-14b/config.json" in joined


@pytest.mark.parametrize("name", ["FooGBCount", "TotalItems", "SlotUsPercent", "NumCPUs"])
def test_field_has_unit_rejects_midword_lookalikes(name):
    # a unit token buried mid-word (not a trailing segment) is NOT a real unit
    assert not vc.field_has_unit(name)


@pytest.mark.parametrize("name", ["MemoryGB", "PeakTBps", "ClockMHz", "TdpMilliWatts",
                                  "GFlopsPeak", "PFlopsPeak"])
def test_field_has_unit_accepts_more_suffixes(name):
    assert vc.field_has_unit(name)


# --- compute-token hole (susiejojo review on blis-catalog#11) --------------- #


@pytest.mark.parametrize(
    "name",
    ["FlopsUtilization", "PeakFlopsRatio", "FlopsEfficiency", "mfuFlops",
     "TFlopsRatio", "TFlopsUtilisation", "BwPeakTBsRatio", "MemoryGiBFraction"],
)
def test_field_has_unit_rejects_flops_and_unit_ratios(name):
    # bare/mid-word Flops without a scale prefix, and any name carrying a
    # dimensionless descriptor even with a real unit token, must be rejected.
    assert not vc.field_has_unit(name)


def test_hardware_flops_utilisation_field_is_flagged(good_catalog):
    hw = good_catalog / "hardware/h100.yaml"
    data = yaml.safe_load(hw.read_text())
    data["FlopsUtilization"] = 0.85  # a learned utilisation carrying "Flops"
    _write_yaml(hw, data)
    _assert_flags(good_catalog, substrings=["hardware/h100.yaml", "FlopsUtilization", "dimensionless"])


# --- networks closed schema (susiejojo review on blis-catalog#11) ----------- #


def test_networks_unknown_field_is_flagged(good_catalog):
    net = good_catalog / "networks/ib-400g.yaml"
    data = yaml.safe_load(net.read_text())
    data["DerateFactor"] = 0.85  # a fitted correction has no place in the catalog
    _write_yaml(net, data)
    _assert_flags(good_catalog, substrings=["networks/ib-400g.yaml", "DerateFactor", "unknown field"])


def test_networks_pd_transfer_base_latency_is_now_rejected(good_catalog):
    # blis-catalog#12 dropped PDTransferBaseLatencyMs from the catalog: the nominal
    # fabric base latency is always 0, and the 0.05 ms --pd-transfer-base-latency
    # modeling placeholder is a registry number (blis-registry#10). The closed fabric
    # schema now rejects the field outright, so a stale entry still carrying it fails.
    net = good_catalog / "networks/ib-400g.yaml"
    data = yaml.safe_load(net.read_text())
    data["PDTransferBaseLatencyMs"] = 0
    _write_yaml(net, data)
    _assert_flags(
        good_catalog,
        substrings=["networks/ib-400g.yaml", "PDTransferBaseLatencyMs", "unknown field"],
    )


# --- duplicate keys + root guard (susiejojo re-review on blis-catalog#11) --- #


def test_duplicate_key_in_yaml_is_rejected(good_catalog):
    # PyYAML would silently keep the last value; a malformed dup must fail.
    net = good_catalog / "networks/ib-400g.yaml"
    net.write_text(
        "Provenance: vendor-spec\nProvenance: vendor_spec\n"
        "InterNodeBwGBps: 50\n",
        encoding="utf-8",
    )
    _assert_flags(good_catalog, substrings=["networks/ib-400g.yaml", "duplicate key"])


def test_nonexistent_root_is_rejected():
    errors = vc.validate_catalog("/path/that/does/not/exist")
    assert errors and "does not exist" in "\n".join(errors)


def test_directory_that_is_not_a_catalog_is_rejected(tmp_path):
    (tmp_path / "unrelated.txt").write_text("hi", encoding="utf-8")
    errors = vc.validate_catalog(tmp_path)
    assert errors and "not a catalog root" in "\n".join(errors)


def test_networks_comment_key_allowed_but_must_be_string(good_catalog):
    net = good_catalog / "networks/ib-400g.yaml"
    data = yaml.safe_load(net.read_text())
    data["_comment_note"] = "prose is fine"
    _write_yaml(net, data)
    assert vc.validate_catalog(good_catalog) == []
    data["_comment_note"] = 5  # a numeric _comment is not prose
    _write_yaml(net, data)
    _assert_flags(good_catalog, substrings=["networks/ib-400g.yaml", "_comment_note"])


# --------------------------------------------------------------------------- #
# Counts: a declared number of physical parts is admitted; a fitted factor is  #
# not, even when it borrows a count-shaped name.                              #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ["SMCount", "GPUsPerNode", "GPUsPerRack"])
def test_declared_part_counts_are_admitted(name):
    assert vc.field_has_unit(name)


@pytest.mark.parametrize(
    "name",
    ["SMCountRatio", "GPUsPerNodeFraction", "ScalingFactor", "achieved", "NodeCount"],
)
def test_a_count_shaped_name_outside_the_closed_list_is_rejected(name):
    """The count vocabulary is a closed list, not a suffix rule: a name that
    merely ends in the word does not become a declared part count."""
    assert not vc.field_has_unit(name)


@pytest.mark.parametrize("value", [131.5, 0, -8, 0.91])
def test_a_count_that_is_not_a_whole_number_of_parts_is_flagged(good_catalog, value):
    chip = good_catalog / "hardware/h100.yaml"
    data = yaml.safe_load(chip.read_text())
    data["SMCount"] = value
    _write_yaml(chip, data)
    _assert_flags(good_catalog, substrings=["hardware/h100.yaml", "whole"])


def test_a_whole_part_count_passes_the_gate(good_catalog):
    chip = good_catalog / "hardware/h100.yaml"
    data = yaml.safe_load(chip.read_text())
    data["SMCount"] = 132
    data["GPUsPerNode"] = 8
    _write_yaml(chip, data)
    assert vc.validate_catalog(good_catalog) == []


def test_the_real_catalog_declares_an_sm_count_for_every_chip():
    """Every chip needs an SM count: the cost model's 12-SM Triton-fetch derate is a
    fraction of it, and a chip missing it cannot price an offload run.

    GPUsPerNode is deliberately NOT required here. How many GPUs sit in a node is a
    property of the deployment, not of the silicon — the same H200 ships on 4- and
    8-GPU baseboards and in PCIe chassis of several widths — and the value the cost
    model actually reads comes from the scenario's cluster block
    (blis-latency-kernel/internal/resolve/layout.go reads s.Cluster.GPUsPerNode, and
    nothing reads Chip.GPUsPerNode). A chip-level copy would be a default that
    disagrees silently with the run it is pricing.

    GB200-NVL72 keeps it, because there a tray boundary IS a hardware fact: its node
    is a Grace-Blackwell tray of 4 GPUs inside a 72-GPU NVLink domain, and
    blis-schemas checks GPUsPerRack against it."""
    root = Path(__file__).resolve().parent.parent
    chips = sorted((root / "hardware").glob("*.yaml"))
    assert chips, "no chips found to check"
    for chip in chips:
        data = yaml.safe_load(chip.read_text())
        assert data.get("SMCount", 0) > 0, f"{chip.name} declares no SMCount"
        if data.get("GPUsPerRack", 0) > 0:
            assert data.get("GPUsPerNode", 0) > 0, (
                f"{chip.name} declares GPUsPerRack but no GPUsPerNode; the rack tier "
                f"is expressed as a multiple of the node tier"
            )


def test_every_sm_count_cites_a_source():
    """An SMCount must be traceable, because it is the one physical figure on these
    chips that no vendor datasheet always publishes.

    The five Hopper/Ampere/Ada entries cite an NVIDIA datasheet URL. The three
    Blackwell parts cannot — NVIDIA publishes no SM count for them, and the AISimulate
    and InferenceX descriptors that source their FLOPs and HBM carry none either — so
    they cite the next most authoritative public source instead. Either way the
    requirement is the same: a reader must be able to chase the number. Review of this
    catalog's first Blackwell entries found 148 asserted with no source at all, which
    is the state this test exists to prevent recurring.

    Deliberately a URL check rather than a wording check: it is the weakest assertion
    that still cannot pass on an uncited figure."""
    root = Path(__file__).resolve().parent.parent
    chips = sorted((root / "hardware").glob("*.yaml"))
    assert chips, "no chips found to check"
    uncited = []
    for chip in chips:
        data = yaml.safe_load(chip.read_text())
        if not data.get("SMCount"):
            continue
        if "http" not in str(data.get("_comment_sm", "")):
            uncited.append(chip.name)
    assert not uncited, (
        f"these chips state an SMCount with no source to chase in _comment_sm: "
        f"{uncited}. Cite the NVIDIA datasheet where one publishes the count, and "
        f"otherwise the most authoritative public source, saying which it is."
    )


def test_an_sm_count_with_no_citation_is_caught(tmp_path):
    """The negative case for the test above: a check that cannot fail proves nothing."""
    chip = tmp_path / "hardware" / "fictional.yaml"
    chip.parent.mkdir(parents=True)
    chip.write_text(yaml.safe_dump({
        "Provenance": "vendor_spec",
        "_comment_sm": "SMCount is the enabled SM count on this part.",
        "SMCount": 148,
    }))
    data = yaml.safe_load(chip.read_text())
    assert "http" not in str(data.get("_comment_sm", "")), (
        "the uncited fixture must be uncited, or the positive test above is vacuous"
    )
