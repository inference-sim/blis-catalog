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
        {"Provenance": "vendor_spec", "InterNodeBwGBps": 50, "PDTransferBaseLatencyMs": 0},
    )
    # workloads/chatbot
    _write_yaml(
        root / "workloads/chatbot.yaml",
        {
            "prefix_tokens": 0,
            "prompt_tokens": 256, "prompt_tokens_stdev": 100,
            "prompt_tokens_min": 2, "prompt_tokens_max": 800,
            "output_tokens": 256, "output_tokens_stdev": 100,
            "output_tokens_min": 1, "output_tokens_max": 1024,
        },
    )
    # workloads/multidoc — the preset that legitimately omits prefix_tokens
    _write_yaml(
        root / "workloads/multidoc.yaml",
        {
            "prompt_tokens": 10240, "prompt_tokens_stdev": 1200,
            "prompt_tokens_min": 500, "prompt_tokens_max": 20480,
            "output_tokens": 1536, "output_tokens_stdev": 300,
            "output_tokens_min": 50, "output_tokens_max": 4096,
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


@pytest.mark.parametrize("field", ["InterNodeBwGBps", "Provenance", "PDTransferBaseLatencyMs"])
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


def test_workloads_missing_required_field_fails(good_catalog):
    wl = good_catalog / "workloads/chatbot.yaml"
    data = yaml.safe_load(wl.read_text())
    del data["output_tokens"]
    _write_yaml(wl, data)
    _assert_flags(good_catalog, substrings=["workloads/chatbot.yaml", "output_tokens"])


def test_workloads_min_exceeds_max_fails(good_catalog):
    wl = good_catalog / "workloads/chatbot.yaml"
    data = yaml.safe_load(wl.read_text())
    data["prompt_tokens_min"] = 900  # > prompt_tokens_max 800
    _write_yaml(wl, data)
    _assert_flags(good_catalog, substrings=["workloads/chatbot.yaml", "prompt_tokens_min"])


def test_workloads_negative_value_fails(good_catalog):
    wl = good_catalog / "workloads/chatbot.yaml"
    data = yaml.safe_load(wl.read_text())
    data["output_tokens_min"] = -1
    _write_yaml(wl, data)
    _assert_flags(good_catalog, substrings=["workloads/chatbot.yaml", "output_tokens_min"])


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
