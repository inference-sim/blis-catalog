"""Tests for the documentation page generator, scripts/docs/catalog_pages.py.

The generator's job is to restate the catalog's data without changing it, so these tests
compare what it renders against the files themselves. They need PyYAML and nothing from
MkDocs.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "docs"))
import catalog_pages as pages  # noqa: E402


@pytest.fixture(scope="module")
def cat():
    return pages.load(ROOT)


def test_loads_every_entry_on_disk(cat):
    assert [m.name for m in cat.models] == sorted(
        p.name for p in (ROOT / "models").iterdir() if p.is_dir())
    assert sorted(cat.chips) == sorted(p.stem for p in (ROOT / "hardware").glob("*.yaml"))
    assert sorted(cat.fabrics) == sorted(p.stem for p in (ROOT / "networks").glob("*.yaml"))
    assert sorted(cat.workloads) == sorted(p.stem for p in (ROOT / "workloads").glob("*.yaml"))


def test_reads_pins_from_the_files_that_own_them(cat):
    ci = (ROOT / ".github/workflows/ci.yml").read_text()
    assert f"cmd/validate-catalog@{cat.validate_pin} " in ci
    deriver = (ROOT / "scripts/derive_graph.py").read_text()
    assert f"\nDERIVER_VERSION = {cat.deriver_version}\n" in deriver


def test_release_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("CATALOG_RELEASE", "9.8.7")
    cat = pages.load(ROOT)
    assert (cat.release, cat.ref) == ("9.8.7", "9.8.7")
    assert "releases/tag/9.8.7" in pages.release_text(cat)


def test_empty_release_means_main(monkeypatch):
    monkeypatch.setenv("CATALOG_RELEASE", " ")
    cat = pages.load(ROOT)
    assert (cat.release, cat.ref) == (None, "main")


def test_malformed_release_fails(monkeypatch):
    monkeypatch.setenv("CATALOG_RELEASE", "v0.2.1")
    with pytest.raises(RuntimeError, match="MAJOR.MINOR.PATCH"):
        pages.load(ROOT)


@pytest.mark.parametrize("text,href", [
    ("see https://example.com/a.", "https://example.com/a"),
    ("'https://example.com/a'", "https://example.com/a"),
    ('"https://example.com/?c=1&d=2"', "https://example.com/?c=1&amp;d=2"),
    ("(https://en.wikipedia.org/wiki/Foo_(bar))", "https://en.wikipedia.org/wiki/Foo_(bar)"),
    ("(see https://example.com/x)", "https://example.com/x"),
])
def test_linkify_keeps_urls_whole(text, href):
    assert f'href="{href}"' in pages.linkify(text)


def test_summary_columns_ignore_speculator_only_kinds(cat):
    # DeepSeek-V4-Pro's MLA layer exists only in its draft module.
    v4 = next(m for m in cat.models if m.name == "deepseek-v4-pro")
    assert "mla" not in v4.attention_kinds()
    assert "sparse_mla" in v4.attention_kinds()


def test_layer_expansion_matches_every_graph(cat):
    # The expansion follows blis-schemas' Stack.Layers: prologue, pattern x repeat,
    # epilogue. Every expanded entry must name a kind the graph declares.
    for m in cat.models:
        s = m.graph["stack"]
        expected = (len(s.get("prologue", [])) + len(s.get("pattern", [])) * s.get("repeat", 0)
                    + len(s.get("epilogue", [])))
        assert len(m.layers) == expected > 0, m.name
        assert set(m.layers) <= {k["id"] for k in m.graph["layer_kinds"]}, m.name


def test_storage_figures_are_numbers(cat):
    # PyYAML reads 7.0e3 as a string; the generator must not.
    for name, tier in cat.tiers.items():
        for key in ("read_bandwidth_mb_s", "write_bandwidth_mb_s", "base_latency_us"):
            assert isinstance(tier[key], float), (name, key)
    assert cat.tiers["nvme_gen4"]["read_bandwidth_mb_s"] == 7000.0


@pytest.mark.parametrize("value,expected", [
    (989.5, "989.5"),
    (2500.0, "2 500"),
    (80, "80"),
    (0.864, "0.864"),
    (None, "—"),
])
def test_num_states_figures_without_rounding(value, expected):
    assert pages.num(value) == expected


def test_every_model_page_renders_its_identity(cat):
    for m in cat.models:
        page = pages.model_page(cat, m)
        assert page.startswith("---\nhide:"), m.name
        assert f"# {m.name}\n" in page
        assert m.identity["source"]["revision"] in page
        assert m.graph["derived_from"]["sha256"] in page
        assert f"{len(m.layers)} layers" in page


def test_hardware_table_has_one_row_per_chip(cat):
    rows = pages.hardware_table(cat).splitlines()[2:]
    assert len(rows) == len(cat.chips)
    h100 = next(r for r in rows if r.startswith("| `h100`"))
    assert "989.5" in h100 and "3.35" in h100


def test_every_placeholder_in_the_docs_is_known(cat):
    frags = pages.fragments(cat)
    for md in (ROOT / "docs").rglob("*.md"):
        for escaped, name in pages.PLACEHOLDER.findall(md.read_text(encoding="utf-8")):
            assert escaped or name in frags, f"{md.relative_to(ROOT)}: [[catalog.{name}]]"


@pytest.mark.parametrize("text", ["[[catalog.nope]]", "[[catalog.Models.table]]",
                                  "[[catalog.models table]]"])
def test_unknown_or_misspelled_placeholder_fails(text):
    with pytest.raises(KeyError):
        pages.substitute(f"a {text} b", {"models.table": "x"}, "page.md")


def test_escaped_placeholder_renders_literally():
    assert pages.substitute("[[!catalog.NAME]]", {}, "p.md") == "[[catalog.NAME]]"


def test_svg_is_well_formed(cat):
    import xml.etree.ElementTree as ET

    for svg in (pages.hardware_figure(cat), pages.workloads_figure(cat),
                pages.layer_strip(cat.models[0])):
        ET.fromstring(svg)  # raises on malformed markup
        assert not re.search(r"\bnan\b|\binf\b", svg)
