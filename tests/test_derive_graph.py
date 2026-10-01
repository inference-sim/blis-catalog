"""Tests for the ModelGraph deriver.

The deriver's output is committed, so a reader inspects graph.yaml rather than rerunning
the translation. That makes two failure modes worth guarding. A graph can be wrong in a
way every per-model check still passes — the layer count right, the parameters positive,
and the layer body copied from a different architecture. And a graph can be edited by
hand, which silently decouples it from the config it claims to come from.
"""

from __future__ import annotations

import collections
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
DERIVER = ROOT / "scripts" / "derive_graph.py"


def model_dirs() -> list[Path]:
    return sorted(p for p in MODELS.iterdir() if p.is_dir() and (p / "config.json").is_file())


def graph_of(d: Path) -> dict:
    return yaml.safe_load((d / "graph.yaml").read_text())


def config_of(d: Path) -> dict:
    return json.loads((d / "config.json").read_text())


def text_config(raw: dict) -> dict:
    return raw.get("text_config", raw)


def layer_sequence(stack: dict) -> list[str]:
    return (list(stack.get("prologue", []))
            + list(stack.get("pattern", [])) * stack.get("repeat", 0)
            + list(stack.get("epilogue", [])))


def test_every_model_has_a_graph():
    missing = [d.name for d in model_dirs() if not (d / "graph.yaml").is_file()]
    assert not missing, f"no derived graph for: {missing}"


def test_rederivation_is_stable():
    """--check must pass: a committed graph equals a fresh derivation.

    This is what makes the committed file trustworthy. It fails if someone edits a graph
    by hand, or changes a config without re-deriving."""
    r = subprocess.run([sys.executable, str(DERIVER), str(ROOT), "--check"],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"re-derivation differs:\n{r.stderr}"


def test_hand_edit_is_detected(tmp_path):
    """The negative case for the check above: a modified graph must be reported.

    A check that cannot fail proves nothing, so this mutates a copy of the tree and
    asserts the deriver notices."""
    import shutil

    tree = tmp_path / "catalog"
    shutil.copytree(ROOT, tree, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    target = next(p for p in (tree / "models").iterdir() if (p / "graph.yaml").is_file())
    g = yaml.safe_load((target / "graph.yaml").read_text())
    g["global"]["hidden_size"] = int(g["global"]["hidden_size"]) + 8
    (target / "graph.yaml").write_text(yaml.safe_dump(g, sort_keys=False))

    r = subprocess.run([sys.executable, str(DERIVER), str(tree), "--check"],
                       capture_output=True, text=True)
    assert r.returncode != 0, "a hand-edited graph was not detected"
    assert target.name in r.stderr


@pytest.mark.parametrize("d", model_dirs(), ids=lambda d: d.name)
def test_layer_count_matches_config(d: Path):
    """Depth must agree with the config, whichever way the config states it.

    Most configs state num_hidden_layers. One shipped hybrid states no layer count at
    all, and its depth is the length of a per-layer type vector."""
    g, t = graph_of(d), text_config(config_of(d))
    derived = len(layer_sequence(g["stack"]))
    stated = t.get("num_hidden_layers")
    vector = t.get("layers_block_type")
    expected = stated if stated is not None else (len(vector) if vector else None)
    if expected is None:
        pytest.skip("the config states neither a layer count nor a layer vector")
    assert derived == expected


@pytest.mark.parametrize("d", model_dirs(), ids=lambda d: d.name)
def test_digest_pins_the_config(d: Path):
    """The recorded digest must be the config's, so a drift between them is detectable."""
    import hashlib

    g = graph_of(d)
    actual = hashlib.sha256((d / "config.json").read_bytes()).hexdigest()
    assert g["derived_from"]["sha256"] == actual


@pytest.mark.parametrize("d", model_dirs(), ids=lambda d: d.name)
def test_attention_shape_matches_config(d: Path):
    """Head counts must come from the config rather than a template.

    A graph whose layer body was copied from another architecture would pass a layer-count
    check and fail here."""
    g, t = graph_of(d), text_config(config_of(d))
    nodes = [n for k in g["layer_kinds"] for n in k["nodes"] if n["op"] == "Attention"]
    if not nodes:
        pytest.skip("no attention layer")
    for n in nodes:
        assert n["n_q"] == t["num_attention_heads"]
        if n["kind"] in ("mla", "sparse_mla"):
            # A latent cache holds one vector per token whatever the config's KV head
            # count says, so the engine pins it to one.
            assert n["n_kv"] == 1
        else:
            assert n["n_kv"] == t["num_key_value_heads"]


@pytest.mark.parametrize("d", model_dirs(), ids=lambda d: d.name)
def test_expert_shape_matches_config(d: Path):
    """Expert count and top-k must come from the config, under whichever alias."""
    g, t = graph_of(d), text_config(config_of(d))
    nodes = [n for k in g["layer_kinds"] for n in k["nodes"] if n["op"] == "GroupedGEMM"]
    experts = (t.get("num_local_experts") or t.get("n_routed_experts")
               or t.get("num_experts"))
    if not nodes:
        assert not experts, "the config declares experts but the graph has no expert layer"
        return
    top_k = t.get("num_experts_per_tok") or t.get("num_experts_per_token")
    for n in nodes:
        assert n["experts"] == experts
        assert n["top_k"] == top_k
        assert n["top_k"] <= n["experts"]


# Architecture families expected to produce identical layer structures, with the reason.
# Every other pair of models must differ: a deriver that emitted one template for
# everything would pass every per-model check above.
EXPECTED_IDENTICAL = {
    frozenset({"nemotron-3-ultra-550b-a55b-bf16", "nemotron-3-ultra-550b-a55b-nvfp4"}):
        "the same architecture at two weight dtypes",
    frozenset({"nemotron-3.5-lightning-30b-a3b-bf16",
               "nemotron-3.5-lightning-30b-a3b-nvfp4"}):
        "the same architecture at two weight dtypes",
    frozenset({"glm-5", "glm-5.2", "glm-5.2-fp8", "glm-5.3"}):
        "one architecture across three GLM-5 generations. Every cost-relevant field is "
        "identical: 78 layers, hidden 6144, 256 experts at top-8, moe_intermediate 2048, "
        "3 leading dense layers, vocab 154880, and the same sparse-MLA geometry. They "
        "differ in quantization_config, rope theta, context length, transformers_version "
        "and — between glm-5 and glm-5.3 — in head_dim, 64 against 192. That last one "
        "looks like it should matter and does not: an MLA layer's per-head width is "
        "kv_lora_rank + qk_rope_head_dim, which both state as 512 + 64, so head_dim is "
        "not a term the cost model reads for this attention kind",
    frozenset({"llama-2-70b-hf", "llama-3.1-70b-instruct"}):
        "identical compute shapes; they differ in vocab, context length and rope theta",
}


def cost_signature(g: dict) -> tuple:
    """What a graph costs to run, as a canonical multiset.

    This is deliberately a BEHAVIOURAL signature rather than a structural one. Two graphs
    collide here only if they price identically: the same work, the same number of times,
    on the same shapes. It ignores everything a cost model does not read — role labels,
    layer-kind identifiers, node ordering within a layer, and how the stack is spelled
    (a prologue-plus-pattern and a literal sequence that expand to the same layers are the
    same deployment).

    A structural comparison would report a difference when a deriver renamed a role or
    reordered two independent nodes, neither of which changes a single step time. It would
    also report sameness for two graphs whose stacks happen to have matching patterns but
    different expansions."""
    kinds = {k["id"]: k for k in g["layer_kinds"]}
    work: collections.Counter = collections.Counter()
    for layer_id in layer_sequence(g["stack"]):
        for n in kinds[layer_id]["nodes"]:
            # The parameters that change what a primitive costs, and nothing else. `when`
            # is excluded: it selects whether a node exists in a given deployment, which is
            # a property of the layout rather than of the model.
            work[(
                n["op"], n.get("n"), n.get("k"), n.get("n_q"), n.get("n_kv"), n.get("d_h"),
                n.get("experts"), n.get("top_k"), n.get("shared_experts"),
                n.get("shared_intermediate_size"), n.get("latent_size"), n.get("kind"),
                n.get("recurrent_kind"), n.get("state_size"), n.get("n_heads"),
                n.get("n_groups"), n.get("conv_kernel"), n.get("intermediate_size"),
                n.get("window"), n.get("index_topk"),
            )] += 1
    return tuple(sorted(work.items(), key=repr))


def test_distinct_architectures_cost_differently():
    """Only the documented groups may price identically.

    Without this, a deriver bug that applied one architecture's template to every model
    would leave every other test passing: the layer counts would still match, the
    parameters would still be positive, and the graphs would still validate."""
    groups = collections.defaultdict(list)
    for d in model_dirs():
        groups[cost_signature(graph_of(d))].append(d.name)

    unexplained = []
    for names in groups.values():
        if len(names) < 2:
            continue
        if frozenset(names) not in EXPECTED_IDENTICAL:
            unexplained.append(sorted(names))
    assert not unexplained, (
        "these models price identically and are not a documented family; "
        f"one of each group is likely derived from the wrong template: {unexplained}")


def test_expected_identical_groups_still_cost_the_same():
    """The exemptions above must stay true.

    If two models in a documented group diverge, either a config changed or the deriver
    started treating them differently — both worth knowing."""
    by_name = {d.name: cost_signature(graph_of(d)) for d in model_dirs()}
    for group, reason in EXPECTED_IDENTICAL.items():
        present = [n for n in group if n in by_name]
        if len(present) < 2:
            continue
        first = by_name[present[0]]
        for other in present[1:]:
            assert by_name[other] == first, (
                f"{present[0]} and {other} were expected to price identically "
                f"({reason}) and no longer do")


def test_weight_dtypes_vary_across_the_catalog():
    """A deriver that defaulted every dtype would pass the per-model checks."""
    seen = {graph_of(d)["global"]["weight_dtype"] for d in model_dirs()}
    assert len(seen) >= 4, f"only {seen} appear; a default is suspected"
    # mxfp4 and nvfp4 are distinct formats, not spellings of one another, and both are in
    # the catalog. Collapsing them would misprice one.
    assert {"mxfp4", "nvfp4"} <= seen, f"expected both 4-bit formats, got {seen}"


# --- The signature's own behaviour -------------------------------------------------
# cost_signature underpins the two collision tests above, so its sensitivity is worth
# establishing rather than assuming. It must ignore changes that cost nothing and notice
# changes that cost something; a signature that failed the first half would report
# differences a cost model cannot observe, and one that failed the second half would let a
# wrong graph pass.

def a_reference_graph() -> dict:
    return graph_of(next(d for d in model_dirs() if d.name == "llama-3.1-8b-instruct"))


@pytest.mark.parametrize("mutate,description", [
    (lambda g: [n.update(role="renamed") for k in g["layer_kinds"] for n in k["nodes"]],
     "renaming every role"),
    (lambda g: g["layer_kinds"][0]["nodes"].reverse(),
     "reordering nodes within a layer"),
    (lambda g: g.update(head=[]),
     "dropping the head, which is priced separately from the stack"),
])
def test_signature_ignores_changes_that_cost_nothing(mutate, description):
    import copy

    g = a_reference_graph()
    before = cost_signature(g)
    variant = copy.deepcopy(g)
    mutate(variant)
    assert cost_signature(variant) == before, (
        f"{description} changed the cost signature, but a cost model cannot observe it")


def test_signature_ignores_how_the_stack_is_spelled():
    """A prologue-plus-pattern and a literal sequence that expand alike are one deployment.

    The deriver compresses a uniform run into a pattern where it can. That is a storage
    choice, so two graphs differing only in the compression must compare equal."""
    import copy

    g = a_reference_graph()
    literal = copy.deepcopy(g)
    literal["stack"] = {"prologue": layer_sequence(g["stack"])}
    assert cost_signature(literal) == cost_signature(g)


def test_signature_ignores_layer_kind_identifiers():
    import copy

    g = a_reference_graph()
    renamed = copy.deepcopy(g)
    old = renamed["layer_kinds"][0]["id"]
    renamed["layer_kinds"][0]["id"] = "some-other-name"
    for field in ("prologue", "pattern", "epilogue"):
        if field in renamed["stack"]:
            renamed["stack"][field] = ["some-other-name" if x == old else x
                                       for x in renamed["stack"][field]]
    assert cost_signature(renamed) == cost_signature(g)


@pytest.mark.parametrize("mutate,description", [
    (lambda g: _widen_first_gemm(g), "widening one projection"),
    (lambda g: g["stack"].update(repeat=g["stack"]["repeat"] - 1), "dropping a layer"),
    (lambda g: _change_kv_heads(g), "changing the KV head count"),
])
def test_signature_notices_changes_that_cost_something(mutate, description):
    import copy

    g = a_reference_graph()
    before = cost_signature(g)
    variant = copy.deepcopy(g)
    mutate(variant)
    assert cost_signature(variant) != before, (
        f"{description} left the cost signature unchanged, so the collision tests would "
        f"not catch it")


def _widen_first_gemm(g: dict) -> None:
    for k in g["layer_kinds"]:
        for n in k["nodes"]:
            if n["op"] == "GEMM":
                n["n"] += 64
                return
    pytest.skip("no GEMM node to widen")


def _change_kv_heads(g: dict) -> None:
    for k in g["layer_kinds"]:
        for n in k["nodes"]:
            if n["op"] == "Attention":
                n["n_kv"] = max(1, n["n_kv"] // 2)
                return
    pytest.skip("no attention node")
