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
        if n.get("role") == "block_index_scores":
            # A block-sparse layer runs a second, much narrower attention to score which
            # blocks the first one reads. Its head geometry is the indexer's own, not the
            # model's, so asserting the model's head count here would require the deriver
            # to misreport it. Checked against the config it does come from instead.
            #
            # Two families spell those fields differently: MiniMax-M3 nests them in
            # sparse_attention_config, DeepSeek-V4 states index_n_heads and
            # index_head_dim at the top level. Both are read, and a config declaring
            # neither fails rather than skipping the check.
            sparse = t.get("sparse_attention_config") or {}
            heads = sparse.get("sparse_num_index_heads", t.get("index_n_heads"))
            dim = sparse.get("sparse_index_dim", t.get("index_head_dim"))
            assert heads is not None and dim is not None, (
                "an indexer node whose config states no index head geometry"
            )
            assert n["n_q"] == heads
            assert n["d_h"] == dim
            continue
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
    frozenset({"minimax-m2.5", "minimax-m2.7"}):
        "one architecture across two MiniMax-M2 generations. Every cost-relevant field is "
        "identical: 62 layers, hidden 3072, 256 experts at top-8, intermediate 1536, 48 "
        "query heads over 8 KV heads at head_dim 128, vocab 200064, fp8 weights. The two "
        "configs differ only in max_position_embeddings (196608 against 204800) and a "
        "dtype label, neither of which the cost model reads",
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


# --- compress(): the stack's spelling -----------------------------------------------
# compress decides how a layer sequence is stored, and the deriver runs it on every
# model. test_signature_ignores_how_the_stack_is_spelled proves the choice is
# cost-neutral, which is the property a cost model cares about; these tests cover the
# other half, that the stored form is CORRECT and says what the sequence actually is.
# A reader inspects the committed stack to see a hybrid's period, so a stack that
# expands correctly but misreports the period is still wrong for the thing compression
# exists to do.


def _dg():
    """The deriver as a module, for the functions that take no catalog on disk."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("dg", DERIVER)
    dg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dg)
    return dg


@pytest.mark.parametrize("sequence,expected,description", [
    (["a"] * 8, {"pattern": ["a"], "repeat": 8},
     "a uniform run collapses to a period of one"),
    (["a", "b"] * 5, {"pattern": ["a", "b"], "repeat": 5},
     "a two-layer period is stored once"),
    (["d", "d", "d"] + ["s"] * 9,
     {"prologue": ["d", "d", "d"], "pattern": ["s"], "repeat": 9},
     "a dense prologue then a uniform sparse tail"),
    ((["k", "k", "k", "m"] * 23) + ["m"],
     {"pattern": ["k", "k", "k", "m"], "repeat": 23, "epilogue": ["m"]},
     "a hybrid period with a trailing odd layer"),
    (["a", "b", "c", "d", "e"], {"prologue": ["a", "b", "c", "d", "e"]},
     "a sequence with no repeating unit is stored literally"),
    (["a"], {"prologue": ["a"]}, "a single layer"),
    ([], {}, "an empty sequence"),
])
def test_compress_stores_the_sequence_it_was_given(sequence, expected, description):
    assert _dg().compress(list(sequence)) == expected, description


def test_compress_prefers_the_true_period_over_a_padded_one():
    """A period-1 tail must be stored as one entry, not as a multiple of itself.

    This is a regression test. The search used to reject any candidate whose prologue
    was longer than its period, which rules out the optimal split on the commonest
    shape in this catalog -- a short dense prologue then a uniform sparse tail -- and
    left a worse one to win. Four committed graphs (glm-5.2, glm-5.2-fp8, glm-5.3,
    minimax-m3) stored `pattern: [attn_moe, attn_moe, attn_moe], repeat: 25` for a
    75-layer run whose period is 1, which reads as a three-layer repeating unit the
    model does not have. It expanded correctly, so every cost-based test passed.

    The GLM shape itself, at its real depth."""
    got = _dg().compress(["attn_dense"] * 3 + ["attn_moe"] * 75)
    assert got == {"prologue": ["attn_dense"] * 3,
                   "pattern": ["attn_moe"], "repeat": 75}, (
        f"stored {got}, which spells a period-1 run as a longer unit")


def test_compress_declines_a_split_that_stores_no_less_than_the_sequence():
    """The docstring's readability rule: past a point the literal form is clearer.

    A sequence whose only repeating unit is surrounded by material longer than the
    saving must come back literal rather than as a split that technically compresses."""
    seq = ["a", "b"] * 3 + ["x", "y", "z"]
    got = _dg().compress(list(seq))
    assert got == {"prologue": list(seq)}, (
        f"stored {got}; a split saving nothing should fall back to the literal form")
    assert "pattern" not in got


@pytest.mark.parametrize("sequence", [
    ["a"] * 8,
    ["a", "b"] * 5,
    ["d", "d", "d"] + ["s"] * 9,
    (["k", "k", "k", "m"] * 23) + ["m"],
    ["m", "e", "m", "e", "m", "a", "e", "m", "e"],
    ["a", "b", "c", "d", "e"],
    ["p", "q", "r", "s", "t"] + ["a", "b"] * 3,
    ["a", "b"] * 3 + ["x", "y", "z"],
    ["a"],
])
def test_compress_round_trips(sequence):
    """However it is spelled, it must expand to what went in.

    The property that makes compression safe at all: layer_sequence is what every
    consumer reads the stack through, so a compression it cannot invert would misprice
    the model however well the stored form reads."""
    assert layer_sequence(_dg().compress(list(sequence))) == list(sequence)


def test_compress_round_trips_every_committed_graph():
    """The same property against the real catalog rather than hand-built shapes."""
    dg = _dg()
    for d in model_dirs():
        seq = layer_sequence(graph_of(d)["stack"])
        assert layer_sequence(dg.compress(list(seq))) == seq, d.name


# --- MiniMax-M3: block-sparse attention ------------------------------------------


def _m3():
    d = ROOT / "models" / "minimax-m3"
    if not d.is_dir():
        pytest.skip("minimax-m3 not in the catalog")
    return graph_of(d), text_config(config_of(d))


def test_m3_bounded_read_is_cheaper_than_full_attention_at_long_context():
    """The point of block-sparse attention is a read that stops growing with context.

    Asserted as a cost difference rather than a field value: a graph that recorded
    the bound but left the kind full would read the whole context, and only a
    comparison catches that. The bound is in tokens, so at any context beyond it the
    sparse layer reads strictly fewer KV bytes than a full-attention layer of the
    same shape.
    """
    g, t = _m3()
    main = [n for k in g["layer_kinds"] for n in k["nodes"]
            if n["op"] == "Attention" and n.get("role") != "block_index_scores"
            and n["kind"] == "swa"]
    assert main, "no bounded attention node; a sparse layer would read the full context"
    sparse = t["sparse_attention_config"]
    bound = (sparse["sparse_topk_blocks"] + sparse["sparse_local_block"]
             + sparse["sparse_init_block"]) * sparse["sparse_block_size"]
    for n in main:
        per_token = 2 * n["n_kv"] * n["d_h"]
        for context in (8192, 131072, 1048576):
            bounded = min(context, n["window"]) * per_token
            full = context * per_token
            assert bounded < full, (
                f"at context {context} the bounded read is not cheaper than a full one"
            )
            assert bounded == bound * per_token, (
                f"the bound should saturate at {bound} tokens, not track context"
            )


def test_m3_indexer_read_is_not_bounded():
    """The indexer scores every block to choose the top k, so its read tracks context.

    If the indexer were given the same bound as the attention it feeds, the model would
    charge nothing for selection and a long-context step would come out too fast. The
    distinction is the whole reason it is a separate node."""
    g, _ = _m3()
    idx = [n for k in g["layer_kinds"] for n in k["nodes"]
           if n.get("role") == "block_index_scores"]
    assert idx, "no indexer node; block selection would be free"
    for n in idx:
        assert "window" not in n, (
            "the indexer carries a window, which would bound a read that must scan the "
            "whole context to rank blocks"
        )
        assert n["kind"] == "gqa"


def test_m3_dense_prologue_is_wider_than_a_routed_expert():
    """The first three layers are dense MLPs at dense_intermediate_size.

    Both widths are declared and they differ by 4x, so a handler that reached for
    intermediate_size would make the prologue a quarter of its true cost. Asserted
    as the inequality the weights show rather than as the literal number."""
    g, t = _m3()
    kinds = {k["id"]: k for k in g["layer_kinds"]}
    dense_up = [n for n in kinds["dense"]["nodes"]
                if n.get("role") == "mlp_gate_up"][0]
    expert = [n for n in kinds["sparse_attn_moe"]["nodes"]
              if n["op"] == "GroupedGEMM"][0]
    assert dense_up["n"] == 2 * t["dense_intermediate_size"]
    assert expert["n"] == t["intermediate_size"]
    assert dense_up["n"] > 2 * expert["n"], (
        "the dense prologue should be wider than a routed expert on this config"
    )


def test_m3_prices_differently_from_its_gqa_sibling():
    """M2.5 and M3 are both MiniMax MoE models and must not price alike.

    M2.5 is dense-attention GQA throughout; M3 bounds its attention read and adds
    an indexer. A handler registered against the wrong architecture would collapse
    the two."""
    g3, _ = _m3()
    d2 = ROOT / "models" / "minimax-m2.5"
    if not d2.is_dir():
        pytest.skip("minimax-m2.5 not in the catalog")
    assert cost_signature(g3) != cost_signature(graph_of(d2))


# --- DeepSeek-V4-Pro: compressed sparse attention at two ratios -------------------


def _v4():
    d = ROOT / "models" / "deepseek-v4-pro"
    if not d.is_dir():
        pytest.skip("deepseek-v4-pro not in the catalog")
    return graph_of(d), text_config(config_of(d))


def test_v4_has_one_layer_kind_per_compression_ratio():
    """The config's compress_ratios vector names two ratios, and they cost differently.

    vLLM's compressor asserts compress_ratio in [4, 128] and sizes each layer's read
    from it, so a graph collapsing them to one kind would price 30 of 61 layers with
    the other one's bound. Asserted through the bound rather than the kind id: an id is
    a label, a window is a cost."""
    g, t = _v4()
    ratios = sorted(set(t["compress_ratios"][: t["num_hidden_layers"]]))
    assert len(ratios) == 2, f"expected two ratios, config states {ratios}"
    windows = set()
    for k in g["layer_kinds"]:
        for n in k["nodes"]:
            if n["op"] == "Attention" and n.get("role") != "block_index_scores":
                windows.add(n["window"])
    assert len(windows) == len(ratios), (
        f"two compression ratios must give two distinct read bounds, got {windows}"
    )


def test_v4_higher_compression_reads_less():
    """A larger compression ratio must bound the read tighter.

    This is the direction the measured tables show -- the ratio-128 kernel is flat in
    context where the ratio-4 one grows -- and a graph that inverted it would make the
    cheap layers expensive and vice versa, while still having two distinct kinds."""
    g, t = _v4()
    layers = t["num_hidden_layers"]
    by_ratio = {}
    seq = layer_sequence(g["stack"])
    kinds = {k["id"]: k for k in g["layer_kinds"]}
    for idx, ratio in enumerate(t["compress_ratios"][:layers]):
        k = kinds[seq[idx]]
        for n in k["nodes"]:
            if n["op"] == "Attention" and n.get("role") != "block_index_scores":
                by_ratio.setdefault(ratio, set()).add(n["window"])
    assert all(len(v) == 1 for v in by_ratio.values()), (
        f"a ratio maps to more than one bound: {by_ratio}"
    )
    flat = {r: v.pop() for r, v in by_ratio.items()}
    lo, hi = min(flat), max(flat)
    assert flat[hi] < flat[lo], (
        f"ratio {hi} should read less than ratio {lo}, got {flat[hi]} vs {flat[lo]}"
    )


def test_v4_head_dim_is_not_summed_with_rope():
    """head_dim is the inclusive per-token width, not the part outside RoPE.

    vLLM's compressor derives nope_head_dim = head_dim - rope_head_dim
    (models/deepseek_v4/compressor.py), so adding qk_rope_head_dim to head_dim would
    overstate this model's KV cache by an eighth. There is no kv_lora_rank to
    reconstruct it from, which is why the MLA branch cannot price this family."""
    g, t = _v4()
    main = [n for k in g["layer_kinds"] for n in k["nodes"]
            if n["op"] == "Attention" and n.get("role") != "block_index_scores"]
    assert main
    for n in main:
        assert n["d_h"] == t["head_dim"]
        assert n["d_h"] != t["head_dim"] + t["qk_rope_head_dim"]


def test_v4_indexer_is_unbounded_where_the_attention_is_bounded():
    """The measured split: the indexer grows 3.53x over a 12x context increase where the
    attention it feeds moves 1.18x. One node cannot carry both."""
    g, _ = _v4()
    for k in g["layer_kinds"]:
        idx = [n for n in k["nodes"] if n.get("role") == "block_index_scores"]
        main = [n for n in k["nodes"]
                if n["op"] == "Attention" and n.get("role") != "block_index_scores"]
        assert len(idx) == 1 and len(main) == 1, f"{k['id']}: expected one of each"
        assert "window" not in idx[0], "a bounded indexer would make selection free"
        assert main[0]["window"] > 0, "an unbounded read would track the full context"


def test_v4_prices_differently_from_deepseek_v3():
    """V3 is dense MLA with kv_lora_rank; V4 is compressed sparse with an indexer.

    Both are DeepSeek MoE models, so a handler registered against the wrong architecture
    would be easy to miss -- and sending V4 through handler_moe is precisely what the
    registry comment warns derives kind=swa window=128 off sliding_window."""
    g4, _ = _v4()
    d3 = ROOT / "models" / "deepseek-v3"
    if not d3.is_dir():
        pytest.skip("deepseek-v3 not in the catalog")
    assert cost_signature(g4) != cost_signature(graph_of(d3))


def test_v4_experts_are_priced_at_their_own_declared_width():
    """A checkpoint storing experts at a different width from the rest must say so.

    DeepSeek-V4-Pro declares expert_dtype fp4 beside an fp8 quantization_config, and
    vLLM resolves that name to MXFP4 experts (models/deepseek_v4/quant_config.py).
    Pricing them at the global fp8 width doubles 384 experts from 720 GiB to 1,441 GiB,
    which puts 180 GiB per rank on a 141 GiB H200 and makes a deployment InferenceX ran
    at tp=8 on 8 GPUs look impossible. Asserted through the occupancy as well as the
    field, because the field is a label and the bytes are the cost."""
    g, t = _v4()
    assert t.get("expert_dtype") == "fp4", "this test is about a config that states one"
    experts = [n for k in g["layer_kinds"] for n in k["nodes"]
               if n["op"] == "GroupedGEMM"]
    assert experts, "no expert node"
    for n in experts:
        assert n.get("weight_dtype") == "mxfp4", (
            f"expert node carries weight_dtype {n.get('weight_dtype')!r}; vLLM "
            f"resolves expert_dtype fp4 to mxfp4"
        )
    # The occupancy that width decides: 0.5 bytes per parameter, against fp8's 1.
    kinds = {k["id"]: k for k in g["layer_kinds"]}
    gib = 0.0
    for lid in layer_sequence(g["stack"]):
        for n in kinds[lid]["nodes"]:
            if n["op"] == "GroupedGEMM":
                gib += 3 * n["n"] * n["k"] * n["experts"] * 0.5 / 2**30
    per_rank = gib / 8
    assert per_rank < 141 * 0.9, (
        f"{per_rank:.1f} GiB per rank at tp=8 does not fit a 141 GiB H200, so this "
        f"deployment would be refused"
    )


def test_an_unmapped_expert_dtype_is_an_error_not_a_fallback():
    """Falling back to the global width is the bug this field exists to fix.

    A width the deriver does not recognize must stop the derivation rather than quietly
    pricing the experts at the checkpoint's other dtype."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "dg", ROOT / "scripts" / "derive_graph.py")
    dg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dg)
    cfg = {"hidden_size": 7168, "num_local_experts": 8, "num_experts_per_tok": 2,
           "intermediate_size": 1024, "expert_dtype": "fp6"}
    try:
        dg.moe(cfg, "synthetic", 7168)
    except dg.DeriveError as exc:
        assert "fp6" in str(exc)
        return
    raise AssertionError("an unmapped expert_dtype was accepted")
