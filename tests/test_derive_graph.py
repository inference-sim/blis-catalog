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


# --- The k-pool indexer, exercised directly -----------------------------------------
# No committed config on this branch declares index_kpool, so a catalog-driven test
# would skip and leave the production branch unrun in CI. These call the deriver's own
# helper against a config built from the published GLM-5.3-Flash values, so the branch
# that model will consume is executed here, with its negative case alongside it.

DERIVE = None


def deriver():
    """The deriver module, imported once for the direct-call tests."""
    global DERIVE
    if DERIVE is None:
        sys.path.insert(0, str(ROOT / "scripts"))
        import derive_graph
        DERIVE = derive_graph
    return DERIVE


def kpool_config(**over) -> dict:
    """A config carrying GLM-5.3-Flash's published indexer and latent geometry.

    Values are the pinned revision's (eb9eb208): hidden 4096, 64 heads, kv_lora_rank
    512, q_lora_rank 1536, qk_nope 256, v_head_dim 256, and the indexer's 32 heads of
    128 with index_topk 2048 and index_kpool 4.

    qk_rope_head_dim is 64 here where that config states 0. The published model declares
    mla_use_nope, and the branch that accepts a zero rope width on that declaration is
    the one adding the model -- not this one. These tests are about the indexer, which
    the rope width does not enter, so a nonzero value keeps them independent of that
    change rather than coupling this branch's CI to it."""
    cfg = {
        "hidden_size": 4096, "num_attention_heads": 64, "num_key_value_heads": 64,
        "kv_lora_rank": 512, "q_lora_rank": 1536,
        "qk_nope_head_dim": 256, "qk_rope_head_dim": 64, "v_head_dim": 256,
        "index_n_heads": 32, "index_head_dim": 128, "index_topk": 2048,
        "index_kpool": 4, "dtype": "bfloat16", "num_hidden_layers": 45,
    }
    cfg.update(over)
    return cfg


def indexer_nodes(cfg: dict) -> list[dict]:
    """Run attention_block + lightning_indexer, as a handler does."""
    d = deriver()
    nodes = d.attention_block(cfg, "synthetic", cfg["hidden_size"], raw=cfg)
    return d.lightning_indexer(cfg, "synthetic", cfg["hidden_size"], nodes, cfg)


def test_kpool_indexer_projections_come_from_the_latent_rank_and_fuse_wk():
    """The three projections vLLM builds, at the widths it builds them.

    wq_b reads q_lora_rank, not hidden -- that was the single largest error in the old
    fused node. wk and weights_proj are ONE GEMM of [head_dim, n_head]
    (deepseek_v32/attention.py:78-85, glm5next/common/attention.py:272-281). Each width
    is confirmed against the pinned GLM-5.3-Flash safetensors headers: wq_b [4096,1536],
    wk [128,4096] + weights_proj [32,4096], gate [128,4096]."""
    cfg = kpool_config()
    gemms = {n.get("role"): (n["n"], n["k"]) for n in indexer_nodes(cfg)
             if n["op"] == "GEMM"}
    heads, dim = cfg["index_n_heads"], cfg["index_head_dim"]
    assert "index_qk_proj" not in gemms, "the fused node sources the query from hidden"
    assert gemms["index_wq_b"] == (heads * dim, cfg["q_lora_rank"])
    assert gemms["index_wk_weights_proj"] == (dim + heads, cfg["hidden_size"])
    assert gemms["index_kpool_compress_gate"] == (dim, cfg["hidden_size"])


def test_kpool_indexer_prices_the_fp32_head_gate_as_its_own_launch():
    """The head gate is a second GEMM at FP32, not part of the fused projection.

    The forward keeps the first head_dim rows of wk_weights_proj for K, then caches the
    remaining n_head rows transposed to FP32 and runs its own matmul against FP32
    activations: `weights = torch.mm(hidden_states.float(), self._wp_fp32)`
    (glm5next/common/attention.py:341-350). That is a distinct launch and a resident
    FP32 weight copy, so modelling only the merged layer undercounts both."""
    nodes = indexer_nodes(kpool_config())
    gate = next((n for n in nodes if n.get("role") == "index_head_gate_proj"), None)
    assert gate is not None, "no head-gate GEMM; the forward runs a second matmul"
    assert (gate["n"], gate["k"]) == (32, 4096), (
        f"the head gate is {gate['n']}x{gate['k']}; it projects n_head from hidden")
    assert gate.get("weight_dtype") == "fp32", (
        "the head-gate weights are cached as FP32; the global width would understate "
        "the resident bytes and the compute")


def test_kpool_scorer_is_bounded_by_pooling_and_not_by_the_topk_it_computes():
    """index_topk must NOT bound the scoring pass.

    The scorer scores every pooled candidate in order to DISCOVER the top-k; the
    selection then bounds the main MLA read, which carries index_topk itself. In the
    schema index_topk means "this node reads only a top-k of its cache", so putting it
    on the scorer makes it read only the answers it exists to find and caps the one
    context-proportional term in the layer at a constant. DeepSeek-V4's representation
    draws the same line: block_index_scores is unbounded and only the downstream sparse
    MLA node carries the top-k."""
    cfg = kpool_config()
    nodes = indexer_nodes(cfg)
    scorer = next(n for n in nodes if n.get("role") == "block_index_scores")
    assert "index_topk" not in scorer, (
        "the scorer carries index_topk, so it is bounded by the selection it is "
        "supposed to produce")
    assert scorer["compress_ratio"] == cfg["index_kpool"], (
        f"the scorer states compress_ratio {scorer.get('compress_ratio')}; pooling is "
        f"{cfg['index_kpool']}:1 and that is its only bound")
    assert "window" not in scorer, "a window would make the selection look bounded"
    # The main attention keeps the selection it really applies.
    main = next(n for n in nodes
                if n["op"] == "Attention" and n.get("role") != "block_index_scores")
    assert main.get("index_topk") == cfg["index_topk"], (
        "the main latent read must carry the top-k the indexer selects for it")


def test_kpool_prices_the_compression_pass_and_its_ape_state():
    """Pooling is real per-token work, not just a projection.

    vLLM softmax-weights each group of index_kpool keys against a FP32
    [index_kpool, index_head_dim] APE table, then rotates, quantizes and writes one
    pooled state to the cache (glm5next/common/attention.py:394-417). A gate GEMM alone
    does not represent that, so the pass is its own node with its traffic stated: the
    key read, the APE table, and the pooled write."""
    cfg = kpool_config()
    nodes = indexer_nodes(cfg)
    comp = next((n for n in nodes if n.get("role") == "index_kpool_compress"), None)
    assert comp is not None, "no compression pass; pooling is work, not just a gate"
    assert comp["op"] == "Elementwise"
    ape = cfg["index_kpool"] * cfg["index_head_dim"] * 4
    key = cfg["index_head_dim"] * 2
    assert comp["bytes_per_token"] == key + ape + cfg["index_head_dim"], (
        f"the compression states {comp['bytes_per_token']} B/token; the pass reads its "
        f"key ({key}) and the FP32 APE table ({ape}) and writes a pooled state")


def test_a_config_without_kpool_gets_no_pooling_nodes_or_bound():
    """The negative case: drop index_kpool and the pooled parts must disappear.

    Same config otherwise, so this isolates the field. A family that declares no pooling
    scans every token, and a compress_ratio there would bound a scan that has nothing
    pooled to bound."""
    cfg = kpool_config()
    cfg.pop("index_kpool")
    nodes = indexer_nodes(cfg)
    roles = {n.get("role") for n in nodes}
    assert "index_kpool_compress_gate" not in roles
    assert "index_kpool_compress" not in roles
    scorer = next(n for n in nodes if n.get("role") == "block_index_scores")
    assert not scorer.get("compress_ratio"), (
        "an unpooled scorer states a compress_ratio, bounding a full-context scan")
    assert scorer["kind"] == "gqa", (
        "an unpooled scorer claims the sparse-latent kind, which implies a bound")
    # The projections are unchanged: vLLM builds the same ones either way.
    gemms = {n.get("role") for n in nodes if n["op"] == "GEMM"}
    assert {"index_wq_b", "index_wk_weights_proj", "index_head_gate_proj"} <= gemms


def test_a_kpool_config_without_a_latent_rank_is_refused():
    """wq_b is sized from q_lora_rank, so a config lacking it must raise.

    Defaulting the width would price the dominant projection on a guess."""
    cfg = kpool_config()
    cfg.pop("q_lora_rank")
    with pytest.raises(deriver().DeriveError, match="q_lora_rank"):
        indexer_nodes(cfg)


def kpool_models() -> list[Path]:
    """Models whose config declares a pooled (k-pool) DSA indexer."""
    return [d for d in model_dirs()
            if int(text_config(config_of(d)).get("index_kpool") or 0) > 1]


@pytest.mark.parametrize("d", kpool_models(), ids=lambda d: d.name)
def test_kpool_indexer_prices_its_own_projections_and_pooled_scan(d: Path):
    """A pooled indexer runs three projections and scores pooled candidates.

    The fused single-GEMM form was wrong three ways for this family. vLLM builds wq_b
    from the LATENT rank, not hidden; it fuses wk and weights_proj into one GEMM of
    [head_dim, n_head]; and it adds a kpool compression gate (common/attention.py
    255-282). Sourcing all of it from hidden at one fused width overstated the
    projection work by 2.3x on GLM-5.3-Flash.

    The scan is the bigger error. index_kpool pools the cache -- one K state per kpool
    tokens, which vLLM sizes as max_model_len // index_kpool -- so an unpooled scorer
    overstates the one term that grows with context by that factor.

    Every width is asserted from the config, so a config update does not silently
    invalidate the test."""
    g, t = graph_of(d), text_config(config_of(d))
    hidden = int(t["hidden_size"])
    heads = int(t["index_n_heads"])
    dim = int(t["index_head_dim"])
    kpool = int(t["index_kpool"])
    topk = int(t["index_topk"])
    q_lora = int(t.get("q_lora_rank") or 0)
    assert q_lora, "a pooled indexer sizes wq_b from the latent rank"

    indexed = [k for k in g["layer_kinds"]
               if any(n.get("role") == "block_index_scores" for n in k["nodes"])]
    assert indexed, f"{d.name}: no layer kind carries an indexer scorer"

    for kind in indexed:
        gemms = {n.get("role"): (n["n"], n["k"]) for n in kind["nodes"]
                 if n["op"] == "GEMM"}
        assert "index_qk_proj" not in gemms, (
            f"{d.name}/{kind['id']}: still carries the fused index_qk_proj, which "
            f"sources the query stage from hidden instead of the latent rank")
        assert gemms["index_wq_b"] == (heads * dim, q_lora), (
            f"{d.name}/{kind['id']}: index_wq_b is {gemms['index_wq_b']}; the config "
            f"gives {heads} * {dim} from the latent rank {q_lora}")
        assert gemms["index_wk_weights_proj"] == (dim + heads, hidden), (
            f"{d.name}/{kind['id']}: index_wk_weights_proj is "
            f"{gemms['index_wk_weights_proj']}; vLLM fuses wk and weights_proj into "
            f"one GEMM of [{dim}, {heads}] from {hidden}")
        assert gemms["index_kpool_compress_gate"] == (dim, hidden), (
            f"{d.name}/{kind['id']}: no kpool compression gate at {dim} from {hidden}")

        scorer = next(n for n in kind["nodes"]
                      if n.get("role") == "block_index_scores")
        assert scorer["compress_ratio"] == kpool, (
            f"{d.name}/{kind['id']}: the scorer states compress_ratio "
            f"{scorer.get('compress_ratio')}; index_kpool is {kpool}, and an unpooled "
            f"scan overstates the dominant long-context term by that factor")
        assert scorer["index_topk"] == topk, (
            f"{d.name}/{kind['id']}: the scorer selects "
            f"{scorer.get('index_topk')}; index_topk is {topk}")
        assert scorer["d_h"] == dim and scorer["n_q"] == heads
        assert scorer["n_kv"] == 1, "the indexer scores against one pooled K stream"


def test_an_unpooled_indexer_prices_no_pooled_compression():
    """A config without index_kpool must not be given the pooled shape.

    Every family reaching lightning_indexer() now gets the same projections, because
    vLLM builds the same ones -- deepseek_v32 and glm5next are identical there. What
    pooling adds is the compression gate, the compression pass, and the compress_ratio
    bound on the scorer, and a config that declares no index_kpool must have none of
    them: there is nothing to pool, so a ratio would bound a scan that reads every
    token."""
    checked = []
    for path in model_dirs():
        t = text_config(config_of(path))
        if "index_topk" not in t or int(t.get("index_kpool") or 0) > 1:
            continue
        g = graph_of(path)
        for kind in g["layer_kinds"]:
            roles = {n.get("role") for n in kind["nodes"]}
            if "block_index_scores" not in roles:
                continue
            checked.append(f"{path.name}/{kind['id']}")
            assert "index_kpool_compress_gate" not in roles, (
                f"{path.name}/{kind['id']}: declares no index_kpool yet prices a "
                f"pooled compression gate")
            assert "index_kpool_compress" not in roles, (
                f"{path.name}/{kind['id']}: declares no index_kpool yet prices the "
                f"pooling pass")
            scorer = next(n for n in kind["nodes"]
                          if n.get("role") == "block_index_scores")
            assert not scorer.get("compress_ratio"), (
                f"{path.name}/{kind['id']}: an unpooled scorer states a compress_ratio, "
                f"so it prices a pooled scan the config never declares")
    assert checked, (
        "no unpooled indexer in the catalog; this test guards their shape and has "
        "nothing to check")


def test_indexer_runs_only_on_the_layers_the_config_indexes():
    """A DSA config may skip the indexer, and the expanded stack must show that.

    vLLM computes a per-layer skip_topk and builds the indexer only when it is false
    (models/deepseek_v32/attention.py:166-200, which is where GlmMoeDsaForCausalLM is
    served -- its __init__ aliases GlmMoeDsaForCausalLM = DeepseekV32ForCausalLM):

        skip_topk = max(layer_id - index_skip_topk_offset + 1, 0) % index_topk_freq != 0

    glm-5.2, glm-5.2-fp8 and glm-5.3 declare freq 4 and offset 3 over 78 layers, which
    indexes 21. Charging all 78 for the scorer and its projections overprices the
    dominant long-context term by 78/21.

    Asserted on the EXPANDED stack, by exact count and exact placement, because a
    per-kind check cannot see how many layers use each kind. The defaults matter too:
    glm-5 declares neither field, so freq 1 and offset 2 index every layer, and this
    test must confirm that case is unchanged rather than silently reduced."""
    checked = 0
    for d in model_dirs():
        t = text_config(config_of(d))
        if not (t.get("index_topk") and t.get("index_n_heads")):
            continue
        if t.get("compress_ratios"):
            # DeepSeek-V4 places its indexer by its per-layer compression-ratio vector
            # instead -- only the ratio-4 layers build one -- which is a different rule
            # from skip_topk and is pinned by its own test. Keyed off the vector rather
            # than the model name so a future family declaring one is also excluded.
            continue
        g = graph_of(d)
        layers = int(t.get("num_hidden_layers") or 0)
        assert layers, f"{d.name}: no layer count to expand against"
        kinds = {k["id"]: k for k in g["layer_kinds"]}
        scores = {kid: any(n.get("role") == "block_index_scores" for n in k["nodes"])
                  for kid, k in kinds.items()}
        seq = layer_sequence(g["stack"])
        assert len(seq) == layers, (
            f"{d.name}: stack expands to {len(seq)} layers for {layers} declared")
        got = [i for i, kid in enumerate(seq) if scores[kid]]

        pattern = t.get("index_topk_pattern")
        if pattern is not None:
            want = [i for i in range(layers) if pattern[i] != "S"]
        else:
            freq = int(t.get("index_topk_freq") or 1)
            offset = t.get("index_skip_topk_offset")
            offset = 2 if offset is None else int(offset)
            want = [i for i in range(layers)
                    if max(i - offset + 1, 0) % freq == 0]
        assert got == want, (
            f"{d.name}: scorers at {len(got)} layers {got[:8]}...; the config's "
            f"skip_topk rule gives {len(want)} at {want[:8]}...")

        # A layer that skips the indexer makes no selection at all -- vLLM passes
        # index_k=None and takes the dense latent path (attention.py:309-315) -- so it
        # must not record a top-k either.
        for i, kid in enumerate(seq):
            if scores[kid]:
                continue
            for n in kinds[kid]["nodes"]:
                if n["op"] != "Attention":
                    continue
                assert not n.get("index_topk"), (
                    f"{d.name}/{kid}: a layer with no indexer records index_topk "
                    f"{n['index_topk']}, asserting a selection it never computes")
                assert n.get("kind") != "sparse_mla", (
                    f"{d.name}/{kid}: a layer with no indexer claims the sparse-latent "
                    f"kind, which implies a bound it does not have")
        checked += 1
    assert checked, "no indexed model in the catalog to check"


def test_kimi_attention_weights_keep_the_width_quantization_skips():
    """A compressed-tensors ignore list naming self_attn keeps those weights at base.

    kimi-k2.5 is globally int4 and kimi-k3 mxfp4, and both name `re:.*self_attn.*` in
    their compressed-tensors ignore list, so every attention projection stays
    unquantized at runtime. Pricing them at the global width charges 4 bits for tensors
    the runtime never quantizes.

    This covers the KDA mixer too, not just the latent path: vLLM builds Kimi-K3's KDA
    as `self.self_attn` with prefix "{...}.self_attn" (kimi_k3/nvidia/model.py:888-891),
    so the same pattern covers it. The routed experts are NOT in the ignore list and
    must keep the quantized global width -- that is the half of the distinction a blanket
    override would destroy."""
    checked = 0
    for name in ("kimi-k2.5", "kimi-k3"):
        d = ROOT / "models" / name
        if not d.is_dir():
            continue
        g, t = graph_of(d), text_config(config_of(d))
        quant = (t.get("quantization_config")
                 or config_of(d).get("quantization_config") or {})
        ignore = [str(x) for x in (quant.get("ignore") or ())]
        assert any("self_attn" in x for x in ignore), (
            f"{name}: this test is about a config excluding self_attn; the ignore list "
            f"is now {ignore}")
        base = str(t.get("dtype") or "")
        assert base.startswith("bfloat16"), (
            f"{name}: expected a bfloat16 base width, got {base!r}")
        glob = g["global"]["weight_dtype"]
        assert glob != "bf16", (
            f"{name}: the global width is already bf16, so this test cannot tell an "
            f"override from the default")

        attention_roles = {
            "qkv_proj", "qkv_a_proj", "kv_a_proj", "q_proj", "q_b_proj", "kv_b_proj",
            "o_proj", "index_wq_b", "index_wk_weights_proj", "index_head_gate_proj",
        }
        seen_attn = seen_expert = 0
        for k in g["layer_kinds"]:
            for n in k["nodes"]:
                role = str(n.get("role") or "")
                if n["op"] not in ("GEMM", "GroupedGEMM"):
                    continue
                if role in attention_roles or role.startswith("kda_"):
                    seen_attn += 1
                    assert n.get("weight_dtype") == "bf16", (
                        f"{name}/{k['id']}: {role} is priced at the global {glob}; the "
                        f"ignore list leaves it unquantized at the bf16 base")
                elif "expert" in role:
                    seen_expert += 1
                    assert n.get("weight_dtype") != "bf16", (
                        f"{name}/{k['id']}: {role} is overridden to bf16, but the "
                        f"experts are what this checkpoint actually quantizes")
        assert seen_attn, f"{name}: no attention projections found to check"
        assert seen_expert, f"{name}: no expert GEMMs found; the negative half is unchecked"
        checked += 1
    if not checked:
        pytest.skip("no Kimi models in the catalog")


def mla_models() -> list[Path]:
    """The models whose config declares a compressed latent KV."""
    return [d for d in model_dirs()
            if text_config(config_of(d)).get("kv_lora_rank")]


@pytest.mark.parametrize("d", mla_models(), ids=lambda d: d.name)
def test_mla_prices_its_low_rank_stages_not_one_fused_projection(d: Path):
    """A latent layer runs a chain of low-rank GEMMs, and each must be priced.

    An MLA layer does not project hidden straight to the query width. It compresses to
    kv_lora_rank (and, where declared, q_lora_rank), normalises, then up-projects: the
    cached latent through kv_b_proj into per-head nope + value widths, and the query
    through q_b_proj. Collapsing that into one query-width `qkv_proj` priced the wrong
    number of GEMMs at the wrong widths -- and for GLM-5.3-Flash it OVERSTATED the
    parameter count by 1.33x while coincidentally matching on bytes, because kv_b_proj
    is BF16 where its siblings are FP8. Byte traffic alone cannot catch this, so every
    stage is asserted by shape.

    Each width comes from the config, so this survives a config update."""
    g, t = graph_of(d), text_config(config_of(d))
    hidden = int(t["hidden_size"])
    nq = int(t["num_attention_heads"])
    nope = int(t["qk_nope_head_dim"])
    rope = int(t["qk_rope_head_dim"])
    v = int(t.get("v_head_dim") or nope)
    kv_lora = int(t["kv_lora_rank"])
    q_lora = int(t.get("q_lora_rank") or 0)

    latent = [k for k in g["layer_kinds"]
              if any(n.get("role") == "kv_b_proj" for n in k["nodes"])]
    assert latent, (
        f"{d.name}: no layer kind prices a kv_b_proj, so the cached latent is never "
        f"up-projected; the layer is still one fused qkv_proj")

    for kind in latent:
        gemms = {n.get("role"): (n["n"], n["k"]) for n in kind["nodes"]
                 if n["op"] == "GEMM"}
        roles = [n.get("role") for n in kind["nodes"]]

        assert "qkv_proj" not in gemms, (
            f"{d.name}/{kind['id']}: still carries the fused qkv_proj alongside the "
            f"low-rank stages, so the projection work is counted twice")

        # The up-projection of the cache: kv_lora_rank -> heads * (nope + v).
        assert gemms["kv_b_proj"] == (nq * (nope + v), kv_lora), (
            f"{d.name}/{kind['id']}: kv_b_proj is {gemms['kv_b_proj']}; the config "
            f"gives {nq} * ({nope} + {v}) from {kv_lora}")
        if q_lora:
            # vLLM fuses the two A-stages into one GEMM over
            # [q_lora_rank, kv_lora_rank + qk_rope_head_dim].
            # An output gate rides in this same launch where the config declares one:
            # vLLM swaps the fused A-projection for KimiK3MergedQKVGateLinear and splits
            # [qkv_a_rows, num_local_heads * v_head_dim] off the single result
            # (kimi_k3/nvidia/mla.py:218-228, 574-576). kimi-k3 declares it, which adds
            # num_heads * v_head_dim rows; no other committed config does.
            gate_rows = nq * v if t.get("mla_use_output_gate") else 0
            assert gemms["qkv_a_proj"] == (
                q_lora + kv_lora + rope + gate_rows, hidden), (
                f"{d.name}/{kind['id']}: qkv_a_proj is {gemms['qkv_a_proj']}; the "
                f"config gives {q_lora} + {kv_lora} + {rope} + {gate_rows} (gate) "
                f"from {hidden}")
            if gate_rows:
                gate = next((n for n in kind["nodes"]
                             if n.get("role") == "mla_output_gate"), None)
                assert gate is not None and gate.get("bytes_per_token"), (
                    f"{d.name}/{kind['id']}: declares mla_use_output_gate but prices "
                    f"no sigmoid-multiply over the attention output")
            assert gemms["q_b_proj"] == (nq * (nope + rope), q_lora), (
                f"{d.name}/{kind['id']}: q_b_proj is {gemms['q_b_proj']}; the config "
                f"gives {nq} * ({nope} + {rope}) from {q_lora}")
            # ONE fused norm over both latents, not two. Kimi-K3 calls
            # fused_q_kv_rmsnorm (kimi_k3/nvidia/mla.py:520) and glm5next asks the
            # shared MLA module for the same via fuse_qkv_rmsnorm=True
            # (glm5next/common/attention.py:604), so two nodes would charge a launch
            # neither family makes.
            fused = next((n for n in kind["nodes"]
                          if n.get("role") == "qkv_a_layernorm"), None)
            assert fused is not None, (
                f"{d.name}/{kind['id']}: no fused latent norm; vLLM normalises both "
                f"latents in one launch for this family")
            assert "q_a_layernorm" not in roles and "kv_a_layernorm" not in roles, (
                f"{d.name}/{kind['id']}: prices separate q and kv latent norms "
                f"alongside the fused one")
            # Sub-hidden traffic must be stated: these widths are the latent ranks, not
            # hidden, so an omitted bytes_per_token prices them as hidden-wide norms.
            assert fused.get("bytes_per_token"), (
                f"{d.name}/{kind['id']}: the fused latent norm states no "
                f"bytes_per_token, so it prices as a hidden-width pass")
            assert fused["bytes_per_token"] < 2 * hidden * 4, (
                f"{d.name}/{kind['id']}: the fused latent norm states "
                f"{fused['bytes_per_token']} B/token, which is not below a "
                f"hidden-width norm's traffic; the latents are narrower than hidden")
            assert "q_proj" not in gemms, (
                f"{d.name}/{kind['id']}: declares q_lora_rank {q_lora} yet prices a "
                f"direct q_proj; the query path is low-rank")
        else:
            # No q_lora_rank: the query is projected directly at full width, which is
            # the branch vLLM takes and the shape deepseek-v2-lite really runs.
            assert gemms["q_proj"] == (nq * (nope + rope), hidden), (
                f"{d.name}/{kind['id']}: q_proj is {gemms['q_proj']}; with no "
                f"q_lora_rank it runs {nq} * ({nope} + {rope}) from {hidden}")
            assert gemms["kv_a_proj"] == (kv_lora + rope, hidden)
            # Only the KV side is compressed, so there is one latent norm to state.
            kvn = next((n for n in kind["nodes"]
                        if n.get("role") == "kv_a_layernorm"), None)
            assert kvn is not None and kvn.get("bytes_per_token"), (
                f"{d.name}/{kind['id']}: the kv latent norm states no bytes_per_token")
            assert "qkv_a_proj" not in gemms, (
                f"{d.name}/{kind['id']}: declares no q_lora_rank, so there is no "
                f"query A-stage to fuse the KV one with")

        # The output projection reads the per-head value width, not the query width.
        assert gemms["o_proj"] == (hidden, nq * v), (
            f"{d.name}/{kind['id']}: o_proj is {gemms['o_proj']}; the config gives "
            f"{hidden} from {nq} * {v}")


@pytest.mark.parametrize("d", model_dirs(), ids=lambda d: d.name)
def test_expert_shape_matches_config(d: Path):
    """Expert count and top-k must come from the config, under whichever alias."""
    g, t = graph_of(d), text_config(config_of(d))
    # The ROUTED expert node. A shared-expert node is a separate grouped GEMM where the
    # checkpoint stores the two at different widths, and it is sized by the shared count
    # rather than the routed one, so it is checked below instead.
    nodes = [n for k in g["layer_kinds"] for n in k["nodes"]
             if n["op"] == "GroupedGEMM" and n.get("role") != "shared_experts"]
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

    # Where a shared-expert node exists, it must carry the config's SHARED count, and
    # every token passes through all of them rather than a routed subset.
    shared_nodes = [n for k in g["layer_kinds"] for n in k["nodes"]
                    if n.get("role") == "shared_experts"]
    shared = t.get("n_shared_experts") or t.get("num_shared_experts")
    for n in shared_nodes:
        assert n["experts"] == shared, (
            f"shared-expert node states {n['experts']}, config says {shared}")
        assert n["top_k"] == n["experts"], (
            "a shared expert is dense: every token passes through all of them")


# Architecture families expected to produce identical layer structures, with the reason.
# Every other pair of models must differ: a deriver that emitted one template for
# everything would pass every per-model check above.
EXPECTED_IDENTICAL = {

    frozenset({"minimax-m2.5", "minimax-m2.7"}):
        "one architecture across two MiniMax-M2 generations. Every cost-relevant field is "
        "identical: 62 layers, hidden 3072, 256 experts at top-8, intermediate 1536, 48 "
        "query heads over 8 KV heads at head_dim 128, vocab 200064, fp8 weights. The two "
        "configs differ only in max_position_embeddings (196608 against 204800) and a "
        "dtype label, neither of which the cost model reads",

    frozenset({"glm-5.2", "glm-5.2-fp8", "glm-5.3"}):
        "one architecture across three GLM-5 generations. Every cost-relevant field is "
        "identical: 78 layers, hidden 6144, 256 experts at top-8, moe_intermediate 2048, "
        "3 leading dense layers, vocab 154880, the same sparse-MLA geometry, and the "
        "same indexer pattern — index_topk_freq 4 with index_skip_topk_offset 3, which "
        "indexes 21 of the 78 layers. They differ in quantization_config, rope theta, "
        "context length, transformers_version and — between glm-5.2 and glm-5.3 — in "
        "head_dim, 64 against 192. That last one looks like it should matter and does "
        "not: an MLA layer's per-head width is kv_lora_rank + qk_rope_head_dim, which "
        "both state as 512 + 64, so head_dim is not a term the cost model reads for this "
        "attention kind. "
        "glm-5 is NOT in this group: it declares neither index_topk_freq nor "
        "index_skip_topk_offset, so vLLM's defaults (freq 1, offset 2) index all 78 "
        "layers against these three's 21. That is a real cost difference — 57 extra "
        "scorers and their projections per forward pass — so grouping it here would "
        "assert away the very distinction the indexer-frequency derivation exists to "
        "capture",
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
    different expansions.

    It covers the speculator's draft stack as well as the target stack, because a draft
    pass is work a deployment pays for. It does NOT cover the head: every model here has
    the same final-norm-plus-lm_head shape, and its cost is driven by vocab_size, which
    test_distinct_architectures_cost_differently is not the right instrument for."""
    kinds = {k["id"]: k for k in g["layer_kinds"]}
    work: collections.Counter = collections.Counter()
    for layer_id in layer_sequence(g["stack"]):
        for n in kinds[layer_id]["nodes"]:
            work[node_cost(n)] += 1

    # The draft stack a speculative decode runs, which is work the target pays for on
    # every accepted token and is NOT part of g["stack"]. Tagged as a separate component
    # rather than folded into the counter above, so a draft layer and a target layer of
    # the same shape stay distinguishable: moving a layer between the two stacks changes
    # what a step costs. Excluded from this and verified nowhere, a wrong draft length or
    # a mirrored-from-the-wrong-kind draft stack collided with a correct graph and passed
    # every collision test.
    spec = g.get("speculator")
    if spec:
        draft: collections.Counter = collections.Counter()
        for layer_id in layer_sequence(spec["stack"]):
            for n in kinds[layer_id]["nodes"]:
                draft[node_cost(n)] += 1
        work[("speculator", spec.get("method"), spec.get("num_spec_tokens"),
              tuple(sorted(draft.items(), key=repr)))] += 1
    return tuple(sorted(work.items(), key=repr))


def node_cost(n: dict) -> tuple:
    """The parameters that change what a primitive costs, and nothing else.

    `emit` is excluded: it selects whether a node exists in a given deployment, which is a
    property of the layout rather than of the model. Those conditions are checked
    structurally instead, by test_collectives_land_where_the_parallelism_needs_them."""
    return (
        n["op"], n.get("n"), n.get("k"), n.get("n_q"), n.get("n_kv"), n.get("d_h"),
        n.get("experts"), n.get("top_k"), n.get("shared_experts"),
        n.get("shared_intermediate_size"), n.get("latent_size"), n.get("kind"),
        n.get("recurrent_kind"), n.get("state_size"), n.get("n_heads"),
        n.get("n_groups"), n.get("conv_kernel"), n.get("intermediate_size"),
        n.get("window"), n.get("index_topk"),
        # compress_ratio bounds the compressed half of a sparse latent read, and a
        # per-node weight_dtype overrides the global width for the tensors it names.
        # Both change what a node costs, and both are distinctions this suite relies on
        # elsewhere -- the C4A/C128A split and gpt-oss's expert-only MXFP4 -- so leaving
        # them out made the collision tests blind to exactly the cases they were added
        # to protect.
        n.get("compress_ratio"), n.get("weight_dtype"),
    )


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
    """A deriver that defaulted every dtype would pass the per-model checks.

    Reads node overrides as well as the global, because a checkpoint that states its
    mixed-precision layout explicitly keeps the BASE dtype global and carries the
    quantized widths on the nodes they apply to. Looking only at the global would miss
    nvfp4 entirely now that the Nemotron variants express it that way."""
    seen = {graph_of(d)["global"]["weight_dtype"] for d in model_dirs()}
    for d in model_dirs():
        g = graph_of(d)
        seen |= {n["weight_dtype"] for k in g["layer_kinds"] for n in k["nodes"]
                 if n.get("weight_dtype")}
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
    (lambda g: _narrow_first_gemm_dtype(g), "narrowing one node's weight dtype"),
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


def _narrow_first_gemm_dtype(g: dict) -> None:
    """A per-node dtype override changes that node's weight bytes fourfold."""
    for k in g["layer_kinds"]:
        for n in k["nodes"]:
            if n["op"] == "GEMM":
                n["weight_dtype"] = "mxfp4"
                return
    pytest.skip("no GEMM node")


def _change_kv_heads(g: dict) -> None:
    for k in g["layer_kinds"]:
        for n in k["nodes"]:
            if n["op"] == "Attention":
                n["n_kv"] = max(1, n["n_kv"] // 2)
                return
    pytest.skip("no attention node")


@pytest.mark.parametrize("model,field,value,description", [
    ("deepseek-v4-pro", "compress_ratio", 4,
     "retuning a compressed stream's ratio"),
    ("gpt-oss-120b", "weight_dtype", "bf16",
     "widening a node's weight dtype"),
])
def test_signature_notices_a_changed_node_parameter(model, field, value, description):
    """cost_signature must see the two fields its contract most depends on.

    Both were omitted from node_cost: mutating either left the signature unchanged, so
    the collision tests were blind to exactly the distinctions this suite relies on --
    the C4A/C128A compression split, and gpt-oss's expert-only MXFP4 scope. A signature
    that cannot see them would let a graph with the wrong one collide with a correct
    graph and pass."""
    import copy

    d = ROOT / "models" / model
    if not d.is_dir():
        pytest.skip(f"{model} is not in the catalog")
    g = graph_of(d)
    before = cost_signature(g)
    variant = copy.deepcopy(g)
    changed = False
    for k in variant["layer_kinds"]:
        for n in k["nodes"]:
            if field in n and n[field] != value:
                n[field] = value
                changed = True
    assert changed, f"no node in {model} carries {field} to mutate"
    assert cost_signature(variant) != before, (
        f"{description} left the cost signature unchanged, so a graph with the wrong "
        f"{field} would pass the collision tests")


# --- Fidelity to the pinned implementations ---------------------------------------
# The gates in this repo establish SELF-CONSISTENCY: a graph re-derives from its config,
# validates against the schema, and prices distinctly from its siblings. None of that
# checks fidelity to the implementation the deriver claims to follow, which is where
# review found five defects. These tests assert against the pinned upstream behaviour,
# so a regeneration cannot quietly restore the old shape.


def test_v4_every_layer_retains_its_sliding_window():
    """The read set is the UNION of the window and the compressed stream.

    vLLM sets `self.window_size = config.sliding_window` with no compress_ratio guard
    (deepseek_v4/attention.py at the pinned commit f9c9e8a), and
    combine_topk_swa_indices merges the window positions with the compressed ones. The
    deriver previously replaced the window with a fixed topk//ratio -- 256 for ratio 4
    and 8 for ratio 128 -- the latter smaller than the 128-token window the config
    declares, so a C128A layer claimed to read 8 tokens where the implementation reads
    at least 128."""
    g, t = _v4()
    declared = t["sliding_window"]
    assert declared > 0, "this test is about a config that declares a window"
    main = [n for k in g["layer_kinds"] for n in k["nodes"]
            if n["op"] == "Attention" and n.get("role") != "block_index_scores"]
    assert main, "no attention nodes"
    for n in main:
        assert n["window"] == declared, (
            f"a layer states window {n['window']} where the config declares "
            f"{declared}; the window is retained at every compression ratio")


def test_v4_compressed_bound_is_a_ratio_not_a_token_count():
    """A ratio-128 layer's compressed term is context/128, which no constant expresses.

    The C128A path reads a positional run -- num_compressed = (position + 1) //
    compress_ratio in the Triton kernel -- rather than selecting a ranked top-k. A graph
    storing a fixed token count there is wrong at every context but one."""
    g, t = _v4()
    ratios = set(t["compress_ratios"][: t["num_hidden_layers"]])
    kinds = {k["id"]: k for k in g["layer_kinds"]}
    seq = layer_sequence(g["stack"])
    for idx, ratio in enumerate(t["compress_ratios"][: t["num_hidden_layers"]]):
        n = next(x for x in kinds[seq[idx]]["nodes"]
                 if x["op"] == "Attention" and x.get("role") != "block_index_scores")
        assert n.get("compress_ratio") == ratio, (
            f"layer {idx} has compress_ratio {n.get('compress_ratio')}, config says "
            f"{ratio}")
    # Only the selecting ratio carries a top-k; the other selects nothing at all.
    for ratio in ratios:
        idx = t["compress_ratios"].index(ratio)
        n = next(x for x in kinds[seq[idx]]["nodes"]
                 if x["op"] == "Attention" and x.get("role") != "block_index_scores")
        if ratio == 4:
            assert n.get("index_topk") == t["index_topk"], (
                "a ratio-4 layer selects up to index_topk compressed positions")
        else:
            assert not n.get("index_topk"), (
                f"a ratio-{ratio} layer reads a positional run and selects no top-k, "
                f"so index_topk {n.get('index_topk')} describes a selection it never "
                f"makes")


def test_v4_only_ratio_4_layers_have_an_indexer():
    """vLLM builds an indexer under `if self.compress_ratio == 4`, noting "Only C4A uses
    sparse attention and hence has indexer" (attention.py:297-317).

    The deriver spliced the indexer into every ratio, so all 31 C128A layers carried an
    index_qk_proj GEMM and a block_index_scores attention. The indexer is the only
    context-proportional term in the layer, so pricing it on a layer that has none
    charges a full context scan 31 times per forward pass."""
    g, t = _v4()
    kinds = {k["id"]: k for k in g["layer_kinds"]}
    seq = layer_sequence(g["stack"])
    seen = {}
    for idx, ratio in enumerate(t["compress_ratios"][: t["num_hidden_layers"]]):
        nodes = kinds[seq[idx]]["nodes"]
        has = any(n.get("role") == "block_index_scores" for n in nodes)
        proj = any(str(n.get("role", "")).startswith("index_wq_proj")
                   for n in nodes)
        assert has == proj, (
            f"layer {idx}: an indexer and its projection must appear together")
        seen.setdefault(ratio, set()).add(has)
    assert seen[4] == {True}, "every ratio-4 layer needs its indexer"
    for ratio, flags in seen.items():
        if ratio == 4:
            continue
        assert flags == {False}, (
            f"ratio-{ratio} layers carry an indexer; the implementation builds one only "
            f"for ratio 4")


def test_v4_draft_module_is_an_uncompressed_layer_of_its_own():
    """An MTP layer takes compress_ratio 1, not a target layer's ratio.

    vLLM reads compress_ratios by layer index and falls to `self.compress_ratio = 1` for
    any index at or beyond num_hidden_layers (attention.py:229-235), so the trailing 0 in
    this config's vector is never used as a divisor. The deriver dropped that entry and
    let the generic speculator fallback mirror kinds[-1], which made the draft stack
    csa128_moe: the tightest-bounded read in the model standing in for the one layer
    with no compression at all, and carrying an indexer it should not have."""
    g, t = _v4()
    assert len(t["compress_ratios"]) == t["num_hidden_layers"] + 1, (
        "this test is about the trailing draft entry; the vector no longer has one")
    spec = g["speculator"]
    drafted = layer_sequence(spec["stack"])
    assert drafted, "no draft stack"
    kinds = {k["id"]: k for k in g["layer_kinds"]}
    target_ids = set(layer_sequence(g["stack"]))
    for lid in drafted:
        assert lid not in target_ids, (
            f"the draft stack reuses target layer kind {lid!r}, whose compression bound "
            f"is not the draft module's")
        n = next(x for x in kinds[lid]["nodes"]
                 if x["op"] == "Attention" and x.get("role") != "block_index_scores")
        assert not n.get("compress_ratio"), (
            f"the draft layer states compress_ratio {n.get('compress_ratio')}; an MTP "
            f"layer is uncompressed (ratio 1)")
        assert not any(x.get("role") == "block_index_scores"
                       for x in kinds[lid]["nodes"]), (
            "the draft layer carries an indexer, which only a ratio-4 layer has")
        assert n["window"] == t["sliding_window"], (
            "the draft layer keeps the model's sliding window")


def test_kimi_k3_leading_layers_are_dense():
    """first_k_dense_replace makes the leading layers dense MLPs, not routed experts.

    KimiDecoderLayer builds a KimiSparseMoeBlock only when
    `layer_idx >= first_k_dense_replace` and otherwise a KimiMLP at the config's own
    intermediate_size (modeling_kimi_linear.py:893-900). The handler defined only routed
    kinds, so layer 0 was priced with a 896-expert GroupedGEMM in place of one dense
    7168 -> 2*33792 -> 7168 MLP."""
    d = ROOT / "models" / "kimi-k3"
    if not d.is_dir():
        pytest.skip("kimi-k3 is not in the catalog")
    g, t = graph_of(d), text_config(config_of(d))
    first_dense = int(t["first_k_dense_replace"])
    assert first_dense > 0, "this test is about a config that declares one"
    seq = layer_sequence(g["stack"])
    kinds = {k["id"]: k for k in g["layer_kinds"]}

    for i in range(first_dense):
        nodes = kinds[seq[i]]["nodes"]
        assert not any(n["op"] == "GroupedGEMM" for n in nodes), (
            f"layer {i} is within first_k_dense_replace {first_dense} and must run a "
            f"dense MLP, but it carries routed experts")
        widths = {n["n"] for n in nodes if n.get("role") == "mlp_gate_up"}
        assert widths == {2 * t["intermediate_size"]}, (
            f"layer {i}'s gate-and-up projection is {widths}; a dense KimiMLP is "
            f"2 * intermediate_size ({2 * t['intermediate_size']}) wide, and uses "
            f"intermediate_size rather than moe_intermediate_size")
    # And the layer just past the prologue IS routed, or the prologue swallowed too much.
    assert any(n["op"] == "GroupedGEMM"
               for n in kinds[seq[first_dense]]["nodes"]), (
        f"layer {first_dense} is the first routed layer and carries no experts")


def test_kimi_k3_routed_experts_run_at_the_latent_width():
    """routed_expert_hidden_size is the latent MoE width, and K is that, not hidden_size.

    KimiSparseMoeBlock projects hidden -> routed_expert_hidden_size before routing, runs
    each expert at that width, and projects back after
    (modeling_kimi_linear.py:776-832). The alias table recognized only the
    DeepSeek/Nemotron spelling, so every Kimi expert was priced at K=7168 instead of
    3584 -- double the expert FLOPs and dispatch volume -- and the two projections were
    missing entirely."""
    d = ROOT / "models" / "kimi-k3"
    if not d.is_dir():
        pytest.skip("kimi-k3 is not in the catalog")
    g, t = graph_of(d), text_config(config_of(d))
    latent = int(t["routed_expert_hidden_size"])
    hidden = int(t["hidden_size"])
    assert latent != hidden, "this test is about a config whose latent width differs"

    experts = [n for k in g["layer_kinds"] for n in k["nodes"]
               if n["op"] == "GroupedGEMM"]
    assert experts, "no routed expert nodes"
    for n in experts:
        assert n.get("latent_size") == latent, (
            f"expert node states latent_size {n.get('latent_size')}, config says "
            f"{latent}")

    # The projections are real work every token pays, so they are nodes.
    for k in g["layer_kinds"]:
        if not any(n["op"] == "GroupedGEMM" for n in k["nodes"]):
            continue
        down = [n for n in k["nodes"] if n.get("role") == "routed_expert_down_proj"]
        up = [n for n in k["nodes"] if n.get("role") == "routed_expert_up_proj"]
        assert len(down) == 1 and len(up) == 1, (
            f"{k['id']}: a latent MoE needs both projections, got "
            f"{len(down)} down and {len(up)} up")
        assert (down[0]["n"], down[0]["k"]) == (latent, hidden), (
            f"{k['id']}: down-projection is {down[0]['n']}x{down[0]['k']}, expected "
            f"{latent}x{hidden}")
        assert (up[0]["n"], up[0]["k"]) == (hidden, latent), (
            f"{k['id']}: up-projection is {up[0]['n']}x{up[0]['k']}, expected "
            f"{hidden}x{latent}")


def test_a_latent_moe_states_its_projections_wherever_it_appears():
    """The property across the catalog, not just for Kimi.

    Any node carrying latent_size describes a projection into and out of that width, so
    the layer must price both. Nemotron-3-Ultra states the same structure as
    moe_latent_size and had the same omission."""
    checked = 0
    for d in model_dirs():
        for k in graph_of(d)["layer_kinds"]:
            latent = {n.get("latent_size") for n in k["nodes"]
                      if n["op"] == "GroupedGEMM" and n.get("latent_size")}
            if not latent:
                continue
            assert len(latent) == 1, f"{d.name}/{k['id']}: mixed latent widths {latent}"
            width = latent.pop()
            roles = {n.get("role") for n in k["nodes"]}
            assert "routed_expert_down_proj" in roles, (
                f"{d.name}/{k['id']}: latent_size {width} with no projection into it")
            assert "routed_expert_up_proj" in roles, (
                f"{d.name}/{k['id']}: latent_size {width} with no projection out of it")
            checked += 1
    assert checked, "no latent MoE layers in the catalog to check"


def test_a_sparse_latent_read_prices_the_selection_it_claims():
    """A graph recording index_topk must also charge for the indexer that selects.

    The representation gap behind two defects in opposite directions: DeepSeek-V4
    priced an indexer on 31 layers that have none, and the GLM-5 family recorded
    index_topk on all 78 layers and emitted no indexer at all. The indexer is the only
    context-proportional term in such a layer, so either way the dominant long-context
    cost is wrong.

    A property over the catalog rather than a per-model assertion, so a future family
    cannot reintroduce it."""
    checked = 0
    for d in model_dirs():
        g = graph_of(d)
        for k in g["layer_kinds"]:
            selects = [n for n in k["nodes"]
                       if n["op"] == "Attention" and n.get("index_topk")]
            if not selects:
                continue
            scorers = [n for n in k["nodes"] if n.get("role") == "block_index_scores"]
            assert scorers, (
                f"{d.name}/{k['id']}: index_topk "
                f"{selects[0]['index_topk']} asserts a top-k selection, but the layer "
                f"has no block_index_scores node to charge for making it")
            # Two spellings, because the catalog holds two indexer shapes: the shared
            # lightning_indexer's wq_b (read from the latent rank, as both deepseek_v32
            # and glm5next build it) and DeepSeek-V4's own per-tensor wq_proj. All that
            # matters here is that SOME projection feeds the scorer.
            assert any(str(n.get("role", "")).startswith(
                ("index_wq_b", "index_wq_proj")) for n in k["nodes"]), (
                f"{d.name}/{k['id']}: an indexer with no projection to feed it")
            for n in scorers:
                assert "window" not in n, (
                    f"{d.name}/{k['id']}: the indexer scores the whole cache, so a "
                    f"window on it would make selection look bounded")
            checked += 1
    assert checked, "no selecting layers in the catalog to check"


def test_glm5_family_prices_its_dsa_indexer():
    """GLM-5 is model_type glm_moe_dsa and ships the full lightning-indexer spec.

    It is registered to handler_moe, which had no indexer path, so the selection work
    was unpriced on every layer of all four generations.

    The shapes are vLLM's: GlmMoeDsaForCausalLM is served by the deepseek_v32 module
    (its __init__ aliases GlmMoeDsaForCausalLM = DeepseekV32ForCausalLM), whose indexer
    builds wq_b from q_lora_rank and one fused wk_weights_proj from hidden
    (models/deepseek_v32/attention.py:71-85). An earlier revision priced a single
    hidden-wide GEMM instead, which had the wrong k on the query stage.

    Only the layers the config indexes are checked, because the same config states a
    skip pattern -- see test_indexer_runs_only_on_the_layers_the_config_indexes."""
    seen = 0
    for name in ("glm-5", "glm-5.2", "glm-5.2-fp8", "glm-5.3"):
        d = ROOT / "models" / name
        if not d.is_dir():
            continue
        g, t = graph_of(d), text_config(config_of(d))
        assert t.get("index_n_heads") and t.get("index_topk"), (
            f"{name}: this test is about a config declaring an indexer")
        heads, dim = t["index_n_heads"], t["index_head_dim"]
        q_lora = int(t["q_lora_rank"])
        indexed_kinds = 0
        for k in g["layer_kinds"]:
            score = [n for n in k["nodes"] if n.get("role") == "block_index_scores"]
            if not score:
                continue
            indexed_kinds += 1
            gemms = {n.get("role"): (n["n"], n["k"]) for n in k["nodes"]
                     if n["op"] == "GEMM"}
            assert len(score) == 1, (
                f"{name}/{k['id']}: {len(score)} scorers in one layer kind")
            assert "index_qk_proj" not in gemms, (
                f"{name}/{k['id']}: still carries the fused index_qk_proj, which "
                f"sources the query stage from hidden instead of the latent rank")
            assert gemms["index_wq_b"] == (heads * dim, q_lora), (
                f"{name}/{k['id']}: index_wq_b is {gemms['index_wq_b']}; vLLM builds "
                f"{heads} * {dim} from the latent rank {q_lora}")
            assert gemms["index_wk_weights_proj"] == (dim + heads, t["hidden_size"]), (
                f"{name}/{k['id']}: index_wk_weights_proj is "
                f"{gemms['index_wk_weights_proj']}; vLLM fuses wk and weights_proj "
                f"into one GEMM of [{dim}, {heads}] from {t['hidden_size']}")
            # Shaped from the config's own indexer dimensions, not the model's heads.
            assert score[0]["n_q"] == heads
            assert score[0]["d_h"] == dim
            # The read it feeds is bounded; the scoring that selects it is not.
            assert "window" not in score[0]
        assert indexed_kinds, f"{name}: no layer kind carries an indexer"
        seen += 1
    if not seen:
        pytest.skip("no GLM-5 models in the catalog")


def test_gpt_oss_quantizes_only_its_experts():
    """modules_to_not_convert names what MXFP4 leaves alone, and it must be honored.

    gpt-oss declares quant_method mxfp4 with self_attn, the router, the embeddings and
    lm_head excluded, so MXFP4 is the expert weights only. Returning it as the global
    width priced every attention projection and the head at 4 bits instead of 16 -- a
    4x understatement on the whole non-expert parameter class, about 2.1 GiB here.

    The mirror of the expert_dtype case: there the experts are narrower than a global
    that is itself quantized; here they are the only narrow thing."""
    d = ROOT / "models" / "gpt-oss-120b"
    if not d.is_dir():
        pytest.skip("gpt-oss-120b is not in the catalog")
    g, t = graph_of(d), text_config(config_of(d))
    quant = t.get("quantization_config") or {}
    excluded = quant.get("modules_to_not_convert")
    assert excluded, "this test is about a config that excludes modules from quantization"
    assert str(quant.get("quant_method")).lower() == "mxfp4"

    assert g["global"]["weight_dtype"] != "mxfp4", (
        "the global width must be the UNQUANTIZED base, since attention, the router, "
        "the embeddings and lm_head are all excluded from the MXFP4 scope")
    assert g["global"]["weight_dtype"] == "bf16"

    # The experts, which ARE in scope, carry the narrow width themselves.
    experts = [n for k in g["layer_kinds"] for n in k["nodes"]
               if n["op"] == "GroupedGEMM"]
    assert experts, "no expert nodes"
    for n in experts:
        assert n.get("weight_dtype") == "mxfp4", (
            f"expert node carries weight_dtype {n.get('weight_dtype')!r}; the MXFP4 "
            f"scope is exactly these weights")

    # And the excluded tensors are NOT narrowed: no override, so they take the global.
    for k in g["layer_kinds"]:
        for n in k["nodes"]:
            if n["op"] == "GEMM":
                assert n.get("weight_dtype") in (None, "bf16"), (
                    f"{k['id']}/{n.get('role')} is priced at "
                    f"{n.get('weight_dtype')!r}, but self_attn is excluded from the "
                    f"MXFP4 scope")
    for n in g["head"]:
        assert n.get("weight_dtype") in (None, "bf16"), (
            f"the head's {n.get('role')} is priced at {n.get('weight_dtype')!r}; "
            f"lm_head is excluded from the MXFP4 scope")


def test_llama4_chunked_attention_is_a_recorded_gap_not_a_silent_one():
    """The decision to price Llama-4 as unbounded must stay deliberate.

    attention_chunk_size 8192 bounds most layers in the implementation, and this
    checkpoint declares no layer_types, no moe_layers and an empty no_rope_layers, so
    the per-layer split is not derivable -- deriving one would mean inventing it. The
    handler says so. This test pins the two halves of that claim: the graph really is
    unbounded, and the config really does lack the fields that would let it not be. If a
    future checkpoint declares layer_types, this fails and the gap gets closed."""
    d = ROOT / "models" / "llama-4-scout-17b-16e-instruct-fp8-dynamic"
    if not d.is_dir():
        pytest.skip("llama-4-scout is not in the catalog")
    g, t = graph_of(d), text_config(config_of(d))
    assert t.get("attention_chunk_size"), "this test is about a chunked-attention config"
    assert not t.get("layer_types"), (
        "the config now declares layer_types, so the chunked/full split IS derivable "
        "and handler_moe's recorded gap should be closed rather than documented")
    assert not t.get("no_rope_layers"), (
        "no_rope_layers is now populated, so the full-attention layers are enumerated "
        "and the split should be derived")
    main = [n for k in g["layer_kinds"] for n in k["nodes"]
            if n["op"] == "Attention" and n.get("role") != "block_index_scores"]
    assert main
    for n in main:
        assert n["kind"] == "gqa" and "window" not in n, (
            "a bound appeared here without the config gaining the fields to derive it")
    # The gap must stay documented in the handler a reader would check.
    src = (ROOT / "scripts" / "derive_graph.py").read_text()
    assert "KNOWN GAP (Llama-4)" in src, (
        "the handler no longer records the chunked-attention gap; either close it or "
        "keep the note that explains why it is open")


BYTES_PER_PARAM = {"bf16": 2.0, "fp16": 2.0, "fp8": 1.0, "int8": 1.0,
                   "nvfp4": 0.5, "mxfp4": 0.5, "int4": 0.5}


def weight_bytes(g: dict) -> float:
    """Total weight bytes the stack holds, honouring per-node dtype overrides."""
    base = g["global"]["weight_dtype"]
    kinds = {k["id"]: k for k in g["layer_kinds"]}
    total = 0.0
    for lid in layer_sequence(g["stack"]):
        for n in kinds[lid]["nodes"]:
            w = BYTES_PER_PARAM[n.get("weight_dtype") or base]
            if n["op"] == "GEMM":
                total += n["n"] * n["k"] * w
            elif n["op"] == "GroupedGEMM":
                total += 3 * n["n"] * n["k"] * n["experts"] * w
    return total


def test_nemotron_nvfp4_keeps_its_declared_mixed_precision_layout():
    """An explicit quantized_layers map states a LAYOUT, not one width.

    nemotron-3-ultra-nvfp4 names 49,152 routed-expert matrices NVFP4, 192 Mamba and
    shared-expert matrices FP8, and ignores 243 more -- attention, the latent
    projections, embeddings and the head -- which stay at the declared bf16 base.
    Collapsing that to a single dominant width priced every ignored tensor at four
    bits, including the latent projections ModelOpt explicitly excludes."""
    d = ROOT / "models" / "nemotron-3-ultra-550b-a55b-nvfp4"
    if not d.is_dir():
        pytest.skip("nemotron-3-ultra-nvfp4 is not in the catalog")
    g, t = graph_of(d), text_config(config_of(d))
    quant = t.get("quantization_config") or {}
    assert quant.get("quantized_layers"), "this test is about a config stating the map"

    assert g["global"]["weight_dtype"] == "bf16", (
        "the base dtype stays global; the quantized widths ride on their own nodes")

    by_role = {}
    for k in g["layer_kinds"]:
        for n in k["nodes"]:
            if n["op"] in ("GEMM", "GroupedGEMM"):
                by_role[n.get("role")] = n.get("weight_dtype")

    assert by_role.get("experts") == "nvfp4", "the routed experts are the NVFP4 tensors"
    assert by_role.get("shared_experts") == "fp8", (
        "the shared experts are stored at FP8, so they cannot ride on the routed node")
    assert by_role.get("mixer_out") == "fp8", "the Mamba projections are FP8"
    for role in ("qkv_proj", "o_proj", "routed_expert_down_proj",
                 "routed_expert_up_proj"):
        if role in by_role:
            assert by_role[role] is None, (
                f"{role} is in the checkpoint's ignore list and must keep the bf16 "
                f"base, not {by_role[role]!r}")


def test_nemotron_nvfp4_occupies_more_than_a_uniform_four_bit_reading():
    """The occupancy the layout decides, which is the cost of getting it wrong.

    Asserted through bytes rather than labels: pricing the whole checkpoint at NVFP4
    understates it, and pricing it all at bf16 overstates it, so the mixed reading must
    land strictly between."""
    d = ROOT / "models" / "nemotron-3-ultra-550b-a55b-nvfp4"
    if not d.is_dir():
        pytest.skip("nemotron-3-ultra-nvfp4 is not in the catalog")
    g = graph_of(d)
    mixed = weight_bytes(g)

    import copy
    all_four_bit = copy.deepcopy(g)
    all_four_bit["global"]["weight_dtype"] = "nvfp4"
    for k in all_four_bit["layer_kinds"]:
        for n in k["nodes"]:
            n.pop("weight_dtype", None)
    uniform = weight_bytes(all_four_bit)

    all_base = copy.deepcopy(g)
    for k in all_base["layer_kinds"]:
        for n in k["nodes"]:
            n.pop("weight_dtype", None)
    base = weight_bytes(all_base)

    assert uniform < mixed < base, (
        f"the mixed layout must sit between a uniform 4-bit reading "
        f"({uniform / 2**30:.1f} GiB) and a uniform bf16 one ({base / 2**30:.1f} GiB), "
        f"got {mixed / 2**30:.1f} GiB")


def test_v4_projections_are_low_rank_stages_not_dense_rectangles():
    """DeepSeek-V4's projection path is low-rank on both sides, with a norm between.

    The input is fused_wqa_wkv (hidden -> q_lora_rank + head_dim), then q_norm, then
    wq_b (q_lora_rank -> n_heads*head_dim). The output is a grouped wo_a then wo_b. A
    single hidden-width rectangle overstated the input side 4.1x and the output side
    2.5x -- about 39B parameters of phantom weight over 61 layers -- and the
    intervening norm means the input stages cannot be fused into one GEMM anyway."""
    g, t = _v4()
    hidden, nq = t["hidden_size"], t["num_attention_heads"]
    hd, q_lora = t["head_dim"], t["q_lora_rank"]
    o_lora, o_groups = t["o_lora_rank"], t["o_groups"]

    for k in g["layer_kinds"]:
        roles = {n.get("role"): n for n in k["nodes"] if n["op"] == "GEMM"}
        assert "qkv_proj" not in roles, (
            f"{k['id']}: a dense QKV rectangle is not this family's projection path")
        assert "o_proj" not in roles, (
            f"{k['id']}: a dense output rectangle is not this family's output path")

        fused, wq_b = roles["fused_wqa_wkv"], roles["wq_b"]
        assert (fused["n"], fused["k"]) == (q_lora + hd, hidden)
        assert (wq_b["n"], wq_b["k"]) == (nq * hd, q_lora)

        wo_a, wo_b = roles["wo_a"], roles["wo_b"]
        assert (wo_a["n"], wo_a["k"]) == (o_groups * o_lora, nq * hd // o_groups)
        assert (wo_b["n"], wo_b["k"]) == (hidden, o_groups * o_lora)

        # The norm between the two input stages, without which they would be one GEMM.
        assert any(n.get("role") == "q_norm" for n in k["nodes"]), (
            f"{k['id']}: no q_norm between the q-LoRA down- and up-projections")

    # The totals, so a future refactor cannot drift while keeping the role names.
    kinds = {k["id"]: k for k in g["layer_kinds"]}
    k0 = kinds[layer_sequence(g["stack"])[0]]
    roles = {n.get("role"): n for n in k0["nodes"] if n["op"] == "GEMM"}
    assert (roles["fused_wqa_wkv"]["n"] * roles["fused_wqa_wkv"]["k"]
            + roles["wq_b"]["n"] * roles["wq_b"]["k"]) == 115_343_360
    assert (roles["wo_a"]["n"] * roles["wo_a"]["k"]
            + roles["wo_b"]["n"] * roles["wo_b"]["k"]) == 184_549_376


def test_kimi_k3_kda_layer_prices_its_mixer_projections():
    """A RecurrentUpdate carries no N or K, so the projections must be their own nodes.

    The KDA mixer emitted only RecurrentUpdate and an AllReduce -- not even a producer
    GEMM for the reduction that followed it -- dropping 443,580,416 parameters per layer
    over 69 KDA layers, about 30.6B, plus their FLOPs.

    Shapes from the pinned KimiDeltaAttention: three separate q/k/v projections, a
    low-rank decay gate through head_dim, one beta scalar per head, a full-rank output
    gate because this config sets use_full_rank_gate, and an output projection."""
    d = ROOT / "models" / "kimi-k3"
    if not d.is_dir():
        pytest.skip("kimi-k3 is not in the catalog")
    g, t = graph_of(d), text_config(config_of(d))
    linear = t["linear_attn_config"]
    hidden = t["hidden_size"]
    heads, dim = linear["num_heads"], linear["head_dim"]
    inner = heads * dim

    kda = next(k for k in g["layer_kinds"] if k["id"].startswith("kda_"))
    roles = {n.get("role"): n for n in kda["nodes"] if n["op"] == "GEMM"}
    expected = {
        "kda_q_proj": (inner, hidden),
        "kda_k_proj": (inner, hidden),
        "kda_v_proj": (inner, hidden),
        "kda_f_a_proj": (dim, hidden),
        "kda_f_b_proj": (inner, dim),
        "kda_b_proj": (heads, hidden),
        "kda_o_proj": (hidden, inner),
    }
    if linear.get("use_full_rank_gate"):
        expected["kda_g_proj"] = (inner, hidden)
    else:
        expected["kda_g_a_proj"] = (dim, hidden)
        expected["kda_g_b_proj"] = (inner, dim)

    for role, (n, k) in expected.items():
        assert role in roles, f"the KDA mixer omits {role}"
        assert (roles[role]["n"], roles[role]["k"]) == (n, k), (
            f"{role} is {roles[role]['n']}x{roles[role]['k']}, expected {n}x{k}")

    total = sum(roles[r]["n"] * roles[r]["k"] for r in expected)
    assert total == 443_580_416, (
        f"the KDA mixer prices {total:,} projection parameters per layer, expected "
        f"443,580,416")

    # The reduction must have something producing the value it reduces.
    ops = [n["op"] for n in kda["nodes"]]
    assert ops.index("GEMM") < ops.index("AllReduce"), (
        "the mixer reduces an output no node in the layer produced")


# --- The hybrid handlers' specific structure ---------------------------------------
# The property tests above confirm the catalog HAS variety -- layer counts agree, dtypes
# differ, attention shapes come from the config. None of them asserts that a given model
# resolved to the RIGHT structure: which layers are attention and which are recurrent,
# and what state geometry the recurrent ones carry. For a handler that detects an index
# base or parses a character vector, that is the part most likely to be off by one.


def test_kimi_k3_resolves_its_full_attention_layers_one_based():
    """The off-by-one this handler detects rather than assumes.

    Kimi-K3 declares linear_attn_config.full_attn_layers as a 1-based list against a
    1-indexed layer numbering. Read as 0-based, every attention layer shifts by one and
    the model is still 93 layers of the right two kinds -- so the layer-count and
    attention-shape property tests both pass on a graph where every KDA and MLA layer
    sits one position off. The handler detects the base (a 0-based list would contain 0)
    and this pins the result: the declared index n must be the MLA layer at 0-based
    position n-1, and the position before it must be KDA."""
    d = ROOT / "models" / "kimi-k3"
    if not d.is_dir():
        pytest.skip("kimi-k3 is not in the catalog")
    g, t = graph_of(d), text_config(config_of(d))
    declared = [int(i) for i in t["linear_attn_config"]["full_attn_layers"]]
    assert 0 not in declared, (
        "this config's list is 1-based; a 0-based one would make the assertions below "
        "describe the wrong positions")
    seq = layer_sequence(g["stack"])
    assert len(seq) == t["num_hidden_layers"]

    mla = {i for i, k in enumerate(seq) if k.startswith("mla_")}
    expected = {n - 1 for n in declared}
    assert mla == expected, (
        f"MLA layers sit at 0-based {sorted(mla)[:6]}...; the config's 1-based list "
        f"{declared[:6]}... puts them at {sorted(expected)[:6]}...")
    # The concrete consequence, stated so a failure reads as the off-by-one it is:
    # index 4 declared means position 3 is attention and position 4 is not.
    assert seq[declared[0] - 1].startswith("mla_")
    assert seq[declared[0]].startswith("kda_"), (
        "the layer after the first declared attention layer is attention too, which is "
        "the signature of a 0-based read")
    assert set(seq) <= {"kda_moe", "mla_moe", "kda_dense", "mla_dense"}


def test_kimi_k3_kda_state_geometry_comes_from_the_config():
    """The recurrent node's shape, which decides its state bytes and its step cost."""
    d = ROOT / "models" / "kimi-k3"
    if not d.is_dir():
        pytest.skip("kimi-k3 is not in the catalog")
    g, t = graph_of(d), text_config(config_of(d))
    kda = [n for k in g["layer_kinds"] if k["id"] == "kda_moe"
           for n in k["nodes"] if n["op"] == "RecurrentUpdate"]
    assert len(kda) == 1, f"expected one recurrent node in a KDA layer, got {len(kda)}"
    n = kda[0]
    assert n["recurrent_kind"] == "kda"
    assert n["n_heads"] == t["num_attention_heads"]
    assert n["state_size"] == t["linear_attn_config"]["head_dim"]
    assert n["state_dtype"] == "fp32", (
        "a recurrent state carried at the weight dtype would understate its bytes")


def test_qwen3_5_resolves_full_attention_on_the_declared_vector():
    """Which layers are GDN and which are full attention, from layer_types.

    The handler reads the vector rather than full_attention_interval, so this asserts
    against the vector: a graph built from the interval would agree here only while the
    two agree, which is the point of preferring the vector."""
    d = ROOT / "models" / "qwen3.5-397b-a17b"
    if not d.is_dir():
        pytest.skip("qwen3.5-397b-a17b is not in the catalog")
    g, t = graph_of(d), text_config(config_of(d))
    types = t["layer_types"]
    seq = layer_sequence(g["stack"])
    assert len(seq) == len(types)
    expected = ["attn_moe" if x == "full_attention" else "gdn_moe" for x in types]
    assert seq == expected, "the stack does not follow the declared layer_types vector"
    # The interval is a generator for the common case; the first full-attention layer
    # sits at the end of the first period, not the start of it.
    interval = int(t["full_attention_interval"])
    assert seq[interval - 1] == "attn_moe"
    assert seq[0] == "gdn_moe", (
        "layer 0 is full attention, which is what reading the interval as 1-based at "
        "the start of each period would produce")


def test_qwen3_5_gdn_state_geometry_follows_vllm_calculator():
    """The GDN state shape, which vLLM's own calculator defines.

    A convolutional state of width 2*k_heads*k_dim + v_heads*v_dim, and a temporal state
    whose per-head geometry is k_dim over v_heads heads. These are the numbers that decide
    the recurrent state's bytes per sequence, so a transposed head count or a key/value
    dim swap misprices every GDN layer and no other test would see it."""
    d = ROOT / "models" / "qwen3.5-397b-a17b"
    if not d.is_dir():
        pytest.skip("qwen3.5-397b-a17b is not in the catalog")
    g, t = graph_of(d), text_config(config_of(d))
    gdn = [n for k in g["layer_kinds"] if k["id"] == "gdn_moe"
           for n in k["nodes"] if n["op"] == "RecurrentUpdate"]
    assert len(gdn) == 1
    n = gdn[0]
    k_heads, v_heads = t["linear_num_key_heads"], t["linear_num_value_heads"]
    k_dim, v_dim = t["linear_key_head_dim"], t["linear_value_head_dim"]
    assert n["recurrent_kind"] == "gdn"
    assert n["n_heads"] == v_heads, (
        f"n_heads is {n['n_heads']}; the temporal state is per VALUE head ({v_heads}), "
        f"not per key head ({k_heads})")
    assert n["state_size"] == k_dim, (
        f"state_size is {n['state_size']}; the temporal state's per-head width is the "
        f"KEY dim ({k_dim})")
    assert n["conv_kernel"] == t["linear_conv_kernel_dim"]
    assert n["intermediate_size"] == 2 * k_heads * k_dim + v_heads * v_dim, (
        "intermediate_size must be the convolution's width, "
        "2*k_heads*k_dim + v_heads*v_dim")


def test_qwen3_5_gdn_state_size_is_the_key_dim_not_the_value_dim():
    """The committed config cannot tell these two apart, so this uses one that can.

    Qwen3.5-397B declares linear_key_head_dim and linear_value_head_dim both 128, so an
    assertion against the real config passes whichever of the two the handler reads --
    verified by mutation: swapping them in the deriver leaves the whole suite green. The
    distinction is real (vLLM's gated_delta_net_state_shape takes the temporal state's
    per-head width from the KEY dim) and it decides the recurrent state's bytes, so it is
    pinned here on a config where the two differ."""
    dg = _dg()
    cfg = {
        "hidden_size": 4096, "vocab_size": 128000, "num_hidden_layers": 4,
        "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
        "intermediate_size": 8192, "num_local_experts": 8, "num_experts_per_tok": 2,
        "moe_intermediate_size": 1024,
        "layer_types": ["linear_attention", "linear_attention",
                        "linear_attention", "full_attention"],
        "full_attention_interval": 4,
        "linear_num_key_heads": 16, "linear_num_value_heads": 32,
        "linear_key_head_dim": 64, "linear_value_head_dim": 256,
        "linear_conv_kernel_dim": 4,
    }
    kinds, _ = dg.handler_qwen3_5_moe(cfg, cfg, "synthetic")
    n = next(x for k in kinds if k["id"] == "gdn_moe"
             for x in k["nodes"] if x["op"] == "RecurrentUpdate")
    assert n["state_size"] == 64, (
        f"state_size is {n['state_size']}; the temporal state's per-head width is the "
        f"key dim (64), not the value dim (256)")
    assert n["n_heads"] == 32, (
        f"n_heads is {n['n_heads']}; the temporal state is per value head (32), not per "
        f"key head (16)")
    assert n["intermediate_size"] == 2 * 16 * 64 + 32 * 256


def test_nemotron_h_depth_and_kinds_come_from_the_layer_vector():
    """Depth from a vector, on a config that states no layer count at all.

    Nemotron-3-Ultra declares no num_hidden_layers: its depth IS the length of
    layers_block_type, and the MoE entries are MLP layers rather than separate blocks,
    which is why the vector is longer than a reader expecting a transformer depth would
    guess. This asserts both the depth and the per-position kind."""
    d = ROOT / "models" / "nemotron-3-ultra-550b-a55b-bf16"
    if not d.is_dir():
        pytest.skip("nemotron-3-ultra is not in the catalog")
    g, t = graph_of(d), text_config(config_of(d))
    vector = t["layers_block_type"]
    assert t.get("num_hidden_layers") is None, (
        "this test is about the config that states no layer count; it now states one")
    seq = layer_sequence(g["stack"])
    assert seq == list(vector), "the stack must be the declared vector, position by position"
    assert {k["id"] for k in g["layer_kinds"]} == set(vector)


def test_nemotron_h_parses_the_character_form_of_the_layer_vector():
    """The M/*/-/E character form, which no committed config exercises.

    handler_nemotron_h accepts either a list of names or a character string, and every
    config in the catalog today uses the list. The character branch is therefore live
    code that real data does not reach, so it is tested directly: M state-space,
    * attention, - MLP, E MoE, and an unknown character must raise rather than default."""
    dg = _dg()
    cfg = {
        "hidden_size": 4096, "vocab_size": 128000, "num_attention_heads": 32,
        "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 8192,
        "num_local_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 1024,
        "mamba_num_heads": 64, "ssm_state_size": 128, "conv_kernel": 4,
        "mamba_head_dim": 64, "n_groups": 8,
        "hybrid_override_pattern": "M-*E",
    }
    kinds, stack = dg.handler_nemotron_h(cfg, cfg, "synthetic")
    assert layer_sequence(stack) == ["mamba", "mlp", "attention", "moe"], (
        "the character form must map M/-/*/E to state-space, MLP, attention and MoE")
    assert {k["id"] for k in kinds} == {"mamba", "mlp", "attention", "moe"}

    bad = dict(cfg, hybrid_override_pattern="M-*X")
    with pytest.raises(dg.DeriveError) as exc:
        dg.handler_nemotron_h(bad, bad, "synthetic")
    assert "X" in str(exc.value), (
        "an unknown layer character must be named in the error rather than defaulted")


# --- Where the collectives land ----------------------------------------------------
# A collective's `emit` condition is excluded from cost_signature on purpose: it selects
# whether the node exists in a given deployment, which is a layout property rather than a
# model one. That leaves it verified nowhere, which these cover structurally. This is the
# dense-vs-MoE distinction the behavioural signature deliberately cannot see.


def test_collectives_land_where_the_parallelism_needs_them():
    """Every layer reduces its MLP output; only a routed layer dispatches to experts.

    A dense layer under tensor parallelism reduces after its MLP. A routed layer also
    exchanges tokens with the ranks holding the experts, which is an All2All under expert
    parallelism. An All2All on a dense layer would price traffic that never moves; a
    missing one on a routed layer would drop the dominant collective of an MoE step."""
    checked_dense = checked_moe = 0
    for d in model_dirs():
        g = graph_of(d)
        for kind in g["layer_kinds"]:
            ops = [n for n in kind["nodes"] if n["op"] in ("AllReduce", "All2All")]
            all2all = [n for n in ops if n["op"] == "All2All"]
            allreduce = [n for n in ops if n["op"] == "AllReduce"]
            has_experts = any(n["op"] == "GroupedGEMM" for n in kind["nodes"])
            where = f"{d.name}/{kind['id']}"

            assert allreduce, f"{where}: a layer that reduces nothing"
            assert any(n["emit"] in ("tensor_parallel", "tensor_parallel_unless_sp_moe")
                       for n in allreduce), (
                f"{where}: no AllReduce conditioned on tensor parallelism")

            if has_experts:
                assert len(all2all) == 1, (
                    f"{where}: a routed layer with {len(all2all)} All2All nodes; an MoE "
                    f"step dispatches to the expert ranks exactly once")
                assert all2all[0]["emit"] == "expert_parallel", (
                    f"{where}: All2All emits on {all2all[0]['emit']!r}, but expert "
                    f"dispatch is what expert parallelism conditions")
                assert any(n["emit"] == "tensor_parallel_unless_sp_moe"
                           for n in allreduce), (
                    f"{where}: a routed layer's MLP reduction must stand down under "
                    f"sequence-parallel MoE, which is what the _unless_sp_moe form says")
                checked_moe += 1
            else:
                assert not all2all, (
                    f"{where}: a layer with no routed experts emits All2All "
                    f"{[n['role'] for n in all2all]}, pricing traffic that never moves")
                checked_dense += 1
    assert checked_dense and checked_moe, (
        f"the catalog must exercise both shapes; saw {checked_dense} dense and "
        f"{checked_moe} routed layer kinds")


def test_every_emit_condition_is_one_a_resolver_knows():
    """The closed set is the point: a graph cannot state a condition nothing evaluates.

    The deriver's three EMIT_ constants are the whole vocabulary, so a committed graph
    naming anything else would reach a resolver with no branch for it."""
    dg = _dg()
    known = {dg.EMIT_TENSOR_PARALLEL, dg.EMIT_EXPERT_PARALLEL,
             dg.EMIT_TENSOR_PARALLEL_UNLESS_SP_MOE}
    for d in model_dirs():
        for kind in graph_of(d)["layer_kinds"]:
            for n in kind["nodes"]:
                if "emit" not in n:
                    continue
                assert n["emit"] in known, (
                    f"{d.name}/{kind['id']}: emit {n['emit']!r} is not in the closed set "
                    f"{sorted(known)}")


def test_a_recurrent_layer_reduces_its_mixer_output():
    """A recurrent mixer is tensor-parallel the same way attention is.

    Its output projection is sharded, so it reduces unconditionally under TP rather than
    under the MoE-aware form an MLP output uses."""
    seen = 0
    for d in model_dirs():
        for kind in graph_of(d)["layer_kinds"]:
            if not any(n["op"] == "RecurrentUpdate" for n in kind["nodes"]):
                continue
            reduces = [n for n in kind["nodes"] if n["op"] == "AllReduce"]
            assert any(n["emit"] == "tensor_parallel" for n in reduces), (
                f"{d.name}/{kind['id']}: a recurrent layer with no unconditional "
                f"tensor-parallel reduction of its mixer output")
            seen += 1
    if not seen:
        pytest.skip("no recurrent layers in the catalog")


# --- The speculator block ----------------------------------------------------------
# A draft stack is work a deployment pays for on every step, so a wrong draft length or
# a draft stack mirrored from the wrong layer kind misprices the model. The source field
# differs by family and the candidates DISAGREE on a shipped config, so which one is read
# is a correctness question rather than a spelling one.


def test_minimax_m3_takes_its_draft_length_from_num_mtp_modules():
    """The case that makes the source field a correctness question.

    MiniMax-M3's config declares BOTH num_mtp_modules (7) and
    num_nextn_predict_layers (1). vLLM reads num_mtp_modules for this family
    (config/speculative.py, minimax_m3_mtp), so reading the other field would understate
    the draft stack sevenfold. Asserted against the config rather than against a literal,
    so the test still means something after a config update."""
    d = ROOT / "models" / "minimax-m3"
    if not d.is_dir():
        pytest.skip("minimax-m3 is not in the catalog")
    g, t = graph_of(d), text_config(config_of(d))
    assert t.get("num_mtp_modules") != t.get("num_nextn_predict_layers"), (
        "this test is about a config whose two draft-length fields disagree; they now "
        "agree, so it no longer pins the distinction it was written for")
    spec = g["speculator"]
    assert spec["method"] == "minimax_m3_mtp"
    assert spec["num_spec_tokens"] == t["num_mtp_modules"], (
        f"draft length is {spec['num_spec_tokens']}, and num_mtp_modules is "
        f"{t['num_mtp_modules']}; num_nextn_predict_layers is "
        f"{t.get('num_nextn_predict_layers')} and reading it here would be the bug")


def test_deepseek_v3_takes_its_draft_length_from_num_nextn_predict_layers():
    """The other branch: the alias table's field, not num_mtp_modules.

    vLLM maps model_type deepseek_v3 to deepseek_mtp and reads n_predict from
    num_nextn_predict_layers. DeepseekV3ForCausalLM is deliberately NOT in
    MTP_MODULE_ARCHS, and this pins that."""
    d = ROOT / "models" / "deepseek-v3"
    if not d.is_dir():
        pytest.skip("deepseek-v3 is not in the catalog")
    g, t = graph_of(d), text_config(config_of(d))
    spec = g["speculator"]
    assert spec["method"] == "deepseek_mtp"
    assert spec["num_spec_tokens"] == t["num_nextn_predict_layers"]
    assert "num_mtp_modules" not in t, (
        "this config states num_mtp_modules too, so the assertion above no longer "
        "distinguishes the two sources")


def test_qwen3_5_takes_its_draft_length_from_mtp_num_hidden_layers():
    """The third spelling. Qwen3.5 states neither of the other two fields: its draft
    length rides in mtp_num_hidden_layers, resolved through the alias table."""
    d = ROOT / "models" / "qwen3.5-397b-a17b"
    if not d.is_dir():
        pytest.skip("qwen3.5-397b-a17b is not in the catalog")
    g, t = graph_of(d), text_config(config_of(d))
    spec = g["speculator"]
    assert spec["method"] == "qwen3_5_mtp"
    assert spec["num_spec_tokens"] == t["mtp_num_hidden_layers"]


def test_a_declared_mtp_module_count_alone_does_not_make_a_speculator():
    """num_mtp_modules is read only for the families whose method consumes it.

    MiniMax-M2.5 declares num_mtp_modules 3 and vLLM registers no minimax_m2_mtp method
    at all -- its only draft path is a separate Eagle3 checkpoint -- so this config must
    derive NO speculator. Reading the field globally would invent a draft stack here,
    and that invention is what MTP_MODULE_ARCHS being a named set rather than a pattern
    prevents."""
    d = ROOT / "models" / "minimax-m2.5"
    if not d.is_dir():
        pytest.skip("minimax-m2.5 is not in the catalog")
    g, t = graph_of(d), text_config(config_of(d))
    assert t.get("num_mtp_modules"), "this test is about a config that declares one"
    assert "speculator" not in g, (
        f"a speculator was derived from num_mtp_modules {t['num_mtp_modules']} for a "
        f"family with no registered MTP method")


def test_a_speculator_mirrors_a_layer_kind_the_graph_defines():
    """A draft stack naming a kind the graph does not define cannot be priced.

    Property test over the catalog: the draft stack is an MoE pass of its own for an MoE
    target, which a scalar draft length would hide, so its entries must resolve."""
    checked = 0
    for d in model_dirs():
        g = graph_of(d)
        spec = g.get("speculator")
        if not spec:
            continue
        defined = {k["id"] for k in g["layer_kinds"]}
        drafted = layer_sequence(spec["stack"])
        assert drafted, f"{d.name}: a speculator with an empty draft stack"
        unknown = set(drafted) - defined
        assert not unknown, f"{d.name}: draft stack names undefined kinds {unknown}"
        assert spec["num_spec_tokens"] > 0, f"{d.name}: non-positive draft length"
        assert spec.get("method"), f"{d.name}: a draft stack with no method"
        checked += 1
    assert checked >= 10, f"only {checked} speculators found; expected the catalog's 12"


def test_a_declared_mtp_vector_is_what_the_draft_stack_mirrors():
    """Where a config declares the draft module's layer composition, it is used.

    NemotronH states mtp_layers_block_type, and its draft stack must follow that vector
    rather than defaulting to the target's last layer kind."""
    d = ROOT / "models" / "nemotron-3.5-lightning-30b-a3b-bf16"
    if not d.is_dir():
        pytest.skip("nemotron-3.5-lightning is not in the catalog")
    g, t = graph_of(d), text_config(config_of(d))
    vector = t.get("mtp_layers_block_type")
    assert vector, "this test is about a config that declares one"
    assert layer_sequence(g["speculator"]["stack"]) == list(vector), (
        f"draft stack is {layer_sequence(g['speculator']['stack'])}, and the config "
        f"declares {vector}")


@pytest.mark.parametrize("mutate,description", [
    (lambda g: g["speculator"].update(num_spec_tokens=g["speculator"]["num_spec_tokens"] + 6),
     "lengthening the draft stack"),
    (lambda g: g["speculator"].update(method="some_other_mtp"),
     "changing the speculative method"),
    (lambda g: g["speculator"]["stack"].update(repeat=2),
     "running the draft stack twice"),
    (lambda g: g.pop("speculator"),
     "dropping the speculator entirely"),
])
def test_signature_notices_a_changed_speculator(mutate, description):
    """The gap this block was reported for: the signature must see a draft change.

    cost_signature iterated only the target stack, so a graph with the wrong draft
    length, the wrong method, or a mis-mirrored draft pattern collided with a correct one
    and passed test_distinct_architectures_cost_differently -- the strongest test here.
    These four mutations are the shapes that bug would take."""
    import copy

    d = ROOT / "models" / "minimax-m3"
    if not d.is_dir():
        pytest.skip("minimax-m3 is not in the catalog")
    g = graph_of(d)
    assert "speculator" in g, "the reference model must have a speculator"
    before = cost_signature(g)
    variant = copy.deepcopy(g)
    mutate(variant)
    assert cost_signature(variant) != before, (
        f"{description} left the cost signature unchanged, so a wrong draft stack would "
        f"pass the collision tests")


def test_signature_separates_draft_work_from_target_work():
    """A layer moved between the target stack and the draft stack must be visible.

    Without the draft stack tagged as its own component, a counter over both would be
    blind to the move: the same layer bodies in the same quantity, priced differently
    because a draft pass runs per speculated token rather than once."""
    import copy

    d = ROOT / "models" / "minimax-m3"
    if not d.is_dir():
        pytest.skip("minimax-m3 is not in the catalog")
    g = graph_of(d)
    moved = copy.deepcopy(g)
    drafted = layer_sequence(g["speculator"]["stack"])
    # Append the draft layers to the target stack and leave the speculator's own stack
    # claiming them too: total layer bodies unchanged in kind, but the split differs.
    moved["stack"] = {"prologue": layer_sequence(g["stack"]) + drafted}
    assert cost_signature(moved) != cost_signature(g), (
        "moving layers into the target stack did not change the signature")


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
    bounds = set()
    for k in g["layer_kinds"]:
        if k["id"] == "mtp_moe":
            continue  # the draft layer, which is not one of the target ratios
        for n in k["nodes"]:
            if n["op"] == "Attention" and n.get("role") != "block_index_scores":
                bounds.add((n.get("compress_ratio"), n.get("index_topk")))
    assert len(bounds) == len(ratios), (
        f"two compression ratios must give two distinct read bounds, got {bounds}"
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
                by_ratio.setdefault(ratio, set()).add(
                    (n["window"], n.get("compress_ratio"), n.get("index_topk")))
    assert all(len(v) == 1 for v in by_ratio.values()), (
        f"a ratio maps to more than one bound: {by_ratio}"
    )
    flat = {r: v.pop() for r, v in by_ratio.items()}
    lo, hi = min(flat), max(flat)
    # Every layer keeps the same sliding window; the ratio is what differs.
    assert flat[lo][0] == flat[hi][0] == t["sliding_window"], (
        f"both ratios retain the declared SWA window, got {flat}")
    assert flat[hi][1] == hi and flat[lo][1] == lo, (
        f"each layer must record its own compression ratio, got {flat}")
    # The read set is window + compressed positions, evaluated rather than compared as
    # a stored constant because the ratio-128 term is context-dependent and no constant
    # expresses it.
    def positions(bound, context):
        window, ratio, topk = bound
        compressed = context // ratio
        if topk:
            compressed = min(topk, compressed)
        return window + compressed
    # Through the context range the measured tables cover, the tighter ratio reads less.
    for context in (1024, 8192, 65536):
        assert positions(flat[hi], context) < positions(flat[lo], context), (
            f"at context {context}, ratio {hi} should read less than ratio {lo}: "
            f"{positions(flat[hi], context)} vs {positions(flat[lo], context)}")
    # And the ordering INVERTS past a crossover, which is a real consequence of the two
    # bounds being different in kind rather than in degree: ratio 4 selects at most
    # index_topk compressed positions, so its read plateaus, while ratio 128 reads a
    # positional run that keeps growing. Stated here because a reader who assumed
    # "higher ratio is always cheaper" would mispredict long-context deployments, and
    # this config permits max_position_embeddings far beyond the crossover.
    crossover = flat[lo][2] * flat[hi][1]
    assert positions(flat[hi], crossover) == positions(flat[lo], crossover), (
        f"the two bounds should meet at context {crossover}")
    assert positions(flat[hi], 2 * crossover) > positions(flat[lo], 2 * crossover), (
        f"past context {crossover} the ratio-{hi} read must exceed the ratio-{lo} one, "
        f"whose top-k has plateaued")


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
    indexed = 0
    for k in g["layer_kinds"]:
        idx = [n for n in k["nodes"] if n.get("role") == "block_index_scores"]
        main = [n for n in k["nodes"]
                if n["op"] == "Attention" and n.get("role") != "block_index_scores"]
        assert len(main) == 1, f"{k['id']}: expected one attention node"
        assert main[0]["window"] > 0, "every layer retains its sliding window"
        if not idx:
            continue  # a layer with no indexer, checked by the test below
        assert len(idx) == 1, f"{k['id']}: expected at most one indexer"
        assert "window" not in idx[0], "a bounded indexer would make selection free"
        indexed += 1
    assert indexed, "no layer carries an indexer"


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
