#!/usr/bin/env python3
"""Derive a BLIS ModelGraph from a vendor config.json.

    python3 scripts/derive_graph.py [ROOT] [--check] [--model NAME]

A ModelGraph expresses a model as a DAG of cost primitives, which is what a cost
model prices. The vendor config speaks one provider's dialect; this script performs
the translation once, mechanically, and writes the result beside the config with a
digest of the source. Committing the output rather than recomputing it means a reader
can inspect what a cost model will price, and the digest makes a drift between the
two detectable.

The translation is not a field rename. Three things make it a derivation:

  * A shape may not be stated. Depth is usually num_hidden_layers, but one shipped
    hybrid states no layer count at all: its depth is the length of a per-layer type
    vector. Head dimension is absent from nine of the committed configs because it is
    hidden_size over head count.
  * A node exists when it launches GPU work, not when a config field exists. An
    architecture-specific scalar that replaces a softmax scale rather than adding an
    operation contributes no node.
  * Collective nodes carry a layout condition rather than being emitted
    unconditionally, because whether a collective runs depends on the parallelism a
    deployment chooses, not on the model.

With --check the script re-derives every graph and reports any that differs from what
is committed, without writing. That is the CI gate: a graph edited by hand, or a
config changed without re-deriving, fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

DERIVER_VERSION = 1

# --- Dialect aliases ------------------------------------------------------------
# Each concept the graph needs, and the keys the committed configs use for it, in
# priority order. A concept absent from every alias is either derived (head
# dimension) or genuinely absent (a dense model has no expert count).
ALIASES: dict[str, tuple[str, ...]] = {
    "hidden_size": ("hidden_size", "d_model", "n_embd"),
    "num_layers": ("num_hidden_layers", "n_layer", "num_layers"),
    "num_q_heads": ("num_attention_heads", "n_head"),
    "num_kv_heads": ("num_key_value_heads", "num_kv_heads", "multi_query_group_num"),
    "head_dim": ("head_dim", "kv_channels"),
    "ffn_size": ("intermediate_size", "ffn_hidden_size"),
    "vocab_size": ("vocab_size",),
    "num_experts": ("num_local_experts", "n_routed_experts", "num_experts"),
    "top_k": ("num_experts_per_tok", "num_experts_per_token", "moe_topk"),
    "moe_ffn_size": ("moe_intermediate_size", "expert_intermediate_size"),
    "num_shared_experts": ("n_shared_experts", "num_shared_experts"),
    "shared_ffn_size": (
        "shared_expert_intermediate_size",
        "moe_shared_expert_intermediate_size",
    ),
    "sliding_window": ("sliding_window", "sliding_window_size"),
    "kv_lora_rank": ("kv_lora_rank",),
    "qk_rope_head_dim": ("qk_rope_head_dim",),
    "qk_nope_head_dim": ("qk_nope_head_dim",),
    "v_head_dim": ("v_head_dim",),
    "num_spec_tokens": ("num_nextn_predict_layers", "mtp_num_hidden_layers"),
    # routed_expert_hidden_size is Kimi's spelling of the same concept: its
    # KimiSparseMoeBlock projects hidden -> this width before routing, runs each expert
    # at K = this width, and projects back after (modeling_kimi_linear.py:776-832,
    # gated on `use_latent_moe = routed_expert_hidden_size is not None`). Reading only
    # the DeepSeek/Nemotron spelling left every Kimi expert priced at the full hidden
    # size, doubling its K.
    "moe_latent_size": ("moe_latent_size", "routed_expert_hidden_size"),
    "index_topk": ("index_topk",),
    "dtype": ("torch_dtype", "dtype"),
}

# Vendor dtype spellings to the graph's vocabulary. A format that changes only
# rounding is not a member: the graph records what changes a byte count or a rate.
DTYPES = {
    "bfloat16": "bf16",
    "bf16": "bf16",
    "float16": "fp16",
    "fp16": "fp16",
    "half": "fp16",
    "float32": "fp32",
    "fp32": "fp32",
    "float8_e4m3fn": "fp8",
    "fp8": "fp8",
    "nvfp4": "nvfp4",
    "mxfp4": "mxfp4",
    "int8": "int8",
}


class DeriveError(Exception):
    """A config this deriver cannot translate. Raised rather than guessed around: a
    graph derived from a misread config prices the wrong model, and a missing branch
    is a deriver gap to fix."""


def pick(cfg: dict[str, Any], concept: str) -> Any:
    """Return the first alias present for a concept, or None."""
    for key in ALIASES[concept]:
        if key in cfg and cfg[key] is not None:
            return cfg[key]
    return None


def require(cfg: dict[str, Any], concept: str, model: str) -> Any:
    val = pick(cfg, concept)
    if val is None:
        raise DeriveError(
            f"{model}: no key for {concept!r} among {ALIASES[concept]}; "
            f"the deriver needs a branch for this dialect"
        )
    return val


def require_key(cfg: dict[str, Any], key: str, model: str) -> Any:
    """Return a config key by its literal name, or raise.

    Distinct from `require`, which resolves a CONCEPT through the alias table. A field
    that only one family declares -- a GDN's head geometry, say -- has no cross-dialect
    concept to alias, so adding it to ALIASES would imply a generality it does not have.
    Reading it by name keeps the alias table a statement about shared concepts.
    """
    val = cfg.get(key)
    if val is None:
        raise DeriveError(
            f"{model}: config declares no {key!r}, which this architecture needs"
        )
    return val


def text_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Return the sub-object holding the language-model shapes.

    A multimodal config nests them under text_config beside a vision or audio tower.
    The graph prices the decoder, and records that it does so."""
    return raw.get("text_config", raw)


def raw_quant(cfg: dict[str, Any], model: str) -> dict[str, Any]:
    """The quantization block as seen from a config slice, or an empty mapping."""
    quant = cfg.get("quantization_config") or {}
    return quant if isinstance(quant, dict) else {}


# A role's dtype class, for a checkpoint that states its mixed-precision layout
# explicitly. The map is keyed by the tensor-name fragment the vendor uses and valued by
# the graph role it prices, so a width read from the checkpoint lands on the right node.
QUANT_ROLE_CLASSES = {
    "experts.": "routed_experts",
    "shared_experts.": "shared_experts",
    "mixer.in_proj": "recurrent_in",
    "mixer.out_proj": "recurrent_out",
}

# Vendor quant-algo spellings to the graph's dtype vocabulary.
QUANT_ALGOS = {
    "NVFP4": "nvfp4",
    "MXFP4": "mxfp4",
    "FP8": "fp8",
    "INT8": "int8",
    "INT4": "int4",
    # A W{weights}A{activations} spelling names both widths; the graph prices WEIGHT
    # bytes, so the W half is what it reads. nemotron-3.5-lightning-nvfp4 ships
    # W4A16_NVFP4 -- four-bit NVFP4 weights against bf16 activations -- which stores
    # the same bytes per parameter as the plain NVFP4 spelling.
    "W4A16_NVFP4": "nvfp4",
    "W4A16_MXFP4": "mxfp4",
    "W8A8_FP8": "fp8",
    "W8A16_INT8": "int8",
}


# The graph roles that price an attention module's weights, i.e. the ones a
# checkpoint's `self_attn` exclusion covers. Listed explicitly rather than matched on a
# substring so that adding a role is a deliberate act: a role missing from here keeps the
# global width, which is the conservative direction for a width override.
ATTENTION_WEIGHT_ROLES = (
    "qkv_proj", "qkv_a_proj", "kv_a_proj", "q_proj", "q_b_proj", "kv_b_proj", "o_proj",
    "index_wq_b", "index_wk_weights_proj", "index_head_gate_proj",
    "index_kpool_compress_gate",
    # DeepSeek-V4 and MiniMax-M3 name their indexer projections differently, because
    # their handlers build them rather than lightning_indexer. index_qk_proj is
    # MiniMax-M3's block-sparse scorer projection, which is a different mechanism from
    # the DSA lightning indexer and keeps its own shape.
    "index_wq_proj", "index_weights_proj", "index_compressor_fused_wkv_wgate",
    "index_qk_proj",
    # A linear-attention mixer is the layer's self_attn too, not a separate module:
    # vLLM builds Kimi-K3's KDA as `self.self_attn` with prefix "{...}.self_attn"
    # (kimi_k3/nvidia/model.py:888-891), so an ignore list naming self_attn covers these
    # projections exactly as it covers the latent ones.
    "kda_q_proj", "kda_k_proj", "kda_v_proj", "kda_b_proj",
    "kda_f_a_proj", "kda_f_b_proj", "kda_g_a_proj", "kda_g_b_proj", "kda_g_proj",
    "kda_o_proj",
)


def compressed_tensors_skips_attention(cfg: dict[str, Any],
                                       raw: dict[str, Any]) -> bool:
    """Whether a compressed-tensors checkpoint leaves the attention modules alone.

    compressed-tensors states an `ignore` list of module names and `re:`-prefixed
    regexes, and both Kimi configs name `re:.*self_attn.*` -- so every MLA projection and
    the indexer stay at the declared base width while the routed experts carry the
    quantized one. kimi-k2.5's global width is int4 and kimi-k3's mxfp4, so pricing those
    projections globally charged 4 bits for tensors the runtime never quantizes.

    Deliberately narrow: this answers only "does the ignore list cover self_attn", which
    is the pattern both committed configs use. A checkpoint ignoring individual attention
    tensors would need its own reading rather than this one."""
    quant = raw.get("quantization_config") or cfg.get("quantization_config") or {}
    if not isinstance(quant, dict):
        return False
    if "compressed-tensors" not in str(quant.get("quant_method", "")).lower():
        return False
    for entry in quant.get("ignore") or ():
        text = str(entry)
        if not text.startswith("re:"):
            continue
        body = text[3:]
        # Only the blanket self_attn form is recognised. A narrower regex may cover some
        # attention tensors and not others, and guessing which would be worse than
        # leaving the global width in place.
        if body in (".*self_attn.*", r".*self_attn.*"):
            return True
    return False


def quantized_classes(cfg: dict[str, Any], model: str) -> dict[str, str]:
    """Map each role class to its stored width, from an explicit per-tensor map.

    A checkpoint that enumerates its quantized tensors states a LAYOUT, not a single
    width: nemotron-3-ultra-nvfp4 names 49,152 routed-expert matrices as NVFP4, 192
    Mamba and shared-expert matrices as FP8, and ignores 243 more -- attention, the
    latent projections, embeddings and the head -- which stay at the declared bf16 base.
    Collapsing that to one dominant width prices every ignored tensor at four bits.

    Returns an empty mapping where the config states no such map, which leaves the
    single-width path untouched for every other model."""
    quant = raw_quant(cfg, model)
    layers = quant.get("quantized_layers")
    if not isinstance(layers, dict) or not layers:
        return {}
    found: dict[str, set[str]] = {}
    for name, spec in layers.items():
        algo = spec.get("quant_algo") if isinstance(spec, dict) else spec
        width = QUANT_ALGOS.get(str(algo).upper())
        if width is None:
            raise DeriveError(
                f"{model}: quantized_layers names algo {algo!r}, which this deriver "
                f"does not map; add it to QUANT_ALGOS rather than letting the tensor "
                f"take a width the checkpoint does not state")
        # Longest fragment wins, so shared_experts does not match the experts rule.
        match = max((frag for frag in QUANT_ROLE_CLASSES if frag in name),
                    key=len, default=None)
        if match is None:
            continue
        found.setdefault(QUANT_ROLE_CLASSES[match], set()).add(width)
    out = {}
    for role, widths in found.items():
        if len(widths) > 1:
            raise DeriveError(
                f"{model}: role class {role!r} is quantized at more than one width "
                f"{sorted(widths)}; a single node cannot price both")
        out[role] = widths.pop()
    return out


def weight_dtype(cfg: dict[str, Any], raw: dict[str, Any], model: str) -> str:
    """Return the graph's dtype name for the model's parameters.

    A quantization block overrides the base dtype: a checkpoint declaring bfloat16
    weights with an fp8 quantization config stores fp8, and pricing it at two bytes
    per parameter would overstate both occupancy and decode-time traffic."""
    quant = raw.get("quantization_config") or cfg.get("quantization_config") or {}
    if isinstance(quant, dict) and quant:
        # A checkpoint that enumerates its quantized tensors states a layout rather than
        # a width. The base dtype stays global and the per-tensor widths ride on the
        # nodes they apply to, so the tensors the map IGNORES keep the base.
        if quantized_classes(cfg, model):
            return base_dtype(cfg, raw, model)
        fmt = str(quant.get("quant_method", "")).lower()
        if "nvfp4" in fmt or "modelopt_fp4" in fmt:
            return "nvfp4"
        if "fp8" in fmt:
            return "fp8"
        # A per-group block states the width per tensor set rather than once. The
        # dominant group decides the stored width: a checkpoint holding 49,152 tensors
        # at four bits and 192 at eight is a four-bit checkpoint, and pricing it at the
        # narrower group's minority width would overstate weight traffic.
        groups = quant.get("config_groups")
        if isinstance(groups, dict) and groups:
            widest = None
            for spec in groups.values():
                weights = (spec or {}).get("weights") or {}
                bits = weights.get("num_bits")
                count = len(spec.get("targets") or ())
                if bits is None:
                    continue
                if widest is None or count > widest[1]:
                    widest = (int(bits), count, weights.get("type"),
                              spec.get("format"), weights.get("group_size"))
            if widest is not None:
                bits, _, kind = widest[0], widest[1], widest[2]
                # Two 4-bit float formats ship in these checkpoints and they are not
                # interchangeable: NVFP4 shares an FP8 scale over 16 elements, MXFP4 an
                # E8M0 scale over 32. The format string distinguishes them; a 4-bit
                # float with no format named is ambiguous and raises.
                if bits == 4 and kind == "float":
                    fmt_name = str(widest[3] or quant.get("format") or "").lower()
                    if "mxfp4" in fmt_name:
                        return "mxfp4"
                    if "nvfp4" in fmt_name:
                        return "nvfp4"
                    if widest[4] == 16:
                        return "nvfp4"
                    if widest[4] == 32:
                        return "mxfp4"
                    raise DeriveError(
                        f"{model}: a 4-bit float weight group names no format and has "
                        f"group_size {widest[4]!r}; NVFP4 and MXFP4 cannot be told apart"
                    )
                if bits == 8 and kind == "float":
                    return "fp8"
                if bits == 8 and kind == "int":
                    return "int8"
                # A 4-bit INT group needs no format string to disambiguate, unlike the
                # 4-bit floats above: there is one integer grid, and the per-group scale
                # count follows group_size rather than being fixed by the format. Kimi-K2.5
                # ships this as num_bits 4, type "int", group_size 32.
                if bits == 4 and kind == "int":
                    return "int4"
                raise DeriveError(
                    f"{model}: a {bits}-bit {kind!r} weight group has no dtype in this "
                    f"schema's vocabulary"
                )
        weight_block = quant.get("weight_block_size")
        if fmt == "compressed-tensors" and weight_block:
            return "fp8"
        # A quant_method that names the format outright needs no group widths, because the
        # name fixes them: MXFP4 is 32-element groups with an E8M0 scale and NVFP4 is
        # 16-element with an FP8 scale, by definition of each format. gpt-oss declares
        # `quant_method: mxfp4` with no config_groups at all.
        #
        # Only these two are accepted by name. A method whose name does not fix a width --
        # "compressed-tensors" is the case in hand -- still needs its groups, which is why
        # this sits after that branch rather than before it.
        if fmt in ("mxfp4", "nvfp4"):
            # ...but only where it covers the whole checkpoint. modules_to_not_convert
            # names the tensor sets the method LEAVES ALONE, and gpt-oss excludes
            # self_attn, the router, the embeddings and lm_head -- so MXFP4 there is the
            # expert weights only, and returning it as the global width prices every
            # attention projection and the head at 4 bits instead of 16. The quantized
            # scope is carried on the expert node instead, the same way an expert_dtype
            # narrower than the global width is.
            if quant.get("modules_to_not_convert"):
                return base_dtype(cfg, raw, model)
            return fmt
        if fmt:
            raise DeriveError(
                f"{model}: quant_method {fmt!r} with no group widths; the deriver "
                f"cannot infer the stored width"
            )
    return base_dtype(cfg, raw, model)


# The width a partially-quantized checkpoint stores its UNQUANTIZED tensors in, for a
# config that states no dtype at all. gpt-oss is the case in hand: it declares neither
# torch_dtype nor dtype, and its quantization_config covers only the experts, so the
# non-expert width has to come from somewhere. OpenAI's model card and the InferenceX
# descriptor this catalog cites for the model both state BF16 for those tensors.
# Recorded per model rather than defaulted, because a default here would silently price
# every future dtype-less config at two bytes.
# Bytes per element, for the widths an activation tensor is stored in. Only the dtypes
# that can carry activations are listed: this sizes elementwise traffic, not weights, so
# the sub-byte quantized formats have no entry rather than a fractional one.
DTYPE_WIDTHS = {
    "fp32": 4,
    "bf16": 2,
    "fp16": 2,
    "fp8": 1,
}


UNQUANTIZED_BASE = {
    "gpt-oss-120b": "bf16",
    "gpt-oss-20b": "bf16",
}


def base_dtype(cfg: dict[str, Any], raw: dict[str, Any], model: str) -> str:
    """The declared parameter dtype, ignoring any quantization block."""
    declared = pick(cfg, "dtype") or pick(raw, "dtype")
    if declared is None:
        known = UNQUANTIZED_BASE.get(model)
        if known:
            return known
        raise DeriveError(f"{model}: no dtype declared")
    name = DTYPES.get(str(declared).lower())
    if name is None:
        raise DeriveError(f"{model}: unrecognized dtype {declared!r}")
    return name


def head_dim(cfg: dict[str, Any], model: str) -> int:
    """Return the attention head dimension.

    Stated where the config says so. Derived as hidden over head count otherwise,
    which is what an engine does and what nine committed configs require."""
    stated = pick(cfg, "head_dim")
    if stated:
        return int(stated)
    hidden = int(require(cfg, "hidden_size", model))
    heads = int(require(cfg, "num_q_heads", model))
    if hidden % heads:
        raise DeriveError(
            f"{model}: hidden_size {hidden} is not divisible by {heads} heads and no "
            f"head_dim is stated"
        )
    return hidden // heads


# --- Node builders --------------------------------------------------------------
# Each returns one primitive with the shape parameters that primitive prices, and
# nothing else. A parameter a node's op does not price is rejected by the schema, so
# the builders keep the unions honest rather than relying on review.


def norm(role: str) -> dict[str, Any]:
    return {"op": "Elementwise", "role": role}


def gemm(role: str, n: int, k: int) -> dict[str, Any]:
    return {"op": "GEMM", "role": role, "n": int(n), "k": int(k)}


def attention(cfg: dict[str, Any], model: str, *, kind: str | None = None,
              window: int | None = None, compress_ratio: int | None = None,
              index_topk: int | None = None,
              latent_width: int | None = None) -> dict[str, Any]:
    """Build the attention node, choosing the kind from the config's own evidence.

    `latent_width` is the per-token cache width for a family that states it directly
    rather than as kv_lora_rank + rope. `compress_ratio` and `index_topk` are the two
    bounds a sparse latent read can carry, and a layer may carry both: the positions it
    reads are the union of its sliding window and its compressed stream, so `window`
    is not an alternative to either."""
    nq = int(require(cfg, "num_q_heads", model))
    nkv = int(require(cfg, "num_kv_heads", model))
    lora = pick(cfg, "kv_lora_rank")

    if kind is None:
        if lora:
            # Every layer of a DSA model reads a top-k, including the ones that build no
            # indexer. vLLM constructs the shared MLA module with use_sparse=True on
            # every layer and passes the common topk_indices_buffer
            # (deepseek_v32/attention.py:202-218), which mla_attention.py:564-566 states
            # outright: "Sparse MLA reads top-k indices from a shared buffer. Pass it
            # explicitly so backbone 'skip' layers (indexer=None) still find it." So a
            # skipped layer omits the SCORING work, not the bounded read -- making it
            # plain mla would overprice its main KV read as unbounded.
            kind = "sparse_mla" if pick(cfg, "index_topk") else "mla"
        elif window:
            kind = "swa"
        else:
            kind = "gqa"

    node: dict[str, Any] = {"op": "Attention", "kind": kind, "n_q": nq}

    if latent_width is not None:
        # A family that states the per-token latent width inclusively, so there is no
        # lora rank to add a rope width to. DeepSeek-V4 is the case in hand: its
        # head_dim already contains qk_rope_head_dim, which is why the branch below
        # cannot size it.
        if kind not in ("mla", "sparse_mla"):
            raise DeriveError(
                f"{model}: a latent width was given for attention kind {kind!r}, which "
                f"stores no latent cache")
        node["n_kv"] = 1
        node["d_h"] = int(latent_width)
        node["qk_rope_head_dim"] = int(require(cfg, "qk_rope_head_dim", model))
        if window:
            node["window"] = int(window)
        if index_topk:
            node["index_topk"] = int(index_topk)
        if compress_ratio:
            node["compress_ratio"] = int(compress_ratio)
        if kind == "sparse_mla" and not (index_topk or compress_ratio):
            raise DeriveError(
                f"{model}: a sparse latent read bounded by neither a top-k selection "
                f"nor a compressed stream describes no real layer")
        return node

    if kind in ("mla", "sparse_mla"):
        # A latent cache holds one vector per token, so the engine pins the KV head
        # count to one whatever the config's num_key_value_heads says. The per-token
        # width is the latent rank plus the RoPE dimension, which is what the cache
        # stores and therefore what a cost model reads.
        nope = int(pick(cfg, "qk_nope_head_dim") or 0)
        rope = int(pick(cfg, "qk_rope_head_dim") or 0)
        # A NoPE variant states it: GLM-5.3-Flash sets mla_use_nope with
        # qk_rope_head_dim 0, so its latent cache holds the rank alone and the width
        # below is correct at rope == 0. Absent that declaration a zero is a missing
        # field rather than a stated one, and accepting it would silently understate
        # every per-token cache read, so the two cases are kept apart.
        if not rope and not cfg.get("mla_use_nope"):
            raise DeriveError(f"{model}: latent attention with no qk_rope_head_dim")
        node["n_kv"] = 1
        node["d_h"] = int(lora) + rope
        node["kv_lora_rank"] = int(lora)
        node["qk_rope_head_dim"] = rope
        if kind == "sparse_mla":
            node["index_topk"] = int(pick(cfg, "index_topk"))
        # nope is the query-side width; it does not enter the cache, so it is not a
        # node parameter. Recorded here so a reader knows it was considered.
        del nope
    else:
        node["n_kv"] = nkv
        node["d_h"] = head_dim(cfg, model)
        if kind == "swa":
            if not window:
                raise DeriveError(f"{model}: sliding-window attention with no window")
            node["window"] = int(window)
    return node


def dense_mlp(cfg: dict[str, Any], model: str, hidden: int) -> list[dict[str, Any]]:
    """Two GEMMs for a gated MLP: the fused gate-and-up projection, then the down.

    The gate and up projections are one kernel in every engine that matters, so they
    are one node at twice the inner width rather than two nodes."""
    ffn = int(require(cfg, "ffn_size", model))
    return [
        gemm("mlp_gate_up", 2 * ffn, hidden),
        gemm("mlp_down", hidden, ffn),
    ]


# A checkpoint storing its routed experts at a different width from the rest states it in
# expert_dtype, and vLLM resolves that NAME rather than reading it as a dtype:
# models/deepseek_v4/quant_config.py documents "fp4" as MXFP4 experts with ue8m0 scales
# and "fp8" as FP8-block experts, the linear and attention layers being FP8 block either
# way. So this mapping is vLLM's, not an inference from the string.
EXPERT_DTYPES = {
    "fp4": "mxfp4",
    "fp8": "fp8",
}


def moe(cfg: dict[str, Any], model: str, hidden: int) -> dict[str, Any]:
    """Build the grouped-GEMM node for a routed expert layer."""
    experts = int(require(cfg, "num_experts", model))
    top_k = int(require(cfg, "top_k", model))
    inner = pick(cfg, "moe_ffn_size") or require(cfg, "ffn_size", model)
    node: dict[str, Any] = {
        "op": "GroupedGEMM",
        "role": "experts",
        "n": int(inner),
        "k": hidden,
        "experts": experts,
        "top_k": top_k,
    }
    shared = pick(cfg, "num_shared_experts")
    shared_inner = pick(cfg, "shared_ffn_size")
    # A config that states a shared-expert WIDTH but no count has one shared expert. The
    # Qwen3.5 configs are the case in hand: they declare
    # shared_expert_intermediate_size and no count, and vLLM's own loader for that family
    # passes n_shared_experts=1 (`models/qwen3_5.py`). Reading the width as "no shared
    # expert" would drop dense work every token pays.
    if shared is None and shared_inner:
        shared = 1
    if shared:
        node["shared_experts"] = int(shared)
        if shared_inner and int(shared_inner) != int(inner):
            node["shared_intermediate_size"] = int(shared_inner)
    # The expert width where the checkpoint states one, which is not always the global
    # weight dtype. DeepSeek-V4-Pro declares expert_dtype fp4 beside an fp8
    # quantization_config: pricing its 384 experts at the global width doubles them from
    # 720 GiB to 1,441 GiB, which puts 180 GiB per rank on a 141 GiB H200 and makes a
    # deployment InferenceX ran at tp=8 on 8 GPUs look impossible. An unrecognized value
    # raises rather than falling back to the global width: falling back is the bug.
    # A method that quantizes only part of the checkpoint leaves the global width at the
    # unquantized base, so the experts -- which ARE quantized -- carry the narrow width
    # here. The mirror of the expert_dtype case below: there the experts are narrower
    # than a global that is itself quantized; here they are the only narrow thing.
    quant = raw_quant(cfg, model)
    if quant:
        fmt = str(quant.get("quant_method", "")).lower()
        if fmt in ("mxfp4", "nvfp4") and quant.get("modules_to_not_convert"):
            node["weight_dtype"] = fmt
    classes = quantized_classes(cfg, model)
    if classes.get("routed_experts"):
        node["weight_dtype"] = classes["routed_experts"]

    expert_dtype = cfg.get("expert_dtype")
    if expert_dtype is not None:
        mapped = EXPERT_DTYPES.get(str(expert_dtype))
        if mapped is None:
            raise DeriveError(
                f"{model}: expert_dtype {expert_dtype!r} is not one this deriver maps; "
                f"add it to EXPERT_DTYPES with the width vLLM resolves it to rather "
                f"than letting the global dtype stand in"
            )
        if mapped != DTYPES.get(str(pick(cfg, "dtype") or "").lower()):
            node["weight_dtype"] = mapped

    latent = pick(cfg, "moe_latent_size")
    if latent:
        # A projection to a narrower space runs before the routed experts, so the
        # expert matmul's K is the latent width rather than the model's hidden size.
        # It changes the expert FLOP count and the dispatch volume together.
        node["latent_size"] = int(latent)
    return node


def moe_block(cfg: dict[str, Any], model: str, hidden: int) -> list[dict[str, Any]]:
    """The routed expert layer: its latent projections, where it has them, and the
    grouped GEMM.

    A latent MoE is three pieces of work, not one. Kimi-K3's KimiSparseMoeBlock
    projects hidden -> latent, routes and runs the experts at the latent width, then
    projects latent -> hidden (modeling_kimi_linear.py:776-832); DeepSeek and Nemotron
    state the same structure as moe_latent_size. The two projections are dense GEMMs
    every token pays, so they are nodes rather than an implication of latent_size.

    The routing gate and the shared experts both run on the PRE-projection hidden
    states -- vLLM gates before the down-projection and adds
    `self.shared_experts(identity)` after the up-projection -- so neither is affected by
    the latent width, which is why shared_intermediate_size stays as moe() states it."""
    node = moe(cfg, model, hidden)
    nodes = [node]

    # A shared expert stored at a DIFFERENT width from the routed ones cannot ride on
    # the routed node: shared_experts/shared_intermediate_size describe work priced at
    # the node's own weight_dtype. nemotron-3-ultra-nvfp4 is the case in hand -- routed
    # experts NVFP4, shared experts FP8 -- so the shared half becomes its own node and
    # the routed node stops claiming it.
    classes = quantized_classes(cfg, model)
    shared_width = classes.get("shared_experts")
    if shared_width and shared_width != node.get("weight_dtype") and node.get(
            "shared_experts"):
        inner = int(node.get("shared_intermediate_size") or node["n"])
        shared = {
            "op": "GroupedGEMM",
            "role": "shared_experts",
            "n": inner,
            "k": node["k"],
            "experts": int(node["shared_experts"]),
            "top_k": int(node["shared_experts"]),
            "weight_dtype": shared_width,
        }
        if node.get("latent_size"):
            shared["latent_size"] = node["latent_size"]
        node.pop("shared_experts", None)
        node.pop("shared_intermediate_size", None)
        nodes.append(shared)

    latent = node.get("latent_size")
    if not latent:
        return nodes
    return [
        gemm("routed_expert_down_proj", int(latent), hidden),
        *nodes,
        gemm("routed_expert_up_proj", hidden, int(latent)),
    ]


# Emit conditions are a closed set naming WHEN a conditional node is part of the graph.
# They are enum members rather than predicate strings: every condition the catalog's models
# need is decided by the collective's role, so a resolver evaluates them with a switch and
# needs no expression parser. A graph cannot express a condition no resolver knows.
EMIT_TENSOR_PARALLEL = "tensor_parallel"
EMIT_EXPERT_PARALLEL = "expert_parallel"
EMIT_TENSOR_PARALLEL_UNLESS_SP_MOE = "tensor_parallel_unless_sp_moe"


def collective(op: str, role: str, emit: str) -> dict[str, Any]:
    return {"op": op, "role": role, "emit": emit}


# The layer-kind id a handler uses for a draft module it prices itself. Reserved rather
# than conventional: derive() prefers it over mirroring a target layer, so a handler that
# defines it is stating that no target layer has the draft module's structure.
DRAFT_KIND_ID = "mtp_moe"


def compress(sequence: list[str]) -> dict[str, Any]:
    """Express a layer sequence as the smallest prologue, repeated pattern and epilogue.

    A literal 108-entry list and a pattern repeated four times describe the same stack, so
    the choice is about storage rather than meaning. Compression is worth doing for two
    reasons: a reader can see a hybrid's period at a glance where a literal list hides it,
    and a diff after a config change stays legible.

    It is NOT what makes step-time computation fast. A kernel collapses the stack to a
    per-kind layer count once at construction, so its per-step work is proportional to the
    number of distinct layer kinds -- three for the widest model here -- whichever way the
    sequence was spelled. Compression saves file size and reading effort, not cycles.

    The search prefers the fewest stored entries, and declines to store a prologue or
    epilogue longer than the pattern it surrounds: past that point the literal form is
    clearer than a split that technically compresses."""
    n = len(sequence)
    if n == 0:
        return {}
    best: tuple[int, int, list[str], int, list[str]] | None = None
    for head in range(0, n + 1):
        rest = sequence[head:]
        if not rest:
            continue
        for period in range(1, len(rest) // 2 + 1):
            repeats = len(rest) // period
            if repeats < 2:
                continue
            body = rest[:period]
            if body * repeats != rest[:period * repeats]:
                continue
            tail = rest[period * repeats:]
            stored = head + period + len(tail)
            # Fewest stored entries wins; the SHORTEST period breaks a tie, so a stack
            # whose true period is 1 is not spelled as a longer multiple of it.
            if best is None or (stored, period) < (best[0], len(best[2])):
                best = (stored, head, body, repeats, tail)
    if best is None or best[0] >= n:
        # No split stores less than the literal sequence, so state it literally. This is
        # also what enforces the docstring's readability rule: a prologue or epilogue so
        # long that the split stores no less than the sequence it describes is not a
        # compression worth reading, and the literal form is clearer.
        return {"prologue": list(sequence)}
    _, head, body, repeats, tail = best
    out: dict[str, Any] = {}
    if head:
        out["prologue"] = sequence[:head]
    out["pattern"] = body
    out["repeat"] = repeats
    if tail:
        out["epilogue"] = tail
    return out


def chain(nodes: list[dict[str, Any]]) -> list[list[int]]:
    """Return edges for a linear chain, which every layer in these architectures is."""
    return [[i, i + 1] for i in range(len(nodes) - 1)]


# How a family lays out its MLA latent stages, because vLLM does not do it one way and
# the differences are launches rather than labels. Keyed by the architecture string, and
# resolved through vLLM's own registry rather than by family resemblance:
#
#   "DeepseekV2ForCausalLM"      -> deepseek_v2.DeepseekV2Attention
#   "DeepseekV3ForCausalLM"      -> deepseek_v2.DeepseekV3ForCausalLM (same attention)
#   "GlmMoeDsaForCausalLM"       -> vllm.models.deepseek_v32  (registry.py:118)
#   "KimiK3ForConditionalGeneration" -> vllm.models.kimi_k3
#   "Glm5NextForConditionalGeneration" -> vllm.models.glm5next
#
# The attention CLASS matters, not the module. deepseek_v2.py defines two, and
# DeepseekV2DecoderLayer picks DeepseekV2MLAAttention whenever model_config.use_mla is
# true, falling back to DeepseekV2Attention otherwise (deepseek_v2.py:1286-1291). Every
# latent-cache model in this catalog takes the MLA class -- which the graphs themselves
# assert, since they price kind=mla with n_kv=1 -- so the MLA class is what to read.
#
# fused_a: one A-projection GEMM over [q_lora_rank, kv_lora_rank + rope].
#   DeepseekV2MLAAttention builds DeepSeekV2FusedQkvAProjLinear over exactly that
#   (deepseek_v2.py:1050-1057), as do KimiK3MergedQKVGateLinear and the deepseek_v32 and
#   glm5next paths. Only the non-MLA DeepseekV2Attention splits them (:493, :605), and
#   no model here takes that class.
# fused_norm: one norm over BOTH latents. kimi_k3 calls fused_q_kv_rmsnorm (mla.py:520)
#   and glm5next opts in with fuse_qkv_rmsnorm=True (glm5next/common/attention.py:604).
#   DeepseekV2MLAAttention does NOT opt in: it builds q_a_layernorm and kv_a_layernorm
#   separately (deepseek_v2.py:1081, 1097) and the wrapper never sets that flag, so the
#   DeepSeek and Kimi-K2.5 graphs keep two norm launches with one fused projection.
#
# An architecture absent here takes {fused_a: True, fused_norm: False}, which is what
# DeepseekV2MLAAttention does -- the shape every MLA family shares unless it opts into
# the norm fusion.
MLA_LAYOUTS = {
    "DeepseekV2ForCausalLM": {"fused_a": True, "fused_norm": False},
    "DeepseekV3ForCausalLM": {"fused_a": True, "fused_norm": False},
    # fused_index_norm: deepseek_v32 calls fused_norm_rope ONCE per layer,
    # unconditionally on the shared code path, passing the q norm, the kv norm, the rope
    # cache AND the indexer's k-norm weight/bias/eps into that single launch
    # (deepseek_v32/attention.py:378-392). glm5next does not: it fuses the two latent
    # norms (fuse_qkv_rmsnorm=True, glm5next/common/attention.py:604) and runs
    # _fused_indexer_k_norm as its own call (:352). So this family launches one norm
    # kernel where glm5next launches two.
    "GlmMoeDsaForCausalLM": {"fused_a": True, "fused_norm": True,
                             "fused_index_norm": True},
    "DeepseekV4ForCausalLM": {"fused_a": True, "fused_norm": True},
    "KimiK3ForConditionalGeneration": {"fused_a": True, "fused_norm": True},
    # kimi-k3's text_config names KimiLinearForCausalLM while the outer config names
    # KimiK3ForConditionalGeneration; both resolve to vllm.models.kimi_k3, so both map
    # here rather than relying on which view the lookup happens to read.
    "KimiLinearForCausalLM": {"fused_a": True, "fused_norm": True},
    "Glm5NextForConditionalGeneration": {"fused_a": True, "fused_norm": True},
    # kimi-k2.5 takes the DeepseekV2MLAAttention shape: its text_config declares
    # DeepseekV3ForCausalLM, the registry sends that to deepseek_v2 (registry.py:90),
    # and KimiK25ForConditionalGeneration initializes its text decoder through
    # DeepseekV2ForCausalLM -- whose decoder layer selects the MLA class because its
    # kimi_k2 text type keeps use_mla true. So: fused projection, separate norms, by the
    # DeepseekV3ForCausalLM entry above.
}


# Architectures whose indexer recomputes the head gate as its own FP32 matmul, rather
# than slicing it out of the wk_weights_proj result. glm5next alone does this
# (common/attention.py:341-350); deepseek_v32 slices and reuses
# (deepseek_v32/attention.py:309-315).
# Architectures that build their whole MLA module -- latent projections and indexer
# alike -- with quant_config=None, so every one of those weights serves at the base
# width whatever the checkpoint's global. glm5next states it in the call:
# `quant_config=None,  # MLA projections are BF16 in checkpoint`
# (glm5next/common/model.py:394). deepseek_v32 passes the real quant_config to its
# indexer's wq_b and only forces None on wk_weights_proj (attention.py:71-86), so it is
# NOT in this set and keeps the narrower override.
MLA_MODULE_UNQUANTIZED = frozenset({
    "Glm5NextForConditionalGeneration",
})


def mla_module_unquantized(raw: dict[str, Any], cfg: dict[str, Any]) -> bool:
    """Whether this architecture builds its entire MLA module unquantized."""
    for arches in (raw.get("architectures"), cfg.get("architectures")):
        for arch in arches or ():
            if arch in MLA_MODULE_UNQUANTIZED:
                return True
    return False


HEAD_GATE_RECOMPUTED = frozenset({
    "Glm5NextForConditionalGeneration",
})


def recomputes_head_gate(raw: dict[str, Any], cfg: dict[str, Any]) -> bool:
    """Whether this architecture runs a separate FP32 head-gate matmul."""
    for arches in (raw.get("architectures"), cfg.get("architectures")):
        for arch in arches or ():
            if arch in HEAD_GATE_RECOMPUTED:
                return True
    return False


def mla_layout(raw: dict[str, Any], cfg: dict[str, Any]) -> dict[str, bool]:
    """The latent-stage layout for this checkpoint's architecture."""
    # The OUTER architecture wins where the two differ: it is what vLLM's registry is
    # keyed on, and it is the module that builds the attention. kimi-k3 declares
    # KimiK3ForConditionalGeneration outside and KimiLinearForCausalLM in its
    # text_config; kimi-k2.5 declares KimiK25ForConditionalGeneration outside and
    # DeepseekV3ForCausalLM inside, and there the INNER one is the text decoder vLLM
    # actually instantiates -- so both views are consulted and an entry for either
    # spelling resolves the same way.
    #
    # Every flag defaults off against the base shape, so an entry states only what it
    # differs in and a new flag cannot silently change a family that never declared it.
    base = {"fused_a": True, "fused_norm": False, "fused_index_norm": False}
    for arches in (raw.get("architectures"), cfg.get("architectures")):
        for arch in arches or ():
            if arch in MLA_LAYOUTS:
                return {**base, **MLA_LAYOUTS[arch]}
    # The DeepseekV2MLAAttention shape, which is what an MLA family does unless it opts
    # into the norm fusion. A latent layer reaching this function already has a latent
    # cache, so the non-MLA class is not the right default for it.
    return base


def attention_block(cfg: dict[str, Any], model: str, hidden: int, *,
                    raw: dict[str, Any] | None = None,
                    kind: str | None = None,
                    window: int | None = None,
                    compress_ratio: int | None = None,
                    index_topk: int | None = None,
                    latent_width: int | None = None,
                    indexed: bool = True) -> list[dict[str, Any]]:
    """The nodes common to every attention layer: norm, QKV, attention, output, reduce."""
    nq = int(require(cfg, "num_q_heads", model))
    nkv = int(require(cfg, "num_kv_heads", model))
    attn = attention(cfg, model, kind=kind, window=window,
                     compress_ratio=compress_ratio, index_topk=index_topk,
                     latent_width=latent_width)
    if latent_width is not None:
        # This family states one inclusive per-token width and no separate nope/v
        # widths, so the projections are sized from that width: a per-head query at the
        # full width, plus the one shared latent KV vector the cache stores. The rope
        # width is NOT added -- latent_width already contains it, which is the whole
        # point of its being inclusive, and adding it again would overstate the
        # projection by an eighth exactly as it would the cache.
        qkv_out = (nq + 1) * int(latent_width)
        o_in = nq * int(latent_width)
    elif attn["kind"] in ("mla", "sparse_mla"):
        # The latent path is a sequence of low-rank stages, not one fused projection,
        # and the config states every width it needs. Each stage below is a GEMM an
        # engine launches and a norm between the two it separates, so the whole chain is
        # returned rather than collapsed into a query-width stand-in.
        nope = int(pick(cfg, "qk_nope_head_dim") or 0)
        rope = int(pick(cfg, "qk_rope_head_dim") or 0)
        v = int(pick(cfg, "v_head_dim") or nope)
        kv_lora = int(pick(cfg, "kv_lora_rank") or 0)
        # q_lora_rank is read by its literal name: it is not a cross-dialect concept in
        # ALIASES, and its ABSENCE is meaningful here -- it selects the direct-q_proj
        # branch rather than defaulting a width.
        q_lora = int(cfg.get("q_lora_rank") or 0)
        qk_head = nope + rope
        if qk_head == 0 or v == 0:
            raise DeriveError(f"{model}: latent attention with no head widths to size "
                              f"its projections")
        if kv_lora == 0:
            raise DeriveError(f"{model}: latent attention with no kv_lora_rank to size "
                              f"its compressed KV stage")
        # A norm's traffic is derivable from hidden_size only when it runs at hidden
        # width. These do not: the q norm is q_lora_rank wide and the kv norm
        # kv_lora_rank, so each states its own traffic or it prices as a hidden-width
        # pass. RMSNorm accumulates in fp32 over a read and a write.
        # raw is needed for the activation width: two committed configs state their
        # dtype nowhere a text_config view can see, so guessing a default here would
        # silently halve or double the traffic on the stages below.
        if raw is None:
            raise DeriveError(
                f"{model}: the latent stages size their norm traffic from the declared "
                f"activation width, so this call must pass raw")
        act = DTYPE_WIDTHS[base_dtype(cfg, raw, model)]

        def latent_norm(role: str, width: int) -> dict[str, Any]:
            return dict(norm(role), bytes_per_token=2 * width * act)

        # Where the checkpoint's quantization skips the attention modules, these stages
        # store at the base width and the global one would misprice them. Both Kimi
        # configs do this: kimi-k2.5 is globally int4 and kimi-k3 mxfp4, and each names
        # `re:.*self_attn.*` in its compressed-tensors ignore list, so every projection
        # below stays unquantized at runtime.
        attn_width = (base_dtype(cfg, raw, model)
                      if compressed_tensors_skips_attention(cfg, raw) else None)

        def staged(node: dict[str, Any]) -> dict[str, Any]:
            if attn_width and node.get("role") in ATTENTION_WEIGHT_ROLES:
                return dict(node, weight_dtype=attn_width)
            return node

        layout = mla_layout(raw, cfg)
        stages = []
        if q_lora:
            # A declared q_lora_rank makes the query low-rank too, and vLLM fuses the
            # two A-stages into one GEMM: DeepSeekV2FusedQkvAProjLinear over
            # [q_lora_rank, kv_lora_rank + qk_rope_head_dim] (glm5next attention.py
            # 464-469, the DeepSeek MLA path it shares). Priced as that one launch,
            # which is what a deployment runs; the checkpoint stores the halves
            # separately, and the weight bytes are identical either way.
            # An output gate rides in the SAME launch where the config declares one.
            # (Only a fused-A family can carry it: it is a shard of that projection.)
            # Kimi-K3 sets mla_use_output_gate, and vLLM then builds
            # KimiK3MergedQKVGateLinear in place of the plain fused A-projection, with
            # the gate as a third shard (kimi_k3/nvidia/mla.py:218-228). The forward
            # splits [qkv_a_rows, num_local_heads * v_head_dim] off the one result and
            # sigmoid-multiplies the attention output by it (mla.py:574-576, 621-622).
            # Those rows are part of this GEMM, so omitting them dropped both the
            # parameters and the work: for kimi-k3 that is 96 * 128 = 12,288 rows, which
            # makes the real launch (14,400, 7,168) rather than (2,112, 7,168).
            gate_rows = nq * v if cfg.get("mla_use_output_gate") else 0
            if layout["fused_a"]:
                # One A-projection over both latents, as DeepSeekV2FusedQkvAProjLinear
                # and KimiK3MergedQKVGateLinear build it.
                stages.append(gemm("qkv_a_proj",
                                   q_lora + kv_lora + rope + gate_rows, hidden))
            else:
                # The generic DeepseekV2Attention builds these separately
                # (deepseek_v2.py:493 q_a_proj, :605 kv_a_proj_with_mqa), which is two
                # launches rather than one. An output gate is a shard of a fused
                # projection, so a family without one cannot be carrying it.
                if gate_rows:
                    raise DeriveError(
                        f"{model}: mla_use_output_gate is set, but this architecture "
                        f"builds its A-projections separately, so there is no fused "
                        f"launch for the gate to be a shard of")
                stages.append(gemm("q_a_proj", q_lora, hidden))
                stages.append(gemm("kv_a_proj", kv_lora + rope, hidden))
            if layout["fused_norm"]:
                # One launch, not two. Kimi-K3 calls fused_q_kv_rmsnorm over both
                # latents (kimi_k3/nvidia/mla.py:520) and glm5next asks the shared MLA
                # module for the same with fuse_qkv_rmsnorm=True
                # (glm5next common/attention.py:604).
                #
                # Where the family ALSO folds the indexer's k-norm into that same call,
                # its width joins this node's traffic. deepseek_v32 passes index_k and
                # the indexer's norm weight/bias/eps into the one fused_norm_rope
                # (deepseek_v32/attention.py:378-392), and that kernel still reads,
                # normalises and writes index_k -- fusion removes a LAUNCH, not the HBM
                # traffic. Suppressing the separate index_k_norm node without adding its
                # width here would delete bytes the kernel really moves.
                # The folded width belongs only to layers that actually RUN the
                # indexer. A skipped layer takes the else at
                # deepseek_v32/attention.py:339-360 with has_indexer = False, and
                # fused_norm_rope substitutes dummies and skips the indexer program
                # under HAS_INDEXER -- so its norm moves the two latents alone. Gated on
                # the per-layer state rather than the architecture, because
                # attention_block() builds the skipped kinds too.
                norm_width = q_lora + kv_lora
                if layout["fused_index_norm"] and indexed:
                    norm_width += int(require_key(cfg, "index_head_dim", model))
                stages.append(latent_norm("qkv_a_layernorm", norm_width))
            else:
                # Two calls, in the order the generic forward makes them: q_a_layernorm
                # right after q_a_proj (deepseek_v2.py:598) and kv_a_layernorm after the
                # KV A-projection (:608).
                stages.append(latent_norm("q_a_layernorm", q_lora))
                stages.append(latent_norm("kv_a_layernorm", kv_lora))
            stages.append(gemm("q_b_proj", nq * qk_head, q_lora))
        else:
            # No q_lora_rank: vLLM takes the else branch and builds a direct q_proj at
            # full width, with only the KV side compressed (deepseek-v2-lite).
            stages.append(gemm("kv_a_proj", kv_lora + rope, hidden))
            # Only the KV side is compressed here, so there is one latent norm and
            # nothing to fuse it with.
            stages.append(latent_norm("kv_a_layernorm", kv_lora))
            stages.append(gemm("q_proj", nq * qk_head, hidden))
        # The up-projection of the cached latent into per-head nope + value widths. Read
        # on every decode step for every cached token, so its width is the one a fused
        # node hid most consequentially.
        stages.append(gemm("kv_b_proj", nq * (nope + v), kv_lora))
        post_attn = []
        if q_lora and cfg.get("mla_use_output_gate"):
            # The sigmoid-multiply itself, which runs on the attention output before
            # o_proj: read the output and the gate, write the product.
            post_attn.append(dict(norm("mla_output_gate"),
                                  bytes_per_token=3 * nq * v * act))
        return [
            norm("input_norm"),
            *[staged(n) for n in stages],
            attn,
            *post_attn,
            staged(gemm("o_proj", hidden, nq * v)),
            collective("AllReduce", "attn_out", EMIT_TENSOR_PARALLEL),
        ]
    else:
        dh = attn["d_h"]
        qkv_out = (nq + 2 * nkv) * dh
        o_in = nq * dh
    return [
        norm("input_norm"),
        gemm("qkv_proj", qkv_out, hidden),
        attn,
        gemm("o_proj", hidden, o_in),
        collective("AllReduce", "attn_out", EMIT_TENSOR_PARALLEL),
    ]


def indexed_layers(cfg: dict[str, Any], model: str, layers: int) -> list[bool] | None:
    """Which layers actually build an indexer, or None where the family has no indexer.

    A DSA config does not necessarily index every layer. vLLM computes a per-layer
    `skip_topk` and constructs the indexer only when it is false
    (models/deepseek_v32/attention.py:166-200, where GlmMoeDsaForCausalLM is served):

        if index_topk_pattern is None:
            skip_topk = max(layer_id - index_skip_topk_offset + 1, 0) % index_topk_freq != 0
        elif 0 <= layer_id < len(index_topk_pattern):
            skip_topk = index_topk_pattern[layer_id] == "S"
        else:
            skip_topk = False

    The defaults matter: freq 1 and offset 2 make every layer indexed, which is why the
    families that declare neither field are unaffected. glm-5.2, glm-5.2-fp8 and glm-5.3
    declare freq 4 with offset 3 over 78 layers, which indexes 21 of them -- so charging
    all 78 for the scorer and its projections overprices that work by 78/21.

    Returns a per-layer list so a handler can key its layer kinds off it rather than
    collapsing indexing to one model-wide flag."""
    if not (pick(cfg, "index_topk") and cfg.get("index_n_heads") is not None):
        return None
    pattern = cfg.get("index_topk_pattern")
    if pattern is not None:
        if len(pattern) < layers:
            raise DeriveError(
                f"{model}: index_topk_pattern has {len(pattern)} entries for {layers} "
                f"layers; a layer past the end would silently take the indexed branch")
        unknown = set(pattern[:layers]) - {"S", "I"}
        if unknown:
            raise DeriveError(
                f"{model}: index_topk_pattern names {sorted(unknown)}; this deriver "
                f"reads 'S' as skipped and anything else as indexed, so an unexpected "
                f"code would be priced as indexed on a guess")
        return [pattern[i] != "S" for i in range(layers)]
    freq = int(cfg.get("index_topk_freq") or 1)
    offset = cfg.get("index_skip_topk_offset")
    offset = 2 if offset is None else int(offset)
    if freq < 1:
        raise DeriveError(
            f"{model}: index_topk_freq is {freq}; the modulus must be at least 1")
    return [max(i - offset + 1, 0) % freq == 0 for i in range(layers)]


def lightning_indexer(cfg: dict[str, Any], model: str, hidden: int,
                      nodes: list[dict[str, Any]],
                      raw: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Splice a DSA lightning indexer in front of the attention node.

    The indexer is what makes a sparse latent read sparse: it scores every cached
    position to select the top-k the attention then reads. That makes it the ONLY
    context-proportional term in such a layer -- measured at 3.53x over a 12x context
    increase where the attention it feeds moves 1.18x -- so it is its own node, with no
    window, rather than folded into a bounded attention.

    A graph recording index_topk without this node asserts a selection it does not
    charge for, which understates the dominant long-context cost. Shared by every family
    that ships the indexer rather than reimplemented per handler, which is how the
    GLM-5 family came to declare the full indexer spec and emit none of it."""
    index_heads = int(require_key(cfg, "index_n_heads", model))
    index_dim = int(require_key(cfg, "index_head_dim", model))
    kpool = int(cfg.get("index_kpool") or 0)
    q_lora = int(cfg.get("q_lora_rank") or 0)
    at = next(i for i, n in enumerate(nodes) if n.get("op") == "Attention")

    # One shape for every family that reaches this helper, because vLLM builds one:
    # GlmMoeDsaForCausalLM is served by the deepseek_v32 module (its __init__ aliases
    # GlmMoeDsaForCausalLM = DeepseekV32ForCausalLM), and that indexer
    # (models/deepseek_v32/attention.py:71-85) constructs exactly what glm5next's does
    # (models/glm5next/common/attention.py:265-282):
    #
    #   wq_b              q_lora_rank -> n_head * head_dim      (ReplicatedLinear)
    #   wk_weights_proj   hidden -> [head_dim, n_head]           (fused, ONE GEMM)
    #
    # The query stage reads the LATENT rank, not hidden. An earlier revision of this
    # helper emitted a single hidden-wide GEMM of n_head*head_dim + head_dim, which
    # overstated the projection work -- 2.3x on GLM-5.3-Flash -- and had the wrong k.
    if not q_lora:
        raise DeriveError(
            f"{model}: a lightning indexer projects its query from the latent rank, "
            f"and the config declares no q_lora_rank to size wq_b from")
    # wk_weights_proj is constructed with quant_config=None in both implementations
    # (deepseek_v32/attention.py:78-86, glm5next/common/attention.py:274-281), so it is
    # served unquantized whatever the checkpoint's global width. The fp8 GLM DSA graphs
    # would otherwise inherit fp8 for a matrix the runtime keeps at the base width.
    base = base_dtype(cfg, raw, model)
    # wk_weights_proj is built quant_config=None in BOTH implementations
    # (deepseek_v32/attention.py:78-86, glm5next/common/attention.py:274-281), so it
    # always takes the base width. wq_b takes it only where the whole module is
    # unquantized: deepseek_v32 hands wq_b the real quant_config.
    module_base = mla_module_unquantized(raw, cfg)
    wq_b = gemm("index_wq_b", index_heads * index_dim, q_lora)
    if module_base:
        wq_b = dict(wq_b, weight_dtype=base)
    projections = [
        wq_b,
        dict(gemm("index_wk_weights_proj", index_dim + index_heads, hidden),
             weight_dtype=base),
    ]
    # The FP32 head gate is glm5next's ALONE. Its forward keeps the first head_dim rows
    # of wk_weights_proj for K, then caches the remaining n_head rows transposed to FP32
    # and runs its own matmul against fp32 activations -- `weights =
    # torch.mm(hidden_states.float(), self._wp_fp32)` (glm5next/common/attention.py
    # 341-350) -- which is a second launch and a resident FP32 copy.
    #
    # deepseek_v32 does NOT: it slices index_weights straight out of the same GEMM's
    # result and hands it to fused_q (deepseek_v32/attention.py:309-315, 433-446), with
    # no FP32 copy and no second matmul. Emitting this node for that family would invent
    # an FP32 launch on every indexed glm-5* layer, so it is gated on the architecture
    # rather than on the presence of an indexer.
    if recomputes_head_gate(raw, cfg):
        projections.append(dict(gemm("index_head_gate_proj", index_heads, hidden),
                                weight_dtype="fp32"))
    if kpool > 1:
        if raw is None:
            raise DeriveError(
                f"{model}: the kpool compression pass sizes its traffic from the "
                f"declared activation width, so this call must pass raw")
        # The compression gate is BF16 in the checkpoint, and glm5next -- the only
        # family with a pooled indexer -- builds the module unquantized anyway.
        projections.append(dict(gemm("index_kpool_compress_gate", index_dim, hidden),
                                weight_dtype=base))
        # The compression kernel runs ONE PROGRAM PER POOL, not per token
        # (nvidia/ops/kpool_compress.py:165-245), so its figure is per-pool traffic
        # amortized over the index_kpool tokens that fill one pool -- bytes_per_token is
        # charged once per model token by a cost model.
        #
        # Per pool, read off the kernel: slot_score is loaded in BOTH passes (the
        # per-dim max for softmax stability, then the weighted sum), the FP32 APE row
        # likewise, slot_k once, and the result is one Hadamard-rotated fp8 vector plus
        # its fp32 scale.
        # Each tensor at its STORED width, not the width the kernel accumulates in.
        # slot_score is bf16: gate_score comes from F.linear over bf16 hidden states and
        # the bf16 compression gate, and the tail cache documents its own copy as the
        # "bf16 gate score" (common/attention.py:172). The Triton load casts to fp32 in
        # registers, which costs no HBM. Only the APE is genuinely fp32 in memory.
        act = DTYPE_WIDTHS[base_dtype(cfg, raw, model)]
        f32 = DTYPE_WIDTHS["fp32"]
        per_pool = (2 * kpool * index_dim * act       # slot_score, bf16, both passes
                    + 2 * kpool * index_dim * f32     # APE rows, fp32, both passes
                    + kpool * index_dim * act         # slot_k, bf16
                    + index_dim + f32)                # fp8 vector + its fp32 scale
        projections.append({
            "op": "Elementwise",
            "role": "index_kpool_compress",
            "bytes_per_token": per_pool // kpool,
        })
        # The APE's read traffic is already inside that figure. It is NOT emitted as a
        # node of its own: the only primitive that could carry its bytes is a GEMM, and
        # a GEMM means 2*n*k FLOPs per token plus a launch in the consuming kernel
        # (blis-latency-kernel internal/price/plan.go:266-269). The APE is a storage-only
        # [index_kpool, index_head_dim] fp32 table consumed inside the pooling kernel, so
        # a GEMM would invent 1,024 FLOPs/token and a launch per indexed layer -- a worse
        # error than the residency it would record. Representing resident parameter bytes
        # that carry no compute needs a schema primitive the ModelGraph does not have;
        # tracked rather than faked.
    # The indexer's k-norm is a node only where it is its OWN launch. deepseek_v32 folds
    # it into the same fused_norm_rope call as the two latent norms
    # (deepseek_v32/attention.py:378-392), so emitting it there would charge a
    # normalisation kernel that family does not launch -- the traffic is already in the
    # fused latent norm. glm5next runs _fused_indexer_k_norm separately (:352), so it
    # keeps its node.
    #
    # Where it is emitted, it reads and writes at the ACTIVATION width:
    # _fused_indexer_k_norm is `F.layer_norm(x.float(), ...).type_as(x)`
    # (glm5next/common/attention.py:55-59), so the fp32 cast is a temporary inside the
    # fused kernel and x is the bf16 output of the unquantized wk_weights_proj. Sizing
    # this from the accumulation dtype would double its HBM traffic.
    if not mla_layout(raw or {}, cfg).get("fused_index_norm"):
        projections.append(dict(norm("index_k_norm"),
                                bytes_per_token=2 * index_dim
                                * DTYPE_WIDTHS[base_dtype(cfg, raw, model)]))

    scorer = {
        "op": "Attention",
        "role": "block_index_scores",
        "kind": "gqa",
        "n_q": index_heads,
        "n_kv": 1,
        "d_h": index_dim,
    }
    if kpool > 1:
        # The scan is over POOLED candidates: one K state per index_kpool tokens, which
        # vLLM expresses as the cache spec's tokens_per_state and sizes as
        # max_model_len // index_kpool (common/attention.py:128-131, 314).
        #
        # index_topk is deliberately NOT set here. The scorer must score EVERY pooled
        # candidate in order to discover the top-k; the selection bounds the main MLA
        # read, which already carries index_topk. Setting it here would make the scorer
        # read only the answers it exists to find, and would cap the one
        # context-proportional term in the layer at a constant. DeepSeek-V4's
        # representation draws the same line: block_index_scores is unbounded and only
        # the downstream sparse MLA node carries the top-k.
        scorer["kind"] = "sparse_mla"
        scorer["compress_ratio"] = kpool
    return nodes[:at] + projections + [scorer] + nodes[at:]


# --- Architecture handlers ------------------------------------------------------
# One handler per family. A family whose layer sequence is uniform returns one layer
# kind and a repeat; a hybrid returns several kinds and the sequence that orders them.
# An architecture with no handler raises rather than falling back on a default, which
# is what keeps a misread config from becoming a plausible-looking graph.


def handler_dense(cfg, raw, model):
    """Llama, Mistral, Qwen2/3, Yi, CodeLlama: GQA attention and a dense MLP.

    A sliding window, where the config declares one, changes the attention kind: the
    per-token read is bounded by the window rather than by context length."""
    hidden = int(require(cfg, "hidden_size", model))
    window = pick(cfg, "sliding_window")
    # A declared window is inert unless the config also switches it on. Qwen2.5 ships
    # use_sliding_window: false with a window value present, and pricing it as
    # windowed would understate every layer's KV read.
    if window and cfg.get("use_sliding_window") is False:
        window = None
    nodes = attention_block(cfg, model, hidden, raw=raw, window=window)
    nodes += [norm("post_attn_norm")]
    nodes += dense_mlp(cfg, model, hidden)
    nodes += [collective("AllReduce", "mlp_out", EMIT_TENSOR_PARALLEL_UNLESS_SP_MOE)]
    kind = {"id": "dense", "nodes": nodes, "edges": chain(nodes)}
    layers = int(require(cfg, "num_layers", model))
    return [kind], {"pattern": ["dense"], "repeat": layers}


def handler_moe(cfg, raw, model):
    """Mixtral, Qwen3-MoE, Llama-4, Granite-MoE, Inkling, DeepSeek-V2, GLM: attention
    plus a routed expert layer.

    Where a config alternates dense and sparse MLP layers, the dense ones become a
    prologue: the sequence has no repeating unit that includes them.

    KNOWN GAP (Llama-4): this family uses chunked local attention bounded by
    attention_chunk_size on most layers, with periodic full-attention NoPE layers, and
    this handler prices every layer as unbounded GQA. The split is NOT derivable from the
    committed Scout checkpoint: transformers reads it from config.layer_types
    (modeling_llama4.py dispatches create_chunked_causal_mask per layer from that list),
    and this config declares no layer_types, no moe_layers, and an EMPTY no_rope_layers.
    Deriving a pattern would mean inventing one, which is what this deriver's handler
    registry exists to prevent.

    The cost of the gap is not uniform across the catalog's workloads, and is larger than
    "chunk size exceeds every context" would suggest. At attention_chunk_size 8192,
    chatbot (512) and contentgen (2048) sit well inside one chunk, where a bounded and an
    unbounded read coincide exactly. summarization reaches 8704 only at its maximum. But
    multidoc's TYPICAL context is 11776 and its maximum 22016, so on that workload this
    handler overstates the per-token read on the chunked layers by roughly the ratio of
    context to chunk -- about 1.4x typical, 2.7x at the tail.

    Recorded rather than silently accepted: a reader comparing a Llama-4 multidoc
    prediction against a measurement should suspect this first. Closing it needs the
    per-layer split, which needs either a checkpoint that declares layer_types or the
    vendor pattern confirmed against an implementation.
    """
    hidden = int(require(cfg, "hidden_size", model))
    layers = int(require(cfg, "num_layers", model))

    window = pick(cfg, "sliding_window")
    if window and cfg.get("use_sliding_window") is False:
        window = None

    # A family in this group may ship a DSA lightning indexer: the GLM-5 generations
    # declare model_type glm_moe_dsa with a full index_n_heads/index_head_dim/index_topk
    # spec, and attention() reads the topk into kind=sparse_mla. The selection work is
    # a node of its own, and omitting it leaves the graph asserting a top-k it never
    # charges for -- which is what happened here across 78 layers per model.
    # ...but not necessarily on every layer. indexed_layers() expands the config's own
    # skip_topk rule, which for glm-5.2/5.2-fp8/5.3 indexes 21 of 78 layers. Keyed per
    # layer rather than per model: a single flag charged all 78 for a scorer that 57 of
    # them never build.
    index_map = indexed_layers(cfg, model, layers)
    indexed_any = index_map is not None and any(index_map)
    indexed_all = index_map is not None and all(index_map)
    # A layer kind must be uniform in whether it indexes, so a partially-indexed model
    # needs the distinction carried in the kind id.
    split_index = indexed_any and not indexed_all

    def attn_nodes(win=None, index=None):
        use = indexed_any if index is None else index
        n = attention_block(cfg, model, hidden, raw=raw, window=win, indexed=use)
        return lightning_indexer(cfg, model, hidden, n, raw) if use else n

    def sparse_nodes(win=None, index=None):
        n = attn_nodes(win, index)
        n += [norm("post_attn_norm"), *moe_block(cfg, model, hidden),
              collective("All2All", "moe_dispatch_combine", EMIT_EXPERT_PARALLEL),
              collective("AllReduce", "mlp_out", EMIT_TENSOR_PARALLEL_UNLESS_SP_MOE)]
        return n

    def dense_nodes(win=None, index=None):
        n = attn_nodes(win, index)
        n += [norm("post_attn_norm")] + dense_mlp(cfg, model, hidden)
        n += [collective("AllReduce", "mlp_out", EMIT_TENSOR_PARALLEL_UNLESS_SP_MOE)]
        return n

    def suffix(index: bool) -> str:
        """The kind-id suffix that keeps an indexed layer distinct from a skipped one."""
        if not split_index:
            return ""
        return "_indexed" if index else "_noindex"

    # The MTP module always builds an indexer, whatever the base skip rule says. vLLM
    # evaluates the draft at layer_id == num_hidden_layers and then constructs under
    # `if not skip_topk or is_mtp_layer` (deepseek_v32/attention.py:179-200), so the
    # is_mtp_layer term forces it even on a frequency-4 config whose layer 78 would skip.
    # Without a declared draft kind, derive()'s fallback mirrors the last DECLARED kind,
    # which on these configs is the unindexed one -- underpricing the draft indexer on
    # every speculative step.
    def declare_draft(kinds: list[dict[str, Any]]) -> None:
        if not split_index:
            return
        if not int(pick(cfg, "num_spec_tokens") or 0):
            return
        nodes = sparse_nodes(window, True)
        kinds.append({"id": DRAFT_KIND_ID, "nodes": nodes, "edges": chain(nodes)})


    # A per-layer ATTENTION-kind vector, which is a different thing from the per-layer
    # MLP-type vector handled below: this one says which layers see a bounded window
    # while every layer keeps the same MLP. gpt-oss declares it as layer_types over
    # "sliding_attention" and "full_attention".
    #
    # Checked against the layer count rather than trusted: a vector of the wrong length
    # would silently price a different depth than the config states.
    attn_types = cfg.get("layer_types")
    if attn_types and window:
        if len(attn_types) != layers:
            raise DeriveError(
                f"{model}: layer_types has {len(attn_types)} entries for {layers} layers"
            )
        unknown = set(attn_types) - {"sliding_attention", "full_attention"}
        if unknown:
            raise DeriveError(
                f"{model}: layer_types names {sorted(unknown)}, which this handler cannot "
                f"price; add the kind rather than defaulting it to full attention"
            )
        if split_index:
            raise DeriveError(
                f"{model}: a per-layer attention-kind vector AND a partial indexer "
                f"pattern would make the layer kind a three-way product; no committed "
                f"config does both, so this handler refuses rather than pricing one "
                f"dimension and dropping the other")
        swa = sparse_nodes(win=window)
        full = sparse_nodes(win=None)
        kinds = [
            {"id": "swa_moe", "nodes": swa, "edges": chain(swa)},
            {"id": "full_moe", "nodes": full, "edges": chain(full)},
        ]
        sequence = ["swa_moe" if t == "sliding_attention" else "full_moe"
                    for t in attn_types]
        return kinds, compress(sequence)

    # Inkling interleaves sliding-window and full-attention layers on a declared
    # index list; every other member of this family is uniform in attention kind.
    local_ids = cfg.get("local_layer_ids")
    if local_ids and window:
        if split_index:
            raise DeriveError(
                f"{model}: a declared local-layer list AND a partial indexer pattern "
                f"would make the layer kind a three-way product; no committed config "
                f"does both, so this handler refuses rather than dropping one")
        swa = sparse_nodes(win=window)
        full = sparse_nodes(win=None)
        kinds = [
            {"id": "swa_moe", "nodes": swa, "edges": chain(swa)},
            {"id": "full_moe", "nodes": full, "edges": chain(full)},
        ]
        local = set(int(i) for i in local_ids)
        sequence = ["swa_moe" if i in local else "full_moe" for i in range(layers)]
        return kinds, compress(sequence)

    # A per-layer MLP type vector marks which layers are sparse.
    mlp_types = cfg.get("mlp_layer_types")
    if mlp_types:
        if len(mlp_types) != layers:
            raise DeriveError(
                f"{model}: mlp_layer_types has {len(mlp_types)} entries for "
                f"{layers} layers"
            )
        # The kind of a layer is the pair (mlp type, does it index), because a skipped
        # layer builds no indexer at all. Only the combinations the sequence actually
        # uses are emitted.
        sequence = []
        for i, t in enumerate(mlp_types):
            idx = True if index_map is None else index_map[i]
            sequence.append(("attn_moe" if t == "sparse" else "attn_dense") + suffix(idx))
        kinds = []
        for mlp_t, builder in (("dense", dense_nodes), ("moe", sparse_nodes)):
            for idx in (True, False):
                kid = ("attn_moe" if mlp_t == "moe" else "attn_dense") + suffix(idx)
                if kid in {k["id"] for k in kinds} or kid not in set(sequence):
                    continue
                nodes = builder(window, idx)
                kinds.append({"id": kid, "nodes": nodes, "edges": chain(nodes)})
        declare_draft(kinds)
        # compress() finds the trailing uniform run itself, which is the common shape
        # here: a short dense prologue then sparse throughout.
        return kinds, compress(sequence)

    # A first_k_dense_replace count does the same job as a type vector.
    first_dense = int(cfg.get("first_k_dense_replace") or 0)
    if split_index:
        # A partially-indexed stack has no single repeated layer, so the sequence is
        # expanded per layer and compress() finds whatever period it really has.
        sequence = []
        for i in range(layers):
            base = "attn_dense" if i < first_dense else "attn_moe"
            sequence.append(base + suffix(index_map[i]))
        kinds = []
        for kid in dict.fromkeys(sequence):
            idx = kid.endswith("_indexed")
            builder = dense_nodes if kid.startswith("attn_dense") else sparse_nodes
            nodes = builder(window, idx)
            kinds.append({"id": kid, "nodes": nodes, "edges": chain(nodes)})
        declare_draft(kinds)
        return kinds, compress(sequence)
    sparse = sparse_nodes(window)
    kinds = [{"id": "attn_moe", "nodes": sparse, "edges": chain(sparse)}]
    if first_dense:
        dense = dense_nodes(window)
        kinds.insert(0, {"id": "attn_dense", "nodes": dense, "edges": chain(dense)})
        return kinds, {
            "prologue": ["attn_dense"] * first_dense,
            "pattern": ["attn_moe"],
            "repeat": layers - first_dense,
        }
    return kinds, {"pattern": ["attn_moe"], "repeat": layers}


def handler_minimax_m2(cfg, raw, model):
    """MiniMax-M2: GQA attention plus a routed expert layer on every layer.

    The config carries attn_type_list, a per-layer vector, and on the published M2.5
    weights every entry is the same. So the stack is uniform and handler_moe prices it
    exactly -- but the vector is checked rather than ignored, because a later variant that
    mixed attention kinds would otherwise be priced as though it did not.
    """
    types = cfg.get("attn_type_list")
    if types:
        layers = int(require(cfg, "num_layers", model))
        if len(types) != layers:
            raise DeriveError(
                f"{model}: attn_type_list has {len(types)} entries for {layers} layers"
            )
        if len(set(types)) != 1:
            raise DeriveError(
                f"{model}: attn_type_list mixes {sorted(set(types))}. This handler prices "
                f"a uniform stack; a mixed one needs its kinds named and priced rather "
                f"than collapsed to one"
            )
    return handler_moe(cfg, raw, model)


def handler_deepseek_v4(cfg, raw, model):
    """DeepSeek-V4: compressed sparse attention at two ratios, a top-k indexer, and a
    routed expert layer.

    Three facts decide the representation, all read from vLLM's own implementation in
    `vllm/models/deepseek_v4/` rather than inferred from field names.

    `head_dim` is INCLUSIVE of the RoPE width. compressor.py derives
    `nope_head_dim = head_dim - rope_head_dim`, so the 512 this config states already
    contains qk_rope_head_dim 64; treating them as additive would overstate the cache
    by an eighth. There is no kv_lora_rank here, which is why the MLA branch of
    `attention()` cannot price this family -- it reconstructs the per-token width as
    lora + rope, and this config states the total directly.

    The per-layer `compress_ratios` vector selects between two bounds on ONE kernel.
    Both are kernel_source FLASHMLA_SPARSE_DSV4 in the measured tables, differing only
    in compress_ratio, and sparse_mla.py sizes the read as
    `next_power_of_2(max_seq_len / compress_ratio)` floored at a 128-token alignment.
    So ratio 128 reads a stream that never exceeds that floor -- measured flat, 126.0
    to 124.6 us across a 16x context increase -- while ratio 4 grows as ctx/4 and
    measures 1.18x over the same range.

    The indexer is a separate node because it is the ONLY context-proportional term,
    measuring 3.53x over a 12x context increase where the attention it feeds moves
    1.18x. Folding it in would charge a context scan at a bounded rate.
    """
    hidden = int(require(cfg, "hidden_size", model))
    layers = int(require(cfg, "num_layers", model))

    ratios = cfg.get("compress_ratios")
    if not ratios:
        raise DeriveError(
            f"{model}: no compress_ratios vector. This family's attention read is "
            f"bounded by a per-layer compression ratio, and pricing every layer alike "
            f"would misprice whichever kind it did not pick"
        )
    # The vector carries a trailing entry for the MTP module, which is not one of the
    # transformer layers num_hidden_layers counts. vLLM indexes it by layer and takes
    # compress_ratio 1 for any index at or beyond num_hidden_layers, NOT the trailing
    # entry's value -- which this config states as 0, a ratio that cannot be read as a
    # divisor at all (attention.py:229-235 at the pinned commit, `else:
    # self.compress_ratio = 1`). So the draft module is an UNCOMPRESSED latent layer,
    # and the entry is dropped rather than consulted.
    draft_ratio = None
    if len(ratios) == layers + 1:
        ratios = ratios[:layers]
        draft_ratio = 1
    elif len(ratios) != layers:
        raise DeriveError(
            f"{model}: compress_ratios has {len(ratios)} entries for {layers} layers; "
            f"expected {layers}, or {layers + 1} with a trailing draft entry"
        )
    unknown = sorted(set(ratios) - {4, 128})
    if unknown:
        raise DeriveError(
            f"{model}: compress_ratios names {unknown}, which this handler cannot "
            f"price. vLLM's compressor asserts compress_ratio in [4, 128] "
            f"(models/deepseek_v4/compressor.py); a third ratio needs its own bound "
            f"rather than the nearer of these two"
        )

    topk = int(require_key(cfg, "index_topk", model))
    index_heads = int(require_key(cfg, "index_n_heads", model))
    index_dim = int(require_key(cfg, "index_head_dim", model))
    rope = int(require(cfg, "qk_rope_head_dim", model))
    total_dim = int(require(cfg, "head_dim", model))
    if total_dim <= rope:
        raise DeriveError(
            f"{model}: head_dim {total_dim} does not exceed qk_rope_head_dim {rope}, "
            f"so it cannot be the inclusive width vLLM's compressor splits"
        )

    indexer = {
        "op": "Attention",
        "role": "block_index_scores",
        "kind": "gqa",
        "n_q": index_heads,
        "n_kv": 1,
        "d_h": index_dim,
    }

    swa_window = int(require(cfg, "sliding_window", model))

    # The projection path is LOW-RANK on both sides, and an RMSNorm sits between the
    # two input stages, so neither side can be folded into one hidden-width rectangle.
    # Shapes from the pinned DeepseekV4Attention (attention.py:248-284):
    #
    #   fused_wqa_wkv : hidden -> q_lora_rank + head_dim   (fused q-LoRA down + KV)
    #   q_norm        : over q_lora_rank
    #   wq_b          : q_lora_rank -> n_heads * head_dim
    #   wo_a          : grouped, (n_heads*head_dim)/o_groups -> o_groups*o_lora_rank
    #   wo_b          : o_groups*o_lora_rank -> hidden
    #
    # Pricing these as one dense QKV and one dense output rectangle overstated the input
    # side 4.1x (473,432,064 against 115,343,360) and the output side 2.5x (469,762,048
    # against 184,549,376), which over 61 layers is tens of billions of parameters in
    # both weight bytes and FLOPs.
    q_lora = int(require_key(cfg, "q_lora_rank", model))
    o_lora = int(require_key(cfg, "o_lora_rank", model))
    o_groups = int(require_key(cfg, "o_groups", model))
    q_heads = int(require(cfg, "num_q_heads", model))
    if (q_heads * total_dim) % o_groups:
        raise DeriveError(
            f"{model}: o_groups {o_groups} does not divide the {q_heads * total_dim}-wide "
            f"head concatenation, so the grouped output projection has no shape")

    def projection_nodes():
        return [
            gemm("fused_wqa_wkv", q_lora + total_dim, hidden),
            norm("q_norm"),
            gemm("wq_b", q_heads * total_dim, q_lora),
        ], [
            # wo_a is a batched matmul over o_groups groups; N and K are the per-call
            # shape the consumer prices, and the group count rides in the role.
            gemm("wo_a", o_groups * o_lora, q_heads * total_dim // o_groups),
            gemm("wo_b", hidden, o_groups * o_lora),
        ]

    def layer_for(ratio):
        # The read set is the UNION of the retained sliding window and the compressed
        # positions, not a choice between them: every layer keeps its SWA window
        # (attention.py:221, `self.window_size = config.sliding_window`, with no
        # compress_ratio guard) and combine_topk_swa_indices merges the two position
        # sets. So a layer states its window AND its compressed bound.
        #
        # The two bounds differ in kind. A ratio-4 layer SELECTS up to index_topk
        # compressed positions through its indexer, so the compressed term is
        # min(index_topk, context/4). A ratio-128 layer selects nothing: it reads a
        # positional run of context/128 (sparse_mla.py, num_compressed =
        # (position + 1) // compress_ratio), which no fixed token count expresses,
        # hence compress_ratio in the graph.
        selects_topk = ratio == 4
        attn = attention(
            cfg, model,
            kind="sparse_mla" if ratio > 1 else "mla",
            latent_width=total_dim,
            window=swa_window,
            compress_ratio=ratio if ratio > 1 else None,
            index_topk=topk if selects_topk else None,
        )
        inp, out = projection_nodes()
        nodes = [norm("input_norm"), *inp]
        if ratio > 1:
            # The compressor runs whenever compress_ratio > 1 -- a broader condition
            # than the indexer's -- and its fused KV-and-gate projection is work of its
            # own, applied to the hidden states (attention.py, fused_wkv_wgate).
            nodes.append(gemm("compressor_fused_wkv_wgate", total_dim + 1, hidden))
        if selects_topk:
            # The indexer exists ONLY on ratio-4 layers: vLLM builds one under
            # `if self.compress_ratio == 4`, noting "Only C4A uses sparse attention and
            # hence has indexer" (attention.py:297-317). A ratio-128 layer has a
            # compressor but no indexer, and pricing one there charges a
            # context-proportional scan the layer never runs.
            #
            # Its wq_b reads the q-LoRA rank, not the hidden size, and it carries its
            # own compressor and a per-head weights projection.
            nodes += [
                gemm("index_wq_proj", index_heads * index_dim, q_lora),
                gemm("index_weights_proj", index_heads, hidden),
                gemm("index_compressor_fused_wkv_wgate", index_dim + 1, hidden),
                dict(indexer),
            ]
        nodes += [
            attn,
            *out,
            collective("AllReduce", "attn_out", EMIT_TENSOR_PARALLEL),
        ]
        nodes += [
            norm("post_attn_norm"),
            *moe_block(cfg, model, hidden),
            collective("All2All", "moe_dispatch_combine", EMIT_EXPERT_PARALLEL),
            collective("AllReduce", "mlp_out", EMIT_TENSOR_PARALLEL_UNLESS_SP_MOE),
        ]
        return nodes

    kinds = []
    for ratio in sorted(set(ratios)):
        nodes = layer_for(ratio)
        kinds.append({"id": f"csa{ratio}_moe", "nodes": nodes, "edges": chain(nodes)})
    # The draft module is its own layer kind rather than a reuse of a target layer's:
    # at ratio 1 it is uncompressed, so it reads the full context within its window and
    # carries no indexer. Without it the speculator's generic fallback mirrors
    # kinds[-1] -- csa128_moe -- pricing the tightest-bounded read in the model for the
    # one layer that has no bound at all.
    if draft_ratio is not None:
        nodes = layer_for(draft_ratio)
        kinds.append({"id": "mtp_moe", "nodes": nodes, "edges": chain(nodes)})
    sequence = [f"csa{r}_moe" for r in ratios]
    return kinds, compress(sequence)


def handler_minimax_m3(cfg, raw, model):
    """MiniMax-M3: block-sparse attention with a learned indexer, over a MoE stack whose
    first three layers are dense.

    Two layer kinds, ordered by the config's own per-layer vectors. `moe_layer_freq`
    and `sparse_attention_config.sparse_attention_freq` are checked to be the same
    vector rather than assumed to be: they are two statements of one fact on the
    published weights, and a variant that decoupled them would need its kinds named
    rather than collapsed.

    The attention kind is `swa`. What the schema's SWA means is a per-token read
    bounded by something other than context length, and M3's bound is
    `sparse_topk_blocks * sparse_block_size` tokens. The bound is the part a KV-byte
    cost model reads, and the arithmetic is the same for a contiguous window as for
    scattered blocks. What is NOT the same is which positions those are: SWA reads the
    most recent window, M3 reads blocks chosen per query. That does not change the bytes
    read, so it does not change this node; it shows up as the indexer below.

    The indexer is its own Attention node rather than a term folded into the main one.
    It has its own projections on the published weights -- index_q_proj [512, 6144] and
    index_k_proj [128, 6144], i.e. `sparse_num_index_heads` heads of `sparse_index_dim`
    for the query and one such head for the key -- so it maintains a second, narrow KV
    cache and scores every block to pick the top k. Its read is bounded by context, not
    by the top-k, which is why it cannot ride inside a node whose read is bounded.

    The dense layers' MLP width is `dense_intermediate_size`, not `intermediate_size`.
    Both are declared and they differ by 4x; the published weights settle which is which
    (`layers.0.mlp.gate_proj` is [12288, 6144] where a routed expert's w1 is
    [3072, 6144]).
    """
    hidden = int(require(cfg, "hidden_size", model))
    layers = int(require(cfg, "num_layers", model))

    sparse = cfg.get("sparse_attention_config")
    if not sparse:
        raise DeriveError(
            f"{model}: no sparse_attention_config. This family's attention read is "
            f"bounded by a block top-k, and pricing it as dense GQA would overstate "
            f"every sparse layer's KV read"
        )
    if sparse.get("use_sparse_attention") is not True:
        raise DeriveError(
            f"{model}: sparse_attention_config present but use_sparse_attention is "
            f"{sparse.get('use_sparse_attention')!r}; a declared-but-off bound is a "
            f"different model and this handler will not guess which"
        )

    attn_freq = sparse.get("sparse_attention_freq")
    moe_freq = cfg.get("moe_layer_freq")
    if not attn_freq or not moe_freq:
        raise DeriveError(
            f"{model}: needs both sparse_attention_freq and moe_layer_freq to order "
            f"its layer kinds"
        )
    pairs = (("sparse_attention_freq", attn_freq),
             ("moe_layer_freq", moe_freq))
    for name, vec in pairs:
        if len(vec) != layers:
            raise DeriveError(
                f"{model}: {name} has {len(vec)} entries for {layers} layers"
            )
    if list(attn_freq) != list(moe_freq):
        raise DeriveError(
            f"{model}: sparse_attention_freq and moe_layer_freq differ. This handler "
            f"prices one dense prologue and one sparse-plus-routed body; a stack that "
            f"mixed them independently needs its kinds named rather than paired"
        )
    unknown = set(attn_freq) - {0, 1}
    if unknown:
        raise DeriveError(
            f"{model}: sparse_attention_freq names {sorted(unknown)}, which this "
            f"handler cannot price; add the kind rather than defaulting it"
        )

    # A stated dense-prologue length and the vector state the same fact. Read by
    # literal name: only this family declares it, so it has no cross-dialect concept.
    first_dense = cfg.get("first_k_dense_replace")
    if first_dense is not None:
        expected = [0] * int(first_dense) + [1] * (layers - int(first_dense))
        if expected != list(attn_freq):
            raise DeriveError(
                f"{model}: first_k_dense_replace {first_dense} disagrees with the "
                f"per-layer vector; the two state one fact and this handler will not "
                f"choose between them"
            )

    topk_blocks = int(require_key(sparse, "sparse_topk_blocks", model))
    block = int(require_key(sparse, "sparse_block_size", model))
    index_heads = int(require_key(sparse, "sparse_num_index_heads", model))
    index_dim = int(require_key(sparse, "sparse_index_dim", model))
    local = int(sparse.get("sparse_local_block") or 0)
    init = int(sparse.get("sparse_init_block") or 0)
    # Every block the kernel is guaranteed to read: the top-k plus the always-resident
    # local and initial blocks. Counted in tokens because that is what a KV read costs.
    bound = (topk_blocks + local + init) * block

    dense_attn = attention_block(cfg, model, hidden, raw=raw)
    dense_ffn = int(require_key(cfg, "dense_intermediate_size", model))
    dense_nodes = dense_attn + [
        norm("post_attn_norm"),
        gemm("mlp_gate_up", 2 * dense_ffn, hidden),
        gemm("mlp_down", hidden, dense_ffn),
        collective("AllReduce", "mlp_out", EMIT_TENSOR_PARALLEL_UNLESS_SP_MOE),
    ]

    sparse_attn = attention_block(cfg, model, hidden, raw=raw, kind="swa", window=bound)
    # The indexer scores the whole context to choose blocks, so its read is bounded by
    # context rather than by the top-k. It is placed BEFORE the bounded attention node,
    # which is the order it runs in: its scores pick the blocks that node then reads.
    indexer = {
        "op": "Attention",
        "role": "block_index_scores",
        "kind": "gqa",
        "n_q": index_heads,
        "n_kv": 1,
        "d_h": index_dim,
    }
    at = next(i for i, n in enumerate(sparse_attn) if n.get("op") == "Attention")
    sparse_nodes = sparse_attn[:at] + [
        gemm("index_qk_proj", index_heads * index_dim + index_dim, hidden),
        indexer,
    ] + sparse_attn[at:]
    sparse_nodes += [
        norm("post_attn_norm"),
        *moe_block(cfg, model, hidden),
        collective("All2All", "moe_dispatch_combine", EMIT_EXPERT_PARALLEL),
        collective("AllReduce", "mlp_out", EMIT_TENSOR_PARALLEL_UNLESS_SP_MOE),
    ]

    kinds = [
        {"id": "dense", "nodes": dense_nodes, "edges": chain(dense_nodes)},
        {"id": "sparse_attn_moe", "nodes": sparse_nodes, "edges": chain(sparse_nodes)},
    ]
    sequence = ["sparse_attn_moe" if f else "dense" for f in attn_freq]
    return kinds, compress(sequence)


def handler_qwen3_5_moe(cfg, raw, model):
    """Qwen3.5-MoE: a Gated DeltaNet linear-attention layer on most layers, full attention
    on every fourth, and a routed expert MLP with a shared expert on all of them.

    The full-attention positions come from the declared layer_types vector rather than
    from full_attention_interval. Both are present and they agree, but the vector is the
    ground truth: an interval is a generator for the common case and the vector is what the
    model was built with, so a variant that broke the period would still be priced right.

    The GDN state shape follows vLLM's own calculator
    (`MambaStateShapeCalculator.gated_delta_net_state_shape`): a convolutional state of
    width `2*k_heads*k_dim + v_heads*v_dim` over `conv_kernel - 1` positions, and a
    temporal state of `v_heads * v_dim * k_dim`. The graph records the temporal state's
    per-head geometry, which is what a cost model reads; the convolution's width rides in
    intermediate_size.
    """
    hidden = int(require(cfg, "hidden_size", model))
    layers = int(require(cfg, "num_layers", model))

    types = cfg.get("layer_types")
    if not types:
        raise DeriveError(
            f"{model}: no layer_types vector. This family mixes linear and full attention "
            f"per layer, and full_attention_interval alone would be a guess at which"
        )
    if len(types) != layers:
        raise DeriveError(
            f"{model}: layer_types has {len(types)} entries for {layers} layers"
        )
    unknown = set(types) - {"linear_attention", "full_attention"}
    if unknown:
        raise DeriveError(
            f"{model}: layer_types names {sorted(unknown)}, which this handler cannot "
            f"price; add the kind rather than defaulting it"
        )

    # The vector and the stated interval must agree. They are two statements of the same
    # fact, and a config where they disagree is one this handler should not guess at.
    interval = int(cfg.get("full_attention_interval") or 0)
    if interval:
        expected = ["full_attention" if (i + 1) % interval == 0 else "linear_attention"
                    for i in range(layers)]
        if expected != list(types):
            raise DeriveError(
                f"{model}: layer_types disagrees with full_attention_interval "
                f"{interval}; the two state the same fact and this handler will not "
                f"choose between them"
            )

    k_heads = int(require_key(cfg, "linear_num_key_heads", model))
    v_heads = int(require_key(cfg, "linear_num_value_heads", model))
    k_dim = int(require_key(cfg, "linear_key_head_dim", model))
    v_dim = int(require_key(cfg, "linear_value_head_dim", model))
    conv = int(require_key(cfg, "linear_conv_kernel_dim", model))

    def with_moe(mixer):
        n = list(mixer)
        n += [norm("post_attn_norm"), *moe_block(cfg, model, hidden),
              collective("All2All", "moe_dispatch_combine", EMIT_EXPERT_PARALLEL),
              collective("AllReduce", "mlp_out", EMIT_TENSOR_PARALLEL_UNLESS_SP_MOE)]
        return n

    gdn_mixer = [
        norm("input_norm"),
        {
            "op": "RecurrentUpdate",
            "recurrent_kind": "gdn",
            "n_heads": v_heads,
            "state_size": k_dim,
            "conv_kernel": conv,
            # The convolution's width, which is what the conv state spans.
            "intermediate_size": 2 * k_heads * k_dim + v_heads * v_dim,
            "state_dtype": DTYPES.get(
                str(cfg.get("mamba_ssm_dtype", "float32")).lower(), "fp32"),
        },
        gemm("mixer_out", hidden, v_heads * v_dim),
        collective("AllReduce", "mixer_out", EMIT_TENSOR_PARALLEL),
    ]
    gdn = with_moe(gdn_mixer)
    full = with_moe(attention_block(cfg, model, hidden, raw=raw))
    kinds = [
        {"id": "gdn_moe", "nodes": gdn, "edges": chain(gdn)},
        {"id": "attn_moe", "nodes": full, "edges": chain(full)},
    ]
    sequence = ["attn_moe" if t == "full_attention" else "gdn_moe" for t in types]
    return kinds, compress(sequence)


def handler_nemotron_h(cfg, raw, model):
    """NemotronH: a declared per-layer vector over state-space, MoE and attention
    layers, with no repeating unit and -- on one variant -- no stated layer count.

    Depth is the vector's length. The MoE entries are MLP layers rather than separate
    transformer blocks, which is why the vector is longer than a reader expecting
    num_hidden_layers would guess."""
    hidden = int(require(cfg, "hidden_size", model))
    vector = cfg.get("layers_block_type") or cfg.get("hybrid_override_pattern")
    if not vector:
        raise DeriveError(f"{model}: no layers_block_type or hybrid_override_pattern")
    if isinstance(vector, str):
        # The character form: M state-space, * attention, - MLP, E MoE.
        chars = {"M": "mamba", "*": "attention", "-": "mlp", "E": "moe"}
        try:
            vector = [chars[c] for c in vector]
        except KeyError as exc:
            raise DeriveError(f"{model}: unknown layer character {exc}") from exc

    stated = pick(cfg, "num_layers")
    if stated is not None and int(stated) != len(vector):
        raise DeriveError(
            f"{model}: num_hidden_layers {stated} disagrees with a {len(vector)}-entry "
            f"layer vector"
        )

    mamba = [
        norm("input_norm"),
        {
            "op": "RecurrentUpdate",
            "recurrent_kind": "mamba2",
            "n_heads": int(cfg["mamba_num_heads"]),
            "state_size": int(cfg["ssm_state_size"]),
            "n_groups": int(cfg.get("n_groups") or cfg.get("mamba_n_groups") or 1),
            "conv_kernel": int(cfg["conv_kernel"]),
            "intermediate_size": int(cfg["mamba_num_heads"]) * int(cfg["mamba_head_dim"]),
            "state_dtype": DTYPES.get(
                str(cfg.get("mamba_ssm_cache_dtype", "float32")).lower(), "fp32"),
        },
        gemm("mixer_out", hidden,
             int(cfg["mamba_num_heads"]) * int(cfg["mamba_head_dim"])),
        collective("AllReduce", "mixer_out", EMIT_TENSOR_PARALLEL),
    ]
    # The recurrent mixer's projections where the checkpoint states their width. The
    # nvfp4 variant stores these at FP8 while its routed experts are NVFP4, so they
    # cannot take the global width either way round.
    classes = quantized_classes(cfg, model)
    if classes.get("recurrent_out"):
        for n in mamba:
            # Scoped to the projection: the AllReduce that follows shares the role name
            # and holds no parameters, so a dtype on it describes nothing.
            if n.get("role") == "mixer_out" and n["op"] == "GEMM":
                n["weight_dtype"] = classes["recurrent_out"]
    moe_nodes = [
        norm("input_norm"),
        *moe_block(cfg, model, hidden),
        collective("All2All", "moe_dispatch_combine", EMIT_EXPERT_PARALLEL),
        collective("AllReduce", "mlp_out", EMIT_TENSOR_PARALLEL_UNLESS_SP_MOE),
    ]
    attn = attention_block(cfg, model, hidden, raw=raw)
    mlp = [norm("input_norm")] + dense_mlp(cfg, model, hidden) + [
        collective("AllReduce", "mlp_out", EMIT_TENSOR_PARALLEL_UNLESS_SP_MOE)]

    catalogue = {
        "mamba": {"id": "mamba", "nodes": mamba, "edges": chain(mamba)},
        "moe": {"id": "moe", "nodes": moe_nodes, "edges": chain(moe_nodes)},
        "attention": {"id": "attention", "nodes": attn, "edges": chain(attn)},
        "mlp": {"id": "mlp", "nodes": mlp, "edges": chain(mlp)},
    }
    used = []
    for name in vector:
        if name not in catalogue:
            raise DeriveError(f"{model}: unknown layer type {name!r}")
        if name not in used:
            used.append(name)
    return [catalogue[n] for n in used], compress(list(vector))


def handler_kimi_k3(cfg, raw, model):
    """Kimi-K3: linear attention on most layers, latent attention on a declared list,
    and a dense MLP on the leading layers.

    The full-attention layer indices are given explicitly rather than by a period, so
    the sequence is built from that list.

    first_k_dense_replace makes the leading layers dense: KimiDecoderLayer builds a
    KimiSparseMoeBlock only when `layer_idx >= first_k_dense_replace`, and otherwise a
    KimiMLP at the config's own intermediate_size rather than moe_intermediate_size
    (modeling_kimi_linear.py:893-900). Pricing layer 0 as routed charges 896 experts
    for a layer that runs one dense MLP."""
    hidden = int(require(cfg, "hidden_size", model))
    layers = int(require(cfg, "num_layers", model))
    linear = cfg.get("linear_attn_config") or {}
    full = linear.get("full_attn_layers")
    if not full:
        raise DeriveError(f"{model}: no linear_attn_config.full_attn_layers")

    def with_moe(mixer_nodes):
        n = list(mixer_nodes)
        n += [norm("post_attn_norm"), *moe_block(cfg, model, hidden),
              collective("All2All", "moe_dispatch_combine", EMIT_EXPERT_PARALLEL),
              collective("AllReduce", "mlp_out", EMIT_TENSOR_PARALLEL_UNLESS_SP_MOE)]
        return n

    def with_dense(mixer_nodes):
        n = list(mixer_nodes)
        n += [norm("post_attn_norm"), *dense_mlp(cfg, model, hidden),
              collective("AllReduce", "mlp_out", EMIT_TENSOR_PARALLEL_UNLESS_SP_MOE)]
        return n

    # The KDA mixer's dense projections. RecurrentUpdate carries no N or K, so it
    # cannot account for these weights: without them the layer had no producer GEMM at
    # all for the reduction that follows it, dropping 443,580,416 parameters per layer
    # over 69 KDA layers (about 30.6B) and their FLOPs with them.
    #
    # Shapes from the pinned KimiDeltaAttention (modeling_kimi_linear.py:495-541). Q, K
    # and V are three separate Linears rather than one fused projection, each followed
    # by its own ShortConvolution; the decay path bottlenecks through head_dim; beta is
    # one scalar per head; and the output gate is a single dense g_proj here because
    # this config sets use_full_rank_gate (the low-rank g_a/g_b pair is the default).
    kda_heads = int(linear.get("num_heads") or require(cfg, "num_q_heads", model))
    kda_dim = int(linear.get("head_dim") or head_dim(cfg, model))
    kda_inner = kda_heads * kda_dim
    kda_proj = [
        gemm("kda_q_proj", kda_inner, hidden),
        gemm("kda_k_proj", kda_inner, hidden),
        gemm("kda_v_proj", kda_inner, hidden),
        # The decay gate, low-rank through head_dim.
        gemm("kda_f_a_proj", kda_dim, hidden),
        gemm("kda_f_b_proj", kda_inner, kda_dim),
        # One beta scalar per head.
        gemm("kda_b_proj", kda_heads, hidden),
    ]
    if linear.get("use_full_rank_gate"):
        kda_proj.append(gemm("kda_g_proj", kda_inner, hidden))
    else:
        kda_proj += [
            gemm("kda_g_a_proj", kda_dim, hidden),
            gemm("kda_g_b_proj", kda_inner, kda_dim),
        ]
    kda_mixer = [
        norm("input_norm"),
        *kda_proj,
        {
            "op": "RecurrentUpdate",
            "recurrent_kind": "kda",
            "n_heads": kda_heads,
            "state_size": kda_dim,
            "conv_kernel": int(linear.get("short_conv_kernel_size") or 0) or None,
            "state_dtype": "fp32",
        },
        gemm("kda_o_proj", hidden, kda_inner),
        collective("AllReduce", "mixer_out", EMIT_TENSOR_PARALLEL),
    ]
    # conv_kernel is omitted rather than zeroed where the config states none.
    kda_mixer = [n if n.get("conv_kernel") is not None or n.get("op") != "RecurrentUpdate"
                 else {k: v for k, v in n.items() if k != "conv_kernel"}
                 for n in kda_mixer]
    # The KDA mixer IS this layer's self_attn (vllm kimi_k3/nvidia/model.py:888-891), so
    # a compressed-tensors ignore list naming self_attn leaves these projections at the
    # base width just as it does the latent ones. kimi-k3's global width is mxfp4, so
    # without this every KDA projection prices at four bits for tensors the runtime
    # never quantizes.
    if compressed_tensors_skips_attention(cfg, raw):
        base = base_dtype(cfg, raw, model)
        kda_mixer = [dict(n, weight_dtype=base)
                     if n.get("role") in ATTENTION_WEIGHT_ROLES else n
                     for n in kda_mixer]
    kda = with_moe(kda_mixer)
    mla = with_moe(attention_block(cfg, model, hidden, raw=raw))
    kinds = [
        {"id": "kda_moe", "nodes": kda, "edges": chain(kda)},
        {"id": "mla_moe", "nodes": mla, "edges": chain(mla)},
    ]
    first_dense = int(cfg.get("first_k_dense_replace") or 0)
    if first_dense:
        kda_dense = with_dense(kda_mixer)
        mla_dense = with_dense(attention_block(cfg, model, hidden, raw=raw))
        kinds += [
            {"id": "kda_dense", "nodes": kda_dense, "edges": chain(kda_dense)},
            {"id": "mla_dense", "nodes": mla_dense, "edges": chain(mla_dense)},
        ]
    full_set = set(int(i) for i in full)
    # The published list is 1-based against a 1-indexed layer numbering; a 0-based
    # reading would shift every attention layer by one. Detected rather than assumed:
    # a 0-based list would contain 0.
    offset = 1 if 0 not in full_set else 0
    # The mixer and the MLP vary independently: which layers are attention comes from
    # the declared list, and which are dense from first_k_dense_replace.
    sequence = []
    for i in range(layers):
        mixer = "mla" if (i + offset) in full_set else "kda"
        sequence.append(f"{mixer}_dense" if i < first_dense else f"{mixer}_moe")
    # A kind no layer uses would be a node a reader has to account for and no step runs.
    kinds = [k for k in kinds if k["id"] in set(sequence)]
    return kinds, compress(sequence)


def handler_glm5_next(cfg, raw, model):
    """GLM-5.3-Flash: KDA on most layers, sparse latent attention on the rest, and a
    dense MLP prologue, with the composition stated by two per-layer vectors.

    Three things differ from handler_kimi_k3, which this otherwise mirrors.

    The layer sequence comes from `layer_types` and `mlp_layer_types` rather than from
    `full_attn_layers` plus `first_k_dense_replace`. Both vectors are length
    num_hidden_layers and they vary independently, so the kind of a layer is the pair:
    34 `linear_attention` against 11 `deepseek_sparse_attention`, and 3 `dense` MLPs
    before 42 `sparse` ones. Reading either vector as a period would misplace layers --
    the attention layers sit at every fourth index, which no single period expresses
    once the dense prologue is also in play.

    The attention layers are `deepseek_sparse_attention`, so they carry a lightning
    indexer. The config declares the full spec (`index_n_heads` 32, `index_head_dim`
    128, `index_topk` 2048) and `lightning_indexer()` emits it. Without that node the
    graph would record a top-k selection it never charges for, which is the omission
    that helper's docstring records against this family.

    `num_nextn_predict_layers` is 1 rather than kimi-k3's 0, so the model carries one
    MTP module. That reaches the graph through the `num_spec_tokens` alias, as it does
    for the DeepSeek-V3 family, rather than through a node here: an MTP module runs as
    its own forward pass, not as part of a base layer.

    The published repo is multimodal and ships a `vision_config`. Only the text path is
    priced, as for the other multimodal entries: the vision tower runs once per image,
    not per decode step, and the deployments this catalog serves are text."""
    hidden = int(require(cfg, "hidden_size", model))
    layers = int(require(cfg, "num_layers", model))

    mixer_types = cfg.get("layer_types")
    if not mixer_types:
        raise DeriveError(f"{model}: no layer_types vector")
    if len(mixer_types) != layers:
        raise DeriveError(
            f"{model}: layer_types has {len(mixer_types)} entries for {layers} layers"
        )
    mlp_types = cfg.get("mlp_layer_types")
    if not mlp_types:
        raise DeriveError(f"{model}: no mlp_layer_types vector")
    if len(mlp_types) != layers:
        raise DeriveError(
            f"{model}: mlp_layer_types has {len(mlp_types)} entries for {layers} layers"
        )

    linear = cfg.get("linear_attn_config") or {}
    if not linear:
        raise DeriveError(f"{model}: no linear_attn_config")

    def with_moe(mixer_nodes):
        n = list(mixer_nodes)
        n += [norm("post_attn_norm"), *moe_block(cfg, model, hidden),
              collective("All2All", "moe_dispatch_combine", EMIT_EXPERT_PARALLEL),
              collective("AllReduce", "mlp_out", EMIT_TENSOR_PARALLEL_UNLESS_SP_MOE)]
        return n

    def with_dense(mixer_nodes):
        n = list(mixer_nodes)
        n += [norm("post_attn_norm"), *dense_mlp(cfg, model, hidden),
              collective("AllReduce", "mlp_out", EMIT_TENSOR_PARALLEL_UNLESS_SP_MOE)]
        return n

    # The KDA mixer's dense projections, as for kimi-k3: RecurrentUpdate carries no N
    # or K, so without these the layer has no producer GEMM for the reduction after it.
    # This config states no use_full_rank_gate, so the low-rank g_a/g_b pair applies.
    kda_heads = int(linear.get("num_heads") or require(cfg, "num_q_heads", model))
    kda_dim = int(linear.get("head_dim") or head_dim(cfg, model))
    kda_inner = kda_heads * kda_dim
    kda_proj = [
        gemm("kda_q_proj", kda_inner, hidden),
        gemm("kda_k_proj", kda_inner, hidden),
        gemm("kda_v_proj", kda_inner, hidden),
        gemm("kda_f_a_proj", kda_dim, hidden),
        gemm("kda_f_b_proj", kda_inner, kda_dim),
        gemm("kda_b_proj", kda_heads, hidden),
    ]
    if linear.get("use_full_rank_gate"):
        kda_proj.append(gemm("kda_g_proj", kda_inner, hidden))
    else:
        kda_proj += [
            gemm("kda_g_a_proj", kda_dim, hidden),
            gemm("kda_g_b_proj", kda_inner, kda_dim),
        ]
    conv = int(linear.get("short_conv_kernel_size") or 0)
    recurrent = {
        "op": "RecurrentUpdate",
        "recurrent_kind": "kda",
        "n_heads": kda_heads,
        "state_size": kda_dim,
        "state_dtype": "fp32",
    }
    if conv:
        recurrent["conv_kernel"] = conv
    kda_mixer = [
        norm("input_norm"),
        *kda_proj,
        recurrent,
        gemm("kda_o_proj", hidden, kda_inner),
        collective("AllReduce", "mixer_out", EMIT_TENSOR_PARALLEL),
    ]
    # The sparse-attention layers are latent attention fronted by the indexer.
    mla_mixer = lightning_indexer(cfg, model, hidden,
                                  attention_block(cfg, model, hidden))

    builders = {
        ("kda", "moe"): lambda: with_moe(kda_mixer),
        ("kda", "dense"): lambda: with_dense(kda_mixer),
        ("mla", "moe"): lambda: with_moe(mla_mixer),
        ("mla", "dense"): lambda: with_dense(mla_mixer),
    }
    sequence = []
    for mixer_t, mlp_t in zip(mixer_types, mlp_types):
        if mixer_t == "linear_attention":
            mixer = "kda"
        elif mixer_t == "deepseek_sparse_attention":
            mixer = "mla"
        else:
            raise DeriveError(f"{model}: unknown layer_types entry {mixer_t!r}")
        if mlp_t not in ("sparse", "dense"):
            raise DeriveError(f"{model}: unknown mlp_layer_types entry {mlp_t!r}")
        mlp = "moe" if mlp_t == "sparse" else "dense"
        sequence.append(f"{mixer}_{mlp}")

    kinds = []
    for key, build in builders.items():
        kid = f"{key[0]}_{key[1]}"
        if kid not in set(sequence):
            continue
        nodes = build()
        kinds.append({"id": kid, "nodes": nodes, "edges": chain(nodes)})
    return kinds, compress(sequence)


HANDLERS = {
    "LlamaForCausalLM": handler_dense,
    "MistralForCausalLM": handler_dense,
    "Qwen2ForCausalLM": handler_dense,
    "Qwen3ForCausalLM": handler_dense,
    "MixtralForCausalLM": handler_moe,
    "Qwen3MoeForCausalLM": handler_moe,
    "Llama4ForConditionalGeneration": handler_moe,
    "GraniteMoeForCausalLM": handler_moe,
    "DeepseekV2ForCausalLM": handler_moe,
    # V3 is the same shape family as V2: MLA attention keyed by kv_lora_rank, a routed
    # expert layer, and a dense prologue named by first_k_dense_replace. Every difference
    # between the committed V2-Lite config and V3 is a VALUE -- depth, hidden size, expert
    # count, scoring function -- not a structure, so the handler prices it with no new node
    # kinds. Registered explicitly rather than by a prefix match, because a default is what
    # this registry exists to prevent.
    #
    # DeepseekV4ForCausalLM has its own handler rather than this one. It declares no
    # kv_lora_rank and states an inclusive head_dim, so the MLA branch here cannot
    # reconstruct its per-token width; and its sliding_window field is 128 where the
    # read is actually bounded by index_topk over a compressed stream, so this handler
    # would derive kind=swa window=128 and misprice every layer. See
    # handler_deepseek_v4, which reads the bound from compress_ratios and index_topk and
    # gives the indexer its own node.
    "DeepseekV3ForCausalLM": handler_moe,
    "DeepseekV4ForCausalLM": handler_deepseek_v4,
    "GlmMoeDsaForCausalLM": handler_moe,
    "InklingForConditionalGeneration": handler_moe,
    "GptOssForCausalLM": handler_moe,
    "MiniMaxM2ForCausalLM": handler_minimax_m2,
    "MiniMaxM3SparseForCausalLM": handler_minimax_m3,
    "MiniMaxM3SparseForConditionalGeneration": handler_minimax_m3,
    "NemotronHForCausalLM": handler_nemotron_h,
    "Qwen3_5MoeForConditionalGeneration": handler_qwen3_5_moe,
    "KimiK3ForConditionalGeneration": handler_kimi_k3,
    "Glm5NextForConditionalGeneration": handler_glm5_next,
    # Kimi-K2.5 is a vision-language model whose TEXT decoder vLLM instantiates as
    # DeepseekV2ForCausalLM over config.text_config
    # (model_executor/models/kimi_k25.py:371-376, architectures=["DeepseekV2ForCausalLM"]),
    # and whose text_config names DeepseekV3ForCausalLM itself. Either way the shape is the
    # DeepSeek family handler_moe already prices: MLA with kv_lora_rank 512, routed experts,
    # a dense prologue from first_k_dense_replace. The vision tower is not priced, which the
    # graph records as modality text_decoder_of_multimodal; the corpus's workloads are all
    # text (1024:1024, 1024:8192, 8192:1024), so the tower never runs in these measurements.
    "KimiK25ForConditionalGeneration": handler_moe,
}


# --- Driver ---------------------------------------------------------------------


def derive(config_path: Path, model: str) -> dict[str, Any]:
    """Derive one graph from one vendor config."""
    source = config_path.read_bytes()
    raw = json.loads(source)
    cfg = text_config(raw)

    arches = raw.get("architectures") or cfg.get("architectures") or []
    if not arches:
        raise DeriveError(f"{model}: config declares no architectures")
    arch = arches[0]
    handler = HANDLERS.get(arch)
    if handler is None:
        raise DeriveError(
            f"{model}: no handler for architecture {arch!r}; add one rather than "
            f"letting a default misprice the model"
        )

    kinds, stack = handler(cfg, raw, model)

    graph: dict[str, Any] = {
        "kind": "ModelGraph",
        "name": model,
        "derived_from": {
            "format": "hf_config_json",
            "path": "config.json",
            "sha256": hashlib.sha256(source).hexdigest(),
            "deriver_version": DERIVER_VERSION,
        },
        "global": {
            "hidden_size": int(require(cfg, "hidden_size", model)),
            "vocab_size": int(require(cfg, "vocab_size", model)),
            "tie_word_embeddings": bool(cfg.get("tie_word_embeddings", False)),
            "weight_dtype": weight_dtype(cfg, raw, model),
        },
        "layer_kinds": kinds,
        "stack": stack,
    }

    # A config that nests its text shapes beside another tower describes more than
    # this graph prices, and the graph says so rather than leaving a reader to find out
    # from a prediction that is low.
    if "text_config" in raw:
        graph["modality"] = "text_decoder_of_multimodal"

    hidden = graph["global"]["hidden_size"]
    graph["head"] = [
        norm("final_norm"),
        gemm("lm_head", graph["global"]["vocab_size"], hidden),
    ]

    spec = pick(cfg, "num_spec_tokens")
    # num_mtp_modules is NOT in the alias table: it is not a cross-family spelling of
    # "draft layers". vLLM reads it for n_predict only on the families whose speculative
    # method consumes it (config/speculative.py:958-984 for MiniMax-M3), and
    # MiniMax-M2.5 declares num_mtp_modules 3 while vLLM registers no minimax_m2_mtp
    # method at all -- its only draft path is a separate Eagle3 checkpoint. Reading the
    # field globally would invent a draft stack for every such config.
    if arch in MTP_MODULE_ARCHS:
        modules = cfg.get("num_mtp_modules")
        if modules is None:
            raise DeriveError(
                f"{model}: {arch} takes its draft length from num_mtp_modules and the "
                f"config declares none"
            )
        spec = modules
    if spec and int(spec) > 0:
        # A draft module is its own stack: for an MoE target the draft pass is a second
        # MoE, which a scalar draft length would hide. Its layer composition comes from
        # three sources, in order of how much the config actually says.
        defined = {k["id"] for k in kinds}
        mtp_vector = cfg.get("mtp_layers_block_type")
        if DRAFT_KIND_ID in defined:
            # A handler that priced the draft module itself, because the family gives it
            # a structure no target layer shares. DeepSeek-V4 is the case in hand: its
            # draft layer is uncompressed where every target layer is compressed, so
            # mirroring any of them would price the wrong read.
            pattern = [DRAFT_KIND_ID]
        elif mtp_vector:
            pattern = [n if n in defined else kinds[-1]["id"] for n in mtp_vector]
        else:
            # The fallback: mirror the target's last layer kind. Sound only where the
            # draft module really does repeat a target layer's structure, which is why a
            # handler whose family differs declares DRAFT_KIND_ID instead.
            pattern = [kinds[-1]["id"]]
        graph["speculator"] = {
            "method": speculative_method(arch, model),
            "num_spec_tokens": int(spec),
            "stack": {"pattern": pattern, "repeat": 1},
        }
    return graph


# The engine's speculative method name per architecture. It is a vLLM vocabulary
# rather than a config field, so it is recorded here per family rather than guessed
# from the model name.
SPEC_METHODS = {
    # glm_moe_dsa is rewritten to deepseek_mtp, NOT to glm4_moe_mtp. vLLM lists it
    # alongside deepseek_v3 and deepseek_v32 in one branch -- `if hf_config.model_type in
    # ("deepseek_v3", "deepseek_v32", "glm_moe_dsa"): hf_config.model_type =
    # "deepseek_mtp"` (config/speculative.py:678-685) -- and all four committed GLM DSA
    # configs declare model_type glm_moe_dsa. glm4_moe_mtp is set on a different branch
    # (:780) for a different family, so naming it here priced the wrong draft method.
    # The comment below already stated the correct mapping while this entry contradicted
    # it, which is how the error survived.
    "GlmMoeDsaForCausalLM": "deepseek_mtp",
    "KimiK3ForConditionalGeneration": "kimi_k3_mtp",
    "NemotronHForCausalLM": "nemotron_h_mtp",
    "InklingForConditionalGeneration": "inkling_mtp",
    "DeepseekV2ForCausalLM": "deepseek_mtp",
    # Both verified in vLLM's own resolution rather than inferred from the family name:
    # config/speculative.py maps model_type deepseek_v3 to deepseek_mtp alongside
    # deepseek_v32 and glm_moe_dsa, and maps deepseek_v4 to the same method on its own
    # branch (architectures DeepSeekMTPModel and DeepSeekV4MTPModel respectively).
    "DeepseekV3ForCausalLM": "deepseek_mtp",
    # vLLM rewrites model_type deepseek_v4 to deepseek_mtp and reads n_predict from
    # num_nextn_predict_layers, NOT from num_mtp_modules the way MiniMax-M3 does
    # (config/speculative.py:664-669, architectures DeepSeekV4MTPModel). The alias table
    # already resolves that field, so this architecture needs no entry in
    # MTP_MODULE_ARCHS.
    "DeepseekV4ForCausalLM": "deepseek_mtp",
    "Qwen3_5MoeForConditionalGeneration": "qwen3_5_mtp",
    # vLLM rewrites model_type glm5_next to glm5_next_mtp and reads n_predict from
    # get_text_config().num_nextn_predict_layers, architectures ["Glm5NextMTPModel"]
    # (config/speculative.py:1060-1065 on vllm-project/vllm main, with
    # "glm5_next_mtp" in its valid-method list at line 66). Read from upstream rather
    # than inferred from the glm4_moe_mtp entry above: the two are distinct methods and
    # this family is not the GLM-4 one. The alias table already resolves
    # num_nextn_predict_layers, so this needs no MTP_MODULE_ARCHS entry.
    "Glm5NextForConditionalGeneration": "glm5_next_mtp",
    # vLLM resolves both the VL wrapper and the text decoder to one method, reading
    # n_predict from num_mtp_modules rather than num_nextn_predict_layers
    # (config/speculative.py:958-974, and "minimax_m3_mtp" in its valid-method list).
    "MiniMaxM3SparseForConditionalGeneration": "minimax_m3_mtp",
    "MiniMaxM3SparseForCausalLM": "minimax_m3_mtp",
}


# Architectures whose speculative method takes its draft length from num_mtp_modules
# rather than from num_nextn_predict_layers. Named rather than pattern-matched: the two
# fields disagree on MiniMax-M3 (7 against 1), so which one is read changes the graph.
MTP_MODULE_ARCHS = frozenset({
    "MiniMaxM3SparseForConditionalGeneration",
    "MiniMaxM3SparseForCausalLM",
})


def speculative_method(arch: str, model: str) -> str:
    method = SPEC_METHODS.get(arch)
    if method is None:
        raise DeriveError(
            f"{model}: {arch} declares draft layers but no speculative method is "
            f"recorded for it"
        )
    return method


def dump(graph: dict[str, Any]) -> str:
    """Render a graph deterministically, so a re-derivation diffs cleanly."""
    return yaml.safe_dump(graph, sort_keys=False, default_flow_style=False, width=100)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default=None)
    ap.add_argument("--check", action="store_true",
                    help="re-derive and report differences without writing")
    ap.add_argument("--model", default=None, help="derive one model only")
    args = ap.parse_args()

    root = Path(args.root) if args.root else Path(__file__).resolve().parent.parent
    models_dir = root / "models"
    if not models_dir.is_dir():
        print(f"{models_dir}: not a directory", file=sys.stderr)
        return 2

    written, unchanged, stale, failed = [], [], [], []
    for entry in sorted(p for p in models_dir.iterdir() if p.is_dir()):
        if args.model and entry.name != args.model:
            continue
        cfg = entry / "config.json"
        if not cfg.is_file():
            failed.append((entry.name, "no config.json"))
            continue
        try:
            graph = derive(cfg, entry.name)
        except DeriveError as exc:
            failed.append((entry.name, str(exc)))
            continue
        except (KeyError, ValueError, TypeError) as exc:
            failed.append((entry.name, f"{type(exc).__name__}: {exc}"))
            continue

        text = dump(graph)
        out = entry / "graph.yaml"
        if args.check:
            if not out.is_file():
                stale.append((entry.name, "graph.yaml is missing"))
            elif out.read_text() != text:
                stale.append((entry.name, "graph.yaml differs from a re-derivation"))
            else:
                unchanged.append(entry.name)
            continue
        if out.is_file() and out.read_text() == text:
            unchanged.append(entry.name)
        else:
            out.write_text(text)
            written.append(entry.name)

    for name, why in failed:
        print(f"{name}: {why}", file=sys.stderr)
    for name, why in stale:
        print(f"{name}: {why}", file=sys.stderr)

    if args.check:
        print(f"checked {len(unchanged) + len(stale)} model(s): "
              f"{len(unchanged)} current, {len(stale)} stale, {len(failed)} failed")
    else:
        print(f"derived {len(written) + len(unchanged)} graph(s): "
              f"{len(written)} written, {len(unchanged)} unchanged, {len(failed)} failed")
    return 1 if (failed or stale) else 0


if __name__ == "__main__":
    sys.exit(main())
