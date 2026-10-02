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
    "moe_latent_size": ("moe_latent_size",),
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
    that only one family declares — a GDN's head geometry, say — has no cross-dialect
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


def weight_dtype(cfg: dict[str, Any], raw: dict[str, Any], model: str) -> str:
    """Return the graph's dtype name for the model's parameters.

    A quantization block overrides the base dtype: a checkpoint declaring bfloat16
    weights with an fp8 quantization config stores fp8, and pricing it at two bytes
    per parameter would overstate both occupancy and decode-time traffic."""
    quant = raw.get("quantization_config") or cfg.get("quantization_config") or {}
    if isinstance(quant, dict) and quant:
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
        # Only these two are accepted by name. A method whose name does not fix a width —
        # "compressed-tensors" is the case in hand — still needs its groups, which is why
        # this sits after that branch rather than before it.
        if fmt in ("mxfp4", "nvfp4"):
            return fmt
        if fmt:
            raise DeriveError(
                f"{model}: quant_method {fmt!r} with no group widths; the deriver "
                f"cannot infer the stored width"
            )
    declared = pick(cfg, "dtype") or pick(raw, "dtype")
    if declared is None:
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
              window: int | None = None) -> dict[str, Any]:
    """Build the attention node, choosing the kind from the config's own evidence."""
    nq = int(require(cfg, "num_q_heads", model))
    nkv = int(require(cfg, "num_kv_heads", model))
    lora = pick(cfg, "kv_lora_rank")

    if kind is None:
        if lora:
            kind = "sparse_mla" if pick(cfg, "index_topk") else "mla"
        elif window:
            kind = "swa"
        else:
            kind = "gqa"

    node: dict[str, Any] = {"op": "Attention", "kind": kind, "n_q": nq}

    if kind in ("mla", "sparse_mla"):
        # A latent cache holds one vector per token, so the engine pins the KV head
        # count to one whatever the config's num_key_value_heads says. The per-token
        # width is the latent rank plus the RoPE dimension, which is what the cache
        # stores and therefore what a cost model reads.
        nope = int(pick(cfg, "qk_nope_head_dim") or 0)
        rope = int(pick(cfg, "qk_rope_head_dim") or 0)
        if not rope:
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


# Emit conditions are a closed set naming WHEN a conditional node is part of the graph.
# They are enum members rather than predicate strings: every condition the catalog's models
# need is decided by the collective's role, so a resolver evaluates them with a switch and
# needs no expression parser. A graph cannot express a condition no resolver knows.
EMIT_TENSOR_PARALLEL = "tensor_parallel"
EMIT_EXPERT_PARALLEL = "expert_parallel"
EMIT_TENSOR_PARALLEL_UNLESS_SP_MOE = "tensor_parallel_unless_sp_moe"


def collective(op: str, role: str, emit: str) -> dict[str, Any]:
    return {"op": op, "role": role, "emit": emit}


def compress(sequence: list[str]) -> dict[str, Any]:
    """Express a layer sequence as the smallest prologue, repeated pattern and epilogue.

    A literal 108-entry list and a pattern repeated four times describe the same stack, so
    the choice is about storage rather than meaning. Compression is worth doing for two
    reasons: a reader can see a hybrid's period at a glance where a literal list hides it,
    and a diff after a config change stays legible.

    It is NOT what makes step-time computation fast. A kernel collapses the stack to a
    per-kind layer count once at construction, so its per-step work is proportional to the
    number of distinct layer kinds — three for the widest model here — whichever way the
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
            if head > period or len(tail) > period:
                continue
            stored = head + period + len(tail)
            if best is None or stored < best[0]:
                best = (stored, head, body, repeats, tail)
    if best is None or best[0] >= n:
        # No split stores less than the literal sequence, so state it literally.
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


def attention_block(cfg: dict[str, Any], model: str, hidden: int, *,
                    kind: str | None = None,
                    window: int | None = None) -> list[dict[str, Any]]:
    """The nodes common to every attention layer: norm, QKV, attention, output, reduce."""
    nq = int(require(cfg, "num_q_heads", model))
    nkv = int(require(cfg, "num_kv_heads", model))
    attn = attention(cfg, model, kind=kind, window=window)
    if attn["kind"] in ("mla", "sparse_mla"):
        # The latent path projects to a compressed KV plus a per-head query, and the
        # projection widths differ enough between published MLA variants that a single
        # formula would be a guess. The QKV node carries the query width, which is the
        # part every variant shares.
        qkv_out = nq * (int(pick(cfg, "qk_nope_head_dim") or 0) +
                        int(pick(cfg, "qk_rope_head_dim") or 0))
        o_in = nq * int(pick(cfg, "v_head_dim") or pick(cfg, "qk_nope_head_dim") or 0)
        if qkv_out == 0 or o_in == 0:
            raise DeriveError(f"{model}: latent attention with no head widths to size "
                              f"its projections")
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
    nodes = attention_block(cfg, model, hidden, window=window)
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
    prologue: the sequence has no repeating unit that includes them."""
    hidden = int(require(cfg, "hidden_size", model))
    layers = int(require(cfg, "num_layers", model))

    window = pick(cfg, "sliding_window")
    if window and cfg.get("use_sliding_window") is False:
        window = None

    def sparse_nodes(win=None):
        n = attention_block(cfg, model, hidden, window=win)
        n += [norm("post_attn_norm"), moe(cfg, model, hidden),
              collective("All2All", "moe_dispatch_combine", EMIT_EXPERT_PARALLEL),
              collective("AllReduce", "mlp_out", EMIT_TENSOR_PARALLEL_UNLESS_SP_MOE)]
        return n

    def dense_nodes(win=None):
        n = attention_block(cfg, model, hidden, window=win)
        n += [norm("post_attn_norm")] + dense_mlp(cfg, model, hidden)
        n += [collective("AllReduce", "mlp_out", EMIT_TENSOR_PARALLEL_UNLESS_SP_MOE)]
        return n

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
        sparse, dense = sparse_nodes(window), dense_nodes(window)
        kinds = [
            {"id": "attn_dense", "nodes": dense, "edges": chain(dense)},
            {"id": "attn_moe", "nodes": sparse, "edges": chain(sparse)},
        ]
        sequence = ["attn_moe" if t == "sparse" else "attn_dense" for t in mlp_types]
        # Compress a trailing uniform run, which is the common shape: a short dense
        # prologue then sparse throughout.
        tail = sequence[-1]
        head_len = len(sequence)
        while head_len > 0 and sequence[head_len - 1] == tail:
            head_len -= 1
        return kinds, compress(sequence)

    # A first_k_dense_replace count does the same job as a type vector.
    first_dense = int(cfg.get("first_k_dense_replace") or 0)
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
    exactly — but the vector is checked rather than ignored, because a later variant that
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
    # transformer layers num_hidden_layers counts. vLLM indexes it by layer, so the
    # extra entry is only ever read for the draft module.
    if len(ratios) == layers + 1:
        ratios = ratios[:layers]
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

    def layer_for(ratio):
        # The read is a top-k over a stream compressed by this ratio, so a larger ratio
        # bounds it tighter. index_topk caps both.
        window = max(1, min(topk, topk // ratio))
        nodes = attention_block(cfg, model, hidden, kind="swa", window=window)
        at = next(i for i, n in enumerate(nodes) if n.get("op") == "Attention")
        nodes = nodes[:at] + [
            gemm("index_qk_proj", index_heads * index_dim + index_dim, hidden),
            dict(indexer),
        ] + nodes[at:]
        nodes += [
            norm("post_attn_norm"),
            moe(cfg, model, hidden),
            collective("All2All", "moe_dispatch_combine", EMIT_EXPERT_PARALLEL),
            collective("AllReduce", "mlp_out", EMIT_TENSOR_PARALLEL_UNLESS_SP_MOE),
        ]
        return nodes

    kinds = []
    for ratio in sorted(set(ratios)):
        nodes = layer_for(ratio)
        kinds.append({"id": f"csa{ratio}_moe", "nodes": nodes, "edges": chain(nodes)})
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

    dense_attn = attention_block(cfg, model, hidden)
    dense_ffn = int(require_key(cfg, "dense_intermediate_size", model))
    dense_nodes = dense_attn + [
        norm("post_attn_norm"),
        gemm("mlp_gate_up", 2 * dense_ffn, hidden),
        gemm("mlp_down", hidden, dense_ffn),
        collective("AllReduce", "mlp_out", EMIT_TENSOR_PARALLEL_UNLESS_SP_MOE),
    ]

    sparse_attn = attention_block(cfg, model, hidden, kind="swa", window=bound)
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
        moe(cfg, model, hidden),
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
        n += [norm("post_attn_norm"), moe(cfg, model, hidden),
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
    full = with_moe(attention_block(cfg, model, hidden))
    kinds = [
        {"id": "gdn_moe", "nodes": gdn, "edges": chain(gdn)},
        {"id": "attn_moe", "nodes": full, "edges": chain(full)},
    ]
    sequence = ["attn_moe" if t == "full_attention" else "gdn_moe" for t in types]
    return kinds, compress(sequence)


def handler_nemotron_h(cfg, raw, model):
    """NemotronH: a declared per-layer vector over state-space, MoE and attention
    layers, with no repeating unit and — on one variant — no stated layer count.

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
    moe_nodes = [
        norm("input_norm"),
        moe(cfg, model, hidden),
        collective("All2All", "moe_dispatch_combine", EMIT_EXPERT_PARALLEL),
        collective("AllReduce", "mlp_out", EMIT_TENSOR_PARALLEL_UNLESS_SP_MOE),
    ]
    attn = attention_block(cfg, model, hidden)
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
    """Kimi-K3: linear attention on most layers, latent attention on a declared list.

    The full-attention layer indices are given explicitly rather than by a period, so
    the sequence is built from that list."""
    hidden = int(require(cfg, "hidden_size", model))
    layers = int(require(cfg, "num_layers", model))
    linear = cfg.get("linear_attn_config") or {}
    full = linear.get("full_attn_layers")
    if not full:
        raise DeriveError(f"{model}: no linear_attn_config.full_attn_layers")

    def with_moe(mixer_nodes):
        n = list(mixer_nodes)
        n += [norm("post_attn_norm"), moe(cfg, model, hidden),
              collective("All2All", "moe_dispatch_combine", EMIT_EXPERT_PARALLEL),
              collective("AllReduce", "mlp_out", EMIT_TENSOR_PARALLEL_UNLESS_SP_MOE)]
        return n

    kda_mixer = [
        norm("input_norm"),
        {
            "op": "RecurrentUpdate",
            "recurrent_kind": "kda",
            "n_heads": int(require(cfg, "num_q_heads", model)),
            "state_size": int(linear.get("head_dim") or head_dim(cfg, model)),
            "state_dtype": "fp32",
        },
        collective("AllReduce", "mixer_out", EMIT_TENSOR_PARALLEL),
    ]
    kda = with_moe(kda_mixer)
    mla = with_moe(attention_block(cfg, model, hidden))
    kinds = [
        {"id": "kda_moe", "nodes": kda, "edges": chain(kda)},
        {"id": "mla_moe", "nodes": mla, "edges": chain(mla)},
    ]
    full_set = set(int(i) for i in full)
    # The published list is 1-based against a 1-indexed layer numbering; a 0-based
    # reading would shift every attention layer by one. Detected rather than assumed:
    # a 0-based list would contain 0.
    offset = 1 if 0 not in full_set else 0
    sequence = ["mla_moe" if (i + offset) in full_set else "kda_moe"
                for i in range(layers)]
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
        # MoE, which a scalar draft length would hide. Its layer composition is declared
        # where the config says so, and otherwise mirrors the target's last layer kind.
        mtp_vector = cfg.get("mtp_layers_block_type")
        if mtp_vector:
            pattern = [n if n in {k["id"] for k in kinds} else kinds[-1]["id"]
                       for n in mtp_vector]
        else:
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
    "GlmMoeDsaForCausalLM": "glm4_moe_mtp",
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
