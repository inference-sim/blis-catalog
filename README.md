# blis-catalog

The authoritative catalog for [BLIS](https://github.com/inference-sim/inference-sim):
models, hardware, workload types, and storage devices. Everything here is a
**declared fact** that a person wrote down — nothing is learned, fitted, or
inferred. Learned numbers (alpha/beta coefficients, LoRA cost defaults) live in
`blis-registry`; deployment choices (GPU type, tensor-parallel degree) are stated
on the command line, not here.

This is the **data half** of the North Star architecture, toward the goal *a
model runs if and only if it is in the catalog* (invariant NS-6). This repository
provides the catalog contents; `inference-sim` reads it via `--catalog` /
`BLIS_CATALOG`, resolving every model against the catalog at run time with no
remote fetch.

## Layout

```
blis-catalog/
├── models/                     # WHAT A MODEL IS
│   └── <name>/
│       ├── config.json         # the vendor's file, committed verbatim, never edited
│       └── model.yaml          # identity + provenance (name, source repo/revision)
├── hardware/                   # WHAT A CHIP CAN DO — one file per GPU
│   ├── h100.yaml  h200.yaml  a100-sxm.yaml  a100-80.yaml  l40s.yaml
├── workloads/                  # WHAT TRAFFIC LOOKS LIKE
│   ├── chatbot.yaml  summarization.yaml  contentgen.yaml  multidoc.yaml
└── devices/                    # WHAT A STORAGE TIER CAN DO
    └── storage.yaml            # nvme_gen4, nvme_gen3, sata_ssd, cpu_dram, s3
```

Three things are deliberately **absent**: GPU type and tensor-parallel degree
(deployment choices, stated on the CLI), learned coefficients (`blis-registry`),
and scenario/candidate objects (introduced by a later release).

## Usage

BLIS locates the catalog by `--catalog` or the `BLIS_CATALOG` environment
variable, with no default and no remote fetch (the flag wins when both are set):

```sh
export BLIS_CATALOG=~/path/to/blis-catalog     # point BLIS at this clone
```

The deployment (GPU type, tensor-parallel degree, and the rest) is stated on the
command line, not here. For how to run a simulation, see the examples in
[inference-sim](https://github.com/inference-sim/inference-sim).

To experiment with a hypothetical model shape, clone the catalog, edit a
`config.json`, and point `BLIS_CATALOG` at the clone — no fork of the simulator,
no pull request. Keep such scratch clones **outside** any repository so
`git status` never reports clean while a run reads uncommitted data.

## A model entry

`model.yaml` says only *which model this is* and *where its config.json came
from*, so a result can be audited and the fetch reproduced by hand:

```yaml
name: qwen3-14b
source:
  provider: huggingface
  repo: Qwen/Qwen3-14B          # case-sensitive, as on the Hub
  revision: <commit-sha>        # the exact revision the config was fetched from
  retrieved: 2026-09-16
```

Each `config.json` was fetched **fresh and verbatim** from its `source.repo` at
the recorded `revision` (gated repos required an authenticated token). The
committed bytes are byte-identical to that revision, so a reviewer with access can
reproduce the fetch and diff.

## Provenance & history

The models were **re-fetched from HuggingFace** rather than copied from the
simulator's older `model_configs/` fixtures, so per-file git history from
`inference-sim` is intentionally not carried over — the authoritative source is
the vendor repo + revision named in each `model.yaml`, not the prior fixture.

**Three C1 fixtures** (`glm-5.2-fp8`, `llama-2-7b-hf`, `llama-3.1-8b-instruct`)
are **fuller** than the hand-trimmed fixtures they replace, but carry
byte-identical values for every field BLIS reads. This was verified through the
real parser (`latency.GetModelConfig` produces an identical `sim.ModelConfig` for
each catalog config vs its fixture); the only read key that differs, `llama-3.1`'s
`rope_scaling`, is type `llama3`, which `applyRopeScaling` treats as a no-op
(already covered permanently by `cmd/root_test.go:TestApplyRopeScaling`). A
catalog-vs-fixture equivalence check is only meaningful until S6 removes the
`model_configs/` fixtures; the durable post-S6 guard is a golden `sim.ModelConfig`
keyed directly off the catalog config, added by the loader S-task.

`kimi-k3` is a **separate case, not a C1 fixture** — it was absent at the
`891facee` design baseline, so it is not one of the 13 pre-existing fixtures the
C1 byte-move rule covers. Its sole provenance is the recorded upstream revision
(`moonshotai/Kimi-K3` @ `f831ab66…`), against which the committed config is
byte-identical and independently auditable. It carries the real-model values
(`kv_lora_rank: 512`, `moe_intermediate_size: 3072`, 24 full-attention layers);
any calibration data fitted against an older stale `kv_lora_rank: 149` copy should
be refit against these.

## Conventions

- **Vendor configs are verbatim.** `config.json` is copied byte-for-byte from the
  vendor and never edited, so it stays diffable against upstream
  (`.gitattributes` disables line-ending normalisation for it).
- **The payload is YAML and JSON.** `.gitignore` deliberately ignores *nothing* by
  extension, and every directory pattern is anchored to the repo root — one
  unanchored or forgotten pattern and a model would silently drop out of the
  catalog, the failure NS-6 exists to prevent.
- **Validation is the simulator's.** There is no separate validate command by
  design: BLIS validates whatever it reads and fails naming the file and the
  problem.
