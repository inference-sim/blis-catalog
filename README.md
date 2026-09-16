# blis-catalog

The authoritative catalog for [BLIS](https://github.com/inference-sim/inference-sim):
models, hardware, workload types, and storage devices. Everything here is a
**declared fact** that a person wrote down — nothing is learned, fitted, or
inferred. Learned numbers (alpha/beta coefficients, LoRA cost defaults) live in
`blis-registry`; deployment choices (GPU type, tensor-parallel degree) are stated
on the command line, not here.

This realises release **R1** of the North Star architecture: *a model runs if and
only if it is in the catalog* (invariant NS-6). BLIS reads the catalog and never
writes it.

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

> **Status:** the catalog data exists; the simulator side that consumes it does
> not yet. The `--catalog` flag, the `BLIS_CATALOG` environment variable, and the
> CI load-check described below are **planned** (the R1 S-tasks in
> `inference-sim`), not yet implemented. Today BLIS still reads its own
> `model_configs/` via `--model-config-folder`. The interface below is the target.

Once the loader lands, BLIS will locate the catalog by `--catalog` or the
`BLIS_CATALOG` environment variable, with no default and no remote fetch:

```sh
export BLIS_CATALOG=~/path/to/blis-catalog     # once, per shell
blis run --model glm-5.2 --hardware H100 --tp 8
```

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
the vendor repo named in each `model.yaml`, not the prior fixture. Three configs
(`glm-5.2-fp8`, `llama-2-7b-hf`, `llama-3.1-8b-instruct`) are therefore **fuller**
than the hand-trimmed simulator fixtures they replace; they are the current
upstream configs and are diffable against them. The S-task that adds the loader
should confirm BLIS parses equivalent values for the keys it reads (R1 acceptance
test #3, byte-identical stdout).

## Conventions

- **Vendor configs are verbatim.** `config.json` is copied byte-for-byte from the
  vendor and never edited, so it stays diffable against upstream
  (`.gitattributes` disables line-ending normalisation for it).
- **The payload is YAML and JSON.** `.gitignore` deliberately ignores *nothing* by
  extension, and every directory pattern is anchored to the repo root — one
  unanchored or forgotten pattern and a model would silently drop out of the
  catalog, the failure NS-6 exists to prevent.
- **Validation is the simulator's (planned).** There is no separate validate
  command by design; once the loader lands, BLIS validates whatever it reads and
  fails naming the file and the problem, and CI runs a load over every catalog
  entry through that same code path. That CI gate does not exist yet.
