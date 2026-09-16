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

BLIS locates the catalog by `--catalog` or the `BLIS_CATALOG` environment
variable. There is no default and no remote fetch.

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
  revision: <commit-sha>        # what was downloaded
  revision_verified: true       # config.json is byte-identical to this revision
  retrieved: 2026-09-16
```

`revision_verified: false` marks an entry whose committed `config.json` could not
be confirmed against the named upstream revision (e.g. a gated repo, or a fixture
that predates this catalog). The repo still records where the config belongs.

## Conventions

- **Vendor configs are verbatim.** `config.json` is copied byte-for-byte from the
  vendor and never edited, so it stays diffable against upstream
  (`.gitattributes` disables line-ending normalisation for it).
- **The payload is YAML and JSON.** `.gitignore` deliberately ignores *nothing* by
  extension — one forgotten whitelist and a model would silently drop out of the
  catalog, the failure NS-6 exists to prevent.
- **Validation is the simulator's.** There is no separate validate command; BLIS
  validates whatever it reads and fails naming the file and the problem. CI runs a
  load over every catalog entry through that same code path.
