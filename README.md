# blis-catalog

The authoritative catalog for [BLIS](https://github.com/inference-sim/inference-sim):
models, hardware, networks, clusters, workload types, and storage devices.
Everything here is a **declared fact** that a person wrote down — a datasheet or
vendor spec — nothing is learned, fitted, measured, or inferred. Learned and
measured numbers (alpha/beta coefficients, LoRA cost defaults, per-GPU MFU
prefill/decode estimates, measured fabric-overhead corrections, PD-transfer
estimates) live in `blis-registry`; deployment choices (which cluster / GPU
type, tensor-parallel degree) are stated on the command line, not here.

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
├── hardware/                   # WHAT A CHIP CAN DO — one file per GPU (vendor specs only)
│   ├── h100.yaml  h200.yaml  a100-sxm.yaml  a100-80.yaml  l40s.yaml
├── networks/                   # WHAT AN INTER-NODE FABRIC CAN DO — one file per reusable fabric class
│   ├── ib-400g.yaml  roce-200g.yaml  ethernet-100gbe.yaml
├── clusters/                   # WHICH CHIP RUNS ON WHICH FABRIC — one file per deployment (1 hardware + 1 network)
│   ├── pok-h100.yaml  pok-h200.yaml  vllm-d-a100.yaml  platform-eval-l40s.yaml  h100-roce-200g.yaml
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

## Networks and clusters

A **Network** is a *reusable fabric class* — `ib-400g`, `roce-200g`,
`ethernet-100gbe` — not one specific pool. A **Cluster** binds one chip
(`hardware/`) to one fabric (`networks/`): `vllm-d-a100` is `a100-80` on
`roce-200g`. Per [inference-sim#1662](https://github.com/inference-sim/inference-sim/issues/1662)
a fabric is a property of the *cluster/pool*, not of the GPU die, so the **same
chip can appear in two clusters on two fabrics** — `pok-h100` (`h100` on
`ib-400g`) and `h100-roce-200g` (`h100` on `roce-200g`) — and pay different
cross-node cost. A cluster is `provenance: deployment` (a real pool) or
`hypothetical` (an illustrative composition, e.g. `h100-roce-200g`).

**Provenance, and the catalog/registry line.** Every fabric bandwidth here is
`provenance: vendor_spec` — a **nominal** datasheet figure (a NIC line rate ÷ 8,
or NVLink bidirectional ÷ 2), **not a measurement**. Real payload throughput
runs below nominal; the *measured overhead/correction factor* (effective ÷
nominal) is a per-cluster number and lives in `blis-registry`, not here. That is
what keeps "nothing measured or fitted" true of this repo.

- **PD KV-transfer** rides the cluster's fabric: its transfer bandwidth **is**
  that fabric's nominal `InterNodeBwGBps` (there is no separate figure), so two
  clusters on different fabrics disaggregate at different cost. The fabric's
  `pd_transfer_base_latency_ms` is `0` (the catalog declares no inherent base
  latency); the `0.05 ms` `--pd-transfer-base-latency` default is a modeling
  placeholder that belongs in `blis-registry`. The `--pd-transfer-*` CLI flags
  remain overrides, so a run naming no cluster is byte-identical.

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
