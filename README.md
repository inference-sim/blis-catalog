# blis-catalog

The authoritative catalog for [BLIS](https://github.com/inference-sim/inference-sim):
models, hardware, networks, workload types, and storage devices.
Everything here is a **declared fact** that a person wrote down — a datasheet or
vendor spec — nothing is learned, fitted, measured, or inferred. Learned and
measured numbers (alpha/beta coefficients, LoRA cost defaults, per-GPU MFU
prefill/decode estimates, measured fabric-overhead corrections, PD-transfer
estimates) live in `blis-registry`; deployment choices (GPU type,
tensor-parallel degree) are stated on the command line, not here.

**Documentation:** <https://inference-sim.github.io/blis-catalog/>: how to use the
catalog, how to add to it, why it is shaped the way it is, and a reference generated from
the data in each release.

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
├── workloads/                  # WHAT TRAFFIC LOOKS LIKE
│   ├── chatbot.yaml  summarization.yaml  contentgen.yaml  multidoc.yaml
└── devices/                    # WHAT A STORAGE TIER CAN DO
    └── storage.yaml            # nvme_gen4, nvme_gen3, sata_ssd, cpu_dram, s3
```

A few things are deliberately **absent**: GPU type and tensor-parallel degree
(deployment choices, stated on the CLI), learned coefficients (`blis-registry`),
Cluster objects that bind a chip to a fabric (deferred — this change focuses on
the fabric classes themselves; see Networks below), and scenario/candidate
objects (introduced by a later release).

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

`graph.yaml` sits beside them: the model expressed as the cost primitives a forward
pass launches, which is what a cost model prices. It is **generated, never
hand-written** — `scripts/derive_graph.py` reads `config.json` and writes it,
recording the SHA-256 of the config it derived from. Re-derive after adding or
changing a config:

```bash
python3 scripts/derive_graph.py --model <name>   # one model
python3 scripts/derive_graph.py                  # all
python3 scripts/derive_graph.py --check          # verify, write nothing (CI runs this)
```

`--check` re-derives every graph and fails on any that differs from what is
committed, which catches both a hand-edited graph and a config changed without
re-deriving. The `ModelGraph` type itself is defined in
[`blis-schemas`](https://github.com/inference-sim/blis-schemas) (`spec/model`); for how
the repositories fit together, see
[The BLIS Repositories](https://github.com/inference-sim/inference-sim/blob/main/docs/concepts/blis-repositories.md).

## Networks

A **Network** is a *reusable fabric class* — `ib-400g`, `roce-200g`,
`ethernet-100gbe` — describing an inter-node fabric, not one specific pool. Per
[inference-sim#1662](https://github.com/inference-sim/inference-sim/issues/1662)
a fabric is a property of the *cluster/pool*, not of the GPU die: two H100 pools
can legitimately differ (InfiniBand vs RoCE vs a 100 GbE uplink), so a fabric is
defined once here and paired with a chip per deployment. (The **Cluster** object
that would bind one chip to one fabric is deferred; this change focuses on
getting the fabric classes themselves right.)

**Provenance, and the catalog/registry line.** Both `hardware/` chips and
`networks/` fabrics carry a structured `Provenance` field: every quantity here is
a **nominal** datasheet figure (a NIC line rate ÷ 8, or NVLink bidirectional ÷ 2),
**not a measurement**. Real payload throughput runs below nominal; the *measured
overhead/correction factor* (effective ÷ nominal) is a per-deployment number and
lives in `blis-registry`, not here. That is what keeps "nothing measured or
fitted" true of this repo.

The set of `Provenance` values is **open** — it is meant to carry any *declared*
(non-measured, non-fitted) provenance. Which values a file may use is a schema
rule, defined and enforced by the schema owner (see [Validation](#validation)),
not here; the values the catalog uses today mean:

- **`vendor_spec`** — a nominal figure taken from a vendor datasheet, including
  the datasheet's own unit normalisations (a bidirectional figure read per-GPU
  per-direction, a dense/sparse selection). Every entry in the catalog today is
  `vendor_spec`.
- **`derived`** — reserved for a figure *computed* by the catalog from spec(s) via
  an arithmetic step the datasheet does not itself state — still declared, still
  non-measured. No entry uses it yet; the open set lets it be adopted later
  without a schema change. (Measured or fitted numbers never join this set — they
  live in `blis-registry`.)

`Provenance` is recorded **per file**, not per field. A finer per-field tag would
let a file that mixes a stated figure (`TFlopsPeak`) with a computed one distinguish
them, but per-field tagging is a strictly *additive* refinement that changes no
value and would alter the single-key contract the strict hardware loader reads
([inference-sim#1831](https://github.com/inference-sim/inference-sim/issues/1831));
because every current figure is `vendor_spec`, a file-level tag loses nothing
today, and the open enum is what lets per-field land later without a value change.

- **PD KV-transfer** rides the fabric: its transfer bandwidth **is** that
  fabric's nominal `InterNodeBwGBps` (there is no separate figure), so different
  fabrics disaggregate at different cost. The fabric carries **no** PD
  base-latency field — the catalog states no inherent fabric base latency, and
  the nominal figure would only ever be `0`; the `0.05 ms`
  `--pd-transfer-base-latency` default is a modeling placeholder that belongs in
  `blis-registry` ([blis-registry#10](https://github.com/inference-sim/blis-registry/issues/10)).
  The `--pd-transfer-*` CLI flags remain overrides, so a run that resolves no
  fabric is byte-identical.

## Conventions

- **Vendor configs are verbatim.** `config.json` is copied byte-for-byte from the
  vendor and never edited, so it stays diffable against upstream
  (`.gitattributes` disables line-ending normalisation for it).
- **The payload is YAML and JSON.** `.gitignore` deliberately ignores *nothing* by
  extension, and every directory pattern is anchored to the repo root — one
  unanchored or forgotten pattern and a model would silently drop out of the
  catalog, the failure NS-6 exists to prevent.
- **Validation is schema-owned.** Every committed entry is validated in CI by
  [blis-schemas'](https://github.com/inference-sim/blis-schemas) `validate-catalog`
  binary, which owns the schema and all its rules (see [Validation](#validation)).
  A malformed entry fails *here*, in the repo that owns the data; the simulator
  additionally validates the *semantics* of whatever it loads at run time.

## Validation

The schema and every structural rule a catalog entry must satisfy are defined and
documented in [blis-schemas](https://github.com/inference-sim/blis-schemas). This
repo re-implements none of them: there is no catalog-local *schema* validation
logic, in code, tests, or prose. (The one catalog-local check that remains,
`derive_graph.py --check` and its test suite, validates graph *derivation* — that
each committed graph re-derives from its vendor config — not the schema.)

CI validates every committed entry by running that owner's binary against the
checkout, pinned by version:

```sh
go run github.com/inference-sim/blis-schemas/cmd/validate-catalog@v0.2.0 .
```

It walks all five namespaces (`models/`, `hardware/`, `networks/`, `workloads/`,
`devices/`), prints a per-entry summary, and fails naming the file when an entry
is malformed. To learn *what* each namespace requires, or to add a rule, go to
blis-schemas — not here. The simulator additionally validates the *semantics* of
whatever it loads at run time ([inference-sim#1750](https://github.com/inference-sim/inference-sim/issues/1750)),
the `blis run` code path; structural facts live in the schema that owns them.
