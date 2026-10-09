# Declared, learned, chosen

Every number behind a BLIS estimate is one of three kinds.

Declared
:   A figure a person read off a source and wrote down: a model's hidden size from its `config.json`, a chip's peak FP8 rate from its datasheet. It is either correct or a transcription error, and once correct it does not change. **Declared numbers live in the catalog.**

Learned
:   A number fitted to measurements, or assumed, for a cost model: the fraction of nominal bandwidth a fabric achieves, the efficiency of a matrix multiply at a given shape. New measurements can revise it, and it applies only within a stated scope, such as a set of chips or parallelism degrees. **Learned numbers live in [blis-registry](https://github.com/inference-sim/blis-registry)** as *coefficients*, each recording how it was obtained and the scope it applies to.

Chosen
:   A setting picked for one run: which chip, how many GPUs, the tensor-parallel width, which fabric connects the nodes. **Chosen settings live in the run's scenario file**, which the user supplies; no BLIS repository owns scenarios.

The test for the catalog: *would a measurement revise this number?* If it would, the number belongs in the registry.

## One quantity, three kinds of number

A single physical quantity often has a number of each kind. Take NVLink bandwidth on an H100:

| Kind | Number | Where it lives |
| :--- | :--- | :--- |
| declared | 450 GB/s per GPU per direction: the datasheet's 900 GB/s bidirectional figure, halved | `hardware/h100.yaml`, `IntraNodeBwGBps` |
| learned | the fraction of that a collective achieves on a real cluster | a coefficient in blis-registry |
| chosen | eight GPUs in one node, tensor-parallel width 8 | the run's scenario |

## Nominal figures

The catalog's figures are *nominal*: what the source states, not what a benchmark achieves. A cost model that used 450 GB/s as the achieved rate would overestimate transfer speed. The correction from nominal to achieved is a learned number, applied where learned numbers are kept.

## Provenance

Each chip and fabric file states its provenance in one field, `Provenance`. The schema accepts two values:

`vendor_spec`
:   The figures are the vendor's, read directly or with a simple step: a unit conversion, half of a bidirectional figure, or the dense figure rather than the sparse one.

`derived`
:   The figures are computed from vendor figures by other arithmetic. They are still declared and still unmeasured.

The label applies to a whole file, and every chip and fabric file is labeled `vendor_spec` today, including files that hold figures of other origin (see below). Read a file's comments, not only its label, before relying on a figure. The [Chips](../reference/hardware.md#sources) and [Fabrics](../reference/networks.md#sources) references reproduce every comment.

## Where the catalog falls short of its rule

The rule is the aim, and some current entries do not yet meet it. As of this release:

- Some chip figures come from published simulator descriptors or vendor APIs rather than datasheets. Several SM counts come from public measurements because no datasheet states them, and one, for B300, is assumed equal to B200's. Each chip's comments say which.
- The storage tiers were copied from the simulator's earlier defaults and cite no source.
- The workloads cite no source.

A pull request that replaces one of these with a cited declared figure, or moves an assumed number to blis-registry, is welcome.

## What the catalog leaves out

- Achieved efficiencies and overheads, such as model FLOPs utilization (MFU), effective bandwidth and fabric overhead, are learned numbers.
- Latency floors no vendor states, such as the fixed cost of moving a KV cache from a prefill server to a decode server, are modeling inputs. That one is an inference-sim setting today (`--pd-transfer-base-latency`); [blis-registry#10](https://github.com/inference-sim/blis-registry/issues/10) tracks moving it to the registry.
- Run choices, such as GPU count, parallelism and engine settings, are stated in the scenario file.
- A cluster that pairs a chip with a fabric is a run choice too. The catalog defines chips and fabrics separately, and the scenario pairs them.
