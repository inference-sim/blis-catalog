---
hide:
  - toc
---

# Chips

[[catalog.count.chips]] chips. Figures are nominal and per GPU. Most come from vendor datasheets; the others come from published simulator descriptors, vendor APIs or public measurements, and each file's notes say which. See [Sources](#sources). What a chip achieves in practice is a coefficient in [blis-registry](https://github.com/inference-sim/blis-registry).

<figure markdown="1">
[[catalog.hardware.figure]]
<figcaption>Chips ordered by dense BF16 peak. Each panel has its own scale, starting at zero. The last panel divides the first by the second: how many BF16 operations a chip can perform per byte it reads from HBM. GPU code that performs fewer operations per byte than this is limited by memory bandwidth rather than by compute. The ratio is computed for this figure; it is not a catalog field.</figcaption>
</figure>

[[catalog.hardware.table]]

Peak rates are dense, in TFLOP/s. A dash means the chip has no native support for the format. Intra-node bandwidth is NVLink where present, per GPU and per direction. `a100-80` is an alias of `a100-sxm`, kept because inference-sim and some blis-registry scopes name the A100 that way.

## Rack-scale chips

On these chips an NVLink domain spans a rack of several nodes, so a third tier sits between the node and the inter-node fabric. Traffic that leaves the rack uses the fabric the scenario names.

[[catalog.hardware.racks]]

## Sources

Each file records the source of its figures, and any conversion applied to them, in its `_comment` keys. They are reproduced here as written, including where a figure is not a datasheet figure.

[[catalog.hardware.notes]]
