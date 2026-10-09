# Workloads

[[catalog.count.workloads]] workloads. A workload says how long requests are, not how many arrive; the arrival rate or concurrency is set by each run.

<figure markdown="1">
[[catalog.workloads.figure]]
<figcaption>Tokens per request, on a logarithmic scale. The thin line spans the stated minimum to maximum, the thick bar one standard deviation either side of the mean (clipped to those bounds), and the dot marks the mean.</figcaption>
</figure>

[[catalog.workloads.table]]

A file states a mean, a standard deviation and bounds, not the form of the distribution. That is the sampler's choice: inference-sim draws from a normal distribution with the stated mean and standard deviation, clamped to the bounds.
