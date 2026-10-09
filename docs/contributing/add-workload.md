# Add a workload

A workload describes the lengths of requests: how long prompts are, how long outputs are, and how long a prefix the requests share. It does not say how many requests arrive. The arrival rate or concurrency is set by each run.

Create `workloads/<name>.yaml`. The file name is the workload's identity; the file has no `name` key.

```yaml
prefix_tokens: 0         # length of the shared prefix; 0 means none
prompt:
  tokens: 256            # mean
  tokens_stdev: 100
  tokens_min: 2
  tokens_max: 800
output:
  tokens: 256
  tokens_stdev: 100
  tokens_min: 1
  tokens_max: 1024
```

All values are token counts per request. The validator enforces:

- each mean (`tokens`) is at least 1;
- no `tokens_stdev`, `tokens_min` or `prefix_tokens` is negative;
- where bounds are given, `tokens_min` ≤ `tokens` ≤ `tokens_max`;
- `prefix_tokens` does not exceed the mean prompt length.

Set the bounds with care. Two workloads with the same means but different tails can saturate a deployment at different loads.

To record where the numbers came from, such as a benchmark or a published trace, add a `_comment` key at the top level or inside `prompt:` or `output:`.

## Then

Run the [checks](checks.md) and open a pull request that cites the source of the numbers.
