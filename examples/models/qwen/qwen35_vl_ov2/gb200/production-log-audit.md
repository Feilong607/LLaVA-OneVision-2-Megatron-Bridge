# Production log audit

`analyze_prod_logs.py` reads the shared production logs using the Python standard
library. It does not import torch, access GPUs, change checkpoints, or modify a
running workload. Run it in a cluster terminal after pulling the synchronized
repository. Keep production logs on the cluster; only the tool is published.

From the repository root:

```bash
python3 examples/models/qwen/qwen35_vl_ov2/gb200/analyze_prod_logs.py --job WORKLOAD_NAME
```

Add `--interval 60` to observe net consumed-sample progress for approximately one
minute. For a more stable wall-clock measurement, use `--interval 300`. The
program waits in the terminal and reads the logs a second time; it does not
submit a workload or create a background monitor. Ctrl-C ends the observation.

## What it measures

- Exact job-name matching excludes other runs, including names that extend the
  requested name. Logs default to `~/train_logs`; override with `--log-dir`.
- A per-pod table shows launcher starts, repeated/decreasing iteration numbers,
  HybridEP TIMEOUT lines (whole file and latest segment), and stream-warning
  lines. These are evidence from retained files, not Kubernetes restart counts.
- The last rank is selected using `nnodes` from the launcher header: worker-10
  for 12 pods, worker-6 for 8 pods. Missing final-rank logs stop the analysis.
  Without a topology header, the highest worker suffix is used with a warning.
- Each launcher start or repeated/decreasing iteration number resets the timing
  segment. A fresh attempt with no completed steps never reports older speed as
  current. Defaults discard 50 completed records after each start and summarize
  up to 100 subsequent step intervals, requiring at least 20 intervals. Override
  with `--warmup`, `--window`, and `--min-steps`.
- Mean, median, nearest-rank P95, maximum, and the fraction above 1.5 times the
  median describe completed-step timing. The 1.5 threshold is a screening aid,
  not a failure criterion. Gaps, invalid durations, malformed iteration records,
  or inconsistent consumed-sample increments suppress timing statistics.
- Completed-step throughput is the sum of logged batch sizes divided by the sum
  of iteration durations. Timestamp throughput uses consumed-sample increments
  and the matching first/last log timestamps, whose resolution is one second.
  Both cover completed intervals only, in the latest identified process segment.
- With `--interval`, net progress is the difference between the latest consumed
  sample counters at the two observations divided by monotonic elapsed time.
  It includes pauses and rollback visible during that observation. Negative net
  progress means the counter regressed, not negative compute throughput. Missing
  records or changes in the target log/total iterations prevent this calculation.

## Interpretation limits

TIMEOUT lines from different SMs and ranks can describe the same event. Do not
multiply line counts by a timeout duration. `AccumulateGrad` warnings may print
once while synchronization continues; warning counts cannot measure its cost.

A later iteration line in an append-only file may belong to a restarted process.
Check the segment, starting iteration, and consumed samples before calling it
recovery. This tool cannot prove CUDA kernel health or checkpoint correctness.

Completed-step timing excludes a currently hung step and downtime after the last
record. Use two observations and check the live pod state as well. No new record
could mean initialization, checkpoint I/O, a slow step, failure, or completion;
logs alone cannot distinguish all of these. Missing or rotated logs also limit
the historical event counts. The separate `prod48_...` preflight log may contain
failures that occur before the production wrapper writes a new launch marker.

Compare 32/48-GPU runs with samples/s because their global batch sizes can differ.
Variable image/video content and checkpoint I/O can cause slow steps. Attribution
to HybridEP or stream synchronization requires controlled runs with the same
data/checkpoint/configuration or a profiler; this tool does not claim causality.
