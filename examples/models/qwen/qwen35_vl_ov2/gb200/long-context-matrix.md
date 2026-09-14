# Qwen3.5 video stage2/3: 48/64-GPU long-context test matrix

Prepared 2026-09-07. This is a GPU experiment launcher, **not a GPU-validated production preset**.
Both stages share the controller, model construction and save/restart test. They select different
mixtures, budgets and optimizer settings, using the full-model
`ov2_qwen35_35b_a3b_midtrain` recipe. The image-alignment recipe named `stage2` is a different stage.

## Matrix and defaults

| GPUs | TP | ETP / EP | dense DP / expert DP | GBS / microbatches per rank | Dispatcher / LLM recompute |
|---|---|---|---|---|---|
| 48 | 4 | 2 / 8 | 12 / 3 | 24 / 2 | HybridEP / selective core_attn + moe |
| 48 | 2 | 2 / 8 | 24 / 3 | 24 / 1 | alltoall / full |
| 64 | 4 | 4 / 8 | 16 / 2 | 32 / 2 | HybridEP / selective core_attn + moe |
| 64 | 2 | 2 / 8 | 32 / 4 | 32 / 1 | alltoall / full |

MBS=1, PP=CP=1, bf16, distributed Muon, trainable vision/adapter/LLM,
vision recompute, 2 Energon workers, shuffle buffer16, parallel shard iter1, sorting disabled.
MTP and Qwen3.5 token/mRoPE settings keep their recipe defaults.

The existing 64k packs use **SEQ_LEN=73728** to accommodate retokenization.
TP4 HybridEP cap is18432; TP2 uses alltoall because36864 exceeds the installed HybridEP cap.
`LC_SEQ_LEN=65536` is available for a strict-cap diagnostic; skipping oversized packs
does not make it equivalent to the73728 production candidate.

| Stage | YAML under qwen3_vl_ov2/gb200 | Budget / full LR schedule | LR | Muon extra / WD / Adam beta2 |
|---|---|---|---|---|
| 2 | mid_training_180s_packed_64k.yaml | 655368; 27307 steps at48 / 20481 at64 | 1e-5 → 1e-6 | .15 / .01 / .95 |
| 3 | stage3_mix_img10.yaml | 617482; 25729 steps at48 / 19297 at64 | 2e-5 → 1e-6 | .2 / 0 / .99 |

Tests retain the full stage's cosine schedule and0.2% warmup, rather than compressing to100steps.
Historical Qwen3 stage2's freeze setting remains unresolved; this tests the current packed
launcher's **full-model** setting.

## Workload form

Use the current Qwen3.5 FLA-capable GB200 image, existing mounts/network/CPU/RAM settings,
and the **same NVL72 rack** for all pods. The script does not submit or stop workloads.

- Distributed PyTorch,4GPUs per pod.
- 48GPUs: **Workers=11** (1master+11workers).
- 64GPUs: **Workers=15** (1master+15workers).
- Fresh workload name for every experiment; the shared output root derives from it.
- Both master and workers: Command `bash`, identical Args.
- Required `INIT_CKPT`: an existing Qwen3.5 `iter_*` directory containing `.metadata`.
  Stage2 production-candidate tests should read a midtrain checkpoint; stage3 should read a video-stage2 checkpoint.
  Earlier Qwen3.5 weights may be used for a hardware smoke, but label that initialization explicitly.

Stage2 Args (one line per argument; replace placeholders with actual absolute paths):

```text
<bridge-export>/examples/models/qwen/qwen35_vl_ov2/gb200/ax_ov2_qwen35_long_context_matrix.sh
INIT_CKPT=<completed-qwen35-checkpoint>/iter_XXXXXXX
```

Stage3 uses the same entry and adds only:

```text
LC_STAGE=3
```

GPU count comes from `PET_NNODES`; no `LC_WORLD` is needed in a real workload.
Default order is **TP4 then TP2**, sequentially on the allocated48 or64GPUs.
Use `LC_TPS=4` or `LC_TPS=2` for one configuration, or `LC_TPS=2,4` for an order check.
Production `SAVE`, `EXTRA_ARGS` and mixture overrides are rejected.
Optional asset overrides: `OV2_STAGE4_POOL`, `OV2_LLM_HF_QWEN35`,
`OV2_HF_PROC_QWEN35_P16M33`, `OV2_EXTRA_PYLIBS`, `OV2_K8S_NAMESPACE`.

## Per-TP execution

1. `reference`: fresh weights,100continuous steps, no model checkpoint writes.
2. `split`: same fresh weights/seed/data, run through step60, synchronously save, exit.
   Target train_iters stays100; `train.exit_interval=60` causes the controlled exit.
3. `resume`: new training processes restore the split SAVE's optimizer, scheduler, RNG and
   every dense-DP dataloader cursor; run61–100. No final large save.

Checkpoint checks cover metadata, both Bridge/Energon trackers, counters, run_config and the exact
DP cursor file set before releasing resume. A hook records every rank's initialized groups,
**inner LLM TP/ETP**, trainable components and initial/resumed counters. Mismatches fail
collectively before consuming a batch.

One complete split checkpoint remains per TP; a default job retains two, potentially hundreds
of GB each. No automatic cleanup. A failure preserves evidence and stops only the controller's
child process groups; later phases do not start. An externally interrupted test must use a fresh
workload/root. The automatic internal restart is the tested recovery scenario; this is not a
general production resume wrapper.

Controls only when changing defaults:

```text
LC_STEPS=100
LC_SPLIT=60
LC_DISCARD=20
LC_TIMEOUT_MIN=360
LC_ROOT=<new-absolute-experiment-directory>
LC_SEQ_LEN=73728
LC_MIN_LONG_TOKENS=60000
```

Require `DISCARD < SPLIT`, `STEPS/2 <= SPLIT < STEPS-DISCARD`, `STEPS <=2000`.
Timeout is per phase. Default work is200steps per TP,400per workload, plus three model startups
per TP. Long-video duration is unmeasured; no ETA is inferred from10k-token midtrain.

## Reports

Output: `$HOME/ckpts_video_sft/qwen35_long_context/<workload>/`.
Each TP has `reference/`, `split/`, `resume/` and `comparison.json`.
Each phase retains controller/training logs, TensorBoard, CPU input metadata, runtime snapshots,
torch peak-memory logs and10-second nvidia-smi samples.
Last-rank log: `train_node11.log` at48GPUs, `train_node15.log` at64GPUs.

From the repo root, no rg or GPU allocation is needed to re-read a completed report:

```bash
python3 examples/models/qwen/qwen35_vl_ov2/gb200/run_long_context_matrix.py --report "$HOME/ckpts_video_sft/qwen35_long_context/<workload>"
```

This refreshes derived JSON only. If TP2 OOMs, completed TP4's `comparison.json` remains available.

Report checks:

- Contiguous iteration/sample accounting, finite loss/grad norm/LR, zero skipped/NaN iterations.
- Every rank's TP/ETP/EP/DP and inner LLM config; HybridEP mode and per-group NVLink-domain evidence.
- Metadata equality within attention TP groups, and continuous-vs-split+resume order per rank.
- Each phase must observe a pack with **nonzero temporal positions and >=60000 payload tokens**.
  Short-only and image-only runs are rejected.
- Reference timing excludes1–20; resumed timing excludes its first20steps.
  Mean/P95, samples/s, payload tokens/s, patches/s and timestamp-window throughput are reported.
- Raw loss/grad-norm/LR series and paired absolute/relative differences.
  Screening tolerances: LR rtol1e-6/atol1e-12, loss1%/.001, grad norm5%/.01.
  Failed tolerances remain visible.
- TIMEOUT and SkipSample **log mentions**, torch allocated/reserved peaks and sampled device peak.
  Sampling misses brief peaks; missing telemetry is null. Skip mentions are not a drop rate.

`complete.json` means execution/report phases completed. Inspect each `resume_screen_passed`
and underlying metrics separately. Pixel values are excluded from hashes; loss/grad-norm agreement
does not prove full gradient/routing or optimizer-tensor equality.
Passing covers observed packs, not all possible expensive packs. This is a controlled
save/relaunch test, not SIGKILL during a write or rack-failure injection.

TP2 vs TP4 changes dispatcher/recompute/data partition as well as TP;48vs64 changes GBS/expert DP.
Compare **complete configurations**, not a pure TP speedup. Never move split checkpoints between
cases or reuse a production SAVE.

## CPU-only plan and tests

No mount checks, GPU access or training:

```bash
LC_WORLD=48 python3 examples/models/qwen/qwen35_vl_ov2/gb200/run_long_context_matrix.py --plan
```

Use `LC_WORLD=64 LC_STAGE=3` for that matrix. Plan does not prove asset presence or GPU viability.

```bash
uv run python -m pytest tests/unit_tests/models/ov2/test_long_context_matrix.py --confcutdir=tests/unit_tests/models/ov2 -q
```

Tests use CPU rank generation, captured arguments from the actual base launcher, synthetic
checkpoint/log fixtures and twelve controller processes. No GPU kernels execute.
