# Qwen3.5 midtrain: 16-GPU routing-map allgather ablation

One workload holds four GB200 pods and runs two **fresh** arms sequentially:
`custom` explicitly enables DeepEP's custom routing-map allgather; `nccl`
disables that operation. Both keep BF16 HybridEP dispatch/combine (`ACCEL=2`).
The existing production defaults are unchanged when the new switch is unset.

## Workload form

Copy the working Qwen3.5 production form, including its **qwen35-fla image,
CPU/RAM, shared mounts and namespace**. Set:

| Field | Value |
|---|---|
| Type | Distributed / PyTorch |
| Workload name | A new name for every attempt, e.g. `qwen35-hep-ab16-a7-01` |
| Workers | **3** (master + 3 workers = 4 pods) |
| GPUs per master / worker | **4** (16 total) |
| Node pool | GB200 NVL72 |
| Node affinity | The **same rack for all four pods**, with 16 free GPUs |
| Master and worker Command | `bash` |
| Master and worker Args | `$HOME/bridge-export/examples/models/qwen/qwen35_vl_ov2/gb200/ax_ov2_qwen35_s15_hep_ab16.sh` |

**Expand `$HOME` to the actual absolute path in the Args field**; the workload
form does not perform shell expansion. Keep the existing `OV2_K8S_NAMESPACE`
setting; if it must be provided via Args, append its actual value on both sides.
No other Args are required for the defaults. This is a launch specification,
not evidence that a GPU experiment has already been run.

Pull the published vendor snapshot in the code-sync workspace before submitting:

```bash
cd ~/bridge-export && git pull --ff-only
```

## Fixed comparison

| Setting | Both arms |
|---|---|
| Model / stage | Qwen3.5-35B-A3B OV2 midtrain; full model, MTP retained |
| Initialization | The same staged stage-2 `iter_0006000`; **no midtrain resume** |
| TP / DP / EP / PP / CP | 1 / 16 / 8 / 1 / 1 |
| GBS / MBS | **80 / 1**, five microbatches per rank |
| Sequence / HybridEP token cap | 10192 / 10240 (existing base launcher) |
| Optimizer | Muon, `muon_split_qkv=false`, no CPU offload |
| Recompute | selective core attention + MoE; vision recompute on |
| Length sorting | patches, window 5 |
| RNG / data workers | seed 1234 / 8 workers per rank |
| Training length | **400 steps per arm** |
| Comparison window | **steps 101–400**; retain earlier steps for numerical checks |
| LR schedule | 66-step warmup, `1e-5` to `1e-6`, **33334-step decay horizon** |
| Checkpoints | Disabled, including final save; independent logs/TensorBoard/input signatures |

The sample budget `2666720 = 80 * 33334` preserves the recipe's long schedule;
the explicit 400-step endpoint and `scheduler.lr_decay_iters=33334` intentionally
truncate the experiment without accelerating the cosine decay. This is a
per-rank workload approximation: production has GBS240, DP48 and six EP8 groups;
this experiment has GBS80, DP16 and two groups. Muon sharding and data partitions
also differ. **Do not extrapolate the measured percentage directly to 48 GPUs.**

Same input seed and sharding are used in both arms. CPU-side fingerprints verify
the per-rank order of tokens, labels, masks and geometry across every batch.
Image shape is included, but image pixels are deliberately not hashed to avoid
large CPU copies. No tensor contents are logged. These signatures and scalar
loss/grad-norm comparisons are useful checks, **not full routing/gradient parity**.

## Output and failure behavior

Default root:

```text
$HOME/ckpts_video_sft/qwen35_hep_ab16/<workload-name>/
  experiment.json
  custom/{train_node0..3.log,controller_node0..3.log,tensorboard/,inputs/,summary.json}
  nccl/{train_node0..3.log,controller_node0..3.log,tensorboard/,inputs/,summary.json}
  control/
  comparison.json
```

`worker-2` contains the final global rank's iteration output. Watch either arm's
`train_node3.log` (which appears after model setup). The master controller prints
the active arm and the final concise report; controller logs include asset,
initialization and backend import failures. All 16 ranks must log the selected
allgather mode and one NVLink domain; otherwise the arm is rejected. A split-rack
EP group must not silently make both arms use NCCL.

Each new process uses a nonce handshake; old success files cannot advance a
restarted pod. All four launchers must finish successfully before the next arm.
Nonzero worker return codes propagate in this experiment. A failed or timed-out
arm stops the experiment; **it does not automatically restart or resume**.
Use a fresh workload name after interruption. No production SAVE is used.

To reprint a completed report without touching GPUs:

```bash
python3 examples/models/qwen/qwen35_vl_ov2/gb200/run_hybridep_ablation.py \
  --report "$HOME/ckpts_video_sft/qwen35_hep_ab16/<workload-name>"
```

The report checks complete iteration/sample accounting, identical metadata
order, finite loss/grad-norm and correct backend selection before comparing
mean, p95 and samples/s. TIMEOUT counts are **printed lines, not independent
faults**. A custom arm with no TIMEOUT tests ordinary overhead; it does not
reproduce or prove a fix for the production stall. Shared-rack contention can
still change during sequential runs, and later steps can diverge numerically.

## Optional Args (same on master and workers)

- `AB_STEPS=800`: longer run if 400 steps do not reproduce the rare stall; max 2000.
- `AB_DISCARD=100`: steps omitted from speed statistics; minimum 100.
- `AB_ORDER=nccl,custom`: reverse order in a **new** workload to check order/cache bias.
- `AB_TIMEOUT_MIN=240`: wall limit **per arm**, including setup/JIT; not a kernel timeout.
- `AB_ROOT=/absolute/fresh/path`: custom experiment root; never a production SAVE.
- `INIT_CKPT=/absolute/stage2/iter_0006000`: override the shared initial weights if needed.
- Existing `OV2_STAGE4_POOL`, model/processor path and namespace overrides are supported.
- `AB_PREFLIGHT_ONLY=1`: validate the experiment contract/checkpoint marker without
  launching; it does **not** prove runtime imports, model fit or topology.

Do not pass raw `ITERS`, `SAVE`, `EXTRA_ARGS`, `ACCEL` or GBS overrides: the
wrapper rejects them so the intended single-variable experiment is preserved.
Keep the working image; its installed HybridEP API already accepts
`enable_custom_allgather`. The repo patch adds opt-in
`OV2_HYBRIDEP_CUSTOM_ALLGATHER=0|1`; an unsupported installed API fails loudly.
No new container build is required for this experiment.
