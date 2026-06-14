# OV2-30B-A3B checkpoint conversion + consistency verification (GB200)

OV2 = `LlavaOnevision2` = language_model (Qwen3-30B-A3B MoE) + vision_model (OV2.1) + adapter.
Verified parallelism: **TP1 / PP1 / EP8** (16 expert-parallel ranks fold into EP8 + DP).

## TL;DR — do you even need to convert?

`use_distributed_optimizer=false` (optimizer is **replicated**) and the checkpoint is `torch_dist`,
so **model + optimizer + RNG are all stored as global, parallelism-independent tensors**.

* **2 GB200 nodes (4+4 = 8 GPU, EP8)** or **4 GB200 nodes (16 GPU, EP8)**: load the existing
  `ov2_30b_a3b_stage2` checkpoint **directly** — Bridge auto-reshards the DP dim on load (EP stays 8).
  This is the same torch_dist resume path your iter_200→400→600 saves already use. **No conversion.**
* **Convert only when you change EP** (e.g. EP8→EP4 for a single 4-GPU node), **export to HF**, or
  **re-bootstrap from the AIAK base**.

Whatever path you take, prove it was lossless with `verify.sh`.

## Supported target parallelism

| dim | support | notes |
|---|---|---|
| **TP** (tensor) | ✅ `--tp N` | LLM shards; vision/adapter shard or replicate. **Set `--etp 1`** unless you also want expert-TP (else expert-TP defaults to TP and the world must cover `TP×EP`). |
| **EP** (expert) | ✅ `--ep N` | EP8 = validated; EP≠8 works but verify it. |
| **ETP** (expert-TP) | ✅ `--etp N` | default follows `--tp`; pass `--etp 1` to keep experts un-TP-sharded. |
| **PP** (pipeline) | ❌ blocked | OV2 is a monolithic VLM; vision tower pinned PP1; vision/adapter built on every rank → PP>1 = unusable ckpt (same as Megatron-MIMO: vision can't PP). Scale with EP/TP. |
| **CP** (context) | ❌ blocked | forward hard-asserts `context_parallel_size==1`. |

### ✅ Proven on real GPUs (A100-2): `TP1/EP8 → TP2/EP4/ETP1`
Resharded the live stage2 ckpt (`iter_0000800`) → loadable torch_dist, then `verify.sh`:
`model key-set identical (33/33)`, `values: 29 identical, 0 mismatched`, **`RESULT: CONSISTENT`**.
(`vision_model.patch_embed.proj.weight` re-cast bf16→fp32 — a lossless upcast, value-identical.)

## convert.sh — produce a *loadable* Bridge torch_dist ckpt

World size (`nnodes * NPROC`) must cover both EP and TP (`world % EP == 0`, `world % TP == 0`, `world >= EP`).
GB200 = 4 GPU/node, so EP8 needs **>= 2 nodes**. Output is written via Bridge `save_megatron_model` →
`iter_0000000/` + `"model"`-wrapped sharded dict + `run_config.yaml` + `latest_checkpointed_iteration.txt`
+ `tokenizer/` (directly usable as `pretrained_checkpoint` / `load`).

```bash
# TP1/EP8 -> TP2/EP8 (isolate TP, keep validated EP8), 2 GB200 nodes -- run on BOTH nodes:
NPROC=4 LIST_IP="<ip0> <ip1>" bash convert/convert.sh reshard --src <ckpt> --out <out> --tp 2 --ep 8 --etp 1
```

```bash
cd /ov2/feilong/gb200/Megatron-Bridge/examples/models/qwen/qwen3_vl_ov2/gb200

# reshard EP8 stage2 -> EP8 at GB200 layout (run the SAME cmd on BOTH nodes; node_rank auto-detected)
NPROC=4 LIST_IP="<ip0> <ip1>" bash convert/convert.sh reshard \
    --src /ov2/feilong/gb200/ckpts_video_sft/ov2_30b_a3b_stage2 \
    --out /ov2/feilong/gb200/ckpts_video_sft/ov2_30b_a3b_stage2_gb200 --ep 8

# reshard EP8 -> EP4 for a single 4-GPU node quickstart  (EP4 is UNVALIDATED -> verify is mandatory)
NPROC=4 bash convert/convert.sh reshard --src <ep8_ckpt> --out <ep4_out> --ep 4

# re-bootstrap from the AIAK assembled base (EP MUST be 8, >= 2 GB200 nodes)
NPROC=4 LIST_IP="<ip0> <ip1>" bash convert/convert.sh from_base --src <base> --out <out> --ep 8

# export to HF (LLM -> HF dir; vision+adapter -> .pt)
NPROC=4 LIST_IP="<ip0> <ip1>" bash convert/convert.sh export_hf --src <ckpt> --out <hf_dir>
```

## verify.sh — before/after weight consistency (CPU only, no GPU contention)

Compares the GLOBAL tensors of two torch_dist checkpoints, so it works across an EP/DP change.
`model.*` tensors drive the exit code; optimizer/extra_state are informational. Big MoE expert
tensors (38 GB) are streamed in bounded-memory chunks, so it runs anywhere.

```bash
# representative tensors, bit-exact (seconds–minutes):
A=<src_ckpt> B=<converted_ckpt> bash convert/verify.sh

# EXHAUSTIVE: every model tensor, bit-exact  (the real "转化前后一致性" gate)
A=<src_ckpt> B=<converted_ckpt> VALUES=full bash convert/verify.sh

# allow tolerance / also diff optimizer state:
A=... B=... ATOL=1e-6 EXTRA="--include-optim" bash convert/verify.sh
```

Exit 0 + `RESULT: CONSISTENT` = every compared model weight is identical. Exit 1 = `INCONSISTENT`
(mismatching keys listed, worst `max|Δ|` first). A lossless reshard must print CONSISTENT.

### Validated CPU-only on the live 30B checkpoints (no 8-GPU job needed)
| test | expectation | result |
|---|---|---|
| iter_600 vs iter_600 (values) | CONSISTENT | 29/29 identical ✓ |
| iter_400 vs iter_600 (values) | INCONSISTENT | 21 trained vision/adapter tensors flagged, 8 frozen LLM tensors identical ✓ |

The convert *run* itself needs the GB200 (8-GPU EP8 process groups); run it there, then gate on `verify.sh`.
