# LLaVA-OneVision-2.1 · 4B (p16m3) on Megatron-Bridge

Two-stage training of **LLaVA-OneVision-2.1 4B** under [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge), reproducing the AIAK (`OV2_public_main`) recipe with standalone data-parallel trainers.

```
LlavaOnevision2 = language_model (Qwen3-4B GPTModel)   # frozen in both stages here
                + vision_model   (OneVision ViT, patch16 / 24L / hidden 1024 / spatial-merge 3)
                + adapter        (PatchMerger: 1024·3² = 9216 → GELU → 2560)
```

- **Stage 1** — *adapter-only* alignment on 558k caption data (LLM + ViT frozen). Optimizer **Adam**, cosine LR.
- **Stage 2** — *vit + adapter* SFT on 780k LLaVA-Next data (LLM frozen). Optimizer **Muon**, constant LR. Online sequence-packing variant available.

> The checkpoints loaded here (assembled mcore `tp1pp1`) carry all three namespaces (`language_model.*` / `vision_model.*` / `adapter.*`) and are byte-identical to the official `lmms-lab/LLaVA-OneVision-2-4B-p16m33-mcore-tp1-pp1` for the LLM+vision tensors.

---

## QuickStart (docker → training)

Everything runs inside the **`mbridge:qwen35`** docker image on the A100 box (`docker images | grep mbridge` → `mbridge:qwen35`, ~29 GB). The image already has torch + Transformer-Engine + Megatron-Core; you only need to bind-mount the data/checkpoints and set `PYTHONPATH`.

### 0. Prerequisites (paths used by the defaults)

| asset | path |
|---|---|
| repo | `/ov2/feilong/Megatron-Bridge` |
| **docker image** | **`mbridge:qwen35`** |
| processor | `/ov2/feilong/preprocessor_p16m33` |
| stage-1 init ckpt (mcore) | `/ov2/feilong/ov2_quickstart/ov_encoder_p16m3_qwen3_mcore_tp1pp1` |
| stage-1 data (558k) | `/vlm/data/blip_laion_cc_sbu_558k_wds` |
| stage-2 data (780k) | `/vlm/data/llava_next_full_mega` |

### 1. Launch stage 1 — explicit `docker run`

This is exactly what `ax_stage_1_alignment_p16m3_adapter_only.sh` runs (detached container `ov2_s1`, 7-way DP on GPUs 1–7):

```bash
cd /ov2/feilong/Megatron-Bridge
OUT=/ov2/feilong/Megatron-Bridge/ov2_1_4b/stage1_alignment_p16m3_adapter_only; mkdir -p "$OUT"

docker run -d --name ov2_s1 \
  --gpus '"device=1,2,3,4,5,6,7"' --ipc=host --shm-size=32g \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e OVCK=/ov2/feilong/ov2_quickstart/ov_encoder_p16m3_qwen3_mcore_tp1pp1 \
  -e OUT="$OUT" -e GBS_TARGET=256 -e TOTAL_SAMPLES=558128 \
  -e LR=2e-5 -e MIN_LR=1e-6 -e WARMUP_FRAC=0.002 -e CLIP_GRAD=1.0 \
  -e DATA=/vlm/data/blip_laion_cc_sbu_558k_wds \
  -e PYTHONPATH=/ov2/feilong/Megatron-Bridge/3rdparty/Megatron-LM:/ov2/feilong/Megatron-Bridge/src:/ov2/feilong/Megatron-Bridge/aiak_shim \
  -v /ov2:/ov2 -v /vlm:/vlm -w /ov2/feilong/Megatron-Bridge \
  mbridge:qwen35 \
  bash -lc "torchrun --standalone --nproc_per_node=7 examples/models/LLaVA_OV_2_1/train_stage1_mp.py >> $OUT/train.log 2>&1"

docker logs -f ov2_s1     # or: tail -f $OUT/train.log
```

**Or just** `bash examples/models/LLaVA_OV_2_1/ax_stage_1_alignment_p16m3_adapter_only.sh` (same thing, every flag overridable via env). Expected: iter-1 `lm loss ≈ 4–6`, descending toward ~3.0 (≈ AIAK reference); ~2155 steps = 1 epoch.

### 2. Launch stage 2 — explicit `docker run`

Same image; loads the stage-1 (or a vit+adapter) checkpoint via `STAGE1_CKPT`, trains vit+adapter with Muon:

```bash
cd /ov2/feilong/Megatron-Bridge
OUT=/ov2/feilong/Megatron-Bridge/ov2_1_4b/stage2_alignment_p16m3_muon_vit_adapter; mkdir -p "$OUT"

docker run -d --name ov2_s2 \
  --gpus '"device=1,2,3,4,5,6,7"' --ipc=host --shm-size=32g \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e STAGE1_CKPT=/ov2/feilong/Megatron-Bridge/ov2_1_4b/stage1_alignment_p16m3_adapter_only/iter_0002155/mp_rank_00/model_optim_rng.pt \
  -e OUT="$OUT" -e DATA=/vlm/data/llava_next_full_mega -e SEQ=32000 -e ACCUM=18 \
  -e LR=2e-5 -e WARMUP=0 -e CLIP_GRAD=1.0 -e MUON_RMS=0.2 -e TOTAL_SAMPLES=779111 \
  -e PYTHONPATH=/ov2/feilong/Megatron-Bridge/3rdparty/Megatron-LM:/ov2/feilong/Megatron-Bridge/src:/ov2/feilong/Megatron-Bridge/aiak_shim \
  -v /ov2:/ov2 -v /vlm:/vlm -w /ov2/feilong/Megatron-Bridge \
  mbridge:qwen35 \
  bash -lc "torchrun --standalone --nproc_per_node=7 examples/models/LLaVA_OV_2_1/train_stage2_mp.py >> $OUT/train.log 2>&1"
```

**Or** `bash examples/models/LLaVA_OV_2_1/ax_stage_2_alignment_p16m3_muon_vit_adapter.sh` (non-packed, matches AIAK date0523), or `…_packed.sh` for online sequence-packing (run `train_stage2_pack_mp.py`; faster on few GPUs — see notes). Expected starting `lm loss ≈ 1.1–1.3`.

### 3. Interactive shell (debugging)

```bash
docker run -it --rm --gpus '"device=0"' --ipc=host --shm-size=32g \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e PYTHONPATH=/ov2/feilong/Megatron-Bridge/3rdparty/Megatron-LM:/ov2/feilong/Megatron-Bridge/src:/ov2/feilong/Megatron-Bridge/aiak_shim \
  -v /ov2:/ov2 -v /vlm:/vlm -w /ov2/feilong/Megatron-Bridge \
  mbridge:qwen35 bash
# inside: torchrun --standalone --nproc_per_node=1 examples/models/LLaVA_OV_2_1/train_stage1_mp.py
```

> **PYTHONPATH order matters**: `3rdparty/Megatron-LM` must come **first** (the image's installed mcore lacks `SelfAttentionSubmodules.apply_rotary_fn` that the OV2 vision layer-spec needs). All launch scripts already set this. Every hyper-parameter is an overridable env var at the top of each `.sh`.

---

## Stage 1 — adapter-only alignment

| | value |
|---|---|
| trainable | adapter only (LLM + ViT frozen, adapter random-init) |
| optimizer | **AdamW** β(0.9, 0.99), eps 1e-5, wd 0 |
| LR | 2e-5 → **cosine** → 1e-6, warmup-fraction 0.002 |
| clip-grad / loss | 1.0 / **token-weighted** mean |
| batch | gbs **256** (mbs 1 + grad-accum), seq 32000 |
| data / epochs | 558k blip_laion_cc_sbu, 1 epoch (~2155 steps) |

`train_stage1_mp.py` · launch `ax_stage_1_alignment_p16m3_adapter_only.sh`.

## Stage 2 — vit + adapter SFT

| | value |
|---|---|
| trainable | adapter + vision_model (LLM frozen) |
| optimizer | **Muon** (mom 0.95, ns-steps 5, matched-adamw-rms 0.2, β 0.9/0.99, eps 1e-5) |
| LR | 2e-5 **constant** (min-lr == lr, warmup 0) |
| clip-grad / loss | 1.0 / token-weighted mean |
| batch | gbs ~128, seq 32000 |
| data / epochs | 780k llava_next_full_mega (multi-turn), 1 epoch |

`train_stage2_mp.py` (non-packed) · `train_stage2_pack_mp.py` (online packing) · launch `ax_stage_2_alignment_p16m3_muon_vit_adapter[_packed].sh`.

---

## Implementation notes (don't skip — these cost real debugging)

- **Next-token label shift is mandatory.** mcore's `compute_language_model_loss` does **not** shift labels internally; the trainers do `labels = torch.roll(labels, -1)` + mask the last position (packed: mask every `cu_seqlens` boundary). Without it the loss starts **>random (~16)** instead of ~4–6. This was the single biggest bug.
- **Token-weighted loss** (sum per-token loss / total supervised tokens across DP), not per-microbatch mean — matches Megatron/AIAK and keeps grad scaling stable.
- **Recompute is LLM-only** (`recompute_granularity=full, method=uniform, num_layers=1`); the ViT/adapter are not recomputed (vision recompute hits an `attn_mask_type` error). Toggle with `RECOMPUTE=0` only if memory allows.
- **Full-model checkpoints**: each `iter_NNNN/mp_rank_00/model_optim_rng.pt` holds all 588 keys (LLM+vision+adapter) so the next stage loads in one shot. Resume is automatic from the latest `iter_*` in the output dir.
- **NCCL timeout** raised to 7200s (`NCCL_TIMEOUT_S`) so a long-sample step can't trip the 10-min watchdog.
- **Online packing** (`train_stage2_pack_mp.py`): greedy-knapsack packs short samples into uniform `PACK_CAP`-token bins with **block-diagonal** attention (`PackedSeqParams`, `cu_seqlens` at *sample* boundaries — within-conversation causal context preserved, cross-sample blocked). Samples longer than `PACK_CAP` get their **own bin**. Caveat: dense packs raise GPU memory; on 80 GB cards keep `PACK_CAP≈8000`, and very long (~32000-token) single-conversation singletons can still OOM — fall back to the non-packed trainer for the long tail if needed.

## Layout

```
examples/models/LLaVA_OV_2_1/
  ax_stage_1_alignment_p16m3_adapter_only.sh          # stage-1 launch
  ax_stage_2_alignment_p16m3_muon_vit_adapter.sh      # stage-2 launch (non-packed)
  ax_stage_2_alignment_p16m3_muon_vit_adapter_packed.sh  # stage-2 launch (online packing)
  train_stage1_mp.py  train_stage2_mp.py  train_stage2_pack_mp.py
  README.md
src/megatron/bridge/models/qwen_vl_ov2/
  llava_ov2_4b.py         # LlavaOnevision2 model + build + checkpoint loader
  onevision_encoder_model.py  adapter.py  layer_spec.py  vision_config.py  ...
  muon_standalone.py      # standalone Muon (used by the stage-2 trainers)
```
