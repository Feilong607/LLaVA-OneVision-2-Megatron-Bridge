#!/usr/bin/env bash
# =============================================================================
# LLaVA-OneVision-2.1 4B (p16m3) · Stage-2 SFT · vit+adapter · ONLINE PACKING
# Same recipe as the non-packed ax_stage_2 (Muon rms0.2, const lr2e-5, clip1.0,
# token-weighted loss, label-shift) but greedy-knapsack-packs samples on the fly
# into uniform <=SEQ sequences with block-diagonal per-sample attention (cu_seqlens,
# qkv_format=thd). Uniform packs => no 7-GPU straggler => steady fast steps.
# ACCUM counts PACKS/opt-step: ACCUM=1 -> gbs = NPROC packs (~140 samples ~ ref gbs 128).
# =============================================================================
set -euo pipefail
REPO="${REPO:-/ov2/feilong/Megatron-Bridge}"
HERE="$REPO/examples/models/LLaVA_OV_2_1"
IMAGE="${IMAGE:-mbridge:qwen35}"
GPUS="${GPUS:-1,2,3,4,5,6,7}"; NPROC="${NPROC:-7}"
STAGE1_CKPT="${STAGE1_CKPT:-/vlm/yinxie/code/OV2/OV2_public_main/checkpoints/date0513-corrected-muon-stage1-vit-adapter/date0511_ax_stage_1_alignment_p16m3_packed_new16_muon/release/mp_rank_00/model_optim_rng.pt}"
OUT="${OUT:-$REPO/ov2_1_4b/stage2_alignment_p16m3_muon_vit_adapter_packed}"
DATA="${DATA:-/vlm/data/llava_next_full_mega}"
SEQ="${SEQ:-32000}"; ACCUM="${ACCUM:-2}"; PACK_BUF="${PACK_BUF:-64}"   # gbs = NPROC*ACCUM packs/step
PACK_CAP="${PACK_CAP:-8000}"   # multi-sample bin cap (memory-bound); samples > PACK_CAP run as their own bin
LR="${LR:-2e-5}"; WARMUP="${WARMUP:-0}"; CLIP_GRAD="${CLIP_GRAD:-1.0}"; MUON_RMS="${MUON_RMS:-0.2}"
TARGET_SAMPLES="${TARGET_SAMPLES:-779111}"; TOTAL_SAMPLES="${TOTAL_SAMPLES:-779111}"; START_CONSUMED="${START_CONSUMED:-0}"
SAVE_EVERY="${SAVE_EVERY:-300}"; LOG_EVERY="${LOG_EVERY:-10}"; KEEP_LAST="${KEEP_LAST:-5}"
mkdir -p "$OUT"
echo "[stage2-pack] out=$OUT accum=$ACCUM (gbs=$((NPROC*ACCUM)) packs) pack_buf=$PACK_BUF lr=$LR(const) muon_rms=$MUON_RMS"
docker rm -f ov2_s2 2>/dev/null || true
docker run -d --name ov2_s2 --gpus "\"device=$GPUS\"" --ipc=host --shm-size=32g \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e STAGE1_CKPT="$STAGE1_CKPT" -e OUT="$OUT" -e DATA="$DATA" -e SEQ="$SEQ" -e ACCUM="$ACCUM" -e PACK_BUF="$PACK_BUF" -e PACK_CAP="$PACK_CAP" \
  -e LR="$LR" -e WARMUP="$WARMUP" -e CLIP_GRAD="$CLIP_GRAD" -e MUON_RMS="$MUON_RMS" \
  -e TARGET_SAMPLES="$TARGET_SAMPLES" -e TOTAL_SAMPLES="$TOTAL_SAMPLES" -e START_CONSUMED="$START_CONSUMED" \
  -e SAVE_EVERY="$SAVE_EVERY" -e LOG_EVERY="$LOG_EVERY" -e KEEP_LAST="$KEEP_LAST" \
  -e PYTHONPATH="$REPO/3rdparty/Megatron-LM:$REPO/src:$REPO/aiak_shim" \
  -v /ov2:/ov2 -v /vlm:/vlm -w "$REPO" \
  "$IMAGE" bash -lc "torchrun --standalone --nproc_per_node=$NPROC $HERE/train_stage2_pack_mp.py >> $OUT/train.log 2>&1"
echo "[stage2-pack] launched container ov2_s2 -> tail -f $OUT/train.log"
