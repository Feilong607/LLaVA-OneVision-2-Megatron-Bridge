#!/usr/bin/env bash
# =============================================================================
# LLaVA-OneVision-2.1 4B (p16m3) · Stage-1 Alignment · adapter-only (Megatron-Bridge)
# Aligned to the AIAK reference ax_stage_1_alignment_p16m3_adapter_only.sh:
#   Adam(0.9,0.99,eps1e-5,wd0) · lr 2e-5 -> cosine -> min 1e-6 (warmup 0.002) ·
#   clip-grad 1.0 · global-batch-size 256 (mbs1 + grad-accum) · GELU adapter ·
#   next-token label shift · token-weighted loss · 1 epoch over 558k · bf16, no FSDP.
# Loads the assembled new-encoder mcore ckpt (LLM+vision frozen); trains adapter only.
# =============================================================================
set -euo pipefail
REPO="${REPO:-/ov2/feilong/Megatron-Bridge}"
HERE="$REPO/examples/models/LLaVA_OV_2_1"
IMAGE="${IMAGE:-mbridge:qwen35}"
GPUS="${GPUS:-1,2,3,4,5,6,7}"; NPROC="${NPROC:-7}"
OVCK="${OVCK:-/ov2/feilong/ov2_quickstart/ov_encoder_p16m3_qwen3_mcore_tp1pp1}"
OUT="${OUT:-$REPO/ov2_1_4b/stage1_alignment_p16m3_adapter_only}"
GBS_TARGET="${GBS_TARGET:-256}"; TOTAL_SAMPLES="${TOTAL_SAMPLES:-558128}"
LR="${LR:-2e-5}"; MIN_LR="${MIN_LR:-1e-6}"; WARMUP_FRAC="${WARMUP_FRAC:-0.002}"; CLIP_GRAD="${CLIP_GRAD:-1.0}"
SAVE_EVERY="${SAVE_EVERY:-200}"; LOG_EVERY="${LOG_EVERY:-10}"; KEEP_LAST="${KEEP_LAST:-5}"
DATA="${DATA:-/vlm/data/blip_laion_cc_sbu_558k_wds}"
mkdir -p "$OUT"
echo "[stage1] out=$OUT gbs_target=$GBS_TARGET lr=$LR->cosine->$MIN_LR clip=$CLIP_GRAD (Adam, GELU adapter)"
docker rm -f ov2_s1 2>/dev/null || true
docker run -d --name ov2_s1 --gpus "\"device=$GPUS\"" --ipc=host --shm-size=32g \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e OVCK="$OVCK" -e OUT="$OUT" -e GBS_TARGET="$GBS_TARGET" -e TOTAL_SAMPLES="$TOTAL_SAMPLES" \
  -e LR="$LR" -e MIN_LR="$MIN_LR" -e WARMUP_FRAC="$WARMUP_FRAC" -e CLIP_GRAD="$CLIP_GRAD" \
  -e SAVE_EVERY="$SAVE_EVERY" -e LOG_EVERY="$LOG_EVERY" -e KEEP_LAST="$KEEP_LAST" -e DATA="$DATA" \
  -e PYTHONPATH="$REPO/3rdparty/Megatron-LM:$REPO/src:$REPO/aiak_shim" \
  -v /ov2:/ov2 -v /vlm:/vlm -w "$REPO" \
  "$IMAGE" bash -lc "torchrun --standalone --nproc_per_node=$NPROC $HERE/train_stage1_mp.py >> $OUT/train.log 2>&1"
echo "[stage1] launched container ov2_s1 -> tail -f $OUT/train.log"
