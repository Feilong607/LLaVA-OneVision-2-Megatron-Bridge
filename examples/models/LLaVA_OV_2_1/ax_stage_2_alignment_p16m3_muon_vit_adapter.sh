#!/usr/bin/env bash
# =============================================================================
# LLaVA-OneVision-2.1 4B (p16m3) · Stage-2 SFT · vit+adapter (Megatron-Bridge)
# Aligned to AIAK date0523_ax_stage_2_alignment_p16m3_4n_muon_true_vit_adapter_from_muon_vit_adapter_lr2e5.sh:
#   trainable adapter+vision_model (LLM frozen) · Muon(mom0.95,ns5,rms0.2,β0.9/0.99,eps1e-5) ·
#   lr 2e-5 constant (min-lr==lr, warmup 0) · clip-grad 1.0 · gbs 128 (mbs1+grad-accum) ·
#   next-token label shift · token-weighted loss · llava_next 780k non-packed · 1 epoch · bf16, no FSDP.
# Loads the colleague's Muon-trained vit+adapter stage-1 ckpt (model-only).
# =============================================================================
set -euo pipefail
REPO="${REPO:-/ov2/feilong/Megatron-Bridge}"
HERE="$REPO/examples/models/LLaVA_OV_2_1"
IMAGE="${IMAGE:-mbridge:qwen35}"
GPUS="${GPUS:-1,2,3,4,5,6,7}"; NPROC="${NPROC:-7}"
STAGE1_CKPT="${STAGE1_CKPT:-/vlm/yinxie/code/OV2/OV2_public_main/checkpoints/date0513-corrected-muon-stage1-vit-adapter/date0511_ax_stage_1_alignment_p16m3_packed_new16_muon/release/mp_rank_00/model_optim_rng.pt}"
OUT="${OUT:-$REPO/ov2_1_4b/stage2_alignment_p16m3_muon_vit_adapter}"
DATA="${DATA:-/vlm/data/llava_next_full_mega}"
SEQ="${SEQ:-32000}"; ACCUM="${ACCUM:-18}"            # gbs = NPROC*ACCUM (7*18=126 ~ ref 128)
LR="${LR:-2e-5}"; WARMUP="${WARMUP:-0}"; CLIP_GRAD="${CLIP_GRAD:-1.0}"; MUON_RMS="${MUON_RMS:-0.2}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-779111}"
SAVE_EVERY="${SAVE_EVERY:-500}"; LOG_EVERY="${LOG_EVERY:-10}"; KEEP_LAST="${KEEP_LAST:-5}"
mkdir -p "$OUT"
echo "[stage2] out=$OUT load=$STAGE1_CKPT gbs=$((NPROC*ACCUM)) lr=$LR(const) clip=$CLIP_GRAD muon_rms=$MUON_RMS"
docker rm -f ov2_s2 2>/dev/null || true
docker run -d --name ov2_s2 --gpus "\"device=$GPUS\"" --ipc=host --shm-size=32g \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e STAGE1_CKPT="$STAGE1_CKPT" -e OUT="$OUT" -e DATA="$DATA" -e SEQ="$SEQ" -e ACCUM="$ACCUM" \
  -e LR="$LR" -e WARMUP="$WARMUP" -e CLIP_GRAD="$CLIP_GRAD" -e MUON_RMS="$MUON_RMS" -e TOTAL_SAMPLES="$TOTAL_SAMPLES" \
  -e SAVE_EVERY="$SAVE_EVERY" -e LOG_EVERY="$LOG_EVERY" -e KEEP_LAST="$KEEP_LAST" \
  -e PYTHONPATH="$REPO/3rdparty/Megatron-LM:$REPO/src:$REPO/aiak_shim" \
  -v /ov2:/ov2 -v /vlm:/vlm -w "$REPO" \
  "$IMAGE" bash -lc "torchrun --standalone --nproc_per_node=$NPROC $HERE/train_stage2_mp.py >> $OUT/train.log 2>&1"
echo "[stage2] launched container ov2_s2 -> tail -f $OUT/train.log"
