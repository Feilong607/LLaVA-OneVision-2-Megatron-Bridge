#!/usr/bin/env bash
# =============================================================================
# LLaVA-OneVision-2.1 4B (p16m33) · Stage-1 Alignment · adapter-only
# BRIDGE-NATIVE: driven entirely by scripts/training/run_recipe.py (provider +
# ov2_step + EnergonProvider + recipe). Replaces the old standalone train_stage1_mp.py.
# Model: Qwen3-4B LLM (frozen) + OV2.1 p16m33 encoder (frozen) + m33 adapter (trained).
# AIAK stage-1: AdamW(0.9,0.99,eps1e-5,wd0) · lr 2e-5 -> cosine -> 1e-6 (warmup-frac 0.002) ·
#   clip 1.0 · gbs 256 (mbs1 + grad-accum) · GELU adapter · token-weighted loss · bf16, 1 epoch / 558k.
# Weights load via the provider's stitch pre_wrap_hook (LLM+vision); native torch_dist checkpoints +
# per-DP-rank dataloader-state save (dataset.dataloader_save) for resume.
# =============================================================================
set -euo pipefail
REPO="${REPO:-/ov2/feilong/gb200/Megatron-Bridge}"
IMAGE="${IMAGE:-mbridge:qwen35}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"; NPROC="${NPROC:-8}"
DATA_PATH="${DATA_PATH:-/vlm/data/blip_laion_cc_sbu_558k_wds}"
SAVE="${SAVE:-/ov2/feilong/gb200/results/ov2_1_stage1_native}"
ITERS="${ITERS:-2181}"          # 1 epoch over 558k @ gbs 256
LOG_EVERY="${LOG_EVERY:-1}"; SAVE_EVERY="${SAVE_EVERY:-200}"
mkdir -p "$SAVE"
docker rm -f ov2_s1 2>/dev/null || true
docker run -d --name ov2_s1 --gpus "\"device=$GPUS\"" --ipc=host --shm-size=32g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e PYTHONPATH="$REPO/src:$REPO/3rdparty/Megatron-LM:$REPO/aiak_shim" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e OMP_NUM_THREADS=8 \
  -v /ov2:/ov2 -v /vlm:/vlm -w "$REPO" \
  "$IMAGE" bash -lc "python -m torch.distributed.run --nproc_per_node=$NPROC scripts/training/run_recipe.py \
     --recipe ov2_1_stage1_adapter_only_config --dataset vlm-energon --step_func ov2_step \
     dataset.path=$DATA_PATH \
     checkpoint.save=$SAVE checkpoint.load=$SAVE dataset.dataloader_save=$SAVE \
     checkpoint.save_interval=$SAVE_EVERY train.train_iters=$ITERS \
     validation.eval_iters=0 logger.log_interval=$LOG_EVERY \
     > $SAVE/train.log 2>&1"
echo "[ov2-native] launched ov2_s1 ($ITERS iters, $NPROC GPUs) -> tail -f $SAVE/train.log"
