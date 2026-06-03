#!/usr/bin/env bash
# =============================================================================
# LLaVA-OneVision-2.1 4B (p16m33) · Stage-2 SFT · true vit + adapter (Muon)
# BRIDGE-NATIVE: driven entirely by scripts/training/run_recipe.py (provider +
# ov2_step + EnergonProvider + recipe). Mirrors ax_stage_1_alignment_p16m3_adapter_only.sh.
# Model: Qwen3-4B LLM (frozen) + OV2.1 p16m33 encoder (TRAINED) + m33 adapter (TRAINED).
# AIAK stage-2 (date0523): Muon(momentum 0.95, ns-steps 5, matched-adamw-rms 0.2) + AdamW
#   (0.9,0.99,eps1e-5) for scalar/1-D params · lr 2e-5 CONSTANT (cosine flat, warmup-frac 0) ·
#   clip 1.0 · wd 0 · gbs 128 (mbs1 + grad-accum) · non-packed LLaVA-Next 780k (MultiMixQASample,
#   OFFLINE_PACKING_BMR=0) · bf16 · 1 epoch (6094 steps).
# Init: loads a TRAINED stage-1 checkpoint MODEL-ONLY via checkpoint.pretrained_checkpoint
#   (== AIAK '--load <stage1> --no-load-optim --no-load-rng'); native torch_dist save +
#   per-DP-rank dataloader-state save (dataset.dataloader_save) for resume.
#
# PREREQUISITES (this is just the launcher, like the stage-1 script):
#   1. recipes/ov2/ov2.py must define `ov2_1_stage2_vit_adapter_muon_config` (Muon optimizer,
#      freeze LLM + train vision_model + adapter; set use_distributed_optimizer=False on BOTH
#      cfg.optimizer and cfg.ddp for the layer-wise Muon DDP path), registered in recipes/ov2/__init__.
#   2. IMAGE must have NVIDIA-NeMo emerging_optimizers v0.2.0 (Muon) -> mbridge:qwen35-muon.
#   3. INIT_CKPT must point at a trained stage-1 output dir (run the stage-1 script first).
# =============================================================================
set -euo pipefail
REPO="${REPO:-/ov2/feilong/gb200/Megatron-Bridge}"
IMAGE="${IMAGE:-mbridge:qwen35-muon}"     # Muon needs emerging_optimizers (NVIDIA-NeMo v0.2.0)
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"; NPROC="${NPROC:-8}"
DATA_PATH="${DATA_PATH:-/vlm/data/llava_next_full_mega}"
INIT_CKPT="${INIT_CKPT:-/ov2/feilong/gb200/results/ov2_1_stage1_native}"   # trained stage-1 (model-only load)
SAVE="${SAVE:-/ov2/feilong/gb200/results/ov2_1_stage2_native}"
ITERS="${ITERS:-6094}"          # 1 epoch over 780k @ gbs 128
LOG_EVERY="${LOG_EVERY:-1}"; SAVE_EVERY="${SAVE_EVERY:-2000}"
mkdir -p "$SAVE"
docker rm -f ov2_s2 2>/dev/null || true
# GPU access on these hosts (A100-22/26): `--privileged` is needed for NVML init, and the docker
# daemon rejects `--gpus "device=..."` ("cannot set both Count and DeviceIDs"). Use `--gpus all`
# + CUDA_VISIBLE_DEVICES for device selection instead.
docker run -d --name ov2_s2 --privileged --gpus all -e CUDA_VISIBLE_DEVICES="$GPUS" --ipc=host --shm-size=32g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e PYTHONPATH="$REPO/src:$REPO/3rdparty/Megatron-LM:$REPO/aiak_shim" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e OMP_NUM_THREADS=8 \
  -v /ov2:/ov2 -v /vlm:/vlm -w "$REPO" \
  "$IMAGE" bash -lc "python -m torch.distributed.run --nproc_per_node=$NPROC scripts/training/run_recipe.py \
     --recipe ov2_1_stage2_vit_adapter_muon_config --dataset vlm-energon --step_func ov2_step \
     dataset.path=$DATA_PATH \
     checkpoint.save=$SAVE checkpoint.load=$SAVE checkpoint.pretrained_checkpoint=$INIT_CKPT \
     dataset.dataloader_save=$SAVE \
     checkpoint.save_interval=$SAVE_EVERY train.train_iters=$ITERS \
     validation.eval_iters=0 logger.log_interval=$LOG_EVERY \
     > $SAVE/train.log 2>&1"
echo "[ov2-native] launched ov2_s2 ($ITERS iters, $NPROC GPUs) -> tail -f $SAVE/train.log"
