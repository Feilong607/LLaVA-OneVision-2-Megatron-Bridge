#!/usr/bin/env bash
# =============================================================================
# OV2-4B (Qwen3-4B, p16m33) · MID-TRAIN (stage 1.5) · FULL model (LLM + vision + adapter), Muon
# Bridge-native (REFACTOR repo): run_recipe.py + ov2_4b_midtrain. Single node (--standalone) OR multi-node (LIST_IP).
#
# Mirrors AIAK date0528: --trainable-modules language_model adapter vision_model; distributed Muon
# (momentum 0.95, ns-steps 5, matched-adamw-rms 0.15), lr 2e-5 constant, gbs 128, recompute ON.
# Dense backbone -> Muon (use_distributed_optimizer=False, layer-wise path); needs the emerging_optimizers
# image (mbridge:qwen35-muon). Chains from a TRAINED stage-2 via INIT_CKPT (model-only load).
# =============================================================================
set -euo pipefail

REPO="${REPO:-/ov2/feilong/gb200/Megatron-Bridge-refactor}"     # midtrain recipe lives in the refactor repo
IMAGE="${IMAGE:-mbridge:qwen35-muon}"                            # dense 4B midtrain uses Muon -> emerging_optimizers
DATA_PATH="${DATA_PATH:-/vlm/data/llava_next_full_mega}"
INIT_CKPT="${INIT_CKPT:-/ov2/feilong/gb200/ckpts_video_sft/llava_ov2_4b_stage2}"  # trained stage-2 (model-only)
SAVE="${SAVE:-/ov2/feilong/gb200/ckpts_video_sft/ov2_4b_midtrain}"
NPROC="${NPROC:-8}"
ITERS="${ITERS:-6094}"          # 1 epoch over 780k @ gbs 128 (override for the real mid-train corpus)
LOG_EVERY="${LOG_EVERY:-1}"; SAVE_EVERY="${SAVE_EVERY:-500}"

if [[ "$INIT_CKPT" != "null" && -n "$INIT_CKPT" && ! -e "$INIT_CKPT" ]]; then
  echo "[ov2-4b-midtrain] ERROR: INIT_CKPT=$INIT_CKPT not found. Run stage-2 first, or set INIT_CKPT=null for a stitch-base smoke." >&2
  exit 1
fi

if [[ -n "${LIST_IP:-}" ]]; then read -ra list_ip <<< "$LIST_IP"; else list_ip=(); fi
NNODES=${#list_ip[@]}
if [[ "$NNODES" -le 1 ]]; then
  RDZV="--standalone"
else
  MASTER_ADDR="${list_ip[0]}"; MASTER_PORT="${MASTER_PORT:-26046}"
  CURRENT_IP="$(hostname -I | awk '{print $1}')"; NODE_RANK=-1
  for i in "${!list_ip[@]}"; do [[ "${list_ip[$i]}" == "$CURRENT_IP" ]] && NODE_RANK=$i && break; done
  [[ "$NODE_RANK" -eq -1 ]] && { echo "ERROR: $CURRENT_IP not in LIST_IP (${list_ip[*]})"; exit 1; }
  RDZV="--nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT"
fi

NCCL_ENV="-e NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0} \
  -e NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8} \
  -e NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3} -e NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}"

PRELOAD=""; [[ "$INIT_CKPT" != "null" && -n "$INIT_CKPT" ]] && PRELOAD="checkpoint.pretrained_checkpoint=$INIT_CKPT"

mkdir -p "$SAVE"; docker rm -f ov2_4b_mid 2>/dev/null || true
echo "[ov2-4b-midtrain] nnodes=${NNODES:-1} init=$INIT_CKPT save=$SAVE (FULL model, Muon)"
docker run -d --name ov2_4b_mid --network=host --privileged --gpus all -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  --ipc=host --shm-size=32g --ulimit memlock=-1 --ulimit stack=67108864 $NCCL_ENV \
  -e PYTHONPATH="$REPO/src:$REPO/3rdparty/Megatron-LM:$REPO/aiak_shim" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e OMP_NUM_THREADS=8 \
  -v /ov2:/ov2 -v /vlm:/vlm -w "$REPO" "$IMAGE" bash -lc "
    python -m torch.distributed.run $RDZV --nproc_per_node=$NPROC scripts/training/run_recipe.py \
      --recipe ov2_4b_midtrain --dataset vlm-energon --step_func ov2_step \
      dataset.path=$DATA_PATH $PRELOAD \
      checkpoint.save=$SAVE checkpoint.load=$SAVE dataset.dataloader_save=$SAVE \
      checkpoint.save_interval=$SAVE_EVERY train.train_iters=$ITERS \
      validation.eval_iters=0 logger.log_interval=$LOG_EVERY \
      > $SAVE/train_node${NODE_RANK:-0}.log 2>&1"
echo "[ov2-4b-midtrain] launched -> tail -f $SAVE/train_node*.log"
