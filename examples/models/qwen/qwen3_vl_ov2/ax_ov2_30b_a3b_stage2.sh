#!/usr/bin/env bash
# =============================================================================
# OV2-30B-A3B (Qwen3-30B-A3B MoE + OV2 vision + m33 adapter) · Stage-2 SFT
# Trains vision tower + adapter (LLM FROZEN), distributed Muon, EP8. Chains from a TRAINED stage-1.
# Bridge-native: run_recipe.py + ov2_35b_a3b_stage2. Single node (--standalone) OR multi-node (LIST_IP).
#
# INIT_CKPT must point at a trained stage-1 output dir (run ax_ov2_30b_a3b_stage1.sh first); it loads
# MODEL-ONLY via checkpoint.pretrained_checkpoint (== AIAK --load <stage1> --no-load-optim --no-load-rng).
# Code = OV2 multi-backbone (refactor) repo; outputs go under /ov2/feilong/gb200/ckpts_video_sft.
# =============================================================================
set -euo pipefail

REPO="${REPO:-/ov2/feilong/gb200/Megatron-Bridge-refactor}"
IMAGE="${IMAGE:-mbridge:qwen35-muon}"                            # stage-2 = distributed Muon (needs emerging_optimizers)
DATA_PATH="${DATA_PATH:-/vlm/data/llava_next_full_mega}"         # stage-2 SFT data (LLaVA-Next 780k)
INIT_CKPT="${INIT_CKPT:-/ov2/feilong/gb200/ckpts_video_sft/ov2_30b_a3b_stage1}"  # trained stage-1 (model-only load)
SAVE="${SAVE:-/ov2/feilong/gb200/ckpts_video_sft/ov2_30b_a3b_stage2}"
NPROC="${NPROC:-8}"
ITERS="${ITERS:-6094}"          # 1 epoch over 780k @ gbs 128
LOG_EVERY="${LOG_EVERY:-1}"; SAVE_EVERY="${SAVE_EVERY:-500}"

if [[ "$INIT_CKPT" != "null" && -n "$INIT_CKPT" && ! -e "$INIT_CKPT" ]]; then
  echo "[ov2-30b-stage2] ERROR: INIT_CKPT=$INIT_CKPT not found. Run ax_ov2_30b_a3b_stage1.sh first, or set INIT_CKPT=null to start from the OV2-30B base." >&2
  exit 1
fi

if [[ -n "${LIST_IP:-}" ]]; then read -ra list_ip <<< "$LIST_IP"; else list_ip=(); fi
NNODES=${#list_ip[@]}
if [[ "$NNODES" -le 1 ]]; then
  RDZV="--standalone"
else
  MASTER_ADDR="${list_ip[0]}"; MASTER_PORT="${MASTER_PORT:-26042}"
  CURRENT_IP="$(hostname -I | awk '{print $1}')"; NODE_RANK=-1
  for i in "${!list_ip[@]}"; do [[ "${list_ip[$i]}" == "$CURRENT_IP" ]] && NODE_RANK=$i && break; done
  [[ "$NODE_RANK" -eq -1 ]] && { echo "ERROR: $CURRENT_IP not in LIST_IP (${list_ip[*]})"; exit 1; }
  RDZV="--nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT"
fi

NCCL_ENV="-e NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0} \
  -e NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8} \
  -e NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3} -e NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}"

PRELOAD=""; [[ "$INIT_CKPT" != "null" && -n "$INIT_CKPT" ]] && PRELOAD="checkpoint.pretrained_checkpoint=$INIT_CKPT"
# RECOMPUTE=1 enables LLM activation recompute (cuts peak mem ~71GB->~35GB at ~30% speed cost) — use
# to fit busy / coexisting GPUs (e.g. sharing a node with another job that left <71GB free).
RECOMPUTE_FLAG=""; [[ "${RECOMPUTE:-0}" == "1" ]] && RECOMPUTE_FLAG="model.recompute_activations=true"

mkdir -p "$SAVE"; docker rm -f ov2_30b_s2 2>/dev/null || true
echo "[ov2-30b-stage2] nnodes=${NNODES:-1} init=$INIT_CKPT save=$SAVE"
docker run -d --name ov2_30b_s2 --network=host --privileged --gpus all -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  --ipc=host --shm-size=32g --ulimit memlock=-1 --ulimit stack=67108864 $NCCL_ENV \
  -e PYTHONPATH="$REPO/src:$REPO/3rdparty/Megatron-LM:$REPO/aiak_shim" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e OMP_NUM_THREADS=8 \
  -v /ov2:/ov2 -v /vlm:/vlm -w "$REPO" "$IMAGE" bash -lc "
    python -m torch.distributed.run $RDZV --nproc_per_node=$NPROC scripts/training/run_recipe.py \
      --recipe ov2_35b_a3b_stage2 --dataset vlm-energon --step_func ov2_step \
      dataset.path=$DATA_PATH $PRELOAD $RECOMPUTE_FLAG \
      checkpoint.save=$SAVE checkpoint.load=$SAVE dataset.dataloader_save=$SAVE \
      checkpoint.save_interval=$SAVE_EVERY train.train_iters=$ITERS \
      validation.eval_iters=0 logger.log_interval=$LOG_EVERY \
      > $SAVE/train_node${NODE_RANK:-0}.log 2>&1"
echo "[ov2-30b-stage2] launched -> tail -f $SAVE/train_node*.log  (loss prints on the LAST node)"
