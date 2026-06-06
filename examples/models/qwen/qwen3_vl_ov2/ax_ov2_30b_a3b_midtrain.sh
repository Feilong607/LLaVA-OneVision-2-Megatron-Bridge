#!/usr/bin/env bash
# =============================================================================
# OV2-30B-A3B (Qwen3-30B-A3B MoE) · MID-TRAIN (stage 1.5) · FULL model (LLM + vision + adapter)
# Bridge-native: run_recipe.py + ov2_35b_a3b_midtrain. Single node (--standalone) OR multi-node (LIST_IP).
#
# Mirrors AIAK date0528 mid-train (--trainable-modules language_model adapter vision_model), full-model
# SFT, lr 2e-5, gbs 128, activation recompute ON. Unlike stage-2 (vit+adapter only), mid-train UNFREEZES
# the LLM -> the MoE experts become trainable, so distributed Muon would deadlock the EP backward
# all-to-all (the stage-2 hang). The recipe therefore AUTO-USES AdamW(distopt=True) for this MoE backbone
# (no OV2_STAGE2_ADAMW needed). dense 4B mid-train keeps Muon.
#
# !! MEMORY: full-model 30B at seq 32000 is MUCH heavier than the vit+adapter stage-2 (optimizer state for
#    ALL params). recompute is ON, but EP8/2-node may still OOM -> you likely need TP>1 (pass TP via the
#    recipe / CLI model.tensor_model_parallel_size) and/or more nodes. Validate on GPU and watch memory.
#    chain from a TRAINED stage-2 via INIT_CKPT (model-only load).
# =============================================================================
set -euo pipefail

REPO="${REPO:-/ov2/feilong/gb200/Megatron-Bridge-refactor}"
IMAGE="${IMAGE:-mbridge:qwen35-muon}"                            # superset image (AdamW path used for MoE)
DATA_PATH="${DATA_PATH:-/vlm/data/llava_next_full_mega}"         # mid-train SFT data (override with the real corpus)
INIT_CKPT="${INIT_CKPT:-/ov2/feilong/gb200/ckpts_video_sft/ov2_30b_a3b_stage2}"  # trained stage-2 (model-only load)
SAVE="${SAVE:-/ov2/feilong/gb200/ckpts_video_sft/ov2_30b_a3b_midtrain}"
NPROC="${NPROC:-8}"
ITERS="${ITERS:-6094}"          # 1 epoch over 780k @ gbs 128 (override for the real mid-train corpus)
LOG_EVERY="${LOG_EVERY:-1}"; SAVE_EVERY="${SAVE_EVERY:-500}"

if [[ "$INIT_CKPT" != "null" && -n "$INIT_CKPT" && ! -e "$INIT_CKPT" ]]; then
  echo "[ov2-30b-midtrain] ERROR: INIT_CKPT=$INIT_CKPT not found. Run stage-2 first, or set INIT_CKPT=null to start from the OV2-30B base (smoke)." >&2
  exit 1
fi

if [[ -n "${LIST_IP:-}" ]]; then read -ra list_ip <<< "$LIST_IP"; else list_ip=(); fi
NNODES=${#list_ip[@]}
if [[ "$NNODES" -le 1 ]]; then
  RDZV="--standalone"
  # FULL-model 30B mid-train (LLM unfrozen) holds optimizer state for ALL params; single-node
  # EP8/TP1 at seq 32000 will very likely OOM. Prefer ax_ov2_30b_a3b_midtrain_2node.sh, raise TP,
  # or reduce seq/gbs. Set MIDTRAIN_OOM_OK=1 to proceed without this pause.
  echo "[ov2-30b-midtrain] WARNING: single-node (--standalone) full-model 30B mid-train will likely OOM at seq 32000; prefer the _2node wrapper or raise TP (set MIDTRAIN_OOM_OK=1 to silence)." >&2
  [[ "${MIDTRAIN_OOM_OK:-0}" == "1" ]] || sleep 5
else
  MASTER_ADDR="${list_ip[0]}"; MASTER_PORT="${MASTER_PORT:-26045}"
  CURRENT_IP="$(hostname -I | awk '{print $1}')"; NODE_RANK=-1
  for i in "${!list_ip[@]}"; do [[ "${list_ip[$i]}" == "$CURRENT_IP" ]] && NODE_RANK=$i && break; done
  [[ "$NODE_RANK" -eq -1 ]] && { echo "ERROR: $CURRENT_IP not in LIST_IP (${list_ip[*]})"; exit 1; }
  RDZV="--nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT"
fi

NCCL_ENV="-e NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0} \
  -e NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8} \
  -e NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3} -e NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}"

PRELOAD=""; [[ "$INIT_CKPT" != "null" && -n "$INIT_CKPT" ]] && PRELOAD="checkpoint.pretrained_checkpoint=$INIT_CKPT"

mkdir -p "$SAVE"; docker rm -f ov2_30b_mid 2>/dev/null || true
echo "[ov2-30b-midtrain] nnodes=${NNODES:-1} init=$INIT_CKPT save=$SAVE (FULL model, AdamW for MoE)"
docker run -d --name ov2_30b_mid --network=host --privileged --gpus all -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  --ipc=host --shm-size=32g --ulimit memlock=-1 --ulimit stack=67108864 $NCCL_ENV \
  -e PYTHONPATH="$REPO/src:$REPO/3rdparty/Megatron-LM:$REPO/aiak_shim" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e OMP_NUM_THREADS=8 \
  -v /ov2:/ov2 -v /vlm:/vlm -w "$REPO" "$IMAGE" bash -lc "
    python -m torch.distributed.run $RDZV --nproc_per_node=$NPROC scripts/training/run_recipe.py \
      --recipe ov2_35b_a3b_midtrain --dataset vlm-energon --step_func ov2_step \
      dataset.path=$DATA_PATH $PRELOAD \
      checkpoint.save=$SAVE checkpoint.load=$SAVE dataset.dataloader_save=$SAVE \
      checkpoint.save_interval=$SAVE_EVERY train.train_iters=$ITERS \
      validation.eval_iters=0 logger.log_interval=$LOG_EVERY \
      > $SAVE/train_node${NODE_RANK:-0}.log 2>&1"
echo "[ov2-30b-midtrain] launched -> tail -f $SAVE/train_node*.log  (loss on the LAST node)"
