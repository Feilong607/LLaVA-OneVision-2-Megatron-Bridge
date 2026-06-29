#!/usr/bin/env bash
# =============================================================================
# OV2 · Qwen3.5-35B-A3B (qwen3_5_moe_text) + OneVision p16m33 · MID-TRAIN (full-model SFT)
# UNFREEZE the FULL model (LLM + vision + adapter). MoE backbone -> recipe AUTO-routes to AdamW
# (distopt) because trainable experts would deadlock Muon's EP backward all-to-all. Chains from the
# TRAINED stage-2 v2 ckpt via OV2_SKIP_BASE_STITCH=1 (all weights from OV2_INIT_CKPT).
# 35B full-unfreeze + AdamW on a single 8x80GB node is TIGHT: recompute_activations is on (recipe),
# and we ADD optimizer-cpu-offload (fraction 1.0) so the fp32 AdamW m/v/master don't OOM at EP8/DP1.
# Target box A100-34 (A800-80GB) where docker --gpus all WORKS (no A100-18 driver-lib workaround needed).
# =============================================================================
set -euo pipefail

REPO="${REPO:-/ov2/feilong/gb200/Megatron-Bridge}"
bash "$REPO/3rdparty/apply_megatron_patch.sh"
IMAGE="${IMAGE:-mbridge:qwen35}"                                  # AdamW path -> base image (no emerging-optimizers needed)
DATA_PATH="${DATA_PATH:-/vlm/data/llava_next_full_mega}"          # LLaVA-Next 780k (AIAK date0528 mid-train data)
SAVE="${SAVE:-/ov2/feilong/gb200/ckpts_video_sft/ov2_qwen35_35b_a3b_p16m33_midtrain_v2}"
INIT="${INIT:-/ov2/feilong/gb200/ckpts_video_sft/ov2_qwen35_35b_a3b_p16m33_stage2_muon_v2/iter_0006094}"
OV2_HF_PROC_QWEN35_P16M33="${OV2_HF_PROC_QWEN35_P16M33:-/ov2/pretrain_models/llava_onevision2/llava_onevision2_qwen35_35b_a3b_p16_m33/auto_model}"
NPROC="${NPROC:-8}"
ITERS="${ITERS:-6094}"
LOG_EVERY="${LOG_EVERY:-1}"; SAVE_EVERY="${SAVE_EVERY:-250}"
CNAME="${CNAME:-ov2_qwen35_mid}"

# A100-34: docker --gpus all works. (A100-18/A800 needs the driver-lib workaround -> override GPU_ARGS.)
GPU_ARGS="${GPU_ARGS:---gpus all}"

# multi-node rendezvous (single node default -> --standalone)
if [[ -n "${LIST_IP:-}" ]]; then read -ra list_ip <<< "$LIST_IP"; else list_ip=(); fi
NNODES=${#list_ip[@]}
if [[ "$NNODES" -le 1 ]]; then
  RDZV="--standalone"
else
  MASTER_ADDR="${list_ip[0]}"; MASTER_PORT="${MASTER_PORT:-26041}"
  CURRENT_IP="$(hostname -I | awk '{print $1}')"; NODE_RANK=-1
  for i in "${!list_ip[@]}"; do [[ "${list_ip[$i]}" == "$CURRENT_IP" ]] && NODE_RANK=$i && break; done
  [[ "$NODE_RANK" -eq -1 ]] && { echo "ERROR: $CURRENT_IP not in LIST_IP (${list_ip[*]})"; exit 1; }
  RDZV="--nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT"
fi

NCCL_ENV="-e NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0} \
  -e NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8} \
  -e NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3} -e NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}"

# recipe sets model.recompute_activations=true for midtrain; keep RECOMPUTE=1 -> also pass the override.
RECOMPUTE_FLAG=""; [[ "${RECOMPUTE:-1}" == "1" ]] && RECOMPUTE_FLAG="model.recompute_activations=true"
# optimizer-cpu-offload: mandatory for 35B full-unfreeze + AdamW at EP8/DP1 (else fp32 m/v OOM).
OFFLOAD_FLAG="optimizer.optimizer_cpu_offload=${OV2_OPT_OFFLOAD:-true} optimizer.optimizer_offload_fraction=${OV2_OFFLOAD_FRACTION:-1.0}"
TIMING_FLAG="${OV2_TIMING_LOG_LEVEL:+logger.timing_log_level=$OV2_TIMING_LOG_LEVEL logger.log_timers_to_tensorboard=false}"

mkdir -p "$SAVE"; docker rm -f "$CNAME" 2>/dev/null || true
echo "[ov2-qwen35-midtrain] nnodes=${NNODES:-1} save=$SAVE init=$INIT image=$IMAGE offload='$OFFLOAD_FLAG' gpu_args='$GPU_ARGS'"
# shellcheck disable=SC2086
docker run -d --name "$CNAME" $GPU_ARGS --network=host --ipc=host --shm-size=32g \
  --ulimit memlock=-1 --ulimit stack=67108864 $NCCL_ENV \
  -e PYTHONPATH="$REPO/src:$REPO/3rdparty/Megatron-LM:$REPO/aiak_shim" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e OMP_NUM_THREADS=8 \
  -e OV2_SKIP_BASE_STITCH=1 -e OV2_INIT_CKPT="$INIT" \
  -e OV2_HF_PROC_QWEN35_P16M33="$OV2_HF_PROC_QWEN35_P16M33" \
  -e OV2_MOE_PERMUTE_FUSION="${OV2_MOE_PERMUTE_FUSION:-0}" \
  -e PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  -e OV2_SEQ_LEN="${OV2_SEQ_LEN:-32768}" \
  -e OV2_RECOMPUTE_FULL="${OV2_RECOMPUTE_FULL:-0}" \
  -e OV2_TIMING_PRINT_INTERVAL="${OV2_TIMING_PRINT_INTERVAL:-50}" \
  -v /ov2:/ov2 -v /vlm:/vlm -w "$REPO" "$IMAGE" bash -lc "
    bash 3rdparty/apply_megatron_patch.sh;
    python -m torch.distributed.run $RDZV --nproc_per_node=$NPROC scripts/training/run_recipe.py \
      --recipe ov2_qwen35_35b_a3b_midtrain --dataset vlm-energon --step_func ov2_step \
      dataset.path=$DATA_PATH $RECOMPUTE_FLAG $OFFLOAD_FLAG \
      checkpoint.pretrained_checkpoint=$INIT \
      checkpoint.save=$SAVE checkpoint.load=$SAVE dataset.dataloader_save=$SAVE \
      checkpoint.save_interval=$SAVE_EVERY train.train_iters=$ITERS \
      validation.eval_iters=0 logger.log_interval=$LOG_EVERY $TIMING_FLAG \
      > $SAVE/train_node${NODE_RANK:-0}.log 2>&1"
echo "[ov2-qwen35-midtrain] launched -> tail -f $SAVE/train_node*.log"
