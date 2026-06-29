#!/usr/bin/env bash
# =============================================================================
# OV2 · Qwen3.5-35B-A3B (qwen3_5_moe_text: GatedDeltaNet hybrid + 256-expert MoE + MTP)
#       + OneVision p16m33 encoder + merge3 adapter · STAGE-2 vit+adapter SFT
# TRAIN vision tower + adapter (LLM FROZEN), distributed Muon, EP8, LLaVA-Next 780k. Chains from the
# TRAINED stage-1 ckpt via OV2_SKIP_BASE_STITCH=1 (all weights from OV2_INIT_CKPT). Parallel to the
# qwen3_vl_ov2/A800 stage-2; the Qwen3.5 stack stays fully separate (do NOT cross recipes/ckpts).
# MoE stage-2 KEEPS Muon by default (LLM/experts frozen -> Muon touches only dense vision+adapter 2-D
# matrices, EP8 all-to-all fires cleanly; recipe-verified). OV2_STAGE2_ADAMW=1 forces AdamW instead.
# GPU PASSTHROUGH (A100-18/A800): docker --gpus hook is BROKEN -> --privileged -v /dev:/dev + host driver libs.
# =============================================================================
set -euo pipefail

REPO="${REPO:-/ov2/feilong/gb200/Megatron-Bridge}"
bash "$REPO/3rdparty/apply_megatron_patch.sh"   # OV2 mcore patch (apply_rotary_fn) -- REQUIRED by the OneVision tower
IMAGE="${IMAGE:-mbridge:qwen35}"
DATA_PATH="${DATA_PATH:-/vlm/data/llava_next_full_mega}"          # stage-2 SFT data (LLaVA-Next 780k)
SAVE="${SAVE:-/ov2/feilong/gb200/ckpts_video_sft/ov2_qwen35_35b_a3b_p16m33_stage2}"
# Chain from the TRAINED stage-1 (mrope) ckpt: all weights (LLM+vision+adapter) load from here.
INIT="${INIT:-/ov2/feilong/gb200/ckpts_video_sft/ov2_qwen35_35b_a3b_p16m33_stage1_mrope}"
OV2_HF_PROC_QWEN35_P16M33="${OV2_HF_PROC_QWEN35_P16M33:-/ov2/pretrain_models/llava_onevision2/llava_onevision2_qwen35_35b_a3b_p16_m33/auto_model}"
NPROC="${NPROC:-8}"
ITERS="${ITERS:-6094}"                          # 1 epoch over 780k @ gbs 128
LOG_EVERY="${LOG_EVERY:-1}"; SAVE_EVERY="${SAVE_EVERY:-250}"
# A100-18/A800 docker --gpus hook broken -> privileged + raw /dev + host driver libs (NVML/libcuda/PTX-JIT).
_NVLIBS=""
for _f in /usr/lib/x86_64-linux-gnu/libcuda.so.*.* /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.*.* /usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.*.* /usr/lib/x86_64-linux-gnu/libnvidia-nvvm.so.*.* /usr/lib/x86_64-linux-gnu/libcudadebugger.so.*.*; do
  [[ -s "$_f" ]] && _NVLIBS="$_NVLIBS -v $_f:$_f:ro"
done
GPU_ARGS="${GPU_ARGS:---privileged -v /dev:/dev $_NVLIBS}"

# node list: node-0 first = rendezvous master. Single node => leave LIST_IP unset (--standalone).
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

# stage-2 trains vit+adapter (LLM frozen) but backprop traverses all 40 LLM layers -> recompute attn to fit.
RECOMPUTE_FLAG=""; [[ "${RECOMPUTE:-1}" == "1" ]] && RECOMPUTE_FLAG="model.recompute_activations=true"
# MoE stage-2 keeps distributed Muon by default (frozen experts -> Muon only on dense vision+adapter).
MUON_NOSPLIT="optimizer.muon_split_qkv=false"
# OV2_TIMING_LOG_LEVEL=1|2 -> print Megatron per-rank (min,max) timing breakdown (forward/backward-compute,
# optimizer-inner-step, all-grads-sync, batch-generator...). Empty/0 (default) = unchanged: only iteration
# time, no extra barrier overhead. MUST be a CONFIG override (logger.timing_log_level): the Timers object is
# built ONCE from cfg in GlobalState and returns no-op DummyTimers for any op above the level -> editing
# train_utils.py timers_to_log can NOT enable it. log_option stays the inherited "minmax" default.
# NOTE: log_timers_to_tensorboard MUST be false here. With it True, training_log() calls
# timers.write_to_wandb/mlflow/comet(reset=True) which reset the per-iter timers (even when
# the writer is None) BEFORE the console timers.log() -> the (min,max) block prints nothing.
TIMING_FLAG="${OV2_TIMING_LOG_LEVEL:+logger.timing_log_level=$OV2_TIMING_LOG_LEVEL logger.log_timers_to_tensorboard=false}"
mkdir -p "$SAVE"; docker rm -f ov2_qwen35_s2 2>/dev/null || true
echo "[ov2-qwen35-stage2] nnodes=${NNODES:-1} save=$SAVE init=$INIT image=$IMAGE adamw=${OV2_STAGE2_ADAMW:-0} gpu_args='$GPU_ARGS'"
# shellcheck disable=SC2086
docker run -d --name ov2_qwen35_s2 $GPU_ARGS --network=host --ipc=host --shm-size=32g \
  --ulimit memlock=-1 --ulimit stack=67108864 $NCCL_ENV \
  -e PYTHONPATH="$REPO/src:$REPO/3rdparty/Megatron-LM:$REPO/aiak_shim" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e OMP_NUM_THREADS=8 \
  -e OV2_SKIP_BASE_STITCH=1 -e OV2_INIT_CKPT="$INIT" \
  -e OV2_HF_PROC_QWEN35_P16M33="$OV2_HF_PROC_QWEN35_P16M33" \
  -e OV2_MOE_PERMUTE_FUSION="${OV2_MOE_PERMUTE_FUSION:-0}" \
  -e OV2_STAGE2_ADAMW="${OV2_STAGE2_ADAMW:-0}" \
  -e OV2_RECOMPUTE_FULL="${OV2_RECOMPUTE_FULL:-0}" \
  -e OV2_TIMING_PRINT_INTERVAL="${OV2_TIMING_PRINT_INTERVAL:-50}" \
  -v /ov2:/ov2 -v /vlm:/vlm -w "$REPO" "$IMAGE" bash -lc "
    bash 3rdparty/apply_megatron_patch.sh;
    python -m torch.distributed.run $RDZV --nproc_per_node=$NPROC scripts/training/run_recipe.py \
      --recipe ov2_qwen35_35b_a3b_stage2 --dataset vlm-energon --step_func ov2_step \
      dataset.path=$DATA_PATH $RECOMPUTE_FLAG $MUON_NOSPLIT \
      checkpoint.pretrained_checkpoint=$INIT \
      checkpoint.save=$SAVE checkpoint.load=$SAVE dataset.dataloader_save=$SAVE \
      checkpoint.save_interval=$SAVE_EVERY train.train_iters=$ITERS \
      validation.eval_iters=0 logger.log_interval=$LOG_EVERY $TIMING_FLAG \
      > $SAVE/train_node${NODE_RANK:-0}.log 2>&1"
echo "[ov2-qwen35-stage2] launched -> tail -f $SAVE/train_node*.log"
