#!/usr/bin/env bash
# =============================================================================
# OV2-30B-A3B (Qwen3-30B-A3B MoE + OV2 vision encoder + m33 adapter) · Stage-1 alignment (p16m33)
# Adapter-only (LLM + vision FROZEN), AdamW, EP8. Bridge-native: run_recipe.py + ov2_35b_a3b_stage1.
# Single node (8 GPU, --standalone) OR multi-node (set LIST_IP). Run the SAME cmd on every node.
#
# Code lives in the OV2 multi-backbone (refactor) repo: it carries the qwen3-30b-a3b backbone, the
# EP-sharded stitch loader + per-expert->grouped remap, fp32 MoE routing, and the always-stitch hook.
# Generated checkpoints + logs go under /ov2/feilong/gb200/ckpts_video_sft.
# Multi-tenant cluster: only launch on nodes whose GPUs are free (check nvidia-smi first).
# =============================================================================
set -euo pipefail

REPO="${REPO:-/ov2/feilong/gb200/Megatron-Bridge}"   # OV2 multi-backbone code + ov2_35b_a3b_* recipes
bash "$REPO/3rdparty/apply_megatron_patch.sh" 2>/dev/null || true   # fresh-clone safety: apply OV2 mcore submodule patch (apply_rotary_fn hook)
IMAGE="${IMAGE:-mbridge:qwen35}"                                # stage-1 = AdamW (no Muon needed)
DATA_PATH="${DATA_PATH:-/vlm/data/blip_laion_cc_sbu_558k_wds}"  # stage-1 alignment data (558k)
SAVE="${SAVE:-/ov2/feilong/gb200/ckpts_video_sft/ov2_30b_a3b_p16m33_stage1}"
# Combined p16m33 base (30B-A3B LLM + patch16 vision tower + fresh merge3 adapter), built via
# A800/convert from_base --vision_hf. It is a Bridge torch_dist ckpt -> load via pretrained_checkpoint
# with OV2_SKIP_BASE_STITCH=1 (skip the AIAK release/mp_rank stitch, which only reads .pt format).
INIT="${INIT:-/ov2/pretrain_models/llava_onevision2/llava_onevision2_30b_a3b_p16_m33/stage_0_tp1_pp1_ep8}"
NPROC="${NPROC:-8}"
ITERS="${ITERS:-2181}"          # 1 epoch over 558k @ gbs 256
LOG_EVERY="${LOG_EVERY:-1}"; SAVE_EVERY="${SAVE_EVERY:-250}"

# node list: node-0 first = rendezvous master. Single node => leave LIST_IP unset (uses --standalone).
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

# this cluster's IB fabric (mlx5_1..8 GPU-attached; mlx5_0 is mgmt). Only used for multi-node NCCL.
NCCL_ENV="-e NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0} \
  -e NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8} \
  -e NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3} -e NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}"

# RECOMPUTE=1 (default) enables LLM activation recompute. stage-1 freezes LLM+vision, but backprop
# still traverses all 48 LLM layers to reach the adapter, so at SEQ_LEN=32000 the attention
# activations OOM an 80GB card without it (same as stage-2). SELECTIVE core_attn by default;
# OV2_RECOMPUTE_FULL=1 forces full-layer recompute.
RECOMPUTE_FLAG=""; [[ "${RECOMPUTE:-1}" == "1" ]] && RECOMPUTE_FLAG="model.recompute_activations=true"
mkdir -p "$SAVE"; docker rm -f ov2_30b_p16m33_s1 2>/dev/null || true
echo "[ov2-30b-stage1] nnodes=${NNODES:-1} save=$SAVE repo=$REPO"
docker run -d --name ov2_30b_p16m33_s1 --network=host --privileged --gpus all -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  --ipc=host --shm-size=32g --ulimit memlock=-1 --ulimit stack=67108864 $NCCL_ENV \
  -e PYTHONPATH="$REPO/src:$REPO/3rdparty/Megatron-LM:$REPO/aiak_shim" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e OMP_NUM_THREADS=8 \
  -e OV2_SKIP_BASE_STITCH=1 -e OV2_INIT_CKPT="$INIT" \
  -e OV2_RECOMPUTE_FULL="${OV2_RECOMPUTE_FULL:-0}" \
  -v /ov2:/ov2 -v /vlm:/vlm -w "$REPO" "$IMAGE" bash -lc "
    python -m torch.distributed.run $RDZV --nproc_per_node=$NPROC scripts/training/run_recipe.py \
      --recipe ov2_30b_a3b_p16m33_stage1 --dataset vlm-energon --step_func ov2_step \
      dataset.path=$DATA_PATH $RECOMPUTE_FLAG \
      checkpoint.pretrained_checkpoint=$INIT \
      checkpoint.save=$SAVE checkpoint.load=$SAVE dataset.dataloader_save=$SAVE \
      checkpoint.save_interval=$SAVE_EVERY train.train_iters=$ITERS \
      validation.eval_iters=0 logger.log_interval=$LOG_EVERY \
      > $SAVE/train_node${NODE_RANK:-0}.log 2>&1"
echo "[ov2-30b-stage1] launched -> tail -f $SAVE/train_node*.log  (loss prints on the LAST node)"
