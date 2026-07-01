#!/usr/bin/env bash
# =============================================================================
# OV2.1 4B (Qwen3-4B dense) · MID-TRAIN · FULL model (LLM + vision + adapter)
# Bridge-native: run_recipe.py + ov2_4b_midtrain. Single node (--standalone) OR multi-node (LIST_IP).
#
# 4B is DENSE (no MoE) -> NO EP, NO AdamW-force: dense midtrain keeps Muon (the recipe default; the
# is_moe AdamW auto-route does NOT fire for 4B). 4B fits easily at TP=1 on 8x80GB -> no CPU offload,
# selective core_attn recompute only. Knobs: OV2_MIDTRAIN_ADAMW=1 to use AdamW instead; OV2_RECOMPUTE_FULL=1
# if a long-video pack spikes the vision tower; TP=2 (auto SP) if you ever need it.
#
# INIT: default null -> the recipe stitches the VERIFIED 4B base (mcore_ckpt) + stage-1 adapter
# (stage1_ckpt on /vlm). For a REAL mid-train, chain from a trained stage-2 via INIT_CKPT=<ckpt>.
# =============================================================================
set -euo pipefail
REPO="${REPO:-/ov2/feilong/gb200/Megatron-Bridge}"
bash "$REPO/3rdparty/apply_megatron_patch.sh" 2>/dev/null || true   # fresh-clone safety: OV2 mcore patch (apply_rotary_fn hook)
IMAGE="${IMAGE:-mbridge:qwen35-muon}"   # has emerging_optimizers (Muon) + fla + megatron.core
DATA_PATH="${DATA_PATH:-$REPO/examples/models/qwen/qwen3_vl_ov2/A800/mid_training_seed85m.yaml}"
INIT_CKPT="${INIT_CKPT:-null}"          # null -> base stitch (4B mcore_ckpt + stage-1 adapter); set a trained stage-2 for a real run
SAVE="${SAVE:-/ov2/feilong/gb200/ckpts_video_sft/ov2_4b_p16m33_midtrain}"
NPROC="${NPROC:-8}"
MIDTRAIN_GBS="${OV2_MIDTRAIN_GBS:-16}"
MIDTRAIN_N_SAMPLES="${OV2_MIDTRAIN_N_SAMPLES:-780000}"
ITERS="${ITERS:-$(( (MIDTRAIN_N_SAMPLES + MIDTRAIN_GBS - 1) / MIDTRAIN_GBS ))}"   # ceil(n/gbs)
LOG_EVERY="${LOG_EVERY:-1}"; SAVE_EVERY="${SAVE_EVERY:-500}"
TP="${TP:-1}"; if [[ "$TP" -gt 1 ]]; then SP=true; else SP=false; fi
SEQ_LEN="${OV2_SEQ_LEN:-10192}"          # matches the seed85m packed length
# 4B+Muon at TP1 holds Muon's ~56GB UNSHARDED fp32 state (master+momentum+fp32 grads) -> full LLM+vision
# recompute is REQUIRED to fit 80GB. VERIFIED 2026-06-30 on A100-22: OV2_RECOMPUTE_FULL=0 OOMs at the first
# MLP forward (~78.9GB); =1 fits (~65GB), iter-1 lm loss 1.04 / grad-norm 1.6 / nan 0, ~2082 tok/s/GPU.
# Set =0 only if you shard the optimizer another way (TP>1 halves Muon state, or OV2_MIDTRAIN_ADAMW=1 -> distopt shards it).
OV2_RECOMPUTE_FULL="${OV2_RECOMPUTE_FULL:-1}"
RECOMPUTE_ARGS=""; [[ "$OV2_RECOMPUTE_FULL" == "1" ]] && RECOMPUTE_ARGS="model.recompute_granularity=full model.recompute_method=uniform model.recompute_num_layers=1"
OV2_VISION_RECOMPUTE="${OV2_VISION_RECOMPUTE:-1}"

if [[ "$INIT_CKPT" != "null" && -n "$INIT_CKPT" && ! -e "$INIT_CKPT" ]]; then
  echo "[ov2-4b-midtrain] ERROR: INIT_CKPT=$INIT_CKPT not found. Set INIT_CKPT=null to start from the 4B base (smoke)." >&2; exit 1
fi
(( MIDTRAIN_GBS % (NPROC / TP) == 0 )) || echo "[ov2-4b-midtrain] WARN: GBS=$MIDTRAIN_GBS not divisible by DP=$((NPROC/TP)) -> mcore microbatch assert may fire." >&2

if [[ -n "${LIST_IP:-}" ]]; then read -ra list_ip <<< "$LIST_IP"; else list_ip=(); fi
NNODES=${#list_ip[@]}
if [[ "$NNODES" -le 1 ]]; then RDZV="--standalone"; NODE_RANK=0
else
  MASTER_ADDR="${list_ip[0]}"; MASTER_PORT="${MASTER_PORT:-26045}"
  CURRENT_IP="$(hostname -I | awk '{print $1}')"; NODE_RANK=-1
  for i in "${!list_ip[@]}"; do [[ "${list_ip[$i]}" == "$CURRENT_IP" ]] && NODE_RANK=$i && break; done
  [[ "$NODE_RANK" -eq -1 ]] && { echo "ERROR: $CURRENT_IP not in LIST_IP (${list_ip[*]})"; exit 1; }
  RDZV="--nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT"
fi
NCCL_ENV="-e NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0} -e NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0} \
  -e NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8} -e NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}"
PRELOAD=""; [[ "$INIT_CKPT" != "null" && -n "$INIT_CKPT" ]] && PRELOAD="checkpoint.pretrained_checkpoint=$INIT_CKPT"

mkdir -p "$SAVE"; docker rm -f ov2_4b_p16m33_mid 2>/dev/null || true
echo "[ov2-4b-midtrain] nnodes=${NNODES:-1} nproc=$NPROC TP=$TP seq=$SEQ_LEN gbs=$MIDTRAIN_GBS iters=$ITERS recompute_full=$OV2_RECOMPUTE_FULL init=$INIT_CKPT save=$SAVE image=$IMAGE (dense, Muon)"
docker run -d --name ov2_4b_p16m33_mid --network=host --privileged --gpus all -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  --ipc=host --shm-size=32g --ulimit memlock=-1 --ulimit stack=67108864 $NCCL_ENV \
  -e PYTHONPATH="$REPO/src:$REPO/3rdparty/Megatron-LM:$REPO/aiak_shim" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e OMP_NUM_THREADS=8 \
  -e OV2_MIDTRAIN_MUON=${OV2_MIDTRAIN_MUON:-0} -e OV2_MIDTRAIN_ADAMW=${OV2_MIDTRAIN_ADAMW:-0} \
  -e OV2_SEQ_LEN=$SEQ_LEN -e OV2_MIDTRAIN_GBS=$MIDTRAIN_GBS -e OV2_MIDTRAIN_N_SAMPLES=$MIDTRAIN_N_SAMPLES \
  -e OV2_RECOMPUTE_FULL=$OV2_RECOMPUTE_FULL -e OV2_VISION_RECOMPUTE=$OV2_VISION_RECOMPUTE \
  -e OV2_FREEZE_LLM=${OV2_FREEZE_LLM:-0} -e OV2_FREEZE_VISION=${OV2_FREEZE_VISION:-0} -e OV2_FREEZE_ADAPTER=${OV2_FREEZE_ADAPTER:-0} \
  -e OV2_PACK_FULL_CAUSAL=${OV2_PACK_FULL_CAUSAL:-0} \
  -e PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True} ${EXTRA_ENV:-} \
  -v /ov2:/ov2 -v /vlm:/vlm -w "$REPO" "$IMAGE" bash -lc "
    python -m torch.distributed.run $RDZV --nproc_per_node=$NPROC scripts/training/run_recipe.py \
      --recipe ov2_4b_midtrain --dataset vlm-energon --step_func ov2_step \
      dataset.path=$DATA_PATH $PRELOAD \
      checkpoint.save=$SAVE checkpoint.load=$SAVE dataset.dataloader_save=$SAVE \
      checkpoint.save_interval=$SAVE_EVERY train.train_iters=$ITERS \
      validation.eval_iters=0 logger.log_interval=$LOG_EVERY \
      model.tensor_model_parallel_size=$TP model.sequence_parallel=$SP $RECOMPUTE_ARGS \
      scheduler.lr_warmup_iters=${OV2_WARMUP_ITERS:-100} ${EXTRA_ARGS:-} \
      > $SAVE/train_node${NODE_RANK:-0}.log 2>&1"
echo "[ov2-4b-midtrain] launched container ov2_4b_p16m33_mid -> tail -f $SAVE/train_node${NODE_RANK:-0}.log"
