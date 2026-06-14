#!/usr/bin/env bash
# =============================================================================
# OV2-30B-A3B (Qwen3-30B-A3B MoE) · MID-TRAIN (stage 1.5) · FULL model (LLM + vision + adapter)
# Bridge-native: run_recipe.py + ov2_35b_a3b_midtrain. Single node (--standalone) OR multi-node (LIST_IP).
#
# Mirrors AIAK date0528 mid-train (--trainable-modules language_model adapter vision_model), full-model
# SFT, lr 2e-5, gbs 128, activation recompute ON. Unlike stage-2 (vit+adapter only), mid-train UNFREEZES
# the LLM -> the MoE experts become trainable, so distributed Muon would deadlock the EP backward
# all-to-all (the stage-2 hang). The recipe therefore AUTO-USES AdamW(distopt=True) for this MoE backbone
# via the is_moe auto-route (and -e OV2_MIDTRAIN_ADAMW=1 below as belt-and-suspenders). dense 4B keeps Muon.
#
# !! MEMORY: full-model 30B at seq 32000 is MUCH heavier than the vit+adapter stage-2 (optimizer state for
#    ALL params). recompute is ON, but EP8/2-node may still OOM -> you likely need TP>1 (pass TP via the
#    recipe / CLI model.tensor_model_parallel_size) and/or more nodes. Validate on GPU and watch memory.
#    chain from a TRAINED stage-2 via INIT_CKPT (model-only load).
# =============================================================================
set -euo pipefail

REPO="${REPO:-/ov2/feilong/gb200/Megatron-Bridge}"
bash "$REPO/3rdparty/apply_megatron_patch.sh" 2>/dev/null || true   # fresh-clone safety: apply OV2 mcore submodule patch (apply_rotary_fn hook)
IMAGE="${IMAGE:-mbridge:qwen35-muon}"                            # superset image (AdamW path used for MoE)
DATA_PATH="${DATA_PATH:-/ov2/feilong/gb200/Megatron-Bridge/examples/models/qwen/qwen3_vl_ov2/A800/mid_training_seed85m.yaml}" 
INIT_CKPT="${INIT_CKPT:-/ov2/feilong/gb200/ckpts_video_sft/ov2_30b_a3b_p16m33_stage2_muon}"  # trained stage-2 (model-only load)
SAVE="${SAVE:-/ov2/feilong/gb200/ckpts_video_sft/ov2_30b_a3b_p16m33_midtrain}"
NPROC="${NPROC:-8}"
# --- OV2 mid-train constants (mirror src/.../recipes/ov2/ov2.py; env-overridable -> this is the tuning
#     surface). Passed into the recipe via -e OV2_* below (same mechanism as OV2_SEQ_LEN) so the
#     recipe-computed train_iters / LR schedule stay in sync with the values used here. ---
MIDTRAIN_GBS="${OV2_MIDTRAIN_GBS:-16}"                  # global batch size; override with OV2_MIDTRAIN_GBS=
MIDTRAIN_N_SAMPLES="${OV2_MIDTRAIN_N_SAMPLES:-780000}"  # LLaVA-Next 780k (1 epoch); raise for the real packed corpus
ITERS="${ITERS:-$(( (MIDTRAIN_N_SAMPLES + MIDTRAIN_GBS - 1) / MIDTRAIN_GBS ))}"   # ceil(n/gbs); default 48750 = ceil(780000/16)
LOG_EVERY="${LOG_EVERY:-1}"; SAVE_EVERY="${SAVE_EVERY:-500}"
# --- memory-fit config (VALIDATED on 22+26 2026-06-14): full-model 30B does NOT fit at TP=1 -- OOMs at
#     the first forward (apply_rotary_pos_emb_vision spike) even at seq10192 (TP=1 keeps params + vision/
#     attn activations unsharded -> >80GB; more nodes can't help, those tensors are DP-replicated).
#     TP=2+SP is the working default: ~27GB/GPU at loop start, huge headroom. TP=2 reshards the (1,1)
#     stage-2 ckpt on load -- benign (weights load; only RNG state is skipped). seq 10192 matches the
#     seed85m packed length. Override TP= / OV2_SEQ_LEN= to retune.
TP="${TP:-2}"; if [[ "$TP" -gt 1 ]]; then SP=true; else SP=false; fi
SEQ_LEN="${OV2_SEQ_LEN:-10192}"
# MoE token capacity control. capacity_factor enables token dropping; pad-to-capacity
# keeps expert shapes stable. Default is enabled for the A800 memory-fit run; set
# MOE_CAPACITY_FACTOR=none (or -1) to disable for ablation/AIAK-parity checks.
MOE_CAPACITY_FACTOR="${MOE_CAPACITY_FACTOR:-1.0}"
MOE_PAD_TO_CAPACITY="${MOE_PAD_TO_CAPACITY:-true}"
# If memory is enough and you want no MoE token dropping / closest no-capacity behavior, use:
#   MOE_CAPACITY_FACTOR=none MOE_PAD_TO_CAPACITY=false bash ...
# If 1.0 still drops too much but disabling capacity OOMs, try MOE_CAPACITY_FACTOR=1.25 or 1.5
# with MOE_PAD_TO_CAPACITY=true as a middle ground.
MOE_CAPACITY_ARGS=""
if [[ -n "$MOE_CAPACITY_FACTOR" && "$MOE_CAPACITY_FACTOR" != "none" && "$MOE_CAPACITY_FACTOR" != "None" && "$MOE_CAPACITY_FACTOR" != "-1" ]]; then
  MOE_CAPACITY_ARGS="model.moe_expert_capacity_factor=$MOE_CAPACITY_FACTOR model.moe_pad_expert_input_to_capacity=$MOE_PAD_TO_CAPACITY"
fi

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

mkdir -p "$SAVE"; docker rm -f ov2_30b_p16m33_mid 2>/dev/null || true
echo "[ov2-30b-midtrain] nnodes=${NNODES:-1} TP=$TP SP=$SP seq=$SEQ_LEN gbs=$MIDTRAIN_GBS iters=$ITERS moe_capacity=$MOE_CAPACITY_FACTOR pad_to_capacity=$MOE_PAD_TO_CAPACITY init=$INIT_CKPT save=$SAVE (FULL model, AdamW for MoE)"
docker run -d --name ov2_30b_p16m33_mid --network=host --privileged --gpus all -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  --ipc=host --shm-size=32g --ulimit memlock=-1 --ulimit stack=67108864 $NCCL_ENV \
  -e PYTHONPATH="$REPO/src:$REPO/3rdparty/Megatron-LM:$REPO/aiak_shim" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e OMP_NUM_THREADS=8 \
  -e OV2_MIDTRAIN_ADAMW=${OV2_MIDTRAIN_ADAMW:-0} -e OV2_MIDTRAIN_MUON=${OV2_MIDTRAIN_MUON:-0} \
  -e OV2_SEQ_LEN=$SEQ_LEN -e OV2_MIDTRAIN_GBS=$MIDTRAIN_GBS -e OV2_MIDTRAIN_N_SAMPLES=$MIDTRAIN_N_SAMPLES \
  -e PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True} ${EXTRA_ENV:-} \
  -v /ov2:/ov2 -v /vlm:/vlm -w "$REPO" "$IMAGE" bash -lc "
    python -m torch.distributed.run $RDZV --nproc_per_node=$NPROC scripts/training/run_recipe.py \
      --recipe ov2_30b_a3b_p16m33_midtrain --dataset vlm-energon --step_func ov2_step \
      dataset.path=$DATA_PATH $PRELOAD \
      checkpoint.save=$SAVE checkpoint.load=$SAVE dataset.dataloader_save=$SAVE \
      checkpoint.save_interval=$SAVE_EVERY train.train_iters=$ITERS \
      validation.eval_iters=0 logger.log_interval=$LOG_EVERY \
      model.tensor_model_parallel_size=$TP model.sequence_parallel=$SP $MOE_CAPACITY_ARGS ${EXTRA_ARGS:-} \
      > $SAVE/train_node${NODE_RANK:-0}.log 2>&1"
echo "[ov2-30b-midtrain] launched -> tail -f $SAVE/train_node*.log  (loss on the LAST node)"
