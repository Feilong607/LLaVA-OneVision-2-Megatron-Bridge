#!/usr/bin/env bash
# =============================================================================
# OV2-30B-A3B (Qwen3-30B-A3B MoE) midtrain · GB200/Blackwell · IN-CONTAINER launcher
# Run this INSIDE the training container (you are ALREADY in docker on GB200): it only
# assembles env + run_recipe overrides and execs torchrun — NO `docker run` wrapper.
#
# GB200 knobs (rationale: see git history / the prior docker version):
#   FREE:    192GB HBM -> recompute OFF (DISABLE_RECOMPUTE=1); expandable_segments; NVLink/NVLS.
#   NEEDS CODE FIX to engage: MXFP8 (mixed_precision) + HybridEP flex dispatcher must reach the
#            built LLM config inside build_llava_ov2 (setting cfg.model alone NO-OPs). Verify in log.
#   BLOCKED: CUDA graphs (OV2 forces cuda_graph_impl=none); EP/DDP comm overlap (grad path).
#   FIXED:   AdamW for the MoE backbone (Muon+EP deadlocks); TP=1 (OV2 SP/vision only TP1-tested).
# NOTE: REPO points at .../Megatron-Bridge (NOT -refactor) — that is where the MFU + dataloader
#       resume fixes live. If you switch REPO, sync those edits there too.
# =============================================================================
set -euo pipefail

REPO="${REPO:-/ov2/feilong/gb200/Megatron-Bridge}"
RECIPE="${RECIPE:-ov2_35b_a3b_midtrain}"        # full-model midtrain (use ov2_35b_a3b_stage2 for frozen-LLM SFT)
DATA_PATH="${DATA_PATH:-$REPO/examples/models/qwen/qwen3_vl_ov2/gb200/mid_training_seed85m.yaml}"  # seed85m offline-packed metadataset
INIT_CKPT="${INIT_CKPT:-/ov2/feilong/gb200/ckpts_video_sft/ov2_30b_a3b_stage2}"
SAVE="${SAVE:-/ov2/feilong/gb200/ckpts_video_sft/ov2_30b_a3b_gb200}"
NPROC="${NPROC:-4}"        # 4 GPUs per GB200 node (2 nodes=8 GPU EP8/DP1, or 4 nodes=16 GPU EP8/DP2)
ITERS="${ITERS:-6094}"; LOG_EVERY="${LOG_EVERY:-1}"; SAVE_EVERY="${SAVE_EVERY:-500}"

# --- TRAINING MODE: PHASE 1 = AIAK date0528 baseline (default) | PHASE 2 = GB200 accel (ACCEL=1) ---
# 30B-A3B is MoE -> optimizer is AdamW, NOT Muon (distributed Muon deadlocks the EP backward
# all-to-all). The recipe keeps AIAK lr 2e-5 constant / clip 1.0 / wd 0 / betas 0.9,0.99 / eps 1e-5
# / gbs 128 / mbs 1 / seq 32000. Phase 1 also matches AIAK: pure bf16 + recompute full/uniform/1.
ACCEL="${ACCEL:-0}"
if [[ "$ACCEL" == "1" ]]; then
  MIXED_PRECISION="${MIXED_PRECISION:-bf16_with_mxfp8_mixed}"   # MXFP8 (verify it reaches the LLM, see header)
  DISABLE_RECOMPUTE="${DISABLE_RECOMPUTE:-1}"; OV2_RECOMPUTE_FULL="${OV2_RECOMPUTE_FULL:-0}"
  DISPATCHER="${DISPATCHER:-flex}"; FLEX_BACKEND="${FLEX_BACKEND:-hybridep}"
else
  MIXED_PRECISION="${MIXED_PRECISION:-bf16}"                    # AIAK baseline: pure bf16
  DISABLE_RECOMPUTE="${DISABLE_RECOMPUTE:-0}"; OV2_RECOMPUTE_FULL="${OV2_RECOMPUTE_FULL:-1}"  # AIAK full/uniform/1
  DISPATCHER="${DISPATCHER:-alltoall}"; FLEX_BACKEND="${FLEX_BACKEND:-}"
fi
export OV2_RECOMPUTE_FULL

# --- MFU display peak (BF16 TFLOP/s/GPU): A100/A800=312, H100=989, GB200 bf16 ~= 2250 ---
export MFU_PEAK_TFLOPS="${MFU_PEAK_TFLOPS:-2250}"             # GB200 bf16 dense; set 312 if on A800, ~4500 if vs FP8

# --- rendezvous: 4 GPU/node. Multi-node via LIST_IP (run the SAME cmd on EACH node) ---
#   2 nodes (8 GPU, EP8/DP1):  LIST_IP="<ip0> <ip1>" bash ax_ov2_30b_a3b_gb200.sh
#   4 nodes (16 GPU, EP8/DP2): LIST_IP="<ip0> <ip1> <ip2> <ip3>" bash ax_ov2_30b_a3b_gb200.sh
#   EP8 spans 2 nodes (4+4) -> MoE all-to-all crosses the node boundary; needs fast inter-node (NVLink5/NVL72 or IB).
if [[ -n "${LIST_IP:-}" ]]; then read -ra list_ip <<< "$LIST_IP"; else list_ip=(); fi
NNODES=${#list_ip[@]}
if [[ "$NNODES" -le 1 ]]; then
  RDZV="--standalone"; NODE_RANK=0
else
  MASTER_ADDR="${list_ip[0]}"; MASTER_PORT="${MASTER_PORT:-26047}"
  CURRENT_IP="$(hostname -I | awk '{print $1}')"; NODE_RANK=-1
  for i in "${!list_ip[@]}"; do [[ "${list_ip[$i]}" == "$CURRENT_IP" ]] && NODE_RANK=$i && break; done
  [[ "$NODE_RANK" -eq -1 ]] && { echo "ERROR: $CURRENT_IP not in LIST_IP (${list_ip[*]})" >&2; exit 1; }
  RDZV="--nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT"
fi

# --- in-container env (were docker -e flags) ---
export PYTHONPATH="$REPO/src:$REPO/3rdparty/Megatron-LM:$REPO/aiak_shim${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OV2_MOE_PERMUTE_FUSION="${OV2_MOE_PERMUTE_FUSION:-0}"  # 30B-A3B: avoid TE Triton MoE-permute wedge (A800 stage2 fix)
export OV2_PACK_FULL_CAUSAL="${OV2_PACK_FULL_CAUSAL:-0}"   # 0=THD block-diagonal (sub-samples isolated, faithful to AIAK code); 1=full-causal (later sub-samples see earlier)
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NCCL_GRAPH_REGISTER="${NCCL_GRAPH_REGISTER:-0}" NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-1}"
[[ -n "${NCCL_IB_HCA:-}" ]] && export NCCL_IB_HCA NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"

# --- run_recipe.py CLI overrides ---
OVERRIDES="dataset.path=$DATA_PATH"
[[ "$INIT_CKPT" != "null" && -n "$INIT_CKPT" ]] && OVERRIDES="$OVERRIDES checkpoint.pretrained_checkpoint=$INIT_CKPT"
OVERRIDES="$OVERRIDES checkpoint.save=$SAVE checkpoint.load=$SAVE dataset.dataloader_save=$SAVE"
OVERRIDES="$OVERRIDES checkpoint.save_interval=$SAVE_EVERY train.train_iters=$ITERS validation.eval_iters=0 logger.log_interval=$LOG_EVERY train.micro_batch_size=1"   # packing REQUIRES mbs=1 (model asserts batch==1)
OVERRIDES="$OVERRIDES scheduler.lr_warmup_iters=0"   # AIAK: warmup 0 -> constant 2e-5 from step 1
OVERRIDES="$OVERRIDES mixed_precision=$MIXED_PRECISION"
[[ "$DISABLE_RECOMPUTE" == "1" ]] && OVERRIDES="$OVERRIDES model.recompute_activations=false model.recompute_granularity=null"
OVERRIDES="$OVERRIDES model.moe_token_dispatcher_type=$DISPATCHER"
[[ "$DISPATCHER" == "flex" && -n "$FLEX_BACKEND" ]] && OVERRIDES="$OVERRIDES model.moe_flex_dispatcher_backend=$FLEX_BACKEND"

mkdir -p "$SAVE"; cd "$REPO"
echo "[ov2-30b-gb200] in-container | repo=$REPO recipe=$RECIPE mp=$MIXED_PRECISION dispatcher=$DISPATCHER/$FLEX_BACKEND accel=$ACCEL recompute_off=$DISABLE_RECOMPUTE recompute_full=$OV2_RECOMPUTE_FULL peak=${MFU_PEAK_TFLOPS}TFLOPs node_rank=$NODE_RANK nnodes=${NNODES:-1}"
# shellcheck disable=SC2086  # $RDZV and $OVERRIDES must word-split into separate args
python -m torch.distributed.run $RDZV --nproc_per_node="$NPROC" scripts/training/run_recipe.py \
  --recipe "$RECIPE" --dataset vlm-energon --step_func ov2_step \
  $OVERRIDES 2>&1 | tee "$SAVE/train_node${NODE_RANK}.log"
