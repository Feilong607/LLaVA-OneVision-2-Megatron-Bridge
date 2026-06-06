#!/usr/bin/env bash
# =============================================================================
# OV2-30B-A3B (Qwen3-30B-A3B MoE) · GB200 / Blackwell acceleration launcher
# Bridge-native: run_recipe.py + an OV2 recipe (RECIPE env). Targets NVIDIA GB200 (Grace-Blackwell,
# NVL72, 192GB HBM3e/GPU, FP8/MXFP8 TE, NVLink5). *** UNTESTED: current cluster is A800 (no FP8);
# this is a forward-looking config for when GB200 is available — validate every GB200 knob on GB200. ***
#
# WHAT THIS TURNS ON vs the A800 scripts, and the OV2-SPECIFIC caveats (see the GB200 analysis):
#   [FREE wins — OV2 can use today, pure config]
#     * 192GB HBM -> DROP activation recompute (A800 needed full/uniform/1; GB200 has the headroom).
#       Recompute is ~25-35% compute overhead; removing it is the single biggest no-risk GB200 win here.
#     * GB200 host/NCCL env (expandable_segments, CUDA_DEVICE_MAX_CONNECTIONS, NVLink topology).
#     * Larger gbs / DP across the NVL72 NVLink domain.
#   [Needs a small CODE propagation fix to actually engage — same trap as calc_ptl/moe_router_dtype]
#     * MXFP8 (mixed_precision=bf16_with_mxfp8_mixed): the fp8 fields must reach the OV2 *runtime* LLM
#       config (language_model.config), which is built from a SEPARATE AutoBridge prov inside
#       build_llava_ov2 — NOT cfg.model. Setting cfg.mixed_precision alone likely NO-OPs on the LLM.
#       FIX: in build_llava_ov2/provide(), copy the fp8_* fields onto the built language_model.config
#       (mirror the moe_router_dtype='fp8'-padding + calc_ptl fixes). Until then MXFP8 below is inert.
#     * HybridEP flex dispatcher (moe_token_dispatcher_type=flex + apply_flex_dispatcher_backend):
#       same propagation concern; the LLM is rebuilt from the bridge. Verify the built model's
#       dispatcher actually changed before trusting the speedup.
#   [BLOCKED for OV2 today]
#     * CUDA graphs (the #1 GB200 host-overhead win, attn+moe_router+moe_preprocess): OV2 forces
#       cuda_graph_impl='none' because graphs bypass the MIMO grad-finalize and mis-scale grads on
#       multi-node. Leave OFF until the MIMO grad path is made graph-safe.
#     * MoE EP comm overlap (overlap_moe_expert_parallel_comm) + DDP overlap: may conflict with the
#       custom finalize_model_grads(pg_collection=None) token-weighted grad path -> verify grad norm
#       stays ~1.3 before enabling. OFF by default here.
#   [Unchanged OV2 constraints] AdamW for the MoE backbone (Muon+EP deadlocks); TP=1 (OV2's SP/vision
#     masked_scatter path is only smoke-tested at TP1 — do NOT raise TP without validating SP).
# =============================================================================
set -euo pipefail

REPO="${REPO:-/ov2/feilong/gb200/Megatron-Bridge-refactor}"
IMAGE="${IMAGE:-mbridge:qwen35-muon}"                            # use a Blackwell/TE-FP8-capable image on GB200
RECIPE="${RECIPE:-ov2_35b_a3b_stage2}"                           # or ov2_35b_a3b_midtrain (full model)
DATA_PATH="${DATA_PATH:-/vlm/data/llava_next_full_mega}"
INIT_CKPT="${INIT_CKPT:-/ov2/feilong/gb200/ckpts_video_sft/ov2_30b_a3b_stage1}"
SAVE="${SAVE:-/ov2/feilong/gb200/ckpts_video_sft/ov2_30b_a3b_gb200}"
NPROC="${NPROC:-8}"
ITERS="${ITERS:-6094}"; LOG_EVERY="${LOG_EVERY:-1}"; SAVE_EVERY="${SAVE_EVERY:-500}"

# --- GB200 toggles (override via env) ---
MIXED_PRECISION="${MIXED_PRECISION:-bf16_with_mxfp8_mixed}"      # MXFP8 (Blackwell). NEEDS the provide() fp8 propagation fix.
DISABLE_RECOMPUTE="${DISABLE_RECOMPUTE:-1}"                      # 192GB HBM -> recompute off (free win)
DISPATCHER="${DISPATCHER:-flex}"                                # GB200 prefers HybridEP flex; alltoall is the A800 default
FLEX_BACKEND="${FLEX_BACKEND:-hybridep}"                         # GB200: hybridep (NVL72 topology-aware)

if [[ -n "${LIST_IP:-}" ]]; then read -ra list_ip <<< "$LIST_IP"; else list_ip=(); fi
NNODES=${#list_ip[@]}
if [[ "$NNODES" -le 1 ]]; then
  RDZV="--standalone"
else
  MASTER_ADDR="${list_ip[0]}"; MASTER_PORT="${MASTER_PORT:-26047}"
  CURRENT_IP="$(hostname -I | awk '{print $1}')"; NODE_RANK=-1
  for i in "${!list_ip[@]}"; do [[ "${list_ip[$i]}" == "$CURRENT_IP" ]] && NODE_RANK=$i && break; done
  [[ "$NODE_RANK" -eq -1 ]] && { echo "ERROR: $CURRENT_IP not in LIST_IP (${list_ip[*]})"; exit 1; }
  RDZV="--nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT"
fi

# GB200 NVL72: intra-rack is all-NVLink (NCCL auto-detects). Host-overhead + allocator tuning matter most.
# CUDA_DEVICE_MAX_CONNECTIONS=1 default; use 32 only if you later enable EP-overlap + CUDA graphs together.
GB200_ENV="-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1} \
  -e NCCL_GRAPH_REGISTER=0 -e NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-1}"
# IB only needed across NVL72 racks; within one rack NVLink suffices. Set NCCL_IB_HCA via env for multi-rack.
[[ -n "${NCCL_IB_HCA:-}" ]] && GB200_ENV="$GB200_ENV -e NCCL_IB_HCA=$NCCL_IB_HCA -e NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}"

# --- assemble run_recipe CLI overrides ---
OVERRIDES="dataset.path=$DATA_PATH"
[[ "$INIT_CKPT" != "null" && -n "$INIT_CKPT" ]] && OVERRIDES="$OVERRIDES checkpoint.pretrained_checkpoint=$INIT_CKPT"
OVERRIDES="$OVERRIDES checkpoint.save=$SAVE checkpoint.load=$SAVE dataset.dataloader_save=$SAVE"
OVERRIDES="$OVERRIDES checkpoint.save_interval=$SAVE_EVERY train.train_iters=$ITERS validation.eval_iters=0 logger.log_interval=$LOG_EVERY"
# MXFP8 (verify it propagates to the LLM — see header):
OVERRIDES="$OVERRIDES mixed_precision=$MIXED_PRECISION"
# 192GB HBM -> recompute off (the midtrain recipe sets recompute_activations=True; override it):
[[ "$DISABLE_RECOMPUTE" == "1" ]] && OVERRIDES="$OVERRIDES model.recompute_activations=false model.recompute_granularity=null"
# HybridEP flex dispatcher (verify it reaches the built LLM):
OVERRIDES="$OVERRIDES model.moe_token_dispatcher_type=$DISPATCHER model.moe_flex_dispatcher_backend=$FLEX_BACKEND"

mkdir -p "$SAVE"; docker rm -f ov2_30b_gb200 2>/dev/null || true
echo "[ov2-30b-gb200] recipe=$RECIPE mixed_precision=$MIXED_PRECISION dispatcher=$DISPATCHER/$FLEX_BACKEND recompute_off=$DISABLE_RECOMPUTE"
echo "[ov2-30b-gb200] WARNING: GB200/FP8 knobs UNTESTED on this A800 cluster; validate fp8 actually engages (grep the run log for fp8/mxfp8) and grad norm stays sane."
docker run -d --name ov2_30b_gb200 --network=host --privileged --gpus all -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  --ipc=host --shm-size=32g --ulimit memlock=-1 --ulimit stack=67108864 $GB200_ENV \
  -e PYTHONPATH="$REPO/src:$REPO/3rdparty/Megatron-LM:$REPO/aiak_shim" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e OMP_NUM_THREADS=8 \
  -v /ov2:/ov2 -v /vlm:/vlm -w "$REPO" "$IMAGE" bash -lc "
    python -m torch.distributed.run $RDZV --nproc_per_node=$NPROC scripts/training/run_recipe.py \
      --recipe $RECIPE --dataset vlm-energon --step_func ov2_step \
      $OVERRIDES \
      > $SAVE/train_node${NODE_RANK:-0}.log 2>&1"
echo "[ov2-30b-gb200] launched -> tail -f $SAVE/train_node*.log"
