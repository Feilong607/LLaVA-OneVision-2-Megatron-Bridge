#!/usr/bin/env bash
# =============================================================================
# OV2 · Qwen3.5-35B-A3B (GatedDeltaNet hybrid + 256-expert MoE + MTP) + OneVision p16m33
# GB200-only IN-CONTAINER launcher (4 GPU/node; EP8 = 2 nodes). Sibling of
# qwen3_vl_ov2/gb200/ax_ov2_30b_a3b_gb200.sh. Do NOT cross 3.5/30B recipes/ckpts/processors
# (3.5 <|image_pad|>=248056 vs 30B 151655).
#
# ACCEL:  0 = bf16 + alltoall (DEFAULT)   1 = MXFP8 + alltoall   2 = bf16 + HybridEP
#         1/2 are UNVALIDATED on the GDN+MTP hybrid -- bring up one at a time, A/B the loss.
# =============================================================================
set -euo pipefail
# Repo root auto-detect from this script's location; explicit REPO= wins.
REPO="${REPO:-$({ __d="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; while [[ "$__d" != "/" && ! -d "$__d/src/megatron/bridge" ]]; do __d="$(dirname "$__d")"; done; echo "$__d"; })}"
[[ -d "$REPO/src/megatron/bridge" ]] || { echo "FATAL: OV2 fork root not found from ${BASH_SOURCE[0]}. Set REPO=/path/to/Megatron-Bridge" >&2; exit 1; }
bash "$REPO/3rdparty/apply_megatron_patch.sh"   # mcore submodule patches (apply_rotary_fn hook, HybridEP pad); idempotent

RECIPE="${RECIPE:-ov2_qwen35_35b_a3b_midtrain}"   # other stages: ov2_qwen35_35b_a3b_stage2 / _stage1
MIDTRAIN_GBS="${OV2_MIDTRAIN_GBS:-16}"
MIDTRAIN_N_SAMPLES="${OV2_MIDTRAIN_N_SAMPLES:-8000000}"   # seed85m budget (matches 30B)
ITERS="${ITERS:-$(( (MIDTRAIN_N_SAMPLES + MIDTRAIN_GBS - 1) / MIDTRAIN_GBS ))}"
WARMUP_ITERS="${OV2_WARMUP_ITERS:-$(( ITERS * 2 / 1000 ))}"
if [ "$WARMUP_ITERS" -lt 1 ]; then WARMUP_ITERS=1; fi
LOG_EVERY="${LOG_EVERY:-1}"; SAVE_EVERY="${SAVE_EVERY:-2000}"

NPROC="${NPROC:-4}"   # GB200 = 4 GPU/node
TP="${TP:-1}"         # 2-node world=8 needs TP=1 so DP=8 satisfies EP=8
if [[ "$TP" -gt 1 ]]; then SP=true; else SP=false; fi
# seed85m packed length. NB: packed with the Qwen2.5-VL tokenizer -> under 3.5 (248056) some packs
# exceed seq_length and get SkipSample'd; check the dropped-pack rate before a long run.
SEQ_LEN="${OV2_SEQ_LEN:-10192}"
MOE_CAPACITY_FACTOR="${MOE_CAPACITY_FACTOR:-none}"
MOE_PAD_TO_CAPACITY="${MOE_PAD_TO_CAPACITY:-false}"
MOE_CAPACITY_ARGS=""
if [[ -n "$MOE_CAPACITY_FACTOR" && "$MOE_CAPACITY_FACTOR" != "none" && "$MOE_CAPACITY_FACTOR" != "None" && "$MOE_CAPACITY_FACTOR" != "-1" ]]; then
  MOE_CAPACITY_ARGS="model.moe_expert_capacity_factor=$MOE_CAPACITY_FACTOR model.moe_pad_expert_input_to_capacity=$MOE_PAD_TO_CAPACITY"
fi

# --- paths (all env-overridable; the recipe reads the OV2_* vars). Home resolved robustly:
# some launch contexts clear $HOME; no username literal is committed. ---
_HOME="${HOME:-}"
[[ -n "$_HOME" ]] || _HOME="$(getent passwd "$(id -un 2>/dev/null)" 2>/dev/null | cut -d: -f6)"
[[ -n "$_HOME" ]] || _HOME="/home/$(id -un 2>/dev/null)"
OV2_PRETRAIN_ROOT="${OV2_PRETRAIN_ROOT:-/datasets/llava/11May}"
OV2_LLM_HF_QWEN35="${OV2_LLM_HF_QWEN35:-$OV2_PRETRAIN_ROOT/Qwen3.5-35B-A3B-text}"
OV2_HF_PROC_QWEN35_P16M33="${OV2_HF_PROC_QWEN35_P16M33:-$OV2_PRETRAIN_ROOT/llava_onevision2/llava_onevision2_qwen35_35b_a3b_p16_m33/auto_model}"
OV2_MCORE_QWEN35_P16M33="${OV2_MCORE_QWEN35_P16M33:-$OV2_PRETRAIN_ROOT/llava_onevision2/llava_onevision2_qwen35_35b_a3b_p16_m33/stage_0_tp1_pp1_ep8}"
DATA_PATH="${DATA_PATH:-$REPO/examples/models/qwen/qwen3_vl_ov2/gb200/mid_training_seed85m.yaml}"
INIT_CKPT="${INIT_CKPT:-$_HOME/ckpts_video_sft/ov2_qwen35_35b_a3b_p16m33_stage2_muon_v2/iter_0006094}"   # trained stage-2 to resume
SAVE="${SAVE:-$_HOME/ckpts_video_sft/ov2_qwen35_35b_a3b_gb200}"
OV2_SKIP_BASE_STITCH="${OV2_SKIP_BASE_STITCH:-1}"   # midtrain from stage-2 -> skip the stage_0 stitch
export OV2_PRETRAIN_ROOT OV2_LLM_HF_QWEN35 OV2_HF_PROC_QWEN35_P16M33 OV2_MCORE_QWEN35_P16M33 OV2_SKIP_BASE_STITCH
export OV2_INIT_CKPT="$INIT_CKPT"   # recipe guard verifies this exists before skipping the stitch

# --- ACCEL. Recompute ON for every lane: the smaller 30B OOMs recompute-OFF at this seq on these
# nodes; DISABLE_RECOMPUTE=1 once fit is proven. ---
ACCEL="${ACCEL:-0}"
if [[ "$ACCEL" == "1" ]]; then          # MXFP8 + alltoall
  MIXED_PRECISION="${MIXED_PRECISION:-bf16_with_mxfp8_mixed}"
  FLEX_BACKEND="${FLEX_BACKEND:-}"
  MFU_PEAK_TFLOPS="${MFU_PEAK_TFLOPS:-4500}"
elif [[ "$ACCEL" == "2" ]]; then        # bf16 + HybridEP
  MIXED_PRECISION="${MIXED_PRECISION:-bf16_mixed}"
  FLEX_BACKEND="${FLEX_BACKEND:-hybridep}"
  MFU_PEAK_TFLOPS="${MFU_PEAK_TFLOPS:-2250}"
else                                    # bf16 baseline -- DEFAULT
  MIXED_PRECISION="${MIXED_PRECISION:-bf16_mixed}"
  FLEX_BACKEND="${FLEX_BACKEND:-}"
  MFU_PEAK_TFLOPS="${MFU_PEAK_TFLOPS:-2250}"
fi
DISABLE_RECOMPUTE="${DISABLE_RECOMPUTE:-0}"; OV2_RECOMPUTE_FULL="${OV2_RECOMPUTE_FULL:-1}"
export OV2_RECOMPUTE_FULL MFU_PEAK_TFLOPS
export OV2_FLEX_BACKEND="$FLEX_BACKEND"

# --- rendezvous: operator env (PET_*/MASTER_ADDR) -> manual LIST_IP -> single-node ---
GPUS_PER_NODE="$NPROC"
if [[ -n "${PET_NNODES:-}" || ( -n "${MASTER_ADDR:-}" && -n "${WORLD_SIZE:-}" ) ]]; then
  NNODES="${PET_NNODES:-$(( WORLD_SIZE / GPUS_PER_NODE ))}"
  NODE_RANK="${PET_NODE_RANK:-$(( ${RANK:-0} / GPUS_PER_NODE ))}"
  MASTER_ADDR="${MASTER_ADDR:-${PET_MASTER_ADDR:-}}"
  MASTER_PORT="${MASTER_PORT:-${PET_MASTER_PORT:-26049}}"
  [[ "$NNODES" -gt 1 && -z "${MASTER_ADDR}" ]] && { echo "[ov2-qwen35] FATAL: multi-node but no MASTER_ADDR/PET_MASTER_ADDR injected." >&2; exit 1; }
  RUN_MODE="multi-node (K8s auto-detected)"
elif [[ -n "${LIST_IP:-}" ]]; then
  read -ra list_ip <<< "$LIST_IP"
  NNODES=${#list_ip[@]}; MASTER_ADDR="${list_ip[0]}"; MASTER_PORT="${MASTER_PORT:-26049}"
  CURRENT_IP="$(hostname -I | awk '{print $1}')"; CURRENT_HOST="$(hostname)"; NODE_RANK=-1
  for i in "${!list_ip[@]}"; do [[ "${list_ip[$i]}" == "$CURRENT_IP" || "${list_ip[$i]}" == "$CURRENT_HOST" ]] && NODE_RANK=$i && break; done
  [[ "$NODE_RANK" -eq -1 ]] && { echo "[ov2-qwen35] ERROR: this host IP($CURRENT_IP)/name($CURRENT_HOST) not in LIST_IP (${list_ip[*]})" >&2; exit 1; }
  RUN_MODE="multi-node (manual LIST_IP)"
else
  NNODES=1; NODE_RANK=0; MASTER_ADDR=127.0.0.1; MASTER_PORT="${MASTER_PORT:-26049}"
  RUN_MODE="single-node TEST"
fi
# k8s DNS: short pod-name MASTER_ADDR -> FQDN if it resolves (avoids rdzv gai timeout).
if [[ "$NNODES" -gt 1 && -n "${MASTER_ADDR:-}" && "$MASTER_ADDR" != *.* && "$MASTER_ADDR" != "127.0.0.1" ]]; then
  _ns="${POD_NAMESPACE:-${OV2_K8S_NAMESPACE:-runai-mv0004}}"
  _fqdn="${MASTER_ADDR}.${_ns}.svc.cluster.local"
  if getent hosts "$_fqdn" >/dev/null 2>&1; then
    echo "[ov2-qwen35-gb200] rdzv: MASTER_ADDR '$MASTER_ADDR' -> FQDN '$_fqdn'" >&2
    MASTER_ADDR="$_fqdn"
  else
    echo "[ov2-qwen35-gb200] WARN: MASTER_ADDR='$MASTER_ADDR' is a short name and '$_fqdn' does not resolve; rdzv may time out. Set OV2_K8S_NAMESPACE=<ns>." >&2
  fi
fi
if [[ "$NNODES" -le 1 ]]; then RDZV="--standalone"; NNODES=1; NODE_RANK=0; else
  RDZV="--nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT"; fi
echo "[ov2-qwen35-gb200] --- rdzv: $RUN_MODE --- master=${MASTER_ADDR:-n/a}:${MASTER_PORT} nnodes=$NNODES node_rank=$NODE_RANK gpus/node=$GPUS_PER_NODE"
WORLD=$(( NPROC * NNODES ))
(( TP >= 1 )) || { echo "[ov2-qwen35] FATAL: TP must be >=1, got TP=$TP" >&2; exit 1; }
(( WORLD % TP == 0 )) || { echo "[ov2-qwen35] FATAL: WORLD=$WORLD must be divisible by TP=$TP." >&2; exit 1; }
DP=$(( WORLD / TP ))
(( MIDTRAIN_GBS % DP == 0 )) || { echo "[ov2-qwen35] FATAL: DP=$DP does not divide GBS=$MIDTRAIN_GBS; adjust TP/NNODES or OV2_MIDTRAIN_GBS." >&2; exit 1; }
# EP=8 fixed in the recipe.
(( DP >= 8 && DP % 8 == 0 )) || { echo "[ov2-qwen35] FATAL: EP=8 needs DP=$DP to be a multiple of 8 (2 GB200 nodes + TP=1 -> DP=8)." >&2; exit 1; }

# --- env ---
export PYTHONPATH="$REPO/_verify_stubs:$REPO/src:$REPO/3rdparty/Megatron-LM:$REPO/aiak_shim${PYTHONPATH:+:$PYTHONPATH}"  # _verify_stubs FIRST (offline stubs)
# deep_ep's .so needs the pip nvidia-nvshmem lib (not CUDA's bundled one); prepend only if present.
_nvshmem_lib="${OV2_NVSHMEM_LIB:-/usr/local/lib/python3.12/dist-packages/nvidia/nvshmem/lib}"
[[ -e "$_nvshmem_lib/libnvshmem_host.so.3" ]] && export LD_LIBRARY_PATH="$_nvshmem_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OV2_SKIP_HELPERS="${OV2_SKIP_HELPERS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"   # Rust tokenizer threads x forked workers -> deadlock
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OV2_MOE_PERMUTE_FUSION="${OV2_MOE_PERMUTE_FUSION:-0}"   # TE Triton MoE-permute wedge
export OV2_MOE_AUX_LOSS_COEFF="${OV2_MOE_AUX_LOSS_COEFF:-0.01}"
export OV2_MTP_LOSS_SCALE="${OV2_MTP_LOSS_SCALE:-}"            # empty -> recipe default 0.1; 0 kills the MTP gradient
export OV2_PACK_FULL_CAUSAL="${OV2_PACK_FULL_CAUSAL:-0}"       # 0 = THD block-diagonal (AIAK-faithful)
export OV2_SEQ_LEN="$SEQ_LEN"
export OV2_MIDTRAIN_GBS="$MIDTRAIN_GBS" OV2_MIDTRAIN_N_SAMPLES="$MIDTRAIN_N_SAMPLES"
export OV2_PARALLEL_SHARD_ITERS="${OV2_PARALLEL_SHARD_ITERS:-1}"  # energon default 16 chokes WekaFS
# GDN/MTP build + NCCL does NOT tolerate expandable_segments:True (observed fault) -> max_split_size_mb.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"
export NCCL_GRAPH_REGISTER="${NCCL_GRAPH_REGISTER:-0}" NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-1}"
[[ -n "${NCCL_IB_HCA:-}" ]] && export NCCL_IB_HCA NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker}"
export NCCL_DEBUG="${OV2_NCCL_DEBUG:-WARN}"
export NCCL_MNNVL_ENABLE="${NCCL_MNNVL_ENABLE:-1}"             # NVL72 cross-node NVLink
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-SYS}"
export NCCL_NET_GDR_C2C="${NCCL_NET_GDR_C2C:-1}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-1}"
export NVLINK_DOMAIN_SIZE="${NVLINK_DOMAIN_SIZE:-72}"
export NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT:-22}" NCCL_IB_RETRY_CNT="${NCCL_IB_RETRY_CNT:-7}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export UCX_TLS="${UCX_TLS:-tcp}"
export NCCL_ALGO="${NCCL_ALGO:-Tree,Ring,NVLSTree}"            # no CollNet (crashed this job before)
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/ov2_triton_cache}"       # node-local JIT caches
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/ov2_inductor_cache}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

# --- HybridEP gates (keyed on the dispatcher; mirrors the 30B launcher) ---
if [[ "$FLEX_BACKEND" == "hybridep" ]]; then
  # THD-padding patch preflight: without it the first MoE dispatch dies with cudaErrorIllegalAddress.
  _fa2a="$REPO/3rdparty/Megatron-LM/megatron/core/transformer/moe/fused_a2a.py"
  grep -q "_HYBRID_EP_PAD_INFO" "$_fa2a" 2>/dev/null || {
    echo "[ov2-qwen35] FATAL: HybridEP needs the fused_a2a.py THD-padding patch (marker _HYBRID_EP_PAD_INFO missing). Run: bash \"$REPO/3rdparty/apply_megatron_patch.sh\". Or use ACCEL=0/1." >&2
    exit 1; }
  # EP ranks per NVLink domain: 8 on NVL72 (one domain). <8 forces the internode-RDMA path (OOM + slower).
  _dom_ext="${NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN:-}"
  export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN="${NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN:-8}"
  (( 8 % NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN == 0 )) || {
    echo "ERROR: NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=$NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN must divide EP=8." >&2; exit 1; }
  [[ -n "$_dom_ext" && "$_dom_ext" != "8" ]] && \
    echo "[ov2-qwen35] WARN: NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=$_dom_ext set in the shell (default 8); 'unset' it unless ranks are genuinely split across NVLink domains." >&2
  export NVSHMEM_DISABLE_CUDA_VMM="${NVSHMEM_DISABLE_CUDA_VMM:-1}"   # nvshmem CUDA-VMM broken on this platform
  # JIT cap: %64 and identical across EP ranks; derive from SEQ_LEN (round up to 64).
  _hep_cap=$(( (SEQ_LEN + 63) / 64 * 64 ))
  export HYBRID_EP_MAX_TOKENS_PER_RANK="${HYBRID_EP_MAX_TOKENS_PER_RANK:-$_hep_cap}"
  (( HYBRID_EP_MAX_TOKENS_PER_RANK >= _hep_cap )) || {
    echo "[ov2-qwen35] FATAL: HYBRID_EP_MAX_TOKENS_PER_RANK=$HYBRID_EP_MAX_TOKENS_PER_RANK < round64(SEQ_LEN)=$_hep_cap -> ranks pad to different targets -> allgather hang." >&2; exit 1; }
  (( HYBRID_EP_MAX_TOKENS_PER_RANK % 64 == 0 )) || {
    echo "[ov2-qwen35] FATAL: HYBRID_EP_MAX_TOKENS_PER_RANK=$HYBRID_EP_MAX_TOKENS_PER_RANK must be a multiple of 64." >&2; exit 1; }
  [[ -n "${OV2_HYBRIDEP_NUM_SMS:-}" ]] && { export OV2_HYBRIDEP_NUM_SMS; echo "[ov2-qwen35] WARN: OV2_HYBRIDEP_NUM_SMS=$OV2_HYBRIDEP_NUM_SMS set (steals SMs from expert GEMMs)." >&2; }
  [[ -n "${NUM_OF_TOKENS_PER_CHUNK_COMBINE_API:-}" ]] && { export NUM_OF_TOKENS_PER_CHUNK_COMBINE_API; echo "[ov2-qwen35] WARN: NUM_OF_TOKENS_PER_CHUNK_COMBINE_API set (can mis-size combine buffers on the pinned deep_ep); 'unset' unless validated." >&2; }
fi
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

# --- run_recipe.py overrides ---
OVERRIDES="dataset.path=$DATA_PATH"
[[ "$INIT_CKPT" != "null" && -n "$INIT_CKPT" ]] && OVERRIDES="$OVERRIDES checkpoint.pretrained_checkpoint=$INIT_CKPT"
OVERRIDES="$OVERRIDES checkpoint.save=$SAVE checkpoint.load=$SAVE dataset.dataloader_save=$SAVE"
OVERRIDES="$OVERRIDES checkpoint.save_interval=$SAVE_EVERY train.train_iters=$ITERS validation.eval_iters=0 logger.log_interval=$LOG_EVERY logger.timing_log_level=${OV2_TIMING_LOG_LEVEL:-2} train.micro_batch_size=1"   # packing REQUIRES mbs=1
OVERRIDES="$OVERRIDES model.tensor_model_parallel_size=$TP model.sequence_parallel=$SP $MOE_CAPACITY_ARGS"
OVERRIDES="$OVERRIDES model.moe_router_dtype=${OV2_ROUTER_DTYPE:-fp32}"   # 256-expert router stability
OVERRIDES="$OVERRIDES scheduler.lr_warmup_iters=$WARMUP_ITERS"
OVERRIDES="$OVERRIDES optimizer.lr=${OV2_LR:-1e-5} optimizer.min_lr=${OV2_MIN_LR:-1e-6}"
# Stage-2 keeps distributed Muon; vision fused-QKV layout needs muon_split_qkv=false. midtrain = AdamW
# (recipe auto-routes) so this is a no-op there. OV2_STAGE2_ADAMW=1 forces AdamW.
export OV2_STAGE2_ADAMW="${OV2_STAGE2_ADAMW:-0}"
if [[ "$RECIPE" == *stage2* && "$OV2_STAGE2_ADAMW" != "1" ]]; then
  OVERRIDES="$OVERRIDES optimizer.muon_split_qkv=false"
fi
# 192GB HBM: whole optimizer on-GPU, NO CPU offload (offload-zero NaN bug class). OV2_OPT_OFFLOAD=true if it does not fit.
if [[ "${OV2_OPT_OFFLOAD:-false}" == "true" ]]; then
  OVERRIDES="$OVERRIDES optimizer.optimizer_cpu_offload=true optimizer.optimizer_offload_fraction=${OV2_OFFLOAD_FRACTION:-1.0} optimizer.use_precision_aware_optimizer=true"
else
  OVERRIDES="$OVERRIDES optimizer.optimizer_cpu_offload=false optimizer.use_precision_aware_optimizer=false"
fi
OVERRIDES="$OVERRIDES dataset.num_workers=${OV2_NUM_WORKERS:-8}"
OVERRIDES="$OVERRIDES dist.distributed_timeout_minutes=${OV2_DIST_TIMEOUT_MIN:-300}"   # first-step JIT + ckpt load exceed 100
# CE fusion OFF: TP=1 -> fused CE materializes full [seq, ~256k-vocab] fp32 logits (~10GB spike) + per-shape recompiles.
OVERRIDES="$OVERRIDES model.cross_entropy_loss_fusion=${OV2_CE_FUSION:-false}"
OVERRIDES="$OVERRIDES logger.tensorboard_dir=$SAVE/tensorboard"
OVERRIDES="$OVERRIDES mixed_precision=$MIXED_PRECISION"
[[ "$DISABLE_RECOMPUTE" == "1" ]] && OVERRIDES="$OVERRIDES model.recompute_activations=false model.recompute_granularity=null"

# OPT-IN Megatron-FSDP: only helps when MODEL-STATE memory is the limit; fsdp_dtensor ckpts are one-way.
if [[ "${OV2_FSDP:-0}" == "1" ]]; then
  unset CUDA_DEVICE_MAX_CONNECTIONS
  OVERRIDES="$OVERRIDES dist.use_megatron_fsdp=true ddp.use_megatron_fsdp=true"
  OVERRIDES="$OVERRIDES ddp.data_parallel_sharding_strategy=optim_grads_params ddp.average_in_collective=false"
  OVERRIDES="$OVERRIDES checkpoint.ckpt_format=fsdp_dtensor"
  echo "[ov2-qwen35-gb200] OV2_FSDP=1: Megatron-FSDP ON; torch_dist INIT_CKPT will mismatch fsdp_dtensor." >&2
fi

mkdir -p "$SAVE"; cd "$REPO"
# The 30B launcher exports OV2_MIDTRAIN_MUON=1; a leftover export flips THIS midtrain to Muon, which
# deadlocks EP backward on trainable 256-expert MoE.
if [[ "$RECIPE" == *midtrain* && "${OV2_MIDTRAIN_MUON:-0}" == "1" ]]; then
  echo "[ov2-qwen35] WARN: OV2_MIDTRAIN_MUON=1 -> midtrain will use Muon, which DEADLOCKS EP backward on trainable experts. 'unset OV2_MIDTRAIN_MUON' (auto-routes to AdamW)." >&2
fi
echo "[ov2-qwen35-gb200] in-container | repo=$REPO recipe=$RECIPE accel=$ACCEL mp=$MIXED_PRECISION flex=${OV2_FLEX_BACKEND:-alltoall} recompute_off=$DISABLE_RECOMPUTE recompute_full=$OV2_RECOMPUTE_FULL peak=${MFU_PEAK_TFLOPS}TF nproc=$NPROC world=$WORLD dp=$DP tp=$TP sp=$SP seq=$SEQ_LEN gbs=$MIDTRAIN_GBS iters=$ITERS warmup=$WARMUP_ITERS lr=${OV2_LR:-1e-5}->${OV2_MIN_LR:-1e-6} router_dtype=${OV2_ROUTER_DTYPE:-fp32} permute_fusion=$OV2_MOE_PERMUTE_FUSION aux_loss=$OV2_MOE_AUX_LOSS_COEFF mtp_scale=${OV2_MTP_LOSS_SCALE:-default} alloc=${PYTORCH_CUDA_ALLOC_CONF} offload=${OV2_OPT_OFFLOAD:-false} node_rank=$NODE_RANK nnodes=$NNODES"
# shellcheck disable=SC2086
python -m torch.distributed.run $RDZV --nproc_per_node="$NPROC" scripts/training/run_recipe.py \
  --recipe "$RECIPE" --dataset vlm-energon --step_func ov2_step \
  $OVERRIDES ${EXTRA_ARGS:-} 2>&1 | tee "$SAVE/train_node${NODE_RANK}.log"
