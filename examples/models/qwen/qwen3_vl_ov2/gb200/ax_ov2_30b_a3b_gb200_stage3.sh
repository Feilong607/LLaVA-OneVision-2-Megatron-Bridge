#!/usr/bin/env bash
# =============================================================================
# OV2-30B-A3B (Qwen3-30B-A3B MoE) STAGE-3 - GB200-only IN-CONTAINER launcher
# (4 GPU/node; TP4 + EP8 needs 8 nodes for DP8). Standalone sibling of
# ax_ov2_30b_a3b_gb200_packed_64k.sh - neither script depends on the other.
#
# Stage-3 = midtrain recipe + video mix (pandas/shortmix/timelens) with 10% 47m_v3 single-image replay
# (stage3_mix_img10.yaml: pandas 56.7% / shortmix 31.5% / timelens 1.8% / image 10.0%),
# hyperparams aligned to the AIAK qwen35-s4 script: Muon lr 2e-5 -> 1e-6 cosine,
# warmup-frac 0.002, wd 0, matched-adamw-rms 0.2, adam_beta2 0.99.
# Init: trained stage-2mix ckpt (weights-only; auto-resumes when SAVE already has ckpts).
# 1 pandas-epoch = 617482 samples; GBS 32 -> 19297 iters; saves every 2000 + final iter.
# Data THD-packed to 64k tokens; seq_length MUST stay >= 65536 or every pack is skipped.
#
# ACCEL:  0 = bf16 + alltoall   1 = MXFP8 + alltoall   2 = bf16 + HybridEP (DEFAULT)
# =============================================================================
set -euo pipefail
# Repo root auto-detect from this script's location; explicit REPO= wins.
REPO="${REPO:-$({ __d="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; while [[ "$__d" != "/" && ! -d "$__d/src/megatron/bridge" ]]; do __d="$(dirname "$__d")"; done; echo "$__d"; })}"
[[ -d "$REPO/src/megatron/bridge" ]] || { echo "FATAL: OV2 fork root not found from ${BASH_SOURCE[0]}. Set REPO=/path/to/LLaVA-OneVision-2-Megatron-Bridge" >&2; exit 1; }
bash "$REPO/3rdparty/apply_megatron_patch.sh"   # mcore submodule patches (apply_rotary_fn hook, HybridEP pad); idempotent

RECIPE="${RECIPE:-ov2_30b_a3b_p16m33_midtrain}"   # the GB200 ckpt is p16m33 -> MUST stay a p16m33 recipe
MIDTRAIN_GBS="${OV2_MIDTRAIN_GBS:-32}"
TOTAL_SAMPLES="${OV2_TOTAL_SAMPLES:-617482}"   # 1 pandas-epoch of the img10 mix (= 555714 x 222230/200000)
EPOCHS="${OV2_EPOCHS:-1}"
MIDTRAIN_N_SAMPLES="${OV2_MIDTRAIN_N_SAMPLES:-$(( TOTAL_SAMPLES * EPOCHS ))}"
ITERS="${ITERS:-$(( (MIDTRAIN_N_SAMPLES + MIDTRAIN_GBS - 1) / MIDTRAIN_GBS ))}"
WARMUP_ITERS="${OV2_WARMUP_ITERS:-$(( ITERS * 2 / 1000 ))}"   # 0.002*iters ramp; OV2_WARMUP_ITERS=0 disables
if [ "$WARMUP_ITERS" -lt 1 ]; then WARMUP_ITERS=1; fi
LOG_EVERY="${LOG_EVERY:-1}"; SAVE_EVERY="${SAVE_EVERY:-2000}"   # 2000..18000 + auto final save at 19297 (train.py post-loop save)
TIMING_LOG_LEVEL="${OV2_TIMING_LOG_LEVEL:-2}"
TIMING_LOG_OPTION="${OV2_TIMING_LOG_OPTION:-minmax}"
TIMING_PRINT_INTERVAL="${OV2_TIMING_PRINT_INTERVAL:-$LOG_EVERY}"
[[ "$TIMING_LOG_LEVEL" =~ ^(-1|0|1|2)$ ]] || { echo "[ov2-30b] FATAL: OV2_TIMING_LOG_LEVEL must be -1, 0, 1, or 2; got $TIMING_LOG_LEVEL" >&2; exit 1; }
[[ "$TIMING_PRINT_INTERVAL" =~ ^[0-9]+$ ]] && (( TIMING_PRINT_INTERVAL > 0 )) || { echo "[ov2-30b] FATAL: OV2_TIMING_PRINT_INTERVAL must be a positive integer; got $TIMING_PRINT_INTERVAL" >&2; exit 1; }
case "$TIMING_LOG_OPTION" in max|minmax|all) ;; *) echo "[ov2-30b] FATAL: OV2_TIMING_LOG_OPTION must be max, minmax, or all; got $TIMING_LOG_OPTION" >&2; exit 1 ;; esac
export OV2_TIMING_PRINT_INTERVAL="$TIMING_PRINT_INTERVAL"

NPROC="${NPROC:-4}"   # GB200 = 4 GPU/node
TP="${TP:-4}"         # explicit TP= wins; SP/DP/HybridEP caps derive from this value
[[ "$TP" =~ ^[0-9]+$ ]] && (( TP >= 1 )) || { echo "[ov2-30b] FATAL: TP must be a positive integer, got TP=$TP" >&2; exit 1; }
if (( TP > 1 )); then SP=true; else SP=false; fi
SEQ_LEN="${OV2_SEQ_LEN:-73728}"    # headroom above the offline-packed 64k samples; shorter than 64k would SkipSample packs
[[ "$SEQ_LEN" =~ ^[0-9]+$ ]] && (( SEQ_LEN > 0 )) || { echo "[ov2-30b] FATAL: OV2_SEQ_LEN must be a positive integer, got $SEQ_LEN" >&2; exit 1; }
MOE_CAPACITY_FACTOR="${MOE_CAPACITY_FACTOR:-none}"
MOE_PAD_TO_CAPACITY="${MOE_PAD_TO_CAPACITY:-false}"
MOE_CAPACITY_ARGS=""
if [[ -n "$MOE_CAPACITY_FACTOR" && "$MOE_CAPACITY_FACTOR" != "none" && "$MOE_CAPACITY_FACTOR" != "None" && "$MOE_CAPACITY_FACTOR" != "-1" ]]; then
  MOE_CAPACITY_ARGS="model.moe_expert_capacity_factor=$MOE_CAPACITY_FACTOR model.moe_pad_expert_input_to_capacity=$MOE_PAD_TO_CAPACITY"
fi

# --- paths (all env-overridable). Home resolved robustly; no username literal committed. ---
_HOME="${HOME:-}"
[[ -n "$_HOME" ]] || _HOME="$(getent passwd "$(id -un 2>/dev/null)" 2>/dev/null | cut -d: -f6)"
[[ -n "$_HOME" ]] || _HOME="/home/$(id -un 2>/dev/null)"
OV2_LLM_HF_30B="${OV2_LLM_HF_30B:-/datasets/qwen-models-ea5jyi/Qwen3-30B-A3B-Instruct-2507}"
OV2_HF_PROC_30B="${OV2_HF_PROC_30B:-/datasets/llava-ov2-30b-a3b-m9lvdn/auto_model}"
OV2_HF_PROC_30B_P16M33="${OV2_HF_PROC_30B_P16M33:-/datasets/llava-ov2-30b-a3b-m9lvdn/auto_model}"
OV2_PRETRAIN_ROOT="${OV2_PRETRAIN_ROOT:-/datasets/llava/11May}"
DATA_PATH="${DATA_PATH:-$REPO/examples/models/qwen/qwen3_vl_ov2/gb200/stage3_mix_img10.yaml}"   # video mix + 10% 47m_v3 image replay
INIT_CKPT="${INIT_CKPT:-$_HOME/ckpts_video_sft/ov2_30b_a3b_stage2mix_v3_gbs32}"   # trained stage-2mix torch_dist ckpt (EP8); weights-only via pretrained_checkpoint
SAVE="${SAVE:-$_HOME/ckpts_video_sft/ov2_30b_a3b_stage3_img10_gbs32}"
OV2_SKIP_BASE_STITCH="${OV2_SKIP_BASE_STITCH:-1}"   # midtrain from a trained ckpt -> skip the stage_0 stitch
export OV2_LLM_HF_30B OV2_PRETRAIN_ROOT OV2_SKIP_BASE_STITCH OV2_HF_PROC_30B OV2_HF_PROC_30B_P16M33
export OV2_INIT_CKPT="$INIT_CKPT"   # recipe guard verifies this exists before skipping the stitch

# --- ACCEL. Muon opt-in on the MoE backbone (default ON per prior runs; OV2_MIDTRAIN_MUON=0 -> AdamW). ---
export OV2_MIDTRAIN_MUON="${OV2_MIDTRAIN_MUON:-1}"
ACCEL="${ACCEL:-2}"
if [[ "$ACCEL" == "1" ]]; then          # MXFP8 + alltoall (HybridEP+fp8-dispatch unsupported -> keep alltoall)
  MIXED_PRECISION="${MIXED_PRECISION:-bf16_with_mxfp8_mixed}"
  FLEX_BACKEND="${FLEX_BACKEND:-}"
  MFU_PEAK_TFLOPS="${MFU_PEAK_TFLOPS:-4500}"
elif [[ "$ACCEL" == "2" ]]; then        # bf16 + HybridEP -- DEFAULT
  MIXED_PRECISION="${MIXED_PRECISION:-bf16_mixed}"
  FLEX_BACKEND="${FLEX_BACKEND:-hybridep}"
  MFU_PEAK_TFLOPS="${MFU_PEAK_TFLOPS:-2250}"
else                                    # bf16 baseline
  MIXED_PRECISION="${MIXED_PRECISION:-bf16_mixed}"
  FLEX_BACKEND="${FLEX_BACKEND:-}"
  MFU_PEAK_TFLOPS="${MFU_PEAK_TFLOPS:-2250}"
fi
# HybridEP's multi-node IB queue-pair depth is limited to 65535. With the THD
# patch's 64-token chunks, each rank must stay at or below 21824 tokens. Keep
# TP user-controlled, but safely adapt the dispatcher when TP is too small.
_hep_safe_tokens=21824
_hep_min_tp=$(( (SEQ_LEN + _hep_safe_tokens - 1) / _hep_safe_tokens ))
if [[ "$FLEX_BACKEND" == "hybridep" ]] && (( TP < _hep_min_tp )); then
  echo "[ov2-30b] WARN: TP=$TP gives ceil(SEQ_LEN/TP)=$(( (SEQ_LEN + TP - 1) / TP )) HybridEP tokens/rank, above $_hep_safe_tokens; falling back to AllToAll (minimum HybridEP TP=$_hep_min_tp)." >&2
  FLEX_BACKEND=""
fi
# Recompute ON (selective core_attn + MoE) for EVERY lane: at seq=65536 activations dwarf the 10k case,
# and even at 10k recompute-OFF OOMs 192GB. DISABLE_RECOMPUTE=1 only if you have freed memory elsewhere.
DISABLE_RECOMPUTE="${DISABLE_RECOMPUTE:-0}"
OV2_RECOMPUTE_FULL="${OV2_RECOMPUTE_FULL:-0}"
OV2_RECOMPUTE_MOE="${OV2_RECOMPUTE_MOE:-1}"

export OV2_RECOMPUTE_FULL OV2_RECOMPUTE_MOE MFU_PEAK_TFLOPS

# ViT full activation recompute；不会冻结 ViT。Default unchanged (=1, the
# validated path); overridable so the tower's recompute can be A/B-ed the same
# way OV2_RECOMPUTE_MOE already can, once memory headroom allows.
export OV2_VISION_RECOMPUTE="${OV2_VISION_RECOMPUTE:-1}"

# 防止父 shell / Pod 遗留变量误冻结模型。
export OV2_FREEZE_VISION=0
export OV2_FREEZE_LLM=0
export OV2_FREEZE_ADAPTER=0

# 强制保留 LLM recompute 开关。
DISABLE_RECOMPUTE=0
export OV2_FLEX_BACKEND="$FLEX_BACKEND"   # read by ov2_provider.provide(); the cfg.model field is dead

# --- rendezvous: operator env (PET_*/MASTER_ADDR) -> manual LIST_IP -> single-node ---
GPUS_PER_NODE="$NPROC"
if [[ -n "${PET_NNODES:-}" || ( -n "${MASTER_ADDR:-}" && -n "${WORLD_SIZE:-}" ) ]]; then
  NNODES="${PET_NNODES:-$(( WORLD_SIZE / GPUS_PER_NODE ))}"
  NODE_RANK="${PET_NODE_RANK:-$(( ${RANK:-0} / GPUS_PER_NODE ))}"
  MASTER_ADDR="${MASTER_ADDR:-${PET_MASTER_ADDR:-}}"
  MASTER_PORT="${MASTER_PORT:-${PET_MASTER_PORT:-26047}}"
  [[ "$NNODES" -gt 1 && -z "${MASTER_ADDR}" ]] && { echo "[ov2-30b] FATAL: multi-node but no MASTER_ADDR/PET_MASTER_ADDR injected." >&2; exit 1; }
  RUN_MODE="multi-node (K8s auto-detected)"
elif [[ -n "${LIST_IP:-}" ]]; then
  read -ra list_ip <<< "$LIST_IP"
  NNODES=${#list_ip[@]}; MASTER_ADDR="${list_ip[0]}"; MASTER_PORT="${MASTER_PORT:-26047}"
  CURRENT_IP="$(hostname -I | awk '{print $1}')"; CURRENT_HOST="$(hostname)"; NODE_RANK=-1
  for i in "${!list_ip[@]}"; do [[ "${list_ip[$i]}" == "$CURRENT_IP" || "${list_ip[$i]}" == "$CURRENT_HOST" ]] && NODE_RANK=$i && break; done
  [[ "$NODE_RANK" -eq -1 ]] && { echo "[ov2-30b] ERROR: this host IP($CURRENT_IP)/name($CURRENT_HOST) not in LIST_IP (${list_ip[*]})" >&2; exit 1; }
  RUN_MODE="multi-node (manual LIST_IP)"
else
  NNODES=1; NODE_RANK=0; MASTER_ADDR=127.0.0.1; MASTER_PORT="${MASTER_PORT:-26047}"
  RUN_MODE="single-node TEST"
fi
# k8s DNS: short pod-name MASTER_ADDR -> FQDN if it resolves (avoids rdzv gai timeout).
if [[ "$NNODES" -gt 1 && -n "${MASTER_ADDR:-}" && "$MASTER_ADDR" != *.* && "$MASTER_ADDR" != "127.0.0.1" ]]; then
  _ns="${POD_NAMESPACE:-${OV2_K8S_NAMESPACE:-runai-mv0004}}"
  _fqdn="${MASTER_ADDR}.${_ns}.svc.cluster.local"
  if getent hosts "$_fqdn" >/dev/null 2>&1; then
    echo "[ov2-30b-gb200] rdzv: MASTER_ADDR '$MASTER_ADDR' -> FQDN '$_fqdn'" >&2
    MASTER_ADDR="$_fqdn"
  else
    echo "[ov2-30b-gb200] WARN: MASTER_ADDR='$MASTER_ADDR' is a short name and '$_fqdn' does not resolve; rdzv may time out. Set OV2_K8S_NAMESPACE=<ns>." >&2
  fi
fi
if [[ "$NNODES" -le 1 ]]; then RDZV="--standalone"; NNODES=1; NODE_RANK=0; else
  RDZV="--nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT"; fi
echo "[ov2-30b-gb200] --- rdzv: $RUN_MODE --- master=${MASTER_ADDR:-n/a}:${MASTER_PORT} nnodes=$NNODES node_rank=$NODE_RANK gpus/node=$GPUS_PER_NODE"
WORLD=$(( NPROC * NNODES ))
[[ "$TP" =~ ^[0-9]+$ ]] && (( TP >= 1 )) || { echo "[ov2-30b] FATAL: TP must be a positive integer, got TP=$TP" >&2; exit 1; }
(( WORLD % TP == 0 )) || { echo "[ov2-30b] FATAL: WORLD=$WORLD must be divisible by TP=$TP." >&2; exit 1; }
DP=$(( WORLD / TP ))
(( MIDTRAIN_GBS % DP == 0 )) || { echo "[ov2-30b] FATAL: GBS=$MIDTRAIN_GBS not divisible by DP=$DP; adjust OV2_MIDTRAIN_GBS / TP / NNODES." >&2; exit 1; }
# EP=8 fixed in the recipe.
(( DP >= 8 && DP % 8 == 0 )) || { echo "[ov2-30b] FATAL: EP=8 needs DP=$DP to be a multiple of 8 (WORLD=32 with TP=4 gives DP=8)." >&2; exit 1; }

# --- env ---
# Offline packages not pip-installed in the image (e.g. emerging_optimizers for Muon): picked up from
# "$REPO/pylibs" or "$HOME/pylibs" (where the base launcher keeps them), or OV2_EXTRA_PYLIBS=/abs/path.
[[ -d "$_HOME/pylibs" ]] && OV2_EXTRA_PYLIBS="$_HOME/pylibs${OV2_EXTRA_PYLIBS:+:$OV2_EXTRA_PYLIBS}"
[[ -d "$REPO/pylibs" ]] && OV2_EXTRA_PYLIBS="$REPO/pylibs${OV2_EXTRA_PYLIBS:+:$OV2_EXTRA_PYLIBS}"
export PYTHONPATH="$REPO/_verify_stubs:$REPO/src:$REPO/3rdparty/Megatron-LM:$REPO/aiak_shim${OV2_EXTRA_PYLIBS:+:$OV2_EXTRA_PYLIBS}${PYTHONPATH:+:$PYTHONPATH}"  # _verify_stubs FIRST (offline stubs)
# deep_ep's .so needs the pip nvidia-nvshmem lib (not CUDA's bundled one); prepend only if present.
_nvshmem_lib="${OV2_NVSHMEM_LIB:-/usr/local/lib/python3.12/dist-packages/nvidia/nvshmem/lib}"
[[ -e "$_nvshmem_lib/libnvshmem_host.so.3" ]] && export LD_LIBRARY_PATH="$_nvshmem_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OV2_SKIP_HELPERS="${OV2_SKIP_HELPERS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"   # Rust tokenizer threads x forked workers -> deadlock
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OV2_MOE_PERMUTE_FUSION="${OV2_MOE_PERMUTE_FUSION:-0}"   # TE Triton MoE-permute wedge
export OV2_MOE_AUX_LOSS_COEFF="${OV2_MOE_AUX_LOSS_COEFF:-0.01}"
export OV2_PACK_FULL_CAUSAL="${OV2_PACK_FULL_CAUSAL:-0}"       # 0 = THD block-diagonal (AIAK-faithful)
export OV2_SEQ_LEN="$SEQ_LEN"
export OV2_MIDTRAIN_GBS="$MIDTRAIN_GBS" OV2_MIDTRAIN_N_SAMPLES="$MIDTRAIN_N_SAMPLES"
export OV2_PARALLEL_SHARD_ITERS="${OV2_PARALLEL_SHARD_ITERS:-1}"  # energon default 16 chokes WekaFS
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NVTE_FWD_LAYERNORM_SM_MARGIN="${NVTE_FWD_LAYERNORM_SM_MARGIN:-0}"   # TP-derived topology; keep 0 unless overlap tuning is validated
export NVTE_BWD_LAYERNORM_SM_MARGIN="${NVTE_BWD_LAYERNORM_SM_MARGIN:-0}"
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

# --- HybridEP gates (keyed on the dispatcher; mirrors the main 30B launcher) ---
if [[ "$FLEX_BACKEND" == "hybridep" ]]; then
  # THD-padding patch preflight: without it the first MoE dispatch dies with cudaErrorIllegalAddress.
  _fa2a="$REPO/3rdparty/Megatron-LM/megatron/core/transformer/moe/fused_a2a.py"
  grep -q "_HYBRID_EP_PAD_INFO" "$_fa2a" 2>/dev/null || {
    echo "[ov2-30b] FATAL: HybridEP needs the fused_a2a.py THD-padding patch (marker _HYBRID_EP_PAD_INFO missing). Run: bash \"$REPO/3rdparty/apply_megatron_patch.sh\". Or use ACCEL=0/1." >&2
    exit 1; }
  # Let DeepEP detect the NVLink domain. A stale fixed value (previously 8)
  # can split a detected 32-rank NVLink domain and incorrectly force IB QPs.
  _hep_domain_override="${OV2_HYBRIDEP_NVLINK_DOMAIN_RANKS:-auto}"
  if [[ "$_hep_domain_override" == "auto" ]]; then
    if [[ -n "${NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN:-}" ]]; then
      echo "[ov2-30b] WARN: unsetting inherited NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=$NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN; DeepEP will auto-detect." >&2
    fi
    unset NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN
  else
    [[ "$_hep_domain_override" =~ ^[0-9]+$ ]] && (( _hep_domain_override > 0 )) || {
      echo "[ov2-30b] FATAL: OV2_HYBRIDEP_NVLINK_DOMAIN_RANKS must be auto or a positive integer, got $_hep_domain_override" >&2; exit 1; }
    _hep_group_ranks=$(( TP * 8 ))   # HybridEP communicates over TP x EP.
    (( _hep_group_ranks % _hep_domain_override == 0 )) || {
      echo "[ov2-30b] FATAL: NVLink domain override=$_hep_domain_override must divide TPxEP=$_hep_group_ranks." >&2; exit 1; }
    export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN="$_hep_domain_override"
    echo "[ov2-30b] HybridEP NVLink domain override: $_hep_domain_override ranks (TPxEP=$_hep_group_ranks)" >&2
  fi
  export NVSHMEM_DISABLE_CUDA_VMM="${NVSHMEM_DISABLE_CUDA_VMM:-1}"   # nvshmem CUDA-VMM broken on this platform
  # HybridEP receives SP-local rows, not the global packed width.
  # Current contract: MBS=1, CP=1.
  _hep_global_tokens="${OV2_HYBRIDEP_GLOBAL_TOKEN_CAP:-$SEQ_LEN}"

  [[ "$_hep_global_tokens" =~ ^[0-9]+$ ]] && (( _hep_global_tokens > 0 )) || {
    echo "[ov2-30b] FATAL: invalid global HybridEP token cap: $_hep_global_tokens" >&2
    exit 1
  }

  _hep_token_shards=1
  [[ "$SP" == "true" ]] && _hep_token_shards="$TP"

  _hep_local_tokens=$(( (_hep_global_tokens + _hep_token_shards - 1) / _hep_token_shards ))
  _hep_cap=$(( (_hep_local_tokens + 63) / 64 * 64 ))

  # Reject inherited global values such as 73728.
  if [[ -n "${HYBRID_EP_MAX_TOKENS_PER_RANK:-}" &&
        "$HYBRID_EP_MAX_TOKENS_PER_RANK" != "$_hep_cap" ]]; then
    echo "[ov2-30b] FATAL: stale HYBRID_EP_MAX_TOKENS_PER_RANK="\
"$HYBRID_EP_MAX_TOKENS_PER_RANK; expected TP-local cap=$_hep_cap "\
"from global=$_hep_global_tokens TP=$TP SP=$SP" >&2
    exit 1
  fi

  export HYBRID_EP_MAX_TOKENS_PER_RANK="$_hep_cap"

  # THD patch uses 64-token chunks. Therefore the effective safe maximum
  # is 21824, not the generic 16-aligned limit 21840.
  (( _hep_cap <= 21824 )) || {
    echo "[ov2-30b] FATAL: HybridEP local cap=$_hep_cap exceeds "\
"the inter-node safe limit 21824; increase TP or use AllToAll." >&2
    exit 1
  }

  echo "[ov2-30b] HybridEP token cap: global=$_hep_global_tokens "\
"TP=$TP SP=$SP local=$_hep_local_tokens cap64=$_hep_cap "\
"ib_depth=$((3 * _hep_cap + 1))" >&2
  [[ -n "${OV2_HYBRIDEP_NUM_SMS:-}" ]] && { export OV2_HYBRIDEP_NUM_SMS; echo "[ov2-30b] WARN: OV2_HYBRIDEP_NUM_SMS=$OV2_HYBRIDEP_NUM_SMS set (steals SMs from expert GEMMs)." >&2; }
  [[ -n "${NUM_OF_TOKENS_PER_CHUNK_COMBINE_API:-}" ]] && { export NUM_OF_TOKENS_PER_CHUNK_COMBINE_API; echo "[ov2-30b] WARN: NUM_OF_TOKENS_PER_CHUNK_COMBINE_API set (can mis-size combine buffers on the pinned deep_ep); 'unset' unless validated." >&2; }
fi
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

# --- run_recipe.py overrides ---
OVERRIDES="dataset.path=$DATA_PATH"
[[ "$INIT_CKPT" != "null" && -n "$INIT_CKPT" ]] && OVERRIDES="$OVERRIDES checkpoint.pretrained_checkpoint=$INIT_CKPT"
OVERRIDES="$OVERRIDES checkpoint.save=$SAVE checkpoint.load=$SAVE dataset.dataloader_save=$SAVE"
OVERRIDES="$OVERRIDES checkpoint.save_interval=$SAVE_EVERY train.train_iters=$ITERS validation.eval_iters=0 logger.log_interval=$LOG_EVERY logger.timing_log_level=$TIMING_LOG_LEVEL logger.timing_log_option=$TIMING_LOG_OPTION logger.log_timers_to_tensorboard=true logger.tensorboard_log_interval=${OV2_TENSORBOARD_LOG_INTERVAL:-$LOG_EVERY} logger.log_throughput=true train.micro_batch_size=1"   # packing REQUIRES mbs=1
OVERRIDES="$OVERRIDES model.tensor_model_parallel_size=$TP model.sequence_parallel=$SP $MOE_CAPACITY_ARGS"
OVERRIDES="$OVERRIDES model.moe_router_dtype=${OV2_ROUTER_DTYPE:-fp32}"   # 128-expert router stability
OVERRIDES="$OVERRIDES scheduler.lr_warmup_iters=$WARMUP_ITERS"
OVERRIDES="$OVERRIDES optimizer.lr=${OV2_LR:-2e-5} optimizer.min_lr=${OV2_MIN_LR:-1e-6}"   # AIAK qwen35-s4: 2e-5 cosine -> 1e-6
# 192GB HBM: whole optimizer on-GPU, NO CPU offload (offload-zero NaN bug class cannot occur).
OVERRIDES="$OVERRIDES optimizer.optimizer_cpu_offload=false optimizer.use_precision_aware_optimizer=false"
# Stage-3 Muon knobs (only when OV2_MIDTRAIN_MUON=1), baked to the AIAK qwen35-s4 values below;
# each knob individually overridable. wd must hit optimizer AND scheduler (scheduler clobbers it per-iter).
if [[ "${OV2_MIDTRAIN_MUON:-0}" == "1" ]]; then
  [[ "${OV2_FSDP:-0}" == "1" ]] && { echo "[ov2-30b-gb200] FATAL: OV2_FSDP=1 is incompatible with Muon (forces use_distributed_optimizer=False)." >&2; exit 1; }
  # Preflight: mcore's distributed Muon needs the 'emerging_optimizers' package (NOT in the base GB200
  # image). Without it the run crashes AFTER full NCCL init with a cryptic ImportError -- probe early.
  if ! python -c "import emerging_optimizers" >/dev/null 2>&1; then
    echo "[ov2-30b-gb200] FATAL: OV2_MIDTRAIN_MUON=1 but 'emerging_optimizers' is not importable." >&2
    echo "  Fix (pick one):" >&2
    echo "    - offline copy on PYTHONPATH:  OV2_EXTRA_PYLIBS=/path/to/pylibs bash \$0  (auto-detected from \$REPO/pylibs or \$HOME/pylibs)" >&2
    echo "    - install it:                  pip install emerging-optimizers" >&2
    echo "    - or use AdamW instead:        OV2_MIDTRAIN_MUON=0 bash \$0" >&2
    exit 1
  fi
  # Stage-3 Muon hyperparams (AIAK qwen35-s4), each env-overridable: matched-adamw-rms 0.2
  # (scale_mode stays the recipe default 'spectral'), wd 0 (set on optimizer AND scheduler
  # start/end, else the scheduler clobbers it back per-iter), adam_beta2 0.99 for Muon's
  # 1-D-param AdamW. Without these the recipe's midtrain block would give 0.15 / 0.01 / 0.95.
  OV2_MUON_EXTRA_SCALE="${OV2_MUON_EXTRA_SCALE:-0.2}"
  OV2_MUON_WD="${OV2_MUON_WD:-0}"
  [[ -n "${OV2_MUON_SCALE_MODE:-}" ]] && OVERRIDES="$OVERRIDES optimizer.muon_scale_mode=$OV2_MUON_SCALE_MODE"
  OVERRIDES="$OVERRIDES optimizer.muon_extra_scale_factor=$OV2_MUON_EXTRA_SCALE"
  OVERRIDES="$OVERRIDES optimizer.weight_decay=$OV2_MUON_WD scheduler.start_weight_decay=$OV2_MUON_WD scheduler.end_weight_decay=$OV2_MUON_WD"
  OVERRIDES="$OVERRIDES optimizer.adam_beta2=${OV2_ADAM_BETA2:-0.99}"
  echo "[ov2-30b-gb200] MUON ENABLED (stage-3): scale_mode=${OV2_MUON_SCALE_MODE:-spectral(recipe)} extra_scale=$OV2_MUON_EXTRA_SCALE wd=$OV2_MUON_WD beta2=${OV2_ADAM_BETA2:-0.99} -- watch iter-1->3 grad-norm/NaN." >&2
  echo "[ov2-30b-gb200] MUON resume CAUTION: Muon cannot cross-optimizer-resume from an AdamW ckpt; use a fresh SAVE or a Muon-saved ckpt." >&2
fi
OVERRIDES="$OVERRIDES dataset.num_workers=${OV2_NUM_WORKERS:-2}"    # conservative WekaFS default; override with OV2_NUM_WORKERS after validation
EXTRA_ARGS="${EXTRA_ARGS:-} dataset.shuffle_buffer_size=16"
OVERRIDES="$OVERRIDES dist.distributed_timeout_minutes=${OV2_DIST_TIMEOUT_MIN:-300}"   # first-step JIT + big-ckpt all_gather exceed 100
# CE fusion OFF: 64k packed sequences make materialized fp32 logits too memory-intensive.
OVERRIDES="$OVERRIDES model.cross_entropy_loss_fusion=${OV2_CE_FUSION:-false}"
OVERRIDES="$OVERRIDES logger.tensorboard_dir=$SAVE/tensorboard"   # never write into a possibly read-only $REPO
OVERRIDES="$OVERRIDES mixed_precision=$MIXED_PRECISION"
[[ "$DISABLE_RECOMPUTE" == "1" ]] && OVERRIDES="$OVERRIDES model.recompute_activations=false model.recompute_granularity=null"

# OPT-IN Megatron-FSDP: only helps when MODEL-STATE memory is the limit; fsdp_dtensor ckpts are one-way.
if [[ "${OV2_FSDP:-0}" == "1" ]]; then
  unset CUDA_DEVICE_MAX_CONNECTIONS
  OVERRIDES="$OVERRIDES dist.use_megatron_fsdp=true ddp.use_megatron_fsdp=true"
  OVERRIDES="$OVERRIDES ddp.data_parallel_sharding_strategy=optim_grads_params ddp.average_in_collective=false"
  OVERRIDES="$OVERRIDES checkpoint.ckpt_format=fsdp_dtensor"
  echo "[ov2-30b-gb200] OV2_FSDP=1: Megatron-FSDP ON; torch_dist INIT_CKPT will mismatch fsdp_dtensor." >&2
fi

mkdir -p "$SAVE"; cd "$REPO"
cp -f "${BASH_SOURCE[0]}" "$DATA_PATH" "$SAVE/" 2>/dev/null || true   # archive launcher + data yaml with the run
# NOTE: the old Muon resume-topology guard was removed -- distributed Muon supports DP-reshard now.
echo "[ov2-30b-gb200] in-container STAGE-3 | repo=$REPO recipe=$RECIPE accel=$ACCEL mp=$MIXED_PRECISION flex=${OV2_FLEX_BACKEND:-alltoall} recompute_off=$DISABLE_RECOMPUTE recompute_full=$OV2_RECOMPUTE_FULL recompute_moe=$OV2_RECOMPUTE_MOE peak=${MFU_PEAK_TFLOPS}TF nproc=$NPROC world=$WORLD dp=$DP tp=$TP sp=$SP seq=$SEQ_LEN gbs=$MIDTRAIN_GBS iters=$ITERS warmup=$WARMUP_ITERS lr=${OV2_LR:-2e-5}->${OV2_MIN_LR:-1e-6} router_dtype=${OV2_ROUTER_DTYPE:-fp32} permute_fusion=$OV2_MOE_PERMUTE_FUSION aux_loss=$OV2_MOE_AUX_LOSS_COEFF moe_capacity=$MOE_CAPACITY_FACTOR pad_to_capacity=$MOE_PAD_TO_CAPACITY muon=$OV2_MIDTRAIN_MUON timing_level=$TIMING_LOG_LEVEL timing_every=$TIMING_PRINT_INTERVAL timing_option=$TIMING_LOG_OPTION node_rank=$NODE_RANK nnodes=$NNODES"
# shellcheck disable=SC2086
python -m torch.distributed.run $RDZV --nproc_per_node="$NPROC" scripts/training/run_recipe.py \
  --recipe "$RECIPE" --dataset vlm-energon --step_func ov2_step \
  $OVERRIDES ${EXTRA_ARGS:-} 2>&1 | tee "$SAVE/train_node${NODE_RANK}.log"
