#!/usr/bin/env bash
# =============================================================================
# OV2 · Qwen3.5-35B-A3B (GatedDeltaNet hybrid + 256-expert MoE + MTP) + OneVision p16m33
# GB200-only IN-CONTAINER launcher (4 GPU/node). Bring-up shape: 2 nodes = 8 GPU = one EP8 group.
# Production shape: 32/64 GPU at TP=2 via ax_ov2_qwen35_s15_prod32.sh (measured). Sibling of
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
MIDTRAIN_GBS="${OV2_MIDTRAIN_GBS:-256}"   # production value (matches 30B); the old default 16 failed GBS%DP at 32 GPU
MIDTRAIN_N_SAMPLES="${OV2_MIDTRAIN_N_SAMPLES:-8000000}"   # seed85m budget (matches 30B)
ITERS="${ITERS:-$(( (MIDTRAIN_N_SAMPLES + MIDTRAIN_GBS - 1) / MIDTRAIN_GBS ))}"
WARMUP_ITERS="${OV2_WARMUP_ITERS:-$(( ITERS * 2 / 1000 ))}"
# Floor the AUTO-derived warmup to 1; an explicit OV2_WARMUP_ITERS=0 is honored (matches the 30B launcher).
if [ -z "${OV2_WARMUP_ITERS:-}" ] && [ "$WARMUP_ITERS" -lt 1 ]; then WARMUP_ITERS=1; fi
LOG_EVERY="${LOG_EVERY:-1}"; SAVE_EVERY="${SAVE_EVERY:-2000}"

NPROC="${NPROC:-4}"   # GB200 = 4 GPU/node
TP="${TP:-1}"         # 1 = the 2-node/8-GPU bring-up MINIMUM (DP must reach EP=8), not a preference:
                      # at 32 GPU TP=2 measured 1.6x FASTER than TP=1; the production wrapper sets 2.
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

# --- ACCEL. Recompute stays ON for every lane here (the 30B stage-3 launcher records recompute-OFF
# OOMing 192GB at this seq); the production wrapper narrows it to selective core_attn+moe, measured to fit. ---
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
# Conservative "fit-first" defaults for a BARE base-launcher run: full LLM recompute and vision-tower
# recompute both ON (the tower + adapter are built TP=1 / sequence_parallel=False on every rank, so
# their activations do not shrink with TP). Production (ax_ov2_qwen35_s15_prod32.sh) overrides to
# selective core_attn+moe with the tower NOT recomputed — measured 63-77 s/iter at peak-live 92.7 G.
# History: the byte-identical 188.4 GB per-pod peak seen at TP=1/2/4 was NOT the tower's activations
# (turning vision recompute on left it unchanged) — it was the allocator fragmentation described at
# PYTORCH_CUDA_ALLOC_CONF below.
export OV2_VISION_RECOMPUTE="${OV2_VISION_RECOMPUTE:-1}"
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
  _ns="${POD_NAMESPACE:-${OV2_K8S_NAMESPACE:-$(cat /var/run/secrets/kubernetes.io/serviceaccount/namespace 2>/dev/null || echo runai-mv0004)}}"
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
(( DP >= 8 && DP % 8 == 0 )) || { echo "[ov2-qwen35] FATAL: EP=8 needs DP=$DP to be a multiple of 8 (bring-up: 2 nodes x 4 GPU at TP=1 -> DP=8; production: 32 GPU at TP=2 -> DP=16)." >&2; exit 1; }

# --- env ---
# Offline packages not pip-installed in the image (e.g. emerging_optimizers for distributed Muon):
# auto-folded from "$_HOME/pylibs" or "$REPO/pylibs" (mirrors the 30B launchers); OV2_EXTRA_PYLIBS= also works.
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
export OV2_MTP_LOSS_SCALE="${OV2_MTP_LOSS_SCALE:-}"            # empty -> recipe default 0.1; 0 kills the MTP gradient
export OV2_PACK_FULL_CAUSAL="${OV2_PACK_FULL_CAUSAL:-0}"       # 0 = THD block-diagonal (AIAK-faithful)
export OV2_SEQ_LEN="$SEQ_LEN"
export OV2_MIDTRAIN_GBS="$MIDTRAIN_GBS" OV2_MIDTRAIN_N_SAMPLES="$MIDTRAIN_N_SAMPLES"
export OV2_PARALLEL_SHARD_ITERS="${OV2_PARALLEL_SHARD_ITERS:-1}"  # energon default 16 chokes WekaFS
# Allocator policy. GDN/MTP + NCCL cannot use expandable_segments:True (observed fault), which is
# what 30B relies on; the substitute used here was max_split_size_mb:256, and MEASUREMENT SHOWS THAT
# SUBSTITUTE WAS THE BUG. OV2_MEM_PROBE at forward #8 (TP2, GBS 256):
#     allocated=26.7G  max_allocated=44.0G  reserved=113-151G
# i.e. live tensors peak at 44 GB while the caching allocator holds 3.4x that. THD packing gives every
# microbatch a different shape, and max_split_size_mb forbids splitting blocks above the limit, so a
# new shape can never reuse a larger free block — the pool only grows. Five runs then reported a
# per-pod peak of 188.4 of 189.5 GB (99.4% of the card), byte-identical at TP=1/2/4, with Muon and
# AdamW, and with vision recompute on and off, dying as `NCCL WARN Cuda failure 2 'out of memory'`
# (NCCL buffers are cudaMalloc'd OUTSIDE the torch pool; one crash failed on a 136-byte calloc).
# So: drop max_split_size_mb (restore normal splitting) and keep a garbage-collection threshold. The
# BASE default 0.6 (collect from ~111 GB) suits the full-recompute lane above (peak-live ~44 G). The
# production wrapper runs selective recompute at peak-live 92.7 G and therefore sets 0.8 — 0.6 there
# would sit on the working set and churn (measured on the TP=1 run). ~37 GB of the card is non-torch
# (CUDA context, NCCL buffers, cuBLAS/TE workspaces), so torch must never approach the full card.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-garbage_collection_threshold:0.6}"
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
  # NVLink domain: let DeepEP auto-detect (a stale fixed 8 can split a detected 32-rank NVL72 domain and
  # force IB QPs). OV2_HYBRIDEP_NVLINK_DOMAIN_RANKS=<n> overrides; must divide TPxEP. Ported from the 30B
  # stage-3 launcher, which learned this at 32 GPUs.
  _hep_domain_override="${OV2_HYBRIDEP_NVLINK_DOMAIN_RANKS:-auto}"
  if [[ "$_hep_domain_override" == "auto" ]]; then
    [[ -n "${NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN:-}" ]] && \
      echo "[ov2-qwen35] WARN: unsetting inherited NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=$NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN; DeepEP will auto-detect." >&2
    unset NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN
  else
    [[ "$_hep_domain_override" =~ ^[0-9]+$ ]] && (( _hep_domain_override > 0 )) || {
      echo "[ov2-qwen35] FATAL: OV2_HYBRIDEP_NVLINK_DOMAIN_RANKS must be auto or a positive integer, got $_hep_domain_override" >&2; exit 1; }
    _hep_group_ranks=$(( TP * 8 ))   # HybridEP communicates over TP x EP.
    (( _hep_group_ranks % _hep_domain_override == 0 )) || {
      echo "[ov2-qwen35] FATAL: NVLink domain override=$_hep_domain_override must divide TPxEP=$_hep_group_ranks." >&2; exit 1; }
    export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN="$_hep_domain_override"
  fi
  export NVSHMEM_DISABLE_CUDA_VMM="${NVSHMEM_DISABLE_CUDA_VMM:-1}"   # nvshmem CUDA-VMM broken on this platform
  # Token cap is per RANK and HybridEP receives SP-LOCAL rows (MBS=1, CP=1): round64(ceil(SEQ_LEN/TP)) when
  # SP is on. The former global-SEQ_LEN cap over-allocated 2x at TP=2 (harmless only at TP=1).
  _hep_global_tokens="${OV2_HYBRIDEP_GLOBAL_TOKEN_CAP:-$SEQ_LEN}"
  [[ "$_hep_global_tokens" =~ ^[0-9]+$ ]] && (( _hep_global_tokens > 0 )) || {
    echo "[ov2-qwen35] FATAL: invalid global HybridEP token cap: $_hep_global_tokens" >&2; exit 1; }
  _hep_token_shards=1; [[ "$SP" == "true" ]] && _hep_token_shards="$TP"
  _hep_local_tokens=$(( (_hep_global_tokens + _hep_token_shards - 1) / _hep_token_shards ))
  _hep_cap=$(( (_hep_local_tokens + 63) / 64 * 64 ))
  if [[ -n "${HYBRID_EP_MAX_TOKENS_PER_RANK:-}" && "$HYBRID_EP_MAX_TOKENS_PER_RANK" != "$_hep_cap" ]]; then
    echo "[ov2-qwen35] FATAL: stale HYBRID_EP_MAX_TOKENS_PER_RANK=$HYBRID_EP_MAX_TOKENS_PER_RANK; expected TP-local cap=$_hep_cap from global=$_hep_global_tokens TP=$TP SP=$SP" >&2; exit 1
  fi
  export HYBRID_EP_MAX_TOKENS_PER_RANK="$_hep_cap"
  (( _hep_cap <= 21824 )) || { echo "[ov2-qwen35] FATAL: HybridEP local cap=$_hep_cap exceeds the inter-node safe limit 21824 (THD patch uses 64-token chunks); increase TP or use ACCEL=0." >&2; exit 1; }
  echo "[ov2-qwen35] HybridEP token cap: global=$_hep_global_tokens TP=$TP SP=$SP local=$_hep_local_tokens cap64=$_hep_cap" >&2
  # UNVALIDATED on the GDN+MTP backbone: run it through ax_ov2_qwen35_s15_smoke.sh with ACCEL=2 first.
  # Candidate extras seen in upstream GB200 release tests (NOT set here until measured):
  #   USE_MNNVL=1 NVSHMEM_IB_ENABLE_IBGDA=0 OV2_HYBRIDEP_NUM_SMS=32
  [[ -n "${OV2_HYBRIDEP_NUM_SMS:-}" ]] && { export OV2_HYBRIDEP_NUM_SMS; echo "[ov2-qwen35] WARN: OV2_HYBRIDEP_NUM_SMS=$OV2_HYBRIDEP_NUM_SMS set (steals SMs from expert GEMMs)." >&2; }
  [[ -n "${NUM_OF_TOKENS_PER_CHUNK_COMBINE_API:-}" ]] && { export NUM_OF_TOKENS_PER_CHUNK_COMBINE_API; echo "[ov2-qwen35] WARN: NUM_OF_TOKENS_PER_CHUNK_COMBINE_API set (can mis-size combine buffers)." >&2; }
fi
# EP comm-overlap (OV2_EP_OVERLAP=1) requires CUDA_DEVICE_MAX_CONNECTIONS>=32; couple them so the lever
# actually engages instead of silently no-op'ing (ported from the 30B launcher). Default path keeps 1.
if [[ "${OV2_EP_OVERLAP:-0}" == "1" ]]; then
  export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-32}"
else
  export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
fi

# --- run_recipe.py overrides ---
OVERRIDES="dataset.path=$DATA_PATH"
[[ "$INIT_CKPT" != "null" && -n "$INIT_CKPT" ]] && OVERRIDES="$OVERRIDES checkpoint.pretrained_checkpoint=$INIT_CKPT"
OVERRIDES="$OVERRIDES checkpoint.save=$SAVE checkpoint.load=$SAVE dataset.dataloader_save=$SAVE"
OVERRIDES="$OVERRIDES checkpoint.save_interval=$SAVE_EVERY train.train_iters=$ITERS validation.eval_iters=0 logger.log_interval=$LOG_EVERY logger.timing_log_level=${OV2_TIMING_LOG_LEVEL:-2} train.micro_batch_size=1"   # packing REQUIRES mbs=1
OVERRIDES="$OVERRIDES model.tensor_model_parallel_size=$TP model.sequence_parallel=$SP $MOE_CAPACITY_ARGS"
OVERRIDES="$OVERRIDES model.moe_router_dtype=${OV2_ROUTER_DTYPE:-fp32}"   # 256-expert router stability
OVERRIDES="$OVERRIDES scheduler.lr_warmup_iters=$WARMUP_ITERS"
OVERRIDES="$OVERRIDES optimizer.lr=${OV2_LR:-1e-5} optimizer.min_lr=${OV2_MIN_LR:-1e-6}"
# Stage-2 keeps distributed Muon; the trainable vision fused-QKV layout needs muon_split_qkv=false.
# Production midtrain also runs Muon (OV2_MIDTRAIN_MUON=1; the wrappers append the same override
# themselves). OV2_STAGE2_ADAMW=1 forces AdamW for stage-2.
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
# CE fusion OFF pending a TP=2 A/B (OV2_CE_FUSION=true; smoke ab-cefusion). The original reason was TP=1: an
# unsharded [seq, 248k] fp32 logits spike (~10GB) plus per-shape recompiles. At TP=2 the vocab is sharded,
# so the 30B launcher's own precondition for re-enabling it now holds here.
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
# Midtrain optimizer: distributed Muon is the s1.5 line's operator-required choice (AIAK parity) and has
# run 800+ qwen3.5 iterations and 11.5k+ 30B stage-3 iterations cleanly; the former "DEADLOCKS EP backward"
# WARN that fired on every production launch was never observed and is gone. Default 1 mirrors the 30B
# launcher; the recipe's MoE auto-route to AdamW applies only when this is 0. Muon cannot resume from an AdamW SAVE.
export OV2_MIDTRAIN_MUON="${OV2_MIDTRAIN_MUON:-1}"
# Preflight (mirrors the 30B launchers): mcore's distributed Muon needs 'emerging_optimizers', which is
# NOT in the base GB200 image. Without it the run dies DEEP -- after full NCCL init -- with a cryptic
# per-rank ImportError. Probe here (PYTHONPATH incl. the pylibs fold-in is set above) and fail loud EARLY.
if [[ ("$RECIPE" == *midtrain* && "${OV2_MIDTRAIN_MUON:-0}" == "1") || ("$RECIPE" == *stage2* && "$OV2_STAGE2_ADAMW" != "1") ]]; then
  [[ "${OV2_FSDP:-0}" == "1" ]] && { echo "[ov2-qwen35] FATAL: OV2_FSDP=1 is incompatible with Muon (Muon forces use_distributed_optimizer=False; FSDP shards optimizer state). Unset one." >&2; exit 1; }
  if ! python -c "import emerging_optimizers" >/dev/null 2>&1; then
    echo "[ov2-qwen35] FATAL: this recipe/optimizer combo uses distributed Muon but 'emerging_optimizers' is not importable." >&2
    echo "  Fix (pick one): stage the offline copy at \$HOME/pylibs or \$REPO/pylibs (auto-folded); or OV2_EXTRA_PYLIBS=/abs/path; or pip install emerging-optimizers; or force AdamW (midtrain: unset OV2_MIDTRAIN_MUON; stage2: OV2_STAGE2_ADAMW=1)." >&2
    exit 1
  fi
fi
echo "[ov2-qwen35-gb200] in-container | repo=$REPO recipe=$RECIPE accel=$ACCEL mp=$MIXED_PRECISION flex=${OV2_FLEX_BACKEND:-alltoall} recompute_off=$DISABLE_RECOMPUTE recompute_full=$OV2_RECOMPUTE_FULL peak=${MFU_PEAK_TFLOPS}TF nproc=$NPROC world=$WORLD dp=$DP tp=$TP sp=$SP seq=$SEQ_LEN gbs=$MIDTRAIN_GBS iters=$ITERS warmup=$WARMUP_ITERS lr=${OV2_LR:-1e-5}->${OV2_MIN_LR:-1e-6} router_dtype=${OV2_ROUTER_DTYPE:-fp32} permute_fusion=$OV2_MOE_PERMUTE_FUSION aux_loss=$OV2_MOE_AUX_LOSS_COEFF mtp_scale=${OV2_MTP_LOSS_SCALE:-default} alloc=${PYTORCH_CUDA_ALLOC_CONF} offload=${OV2_OPT_OFFLOAD:-false} node_rank=$NODE_RANK nnodes=$NNODES"
# shellcheck disable=SC2086
python -m torch.distributed.run $RDZV --nproc_per_node="$NPROC" scripts/training/run_recipe.py \
  --recipe "$RECIPE" --dataset vlm-energon --step_func ov2_step \
  $OVERRIDES ${EXTRA_ARGS:-} 2>&1 | tee -a "$SAVE/train_node${NODE_RANK}.log"
