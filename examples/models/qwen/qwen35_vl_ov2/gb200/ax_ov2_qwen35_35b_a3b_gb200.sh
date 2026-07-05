#!/usr/bin/env bash
# =============================================================================
# OV2 · Qwen3.5-35B-A3B (qwen3_5_moe_text: GatedDeltaNet hybrid + 256-expert MoE + MTP)
#       + OneVision p16m33 · midtrain - GB200/Blackwell - IN-CONTAINER launcher
# Run this INSIDE the training container (you are ALREADY in docker on GB200): it only assembles
# env + run_recipe overrides and execs torchrun -- NO `docker run` wrapper (unlike the A800 launchers
# in ../A800, which docker-run mbridge:qwen35). This is the gb200 sibling of qwen3_vl_ov2/gb200/
# ax_ov2_30b_a3b_gb200.sh, kept structurally IDENTICAL; only the qwen3.5-specific deltas differ
# (marked "Q3.5:"). The Qwen3.5 stack stays fully SEPARATE from the Qwen3-30B stack -- do NOT cross
# recipes/ckpts/processors (3.5's <|image_pad|>=248056 vs 30B's 151655).
#
# HARDWARE: auto-detected (A100/A800 sm_80 | H100 sm_90 | GB200 sm_100); override HW=a100|h100|gb200.
#
# ACCEL MODES (set ACCEL=0|1|2):
#   0  bf16 baseline + alltoall + recompute (DEFAULT).
#   1  MXFP8 expert/matmul GEMMs + alltoall.
#   2  bf16 + HybridEP flex dispatcher (NVL72 topology-aware token comm).
#   Q3.5: ACCEL=1/2 are UNVALIDATED on the GatedDeltaNet+MTP hybrid (30B has neither). The MoE path
#         should be fine, but GDN linear-attention + the MTP head are not exercised by fp8/HybridEP
#         here -- bring them up one at a time and A/B the loss. Default 0.
#
# Q3.5 deltas vs the 30B gb200 launcher:
#   * RECIPE = ov2_qwen35_35b_a3b_midtrain (256-expert qwen3_5_moe; 40 layers, 30 GDN + MTP).
#   * model/proc/stage_0 via OV2_LLM_HF_QWEN35 / OV2_MCORE_QWEN35_P16M33 / OV2_HF_PROC_QWEN35_P16M33.
#   * image_token_id=248056 / adapter_init_scale=0.0184 / mrope_section=[11,11,10] come from the RECIPE
#     backbone (ov2_qwen35.py) -- NOT set here, so this launcher stays generic.
#   * PYTORCH_CUDA_ALLOC_CONF uses max_split_size_mb, NOT expandable_segments:True (the latter breaks
#     NCCL on the qwen3.5 GDN/MTP build -- observed OOM/NCCL fault; see memory).
#   * OV2_MTP_LOSS_SCALE exposed (midtrain KEEPS MTP=0.1; set 0 only to disable the MTP gradient).
#   * Stage-2 (frozen-LLM) keeps distributed Muon by default (+ muon_split_qkv=false for vision QKV);
#     midtrain is MoE-full-unfreeze -> recipe AUTO-routes to AdamW (Muon+trainable-experts deadlocks EP).
# NOTE: REPO must point at the repo holding the OV2 fixes (MFU + dataloader-resume + provider
#       fp8/dispatcher + mrope/fp32-softmax). If you switch REPO, sync those edits there too.
# =============================================================================
set -euo pipefail
REPO="${REPO:-/ov2/feilong/gb200/Megatron-Bridge}"

# Auto-detect repo root from this script's location (works on /ov2 AND a GB200 ~/LLaVA-... clone).
REPO="${REPO:-$({ __d="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; while [[ "$__d" != "/" && ! -d "$__d/src/megatron/bridge" ]]; do __d="$(dirname "$__d")"; done; echo "$__d"; })}"
[[ -d "$REPO/src/megatron/bridge" ]] || { echo "FATAL: OV2 fork root not found from ${BASH_SOURCE[0]} (no src/megatron/bridge above it). Set REPO=/path/to/Megatron-Bridge" >&2; exit 1; }
bash "$REPO/3rdparty/apply_megatron_patch.sh"   # OV2 mcore submodule patch (apply_rotary_fn hook). FAIL LOUD: a silently-missing hook -> cryptic "unexpected keyword argument 'apply_rotary_fn'" at build. Idempotent.
# Q3.5: midtrain default. Frozen-LLM SFT -> ov2_qwen35_35b_a3b_stage2; adapter alignment -> ov2_qwen35_35b_a3b_stage1.
RECIPE="${RECIPE:-ov2_qwen35_35b_a3b_midtrain}"
# DATA_PATH / INIT_CKPT / SAVE / model paths: set per-card by the CARD PATH PROFILE (after HW detect).
MIDTRAIN_GBS="${OV2_MIDTRAIN_GBS:-16}"                  # global batch size; override with OV2_MIDTRAIN_GBS=
MIDTRAIN_N_SAMPLES="${OV2_MIDTRAIN_N_SAMPLES:-780000}"  # LLaVA-Next 780k default
ITERS="${ITERS:-$(( (MIDTRAIN_N_SAMPLES + MIDTRAIN_GBS - 1) / MIDTRAIN_GBS ))}"
WARMUP_ITERS="${OV2_WARMUP_ITERS:-$(( ITERS * 2 / 1000 ))}"
if [ "$WARMUP_ITERS" -lt 1 ]; then WARMUP_ITERS=1; fi
LOG_EVERY="${LOG_EVERY:-1}"; SAVE_EVERY="${SAVE_EVERY:-2000}"

# --- HARDWARE PROFILE (auto-detected from compute capability; override HW=a100|a800|h100|gb200). ---
HW="${HW:-auto}"
case "$HW" in
  gb200) _cc=100;; h100|hopper) _cc=90;; a100|a800|ampere) _cc=80;;
  auto) _cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')";;
  *) _cc="";;
esac
_cc="${_cc:-80}"
[[ "$_cc" =~ ^[0-9]+$ ]] || _cc=80
if   [[ "$_cc" -ge 100 ]]; then HWNAME=gb200;  HW_NPROC=4; PEAK_BF16=2250; PEAK_FP8=4500; HW_NVLS=1; HW_FP8=1
elif [[ "$_cc" -ge 90  ]]; then HWNAME=hopper; HW_NPROC=8; PEAK_BF16=989;  PEAK_FP8=1979; HW_NVLS=1; HW_FP8=1
else                            HWNAME=ampere; HW_NPROC=8; PEAK_BF16=312;  PEAK_FP8=312;  HW_NVLS=0; HW_FP8=0
fi
NPROC="${NPROC:-$HW_NPROC}"        # GB200=4 GPU/node, A100/H100=8 GPU/node
TP="${TP:-1}"                      # GB200 2-node world=8 needs TP=1 so DP=8 can satisfy EP=8.
if [[ "$TP" -gt 1 ]]; then SP=true; else SP=false; fi
# Q3.5: seed85m offline-packed length (matches 30B gb200). NB: seed85m was packed with the Qwen2.5-VL
# tokenizer; under the Qwen3.5 tokenizer (248056) some packs may exceed seq_length and get SkipSample'd
# (lossy, not fatal). For 3.5-validated data use the A800 llava_next path at OV2_SEQ_LEN=32768 (see
# CARD PATH PROFILE). Verify the dropped-pack rate in the log before a long run.
SEQ_LEN="${OV2_SEQ_LEN:-10192}"
MOE_CAPACITY_FACTOR="${MOE_CAPACITY_FACTOR:-none}"
MOE_PAD_TO_CAPACITY="${MOE_PAD_TO_CAPACITY:-false}"
MOE_CAPACITY_ARGS=""
if [[ -n "$MOE_CAPACITY_FACTOR" && "$MOE_CAPACITY_FACTOR" != "none" && "$MOE_CAPACITY_FACTOR" != "None" && "$MOE_CAPACITY_FACTOR" != "-1" ]]; then
  MOE_CAPACITY_ARGS="model.moe_expert_capacity_factor=$MOE_CAPACITY_FACTOR model.moe_pad_expert_input_to_capacity=$MOE_PAD_TO_CAPACITY"
fi

# --- CARD PATH PROFILE: GB200 (/datasets or /home) <-> A100 (/ov2). All env-overridable.
#     The recipe (ov2_qwen35.py) reads OV2_PRETRAIN_ROOT + the OV2_*_QWEN35 vars below. ---
if [[ "${HWNAME:-}" == "gb200" || -d /datasets/qwen-models-ea5jyi ]]; then
  OV2_PRETRAIN_ROOT="${OV2_PRETRAIN_ROOT:-/datasets/llava/11May}"
  # Q3.5: text-only HF dir (extract_qwen35_text.py --weights) + p16m33 processor + stage_0 base.
  OV2_LLM_HF_QWEN35="${OV2_LLM_HF_QWEN35:-$OV2_PRETRAIN_ROOT/Qwen3.5-35B-A3B-text}"
  OV2_HF_PROC_QWEN35_P16M33="${OV2_HF_PROC_QWEN35_P16M33:-$OV2_PRETRAIN_ROOT/llava_onevision2/llava_onevision2_qwen35_35b_a3b_p16_m33/auto_model}"
  OV2_MCORE_QWEN35_P16M33="${OV2_MCORE_QWEN35_P16M33:-$OV2_PRETRAIN_ROOT/llava_onevision2/llava_onevision2_qwen35_35b_a3b_p16_m33/stage_0_tp1_pp1_ep8}"
  DATA_PATH="${DATA_PATH:-$REPO/examples/models/qwen/qwen3_vl_ov2/gb200/mid_training_seed85m.yaml}"   # see SEQ_LEN caveat above
  INIT_CKPT="${INIT_CKPT:-/home/ftan0055/ckpts_video_sft/ov2_qwen35_35b_a3b_p16m33_stage2_muon_v2/iter_0006094}"   # trained stage-2 to resume (stage the v2 ckpt onto the GB200 box)
  SAVE="${SAVE:-/home/ftan0055/ckpts_video_sft/ov2_qwen35_35b_a3b_gb200}"
  OV2_SKIP_BASE_STITCH="${OV2_SKIP_BASE_STITCH:-1}"   # midtrain from stage-2 -> skip the stage_0 stitch
else
  OV2_PRETRAIN_ROOT="${OV2_PRETRAIN_ROOT:-/ov2/pretrain_models}"
  OV2_LLM_HF_QWEN35="${OV2_LLM_HF_QWEN35:-$OV2_PRETRAIN_ROOT/Qwen3.5-35B-A3B-text}"
  OV2_HF_PROC_QWEN35_P16M33="${OV2_HF_PROC_QWEN35_P16M33:-$OV2_PRETRAIN_ROOT/llava_onevision2/llava_onevision2_qwen35_35b_a3b_p16_m33/auto_model}"
  OV2_MCORE_QWEN35_P16M33="${OV2_MCORE_QWEN35_P16M33:-$OV2_PRETRAIN_ROOT/llava_onevision2/llava_onevision2_qwen35_35b_a3b_p16_m33/stage_0_tp1_pp1_ep8}"
  DATA_PATH="${DATA_PATH:-$REPO/examples/models/qwen/qwen3_vl_ov2/mid_training_seed85m.yaml}"   # seed85m (matches 30B); see SEQ_LEN caveat. 3.5-validated alt: DATA_PATH=/vlm/data/llava_next_full_mega OV2_SEQ_LEN=32768
  INIT_CKPT="${INIT_CKPT:-/ov2/feilong/gb200/ckpts_video_sft/ov2_qwen35_35b_a3b_p16m33_stage2_muon_v2/iter_0006094}"
  SAVE="${SAVE:-/ov2/feilong/gb200/ckpts_video_sft/ov2_qwen35_35b_a3b_gb200}"
  OV2_SKIP_BASE_STITCH="${OV2_SKIP_BASE_STITCH:-1}"
fi
export OV2_PRETRAIN_ROOT OV2_LLM_HF_QWEN35 OV2_HF_PROC_QWEN35_P16M33 OV2_MCORE_QWEN35_P16M33 OV2_SKIP_BASE_STITCH
export OV2_INIT_CKPT="$INIT_CKPT"   # recipe guard verifies this exists before skipping the stitch

# --- TRAINING MODE (ACCEL legend above). 256-expert MoE -> midtrain optimizer AdamW (distributed Muon
# deadlocks EP backward on trainable experts). Recipe keeps the AIAK LR/clip/wd/betas/eps/gbs/mbs. ---
ACCEL="${ACCEL:-0}"
if [[ "$ACCEL" == "1" && "$HW_FP8" != "1" ]]; then
  echo "[ov2-qwen35] WARN: ACCEL=1 (MXFP8) needs fp8 HW; HW=$HWNAME (cc=$_cc) has none -> bf16 baseline (ACCEL=0)." >&2
  ACCEL=0
fi
if [[ "$ACCEL" == "1" ]]; then          # MXFP8 + alltoall (fp8 HW only)
  MIXED_PRECISION="${MIXED_PRECISION:-bf16_with_mxfp8_mixed}"
  DISABLE_RECOMPUTE="${DISABLE_RECOMPUTE:-1}"; OV2_RECOMPUTE_FULL="${OV2_RECOMPUTE_FULL:-0}"
  FLEX_BACKEND="${FLEX_BACKEND:-}"
  MFU_PEAK_TFLOPS="${MFU_PEAK_TFLOPS:-$PEAK_FP8}"
elif [[ "$ACCEL" == "2" ]]; then        # bf16 + HybridEP
  MIXED_PRECISION="${MIXED_PRECISION:-bf16_mixed}"
  DISABLE_RECOMPUTE="${DISABLE_RECOMPUTE:-1}"; OV2_RECOMPUTE_FULL="${OV2_RECOMPUTE_FULL:-0}"
  FLEX_BACKEND="${FLEX_BACKEND:-hybridep}"
  MFU_PEAK_TFLOPS="${MFU_PEAK_TFLOPS:-$PEAK_BF16}"
else                                    # bf16 baseline -- DEFAULT
  MIXED_PRECISION="${MIXED_PRECISION:-bf16_mixed}"
  # Q3.5: 35B (256 experts) > 30B; keep recompute ON on GB200 by default (memory headroom is tighter
  # than 30B). DISABLE_RECOMPUTE=1 / OV2_RECOMPUTE_FULL=0 to turn it off once you've confirmed the fit.
  DISABLE_RECOMPUTE="${DISABLE_RECOMPUTE:-0}"; OV2_RECOMPUTE_FULL="${OV2_RECOMPUTE_FULL:-1}"
  FLEX_BACKEND="${FLEX_BACKEND:-}"
  MFU_PEAK_TFLOPS="${MFU_PEAK_TFLOPS:-$PEAK_BF16}"
fi
export OV2_RECOMPUTE_FULL MFU_PEAK_TFLOPS
export OV2_FLEX_BACKEND="$FLEX_BACKEND"

# --- rendezvous: AUTO-DETECT master/worker (K8s PET_*/MASTER_ADDR -> manual LIST_IP -> single-node). ---
GPUS_PER_NODE="$NPROC"
if [[ -n "${PET_NNODES:-}" || ( -n "${MASTER_ADDR:-}" && -n "${WORLD_SIZE:-}" ) ]]; then
  NNODES="${PET_NNODES:-$(( WORLD_SIZE / GPUS_PER_NODE ))}"
  NODE_RANK="${PET_NODE_RANK:-$(( ${RANK:-0} / GPUS_PER_NODE ))}"
  MASTER_ADDR="${MASTER_ADDR:-${PET_MASTER_ADDR:-}}"
  MASTER_PORT="${MASTER_PORT:-${PET_MASTER_PORT:-26049}}"
  [[ "$NNODES" -gt 1 && -z "${MASTER_ADDR}" ]] && { echo "[ov2-qwen35] FATAL: multi-node but neither MASTER_ADDR nor PET_MASTER_ADDR injected by the operator." >&2; exit 1; }
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
# k8s DNS hardening: short MASTER_ADDR pod name -> FQDN if it resolves (avoids gai-error rdzv timeout).
if [[ "$NNODES" -gt 1 && -n "${MASTER_ADDR:-}" && "$MASTER_ADDR" != *.* && "$MASTER_ADDR" != "127.0.0.1" ]]; then
  _ns="${POD_NAMESPACE:-${OV2_K8S_NAMESPACE:-runai-mv0004}}"
  _fqdn="${MASTER_ADDR}.${_ns}.svc.cluster.local"
  if getent hosts "$_fqdn" >/dev/null 2>&1; then
    echo "[ov2-qwen35-gb200] rdzv: short MASTER_ADDR '$MASTER_ADDR' -> FQDN '$_fqdn'" >&2
    MASTER_ADDR="$_fqdn"
  else
    echo "[ov2-qwen35-gb200] WARN: MASTER_ADDR='$MASTER_ADDR' short name and FQDN '$_fqdn' does not resolve here; rdzv may time out. Set OV2_K8S_NAMESPACE=<ns> or pass a resolvable MASTER_ADDR." >&2
  fi
fi
if [[ "$NNODES" -le 1 ]]; then RDZV="--standalone"; NNODES=1; NODE_RANK=0; else
  RDZV="--nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT"; fi
echo "[ov2-qwen35-gb200] --- rdzv: $RUN_MODE --- master=${MASTER_ADDR:-n/a}:${MASTER_PORT} nnodes=$NNODES node_rank=$NODE_RANK gpus/node=$GPUS_PER_NODE"
WORLD=$(( NPROC * NNODES ))
(( TP >= 1 )) || { echo "[ov2-qwen35] FATAL: TP must be >=1, got TP=$TP" >&2; exit 1; }
(( WORLD % TP == 0 )) || { echo "[ov2-qwen35] FATAL: WORLD=$WORLD must be divisible by TP=$TP." >&2; exit 1; }
DP=$(( WORLD / TP ))
(( MIDTRAIN_GBS % DP == 0 )) || echo "[ov2-qwen35] WARN: DP=$DP (WORLD=$WORLD / TP=$TP) does not divide GBS=$MIDTRAIN_GBS -> mcore microbatch assert will fire; adjust TP/NNODES or OV2_MIDTRAIN_GBS." >&2
# EP=8 fixed in the recipe -> DP must be a multiple of EP and >= EP.
(( DP >= 8 && DP % 8 == 0 )) || { echo "[ov2-qwen35] FATAL: EP=8 needs DP=$DP (WORLD=$WORLD / TP=$TP) to be a multiple of 8 and >=8. For 2 GB200 nodes (WORLD=8) keep TP=1; TP=2 needs >=4 GB200 nodes." >&2; exit 1; }

# --- in-container env (were docker -e flags) ---
export PYTHONPATH="$REPO/src:$REPO/3rdparty/Megatron-LM:$REPO/aiak_shim${PYTHONPATH:+:$PYTHONPATH}"
export OV2_SKIP_HELPERS="${OV2_SKIP_HELPERS:-1}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OV2_MOE_PERMUTE_FUSION="${OV2_MOE_PERMUTE_FUSION:-0}"  # avoid TE Triton MoE-permute wedge
export OV2_MOE_AUX_LOSS_COEFF="${OV2_MOE_AUX_LOSS_COEFF:-0.01}"  # AIAK midtrain load-balance coeff; build_llava_ov2 forces it on the built LLM (provider zeros it again when the LLM is frozen -> stage-1/2)
# Q3.5: MTP loss weight. midtrain KEEPS MTP (empty -> recipe/qwen35_bridge default 0.1). Set 0 to kill
# the MTP gradient (e.g. if it fights convergence); read by build_llava_ov2 onto the runtime config.
export OV2_MTP_LOSS_SCALE="${OV2_MTP_LOSS_SCALE:-}"
export OV2_PACK_FULL_CAUSAL="${OV2_PACK_FULL_CAUSAL:-0}"      # 0=THD block-diagonal (AIAK-faithful)
export OV2_SEQ_LEN="$SEQ_LEN"                                # recipe reads this at import -> model+dataset+task_encoder
export OV2_MIDTRAIN_GBS="$MIDTRAIN_GBS" OV2_MIDTRAIN_N_SAMPLES="$MIDTRAIN_N_SAMPLES"
export OV2_PARALLEL_SHARD_ITERS="${OV2_PARALLEL_SHARD_ITERS:-1}"  # energon per-worker concurrent open shards (default 16 chokes WekaFS); 1 keeps 4xWx1 streams/node
# Q3.5: GDN/MTP build + NCCL did NOT tolerate expandable_segments:True (observed NCCL fault/OOM); use
# max_split_size_mb instead. Override with PYTORCH_CUDA_ALLOC_CONF= if your build is fine with expandable.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"
export NCCL_GRAPH_REGISTER="${NCCL_GRAPH_REGISTER:-0}" NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-$HW_NVLS}"
[[ -n "${NCCL_IB_HCA:-}" ]] && export NCCL_IB_HCA NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
# --- GB200 cross-node NCCL tuning (MNNVL + NVLink). All env-overridable. ---
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker}"
export NCCL_DEBUG="${OV2_NCCL_DEBUG:-WARN}"
export NCCL_MNNVL_ENABLE="${NCCL_MNNVL_ENABLE:-1}"                   # NVL72 Multi-Node NVLink
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-SYS}"
export NCCL_NET_GDR_C2C="${NCCL_NET_GDR_C2C:-1}"                     # Grace<->Blackwell C2C GPUDirect
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-1}"
export NVLINK_DOMAIN_SIZE="${NVLINK_DOMAIN_SIZE:-72}"
export NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT:-22}" NCCL_IB_RETRY_CNT="${NCCL_IB_RETRY_CNT:-7}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export UCX_TLS="${UCX_TLS:-tcp}"
export NCCL_ALGO="${NCCL_ALGO:-Tree,Ring,NVLSTree}"                  # SAFE set; CollNet (IB SHARP) crashed this job before
# node-local JIT caches (NOT shared /ov2): a networked JIT cache is slow and races.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/ov2_triton_cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/ov2_inductor_cache}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

# --- HybridEP topology (ACCEL=2 only): # of EP-communicating ranks sharing one NVLink domain. Must
# DIVIDE EP=8 (mcore asserts EP % value == 0). Full NVL72 rack -> 8; EP8 split 4+4 across domains -> 4. ---
if [[ "$ACCEL" == "2" ]]; then
  export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN="${NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN:-8}"
  (( 8 % NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN == 0 )) || {
    echo "ERROR: NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=$NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN must divide EP=8." >&2; exit 1; }
fi
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

# --- run_recipe.py CLI overrides ---
OVERRIDES="dataset.path=$DATA_PATH"
[[ "$INIT_CKPT" != "null" && -n "$INIT_CKPT" ]] && OVERRIDES="$OVERRIDES checkpoint.pretrained_checkpoint=$INIT_CKPT"
OVERRIDES="$OVERRIDES checkpoint.save=$SAVE checkpoint.load=$SAVE dataset.dataloader_save=$SAVE"
OVERRIDES="$OVERRIDES checkpoint.save_interval=$SAVE_EVERY train.train_iters=$ITERS validation.eval_iters=0 logger.log_interval=$LOG_EVERY logger.timing_log_level=${OV2_TIMING_LOG_LEVEL:-2} train.micro_batch_size=1"   # packing REQUIRES mbs=1
OVERRIDES="$OVERRIDES model.tensor_model_parallel_size=$TP model.sequence_parallel=$SP $MOE_CAPACITY_ARGS"
# MoE router fp32 for 256-expert stability. CLI override is the RELIABLE path (provider field may not survive build_llava_ov2's HF rebuild).
OVERRIDES="$OVERRIDES model.moe_router_dtype=${OV2_ROUTER_DTYPE:-fp32}"
OVERRIDES="$OVERRIDES scheduler.lr_warmup_iters=$WARMUP_ITERS"
OVERRIDES="$OVERRIDES optimizer.lr=${OV2_LR:-1e-5} optimizer.min_lr=${OV2_MIN_LR:-1e-6}"
# Q3.5: stage-2 keeps distributed Muon (frozen LLM -> Muon only on dense vision+adapter); the vision
# fused-QKV layout differs from the LLM so muon_split_qkv MUST be false (mcore defaults True). midtrain
# is AdamW (recipe auto-routes) so this is a harmless no-op there. OV2_STAGE2_ADAMW=1 forces AdamW.
export OV2_STAGE2_ADAMW="${OV2_STAGE2_ADAMW:-0}"
if [[ "$RECIPE" == *stage2* && "$OV2_STAGE2_ADAMW" != "1" ]]; then
  OVERRIDES="$OVERRIDES optimizer.muon_split_qkv=false"
fi
# GB200 (192GB HBM): keep the WHOLE optimizer on-GPU as plain fp32 AdamW -- NO CPU offload (the A800
# 35B launcher offloads to fit 80GB). Defensive: the recipe already defaults these off for the midtrain
# AdamW branch, so this GUARANTEES GB200 never walks the offload path (-> the offload-zero NaN bug class
# #1842/#1872/#1986 cannot occur). If 35B+256-expert does not fit 192GB, set OV2_OPT_OFFLOAD=true.
if [[ "${OV2_OPT_OFFLOAD:-false}" == "true" ]]; then
  OVERRIDES="$OVERRIDES optimizer.optimizer_cpu_offload=true optimizer.optimizer_offload_fraction=${OV2_OFFLOAD_FRACTION:-1.0} optimizer.use_precision_aware_optimizer=true"
else
  OVERRIDES="$OVERRIDES optimizer.optimizer_cpu_offload=false optimizer.use_precision_aware_optimizer=false"
fi
OVERRIDES="$OVERRIDES dataset.num_workers=${OV2_NUM_WORKERS:-16}"   # Grace ~1TB RAM; lower if host mem tight
OVERRIDES="$OVERRIDES dist.distributed_timeout_minutes=${OV2_DIST_TIMEOUT_MIN:-100}"
OVERRIDES="$OVERRIDES logger.tensorboard_dir=$SAVE/tensorboard"
OVERRIDES="$OVERRIDES mixed_precision=$MIXED_PRECISION"
[[ "$DISABLE_RECOMPUTE" == "1" ]] && OVERRIDES="$OVERRIDES model.recompute_activations=false model.recompute_granularity=null"

# --- OPT-IN: Megatron-FSDP (OV2_FSDP=1). Shards params+grads+optim across DP. Only helps when
# MODEL-STATE (not activation) memory is the limit. fsdp_dtensor ckpts are a ONE-WAY door. ---
if [[ "${OV2_FSDP:-0}" == "1" ]]; then
  unset CUDA_DEVICE_MAX_CONNECTIONS
  OVERRIDES="$OVERRIDES dist.use_megatron_fsdp=true ddp.use_megatron_fsdp=true"
  OVERRIDES="$OVERRIDES ddp.data_parallel_sharding_strategy=optim_grads_params ddp.average_in_collective=false"
  OVERRIDES="$OVERRIDES checkpoint.ckpt_format=fsdp_dtensor"
  echo "[ov2-qwen35-gb200] OV2_FSDP=1: Megatron-FSDP ON (world=$WORLD dp=$DP tp=$TP). fsdp_dtensor ckpt is ONE-WAY; torch_dist INIT_CKPT will mismatch -> convert it or point INIT_CKPT at an fsdp ckpt." >&2
fi

mkdir -p "$SAVE"; cd "$REPO"
# --- Muon resume-topology guard: distributed Muon shards momentum across DP with NO reshard support,
# so resuming a Muon ckpt at a DIFFERENT world size silently loads mismatched momentum. AdamW reshards
# fine -> guard Muon only (stage2 default). Marker lives in $SAVE so it travels with the ckpt dir. ---
_is_muon=0; [[ "$RECIPE" == *stage2* && "${OV2_STAGE2_ADAMW:-0}" != "1" ]] && _is_muon=1
[[ "$RECIPE" == *midtrain* && "${OV2_MIDTRAIN_MUON:-0}" == "1" ]] && _is_muon=1   # midtrain Muon momentum is ALSO DP-sharded -> same reshard guard
_wf="$SAVE/.ov2_train_world"
_has_ckpt=0; { [[ -f "$SAVE/latest_checkpointed_iteration.txt" ]] || compgen -G "$SAVE/iter_*" >/dev/null 2>&1; } && _has_ckpt=1
if [[ "$_is_muon" == "1" && "$_has_ckpt" == "1" && -f "$_wf" ]]; then
  _saved_world="$(cat "$_wf" 2>/dev/null || echo "")"
  if [[ -n "$_saved_world" && "$_saved_world" != "$WORLD" ]]; then
    if [[ "${OV2_ALLOW_DP_RESHARD:-0}" == "1" ]]; then
      echo "[ov2-qwen35] WARN: Muon ckpt in $SAVE saved at WORLD=$_saved_world, resuming at WORLD=$WORLD -> DP-sharded momentum MISMATCH; OV2_ALLOW_DP_RESHARD=1 set, continuing (optimizer state WILL be wrong)." >&2
    else
      echo "[ov2-qwen35] FATAL: distributed-Muon ckpt in $SAVE was saved at WORLD=$_saved_world but resuming at WORLD=$WORLD. Muon momentum is DP-sharded with NO reshard support. Resume at WORLD=$_saved_world, OR set OV2_ALLOW_DP_RESHARD=1 to override (momentum garbage)." >&2
      exit 1
    fi
  fi
fi
[[ "$NODE_RANK" -eq 0 ]] && { echo "$WORLD" > "$_wf" 2>/dev/null || true; }
echo "[ov2-qwen35-gb200] in-container | hw=$HWNAME(cc=$_cc) repo=$REPO recipe=$RECIPE accel=$ACCEL mp=$MIXED_PRECISION flex=${OV2_FLEX_BACKEND:-alltoall} recompute_off=$DISABLE_RECOMPUTE recompute_full=$OV2_RECOMPUTE_FULL peak=${MFU_PEAK_TFLOPS}TF nproc=$NPROC world=$WORLD dp=$DP tp=$TP sp=$SP seq=$SEQ_LEN gbs=$MIDTRAIN_GBS iters=$ITERS warmup=$WARMUP_ITERS lr=${OV2_LR:-1e-5}->${OV2_MIN_LR:-1e-6} router_dtype=${OV2_ROUTER_DTYPE:-fp32} permute_fusion=$OV2_MOE_PERMUTE_FUSION aux_loss=$OV2_MOE_AUX_LOSS_COEFF mtp_scale=${OV2_MTP_LOSS_SCALE:-default} alloc=${PYTORCH_CUDA_ALLOC_CONF} offload=${OV2_OPT_OFFLOAD:-false} node_rank=$NODE_RANK nnodes=$NNODES"
# shellcheck disable=SC2086
python -m torch.distributed.run $RDZV --nproc_per_node="$NPROC" scripts/training/run_recipe.py \
  --recipe "$RECIPE" --dataset vlm-energon --step_func ov2_step \
  $OVERRIDES ${EXTRA_ARGS:-} 2>&1 | tee "$SAVE/train_node${NODE_RANK}.log"
