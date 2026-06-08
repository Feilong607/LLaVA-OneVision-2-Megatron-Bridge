#!/usr/bin/env bash
# =============================================================================
# OV2-30B-A3B (Qwen3-30B-A3B MoE) midtrain · GB200/Blackwell · IN-CONTAINER launcher
# Run this INSIDE the training container (you are ALREADY in docker on GB200): it only
# assembles env + run_recipe overrides and execs torchrun — NO `docker run` wrapper.
#
# HARDWARE: auto-detected (A100/A800 sm_80 | H100 sm_90 | GB200 sm_100); override HW=a100|h100|gb200.
#   Nothing is hardwired to GB200 -- NPROC (4 vs 8), MFU peak, NVLS, and fp8 availability switch per HW,
#   so the SAME script runs on A-cards too. On Ampere (no fp8) ACCEL=1/MXFP8 auto-falls back to bf16.
#
# ACCEL MODES (set ACCEL=0|1|2):
#   0  PHASE-1 baseline  : pure bf16 + alltoall + recompute full/uniform/1   (AIAK date0528 parity)
#   1  PHASE-2a MXFP8    : MXFP8 expert/matmul GEMMs + alltoall dispatch (fp8 compute, bf16 comm)
#   2  PHASE-2b HybridEP : bf16 + HybridEP flex dispatcher (NVL72 topology-aware token comm)
#   (MXFP8 + HybridEP-fp8-dispatch is UNSUPPORTED — mcore asserts in fused_a2a — so 1 and 2 are
#    distinct modes; the provider raises a clear error if you force both.)
#
# GB200 knobs:
#   FREE:    192GB HBM -> recompute OFF for ACCEL=1/2 (DISABLE_RECOMPUTE); expandable_segments; NVLS.
#   FIXED (code, ov2_provider.provide): MXFP8 fp8 fields + HybridEP flex dispatcher are now force-set
#            onto the RUNTIME LLM config (cfg.model alone NO-OPs because build_llava_ov2 rebuilds the
#            LLM from HF). Driven by mixed_precision=... (fp8) + OV2_FLEX_BACKEND=... (dispatcher).
#            Verify in the log: "[ov2 provider] fp8 wired ..." / "flex dispatcher wired ...".
#   BLOCKED: CUDA graphs (OV2 forces cuda_graph_impl=none; THD-packed variable seqlens break capture).
#   OPT-IN:  EP comm-overlap (OV2_EP_OVERLAP=1, ~1.3x on exposed a2a but re-validate the grad path);
#            Megatron-FSDP (OV2_FSDP=1, only helps at >=4 nodes/DP>1 — see block below).
#   FIXED:   AdamW for the MoE backbone (Muon+EP deadlocks); TP=1 (OV2 SP/vision only TP1-tested).
# NOTE: REPO points at .../Megatron-Bridge (NOT -refactor) — where the MFU + dataloader-resume +
#       provider fp8/dispatcher fixes live. If you switch REPO, sync those edits there too.
# =============================================================================
set -euo pipefail

# Auto-detect repo root from this script's location (works on A100-2 /ov2 AND GB200 ~/LLaVA-...).
_SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$({ __d="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; while [[ "$__d" != "/" && ! -d "$__d/src/megatron/bridge" ]]; do __d="$(dirname "$__d")"; done; echo "$__d"; })}"
bash "$REPO/3rdparty/apply_megatron_patch.sh" 2>/dev/null || true   # fresh-clone safety: apply OV2 mcore submodule patch (apply_rotary_fn hook)
[[ -d "$REPO/src/megatron/bridge" ]] || { echo "FATAL: OV2 fork root not found from ${BASH_SOURCE[0]} (no src/megatron/bridge above it). Set REPO=/path/to/LLaVA-OneVision-2-Megatron-Bridge" >&2; exit 1; }
RECIPE="${RECIPE:-ov2_35b_a3b_midtrain}"        # full-model midtrain (use ov2_35b_a3b_stage2 for frozen-LLM SFT)
# DATA_PATH / INIT_CKPT / SAVE / model paths: set per-card by the CARD PATH PROFILE (after HW detect).
ITERS="${ITERS:-6094}"; LOG_EVERY="${LOG_EVERY:-1}"; SAVE_EVERY="${SAVE_EVERY:-500}"
# NPROC (GPUs/node) is set by the HARDWARE PROFILE block below (GB200=4, A100/H100=8); override via NPROC=.

# --- HARDWARE PROFILE: A100/A800 (sm_80) <-> Hopper/H100 (sm_90) <-> GB200 (sm_100). Auto-detected
# from compute capability; override with HW=a100|a800|h100|gb200. Sets per-HW DEFAULTS only -- nothing
# is hardwired to GB200, and every value stays env-overridable. This is what lets the SAME script run
# (and be tested) on A-cards as well as GB200. ---
HW="${HW:-auto}"
case "$HW" in
  gb200) _cc=100;; h100|hopper) _cc=90;; a100|a800|ampere) _cc=80;;
  auto) _cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')";;
  *) _cc="";;
esac
_cc="${_cc:-80}"
[[ "$_cc" =~ ^[0-9]+$ ]] || _cc=80    # nvidia-smi may return 'N/A' (MIG/unhealthy); fall back to ampere, don't crash set -u arithmetic
if   [[ "$_cc" -ge 100 ]]; then HWNAME=gb200;  HW_NPROC=4; PEAK_BF16=2250; PEAK_FP8=4500; HW_NVLS=1; HW_FP8=1
elif [[ "$_cc" -ge 90  ]]; then HWNAME=hopper; HW_NPROC=8; PEAK_BF16=989;  PEAK_FP8=1979; HW_NVLS=1; HW_FP8=1
else                            HWNAME=ampere; HW_NPROC=8; PEAK_BF16=312;  PEAK_FP8=312;  HW_NVLS=0; HW_FP8=0
fi
NPROC="${NPROC:-$HW_NPROC}"        # GB200=4 GPU/node, A100/H100=8 GPU/node

# --- CARD PATH PROFILE: A100 (/ov2) <-> GB200 (/datasets). Per-card path DEFAULTS; all env-overridable.
#     The recipe reads OV2_PRETRAIN_ROOT (llava processor+stage_0 ckpt root) and OV2_LLM_HF_30B (Qwen LLM). ---
if [[ "${HWNAME:-}" == "gb200" || -d /datasets/qwen-models-ea5jyi ]]; then
  OV2_LLM_HF_30B="${OV2_LLM_HF_30B:-/datasets/qwen-models-ea5jyi/Qwen3-30B-A3B-Instruct-2507}"
  OV2_PRETRAIN_ROOT="${OV2_PRETRAIN_ROOT:-/datasets/llava/11May}"      # GB200: now only the processor root (stage_0 SKIPPED); better set OV2_HF_PROC_30B directly — PENDING from you
  DATA_PATH="${DATA_PATH:-$REPO/examples/models/qwen/qwen3_vl_ov2/gb200/mid_training_seed85m.yaml}"   # /datasets/llava/11May data
  INIT_CKPT="${INIT_CKPT:-/datasets/stage2}"                           # stage2 resume ckpt (GB200, user-set)
  SAVE="${SAVE:-/home/ftan0055/ckpts_video_sft/ov2_30b_a3b_gb200}"     # output dir (GB200, user-set)
  OV2_SKIP_BASE_STITCH="${OV2_SKIP_BASE_STITCH:-1}"   # GB200: mid-train from stage2 -> skip the stage_0 stitch
else
  OV2_LLM_HF_30B="${OV2_LLM_HF_30B:-/ov2/pretrain_models/Qwen3-30B-A3B-Instruct-2507}"
  OV2_PRETRAIN_ROOT="${OV2_PRETRAIN_ROOT:-/ov2/pretrain_models}"
  DATA_PATH="${DATA_PATH:-$REPO/examples/models/qwen/qwen3_vl_ov2/mid_training_seed85m.yaml}"         # /ov2/dataset_sft data (backup yaml)
  INIT_CKPT="${INIT_CKPT:-/ov2/feilong/gb200/ckpts_video_sft/ov2_30b_a3b_stage2}"
  SAVE="${SAVE:-/ov2/feilong/gb200/ckpts_video_sft/ov2_30b_a3b_gb200}"
  OV2_SKIP_BASE_STITCH="${OV2_SKIP_BASE_STITCH:-0}"   # A100: keep the stage_0 stitch
fi
OV2_HF_PROC_30B="${OV2_HF_PROC_30B:-$OV2_PRETRAIN_ROOT/llava_onevision2/llava_onevision2_30b_a3b/auto_model}"
export OV2_LLM_HF_30B OV2_PRETRAIN_ROOT OV2_SKIP_BASE_STITCH OV2_HF_PROC_30B
export OV2_INIT_CKPT="$INIT_CKPT"   # recipe guard verifies this exists before skipping the stitch

# --- TRAINING MODE (see ACCEL legend above). 30B-A3B is MoE -> optimizer AdamW (distributed Muon
# deadlocks EP backward). Recipe keeps AIAK lr 2e-5 const / clip 1.0 / wd 0 / betas .9,.99 / eps 1e-5
# / gbs 128 / mbs 1 / seq 32000. ---
ACCEL="${ACCEL:-0}"
# MXFP8 needs fp8 tensor cores (Hopper sm_90 / Blackwell sm_100). On Ampere (A100/A800) there are none
# -> auto-fall back to the bf16 baseline so the SAME command is safe to run on A-cards.
if [[ "$ACCEL" == "1" && "$HW_FP8" != "1" ]]; then
  echo "[ov2-30b] WARN: ACCEL=1 (MXFP8) needs fp8 HW; HW=$HWNAME (cc=$_cc) has none -> bf16 baseline (ACCEL=0)." >&2
  ACCEL=0
fi
if [[ "$ACCEL" == "1" ]]; then          # Phase-2a: MXFP8 + alltoall (fp8 HW only)
  MIXED_PRECISION="${MIXED_PRECISION:-bf16_with_mxfp8_mixed}"
  DISABLE_RECOMPUTE="${DISABLE_RECOMPUTE:-1}"; OV2_RECOMPUTE_FULL="${OV2_RECOMPUTE_FULL:-0}"
  FLEX_BACKEND="${FLEX_BACKEND:-}"                       # MUST stay empty: alltoall (HybridEP+fp8 unsupported)
  MFU_PEAK_TFLOPS="${MFU_PEAK_TFLOPS:-$PEAK_FP8}"        # fp8 tensor-core peak (MFU vs fp8)
elif [[ "$ACCEL" == "2" ]]; then        # Phase-2b: bf16 + HybridEP (best on NVL72; allowed elsewhere)
  MIXED_PRECISION="${MIXED_PRECISION:-bf16_mixed}"   # registry key is 'bf16_mixed' (plain 'bf16' is NOT a recipe -> ValueError)
  DISABLE_RECOMPUTE="${DISABLE_RECOMPUTE:-1}"; OV2_RECOMPUTE_FULL="${OV2_RECOMPUTE_FULL:-0}"
  FLEX_BACKEND="${FLEX_BACKEND:-hybridep}"
  MFU_PEAK_TFLOPS="${MFU_PEAK_TFLOPS:-$PEAK_BF16}"
else                                    # Phase-1: AIAK bf16 baseline (the A-card default)
  MIXED_PRECISION="${MIXED_PRECISION:-bf16_mixed}"   # registry key is 'bf16_mixed' (plain 'bf16' is NOT a recipe -> ValueError)
  DISABLE_RECOMPUTE="${DISABLE_RECOMPUTE:-0}"; OV2_RECOMPUTE_FULL="${OV2_RECOMPUTE_FULL:-1}"  # AIAK full/uniform/1
  FLEX_BACKEND="${FLEX_BACKEND:-}"
  MFU_PEAK_TFLOPS="${MFU_PEAK_TFLOPS:-$PEAK_BF16}"
fi
export OV2_RECOMPUTE_FULL MFU_PEAK_TFLOPS
# OV2_FLEX_BACKEND is what ov2_provider.provide() reads to wire the runtime dispatcher (the cfg.model
# field is dead). Empty -> verified alltoall; "hybridep" -> HybridEP (also disables shared-expert-overlap).
export OV2_FLEX_BACKEND="$FLEX_BACKEND"

# --- rendezvous: 4 GPU/node. Multi-node via LIST_IP (run the SAME cmd on EACH node) ---
#   2 nodes (8 GPU, EP8/DP1):  LIST_IP="<ip0> <ip1>" bash ax_ov2_30b_a3b_gb200.sh
#   4 nodes (16 GPU, EP8/DP2): LIST_IP="<ip0> <ip1> <ip2> <ip3>" bash ax_ov2_30b_a3b_gb200.sh
#   EP8 spans 2 nodes (4+4) -> MoE all-to-all crosses the node boundary; needs NVLink5/NVL72 or IB.
if [[ -n "${LIST_IP:-}" ]]; then read -ra list_ip <<< "$LIST_IP"; else list_ip=(); fi
NNODES=${#list_ip[@]}
if [[ "$NNODES" -le 1 ]]; then
  RDZV="--standalone"; NODE_RANK=0; NNODES=1
else
  MASTER_ADDR="${list_ip[0]}"; MASTER_PORT="${MASTER_PORT:-26047}"
  CURRENT_IP="$(hostname -I | awk '{print $1}')"; NODE_RANK=-1
  for i in "${!list_ip[@]}"; do [[ "${list_ip[$i]}" == "$CURRENT_IP" ]] && NODE_RANK=$i && break; done
  [[ "$NODE_RANK" -eq -1 ]] && { echo "ERROR: $CURRENT_IP not in LIST_IP (${list_ip[*]})" >&2; exit 1; }
  RDZV="--nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT"
fi
WORLD=$(( NPROC * NNODES ))
# midtrain GBS=128 -> the microbatch calc needs GBS % (mbs * DP) == 0 with DP=WORLD (TP1/PP1/CP1).
# Validated: 2 nodes (WORLD 8 -> 16 microbatches) and 4 nodes (WORLD 16 -> 8). Odd counts (e.g. 3 nodes,
# WORLD 12) hard-assert in mcore -> warn (use 2/4 nodes, or override train.global_batch_size).
(( 128 % WORLD == 0 )) || echo "[ov2-30b] WARN: world=$WORLD does not divide default GBS=128 -> mcore microbatch assert will fire; use 2 or 4 nodes (WORLD 8/16) or override train.global_batch_size." >&2

# --- in-container env (were docker -e flags) ---
export PYTHONPATH="$REPO/src:$REPO/3rdparty/Megatron-LM:$REPO/aiak_shim${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OV2_MOE_PERMUTE_FUSION="${OV2_MOE_PERMUTE_FUSION:-0}"  # avoid TE Triton MoE-permute wedge (30B-A3B fix)
export OV2_PACK_FULL_CAUSAL="${OV2_PACK_FULL_CAUSAL:-0}"      # 0=THD block-diagonal (AIAK-faithful); 1=full-causal
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_GRAPH_REGISTER="${NCCL_GRAPH_REGISTER:-0}" NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-$HW_NVLS}"  # NVLS: GB200/H100 on, Ampere off
[[ -n "${NCCL_IB_HCA:-}" ]] && export NCCL_IB_HCA NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
# --- no-internet (GB200) JIT hygiene: keep Triton/Inductor caches on NODE-LOCAL /tmp (NOT the shared
# /ov2 FS — a networked JIT cache is slow and races). moe_permute_fusion=0 already avoids the one TE
# Triton kernel that wedged 30B-A3B. If a same-node first-touch compile race ever appears, switch to a
# per-rank dir via a torchrun --no-python wrapper that sets TRITON_CACHE_DIR=/tmp/ov2_triton/r$RANK. ---
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/ov2_triton_cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/ov2_inductor_cache}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

# --- HybridEP topology (ACCEL=2 only): # of EP-communicating ranks sharing one NVLink domain. On a
# single NVL72 rack ALL GPUs are one domain -> = world size. Verify with gb200_check.sh topo; override
# (e.g. =4) if EP ranks are split across racks/domains — a wrong value hurts perf and sometimes
# CORRECTNESS (mcore HybridEP). ---
if [[ "$ACCEL" == "2" ]]; then
  # # of EP ranks (of the EP=8 group) sharing one NVLink domain. mcore asserts EP(8) % value == 0,
  # so it must DIVIDE 8 (NOT equal world size). On a full NVL72 rack all 8 EP ranks share one domain -> 8;
  # if EP8 is split 4+4 across two NVLink domains, set 4. A wrong value -> mcore assert / perf-correctness loss.
  export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN="${NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN:-8}"
  (( 8 % NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN == 0 )) || {
    echo "ERROR: NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=$NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN must divide EP=8." >&2; exit 1; }
fi
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

# --- run_recipe.py CLI overrides ---
OVERRIDES="dataset.path=$DATA_PATH"
[[ "$INIT_CKPT" != "null" && -n "$INIT_CKPT" ]] && OVERRIDES="$OVERRIDES checkpoint.pretrained_checkpoint=$INIT_CKPT"
OVERRIDES="$OVERRIDES checkpoint.save=$SAVE checkpoint.load=$SAVE dataset.dataloader_save=$SAVE"
OVERRIDES="$OVERRIDES checkpoint.save_interval=$SAVE_EVERY train.train_iters=$ITERS validation.eval_iters=0 logger.log_interval=$LOG_EVERY train.micro_batch_size=1"   # packing REQUIRES mbs=1 (model asserts batch==1)
OVERRIDES="$OVERRIDES scheduler.lr_warmup_iters=0"   # AIAK: warmup 0 -> constant 2e-5 from step 1
OVERRIDES="$OVERRIDES mixed_precision=$MIXED_PRECISION"
[[ "$DISABLE_RECOMPUTE" == "1" ]] && OVERRIDES="$OVERRIDES model.recompute_activations=false model.recompute_granularity=null"
# NB: the dispatcher is wired in ov2_provider.provide() via OV2_FLEX_BACKEND (above) — setting
# model.moe_token_dispatcher_type here would be a DEAD field (build_llava_ov2 rebuilds the LLM).

# --- NOTE on EP comm-overlap (a ~1.3x step-time lever on exposed EP a2a, per the perf skills): it is
# NOT exposed as a flag here because the OV2 recipe sets cfg.comm_overlap=None, so a CLI override
# (comm_overlap.overlap_moe_expert_parallel_comm=true) would crash (attr-on-None). Enabling it needs a
# RECIPE change: build a CommOverlapConfig in ov2.py AND re-validate the grad path (the prior OV2 run
# flagged EP/DDP grad-overlap as fragile here). Prereqs once wired: PyTorch>=2.6 (else silent hang),
# selective (NOT full) recompute, moe_shared_expert_overlap=False (HybridEP sets this automatically). ---

# --- OPT-IN: Megatron-FSDP (OV2_FSDP=1). Shards params+grads+optim across the DATA-parallel dim. With
# EP8 fixed: DP = world/EP, so on 8 GPU (2 nodes) DP=1 -> FSDP is a NO-OP that only adds overhead; it
# only helps at >=4 nodes (DP>=2) AND when MODEL-STATE (not activation) memory is the limit. OV2's
# 30B-A3B@seq32k pain is activation-bound -> FSDP will NOT fix that (use recompute/CP/offload instead).
# fsdp_dtensor ckpts are a ONE-WAY door: NOT loadable by the torch_dist recipe or the convert/ tools. ---
if [[ "${OV2_FSDP:-0}" == "1" ]]; then
  unset CUDA_DEVICE_MAX_CONNECTIONS            # Megatron-FSDP asserts CUDA_DEVICE_MAX_CONNECTIONS != 1
  OVERRIDES="$OVERRIDES dist.use_megatron_fsdp=true ddp.use_megatron_fsdp=true"
  OVERRIDES="$OVERRIDES ddp.data_parallel_sharding_strategy=optim_grads_params ddp.average_in_collective=false"
  OVERRIDES="$OVERRIDES checkpoint.ckpt_format=fsdp_dtensor"   # FSDP forces this; see ckpt caveat below
  # CKPT CAVEAT: ckpt_format=fsdp_dtensor makes Bridge's save/load expect fsdp_dtensor. The DEFAULT
  # INIT_CKPT (ov2_30b_a3b_stage2) is torch_dist -> loading it as pretrained_checkpoint will MISMATCH.
  # For an FSDP run you must EITHER (a) first convert the init ckpt to fsdp_dtensor, OR (b) point
  # INIT_CKPT/SAVE at an fsdp_dtensor ckpt. fsdp_dtensor saves are NOT loadable by the torch_dist
  # recipe or the convert/ tools (one-way). Keep your torch_dist baseline for portability.
  echo "[ov2-30b-gb200] OV2_FSDP=1: Megatron-FSDP ON (world=$WORLD nnodes=$NNODES). FSDP only shards when \
DP=world/EP8 >= 2 (i.e. >=4 nodes at EP8); fewer -> NO-OP that only adds overhead. fsdp_dtensor ckpt is \
ONE-WAY; torch_dist INIT_CKPT will mismatch -> convert it to fsdp_dtensor or point INIT_CKPT at an fsdp ckpt." >&2
fi

mkdir -p "$SAVE"; cd "$REPO"
echo "[ov2-30b-gb200] in-container | hw=$HWNAME(cc=$_cc) repo=$REPO recipe=$RECIPE accel=$ACCEL mp=$MIXED_PRECISION flex=${OV2_FLEX_BACKEND:-alltoall} recompute_off=$DISABLE_RECOMPUTE recompute_full=$OV2_RECOMPUTE_FULL peak=${MFU_PEAK_TFLOPS}TF nproc=$NPROC world=$WORLD node_rank=$NODE_RANK nnodes=$NNODES"
# shellcheck disable=SC2086  # $RDZV and $OVERRIDES must word-split into separate args
python -m torch.distributed.run $RDZV --nproc_per_node="$NPROC" scripts/training/run_recipe.py \
  --recipe "$RECIPE" --dataset vlm-energon --step_func ov2_step \
  $OVERRIDES 2>&1 | tee "$SAVE/train_node${NODE_RANK}.log"
