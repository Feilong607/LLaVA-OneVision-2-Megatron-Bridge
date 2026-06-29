#!/usr/bin/env bash
# =============================================================================
# OV2-30B-A3B (Qwen3-30B-A3B MoE) midtrain - B200 (x86 Blackwell, 8 GPU/node) - IN-CONTAINER launcher
# Run this INSIDE the training container (you are ALREADY in docker on the B200 node): it only
# assembles env + run_recipe overrides and execs torchrun -- NO `docker run` wrapper.
#
# HARDWARE: auto-detected (A100/A800 sm_80 | H100 sm_90 | B200 sm_100, 8 GPU/node); override HW=a100|h100|b200.
#   Nothing is hardwired -- NPROC, MFU peak, NVLS, and fp8 availability switch per HW (B200=8 GPU/node),
#   so the SAME script runs on A-cards too. On Ampere (no fp8) ACCEL=1/MXFP8 auto-falls back to bf16.
#
# ACCEL MODES (set ACCEL=0|1|2):
#   0  PHASE-1 baseline  : pure bf16 + alltoall + recompute full/uniform/1 (AIAK date0528 parity, on B200 too). DEFAULT mode.
#   1  PHASE-2a MXFP8    : MXFP8 expert/matmul GEMMs + alltoall dispatch (fp8 compute, bf16 comm)
#   2  PHASE-2b HybridEP : bf16 + HybridEP flex dispatcher (intra-node 8-GPU NVSwitch domain on B200; EP8 fits 1 node)
#   (MXFP8 + HybridEP-fp8-dispatch is UNSUPPORTED -- mcore asserts in fused_a2a -- so 1 and 2 are
#    distinct modes; the provider raises a clear error if you force both.)
#
# B200 knobs:
#   FREE:    192GB HBM -> recompute OFF for ACCEL=1/2 on B200 (DISABLE_RECOMPUTE=1); ACCEL=0 keeps AIAK recompute; expandable_segments; NVLS.
#   FIXED (code, ov2_provider.provide): MXFP8 fp8 fields + HybridEP flex dispatcher are now force-set
#            onto the RUNTIME LLM config (cfg.model alone NO-OPs because build_llava_ov2 rebuilds the
#            LLM from HF). Driven by mixed_precision=... (fp8) + OV2_FLEX_BACKEND=... (dispatcher).
#            Verify in the log: "[ov2 provider] fp8 wired ..." / "flex dispatcher wired ...".
#   BLOCKED: CUDA graphs (OV2 forces cuda_graph_impl=none; THD-packed variable seqlens break capture).
#   OPT-IN:  EP comm-overlap (OV2_EP_OVERLAP=1, ~1.3x on exposed a2a but re-validate the grad path);
#            Megatron-FSDP (OV2_FSDP=1, only helps at >=4 nodes/DP>1 -- see block below).
#   FIXED:   AdamW for the MoE backbone (Muon+EP deadlocks); TP=1 (OV2 SP/vision only TP1-tested).
# NOTE: REPO points at .../Megatron-Bridge (NOT -refactor) -- where the MFU + dataloader-resume +
#       provider fp8/dispatcher fixes live. If you switch REPO, sync those edits there too.
# =============================================================================
set -euo pipefail
REPO=/home/ftan0055/LLaVA-OneVision-2-Megatron-Bridge

# Auto-detect repo root from this script's location (works on A100-2 /ov2 AND B200 ~/LLaVA-...).
_SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$({ __d="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; while [[ "$__d" != "/" && ! -d "$__d/src/megatron/bridge" ]]; do __d="$(dirname "$__d")"; done; echo "$__d"; })}"
[[ -d "$REPO/src/megatron/bridge" ]] || { echo "FATAL: OV2 fork root not found from ${BASH_SOURCE[0]} (no src/megatron/bridge above it). Set REPO=/path/to/LLaVA-OneVision-2-Megatron-Bridge" >&2; exit 1; }
bash "$REPO/3rdparty/apply_megatron_patch.sh"   # fresh-clone safety: apply OV2 mcore submodule patch (apply_rotary_fn hook). FAIL LOUD (no 2>/dev/null||true): a silently-missing hook -> cryptic "unexpected keyword argument 'apply_rotary_fn'" at build. Script is idempotent (no-op if already applied).
RECIPE="${RECIPE:-ov2_30b_a3b_p16m33_midtrain}"  # p16m33 full-model midtrain; for frozen-LLM SFT use ov2_30b_a3b_p16m33_stage2. NB: /home/ftan0055/llava-ov2-30b-a3b-m9lvdn is a p16m33 ckpt -> MUST use a p16m33 recipe (the merge2 ov2_35b_a3b_* binds patch14/merge2 + the non-p16m33 processor -> vision-config/ckpt MISMATCH).
# DATA_PATH / INIT_CKPT / SAVE / model paths: set per-card by the CARD PATH PROFILE (after HW detect).
# Match the A800 midtrain launcher's tunable constants. The recipe reads the exported OV2_*
# values below, so train_iters / LR schedule / task_encoder seq_length stay in sync.
MIDTRAIN_GBS="${OV2_MIDTRAIN_GBS:-16}"                  # global batch size; override with OV2_MIDTRAIN_GBS=
MIDTRAIN_N_SAMPLES="${OV2_MIDTRAIN_N_SAMPLES:-780000}"  # LLaVA-Next 780k default
ITERS="${ITERS:-$(( (MIDTRAIN_N_SAMPLES + MIDTRAIN_GBS - 1) / MIDTRAIN_GBS ))}"
# LR warmup = 0.002 * train_iters (AIAK stage-1 fraction; ~97 @ 48750 iters). Gentle ramp 0->2e-5 then
# constant (min_lr==max_lr). Override OV2_WARMUP_ITERS= (e.g. =0 for AIAK pure-constant-LR).
WARMUP_ITERS="${OV2_WARMUP_ITERS:-$(( ITERS * 2 / 1000 ))}"
if [ "$WARMUP_ITERS" -lt 1 ]; then WARMUP_ITERS=1; fi
LOG_EVERY="${LOG_EVERY:-1}"; SAVE_EVERY="${SAVE_EVERY:-2000}"
# NPROC (GPUs/node) is set by the HARDWARE PROFILE block below (B200/A100/H100=8); override via NPROC=.

# --- HARDWARE PROFILE: A100/A800 (sm_80) <-> Hopper/H100 (sm_90) <-> B200 (sm_100, 8 GPU/node). Auto-detected
# from compute capability; override with HW=a100|a800|h100|b200. Sets per-HW DEFAULTS only -- nothing
# is hardwired to B200, and every value stays env-overridable. This is what lets the SAME script run
# (and be tested) on A-cards as well as B200. ---
HW="${HW:-auto}"
case "$HW" in
  b200|blackwell) _cc=100;; h100|hopper) _cc=90;; a100|a800|ampere) _cc=80;;
  auto) _cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')";;
  *) _cc="";;
esac
_cc="${_cc:-80}"
[[ "$_cc" =~ ^[0-9]+$ ]] || _cc=80    # nvidia-smi may return 'N/A' (MIG/unhealthy); fall back to ampere, don't crash set -u arithmetic
if   [[ "$_cc" -ge 100 ]]; then HWNAME=b200;   HW_NPROC=8; PEAK_BF16=2250; PEAK_FP8=4500; HW_NVLS=1; HW_FP8=1
elif [[ "$_cc" -ge 90  ]]; then HWNAME=hopper; HW_NPROC=8; PEAK_BF16=989;  PEAK_FP8=1979; HW_NVLS=1; HW_FP8=1
else                            HWNAME=ampere; HW_NPROC=8; PEAK_BF16=312;  PEAK_FP8=312;  HW_NVLS=0; HW_FP8=0
fi
NPROC="${NPROC:-$HW_NPROC}"        # B200/A100/H100 = 8 GPU/node
TP="${TP:-1}"                      # 1 B200 node (world=8) -> TP=1 so DP=8 satisfies EP=8 ENTIRELY intra-node. TP=2 needs >=2 nodes.
if [[ "$TP" -gt 1 ]]; then SP=true; else SP=false; fi
SEQ_LEN="${OV2_SEQ_LEN:-10192}"    # seed85m offline-packed length; matches A800 launcher
# MoE token capacity control. B200 has enough memory, so default to no token dropping / closest
# no-capacity behavior. If memory gets tight, try MOE_CAPACITY_FACTOR=1.0 MOE_PAD_TO_CAPACITY=true,
# or use 1.25 / 1.5 with pad-to-capacity as a middle ground.
MOE_CAPACITY_FACTOR="${MOE_CAPACITY_FACTOR:-none}"
MOE_PAD_TO_CAPACITY="${MOE_PAD_TO_CAPACITY:-false}"
MOE_CAPACITY_ARGS=""
if [[ -n "$MOE_CAPACITY_FACTOR" && "$MOE_CAPACITY_FACTOR" != "none" && "$MOE_CAPACITY_FACTOR" != "None" && "$MOE_CAPACITY_FACTOR" != "-1" ]]; then
  MOE_CAPACITY_ARGS="model.moe_expert_capacity_factor=$MOE_CAPACITY_FACTOR model.moe_pad_expert_input_to_capacity=$MOE_PAD_TO_CAPACITY"
fi

# --- CARD PATH PROFILE: A100 (/ov2) <-> B200 (/datasets). Per-card path DEFAULTS; all env-overridable.
#     !! VERIFY ON THE B200 BOX: these /datasets defaults are placeholders and may differ on B200. Set
#        OV2_LLM_HF_30B / OV2_HF_PROC_30B / DATA_PATH / INIT_CKPT / SAVE explicitly if the mounts differ. !!
#     The recipe reads OV2_PRETRAIN_ROOT (llava processor+stage_0 ckpt root) and OV2_LLM_HF_30B (Qwen LLM). ---
if [[ "${HWNAME:-}" == "b200" || -d /datasets/qwen-models-ea5jyi ]]; then
  OV2_LLM_HF_30B="${OV2_LLM_HF_30B:-/datasets/qwen-models-ea5jyi/Qwen3-30B-A3B-Instruct-2507}"
  OV2_HF_PROC_30B="${OV2_HF_PROC_30B:-/home/ftan0055/llava-ov2-30b-a3b-m9lvdn/auto_model}"            # bundled processor (B200)
  OV2_HF_PROC_30B_P16M33="${OV2_HF_PROC_30B_P16M33:-/home/ftan0055/llava-ov2-30b-a3b-m9lvdn/auto_model}"   # bundled processor (B200, p16m33 recipe)
  OV2_PRETRAIN_ROOT="${OV2_PRETRAIN_ROOT:-/datasets/llava/11May}"      # B200: now only the processor root (stage_0 SKIPPED); OV2_HF_PROC_30B set directly above to the bundled auto_model
  DATA_PATH="${DATA_PATH:-$REPO/examples/models/qwen/qwen3_vl_ov2/b200/mid_training_seed85m.yaml}"   # /datasets/llava/11May data (b200/ copy)
  INIT_CKPT="${INIT_CKPT:-/home/ftan0055/llava-ov2-30b-a3b-m9lvdn}"   # trained ckpt to resume (B200; has iter_0001000 + auto_model)
  SAVE="${SAVE:-/home/ftan0055/ckpts_video_sft/ov2_30b_a3b_b200}"     # output dir (B200, user-set)
  OV2_SKIP_BASE_STITCH="${OV2_SKIP_BASE_STITCH:-1}"   # B200: mid-train from stage2 -> skip the stage_0 stitch
else
  OV2_LLM_HF_30B="${OV2_LLM_HF_30B:-/ov2/pretrain_models/Qwen3-30B-A3B-Instruct-2507}"
  OV2_PRETRAIN_ROOT="${OV2_PRETRAIN_ROOT:-/ov2/pretrain_models}"
  DATA_PATH="${DATA_PATH:-$REPO/examples/models/qwen/qwen3_vl_ov2/mid_training_seed85m.yaml}"         # /ov2/dataset_sft data (backup yaml)
  INIT_CKPT="${INIT_CKPT:-/ov2/feilong/gb200/ckpts_video_sft/ov2_30b_a3b_stage2}"
  SAVE="${SAVE:-/ov2/feilong/gb200/ckpts_video_sft/ov2_30b_a3b_b200}"
  OV2_SKIP_BASE_STITCH="${OV2_SKIP_BASE_STITCH:-0}"   # A100: keep the stage_0 stitch
fi
OV2_HF_PROC_30B="${OV2_HF_PROC_30B:-$OV2_PRETRAIN_ROOT/llava_onevision2/llava_onevision2_30b_a3b/auto_model}"
# p16m33 processor (patch16/merge3) - platform-aware via OV2_PRETRAIN_ROOT; override directly if B200 puts it elsewhere.
OV2_HF_PROC_30B_P16M33="${OV2_HF_PROC_30B_P16M33:-$OV2_PRETRAIN_ROOT/llava_onevision2/llava_onevision2_30b_a3b_p16_m33/auto_model}"
export OV2_LLM_HF_30B OV2_PRETRAIN_ROOT OV2_SKIP_BASE_STITCH OV2_HF_PROC_30B OV2_HF_PROC_30B_P16M33
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
else                                    # Phase-1: bf16 baseline -- the DEFAULT mode (also the A-card default)
  MIXED_PRECISION="${MIXED_PRECISION:-bf16_mixed}"   # registry key is 'bf16_mixed' (plain 'bf16' is NOT a recipe -> ValueError)
  # ACCEL=0 baseline: recompute ON (AIAK full/uniform/1 parity). B200 192GB fits it comfortably; recompute
  # is mathematically identical to recompute-off (only trades compute for memory). Set DISABLE_RECOMPUTE=1
  # to turn it OFF for speed on B200 memory headroom.
  DISABLE_RECOMPUTE="${DISABLE_RECOMPUTE:-0}"; OV2_RECOMPUTE_FULL="${OV2_RECOMPUTE_FULL:-1}"   # B200: recompute ON (AIAK parity)
  FLEX_BACKEND="${FLEX_BACKEND:-}"
  MFU_PEAK_TFLOPS="${MFU_PEAK_TFLOPS:-$PEAK_BF16}"
fi
export OV2_RECOMPUTE_FULL MFU_PEAK_TFLOPS
# OV2_FLEX_BACKEND is what ov2_provider.provide() reads to wire the runtime dispatcher (the cfg.model
# field is dead). Empty -> verified alltoall; "hybridep" -> HybridEP (also disables shared-expert-overlap).
export OV2_FLEX_BACKEND="$FLEX_BACKEND"

# --- rendezvous: AUTO-DETECT master/worker. NO hardcoded IPs -> survives pod reschedules (k8s pod IPs
#   change every restart!). GPUS_PER_NODE = $NPROC (B200=8 from the HW profile above).
#   Priority:
#     (1) operator-injected env (PyTorchJob / Run:AI auto-inject PET_* + MASTER_ADDR/WORLD_SIZE) -> TRUE auto
#     (2) manual LIST_IP="<ip0> <ip1> ..." + hostname/IP match (run SAME cmd on each node, no operator)
#     (3) single-node test
#   EP8 FITS IN ONE 8-GPU B200 node -> MoE all-to-all stays INTRA-node over NVSwitch (no cross-node a2a). Multi-node only scales DP over IB.
GPUS_PER_NODE="$NPROC"
if [[ -n "${PET_NNODES:-}" || ( -n "${MASTER_ADDR:-}" && -n "${WORLD_SIZE:-}" ) ]]; then
  # (1) K8s auto: prefer the exact PET_* the operator injects; fall back to WORLD_SIZE/RANK arithmetic.
  NNODES="${PET_NNODES:-$(( WORLD_SIZE / GPUS_PER_NODE ))}"
  NODE_RANK="${PET_NODE_RANK:-$(( ${RANK:-0} / GPUS_PER_NODE ))}"
  MASTER_ADDR="${MASTER_ADDR:-${PET_MASTER_ADDR:-}}"
  MASTER_PORT="${MASTER_PORT:-${PET_MASTER_PORT:-26047}}"
  [[ "$NNODES" -gt 1 && -z "${MASTER_ADDR}" ]] && { echo "[ov2-30b] FATAL: multi-node but neither MASTER_ADDR nor PET_MASTER_ADDR injected by the operator." >&2; exit 1; }
  RUN_MODE="multi-node (K8s auto-detected)"
elif [[ -n "${LIST_IP:-}" ]]; then
  # (2) manual: derive NODE_RANK by matching this host's IP/name against the list; master = list[0].
  read -ra list_ip <<< "$LIST_IP"
  NNODES=${#list_ip[@]}; MASTER_ADDR="${list_ip[0]}"; MASTER_PORT="${MASTER_PORT:-26047}"
  CURRENT_IP="$(hostname -I | awk '{print $1}')"; CURRENT_HOST="$(hostname)"; NODE_RANK=-1
  for i in "${!list_ip[@]}"; do [[ "${list_ip[$i]}" == "$CURRENT_IP" || "${list_ip[$i]}" == "$CURRENT_HOST" ]] && NODE_RANK=$i && break; done
  [[ "$NODE_RANK" -eq -1 ]] && { echo "[ov2-30b] ERROR: this host IP($CURRENT_IP)/name($CURRENT_HOST) not in LIST_IP (${list_ip[*]})" >&2; exit 1; }
  RUN_MODE="multi-node (manual LIST_IP)"
else
  # (3) single-node test (no operator env, no LIST_IP)
  NNODES=1; NODE_RANK=0; MASTER_ADDR=127.0.0.1; MASTER_PORT="${MASTER_PORT:-26047}"
  RUN_MODE="single-node TEST"
fi
# --- k8s DNS hardening: the operator injects MASTER_ADDR as a SHORT pod name (feilong-...-master-0),
# which intermittently fails to resolve from worker pods (gai error -2 / IPv6) -> the c10d store
# rendezvous times out at 600000ms and the whole job dies. If MASTER_ADDR is a bare short name and its
# k8s FQDN resolves, switch to the FQDN (reliably resolvable from every pod). Namespace via POD_NAMESPACE
# (k8s downward API) or OV2_K8S_NAMESPACE; defaults to runai-mv0004. ---
if [[ "$NNODES" -gt 1 && -n "${MASTER_ADDR:-}" && "$MASTER_ADDR" != *.* && "$MASTER_ADDR" != "127.0.0.1" ]]; then
  _ns="${POD_NAMESPACE:-${OV2_K8S_NAMESPACE:-runai-mv0004}}"
  _fqdn="${MASTER_ADDR}.${_ns}.svc.cluster.local"
  if getent hosts "$_fqdn" >/dev/null 2>&1; then
    echo "[ov2-30b-b200] rdzv: short MASTER_ADDR '$MASTER_ADDR' -> FQDN '$_fqdn' (avoids gai-error rendezvous timeout)" >&2
    MASTER_ADDR="$_fqdn"
  else
    echo "[ov2-30b-b200] WARN: MASTER_ADDR='$MASTER_ADDR' is a short name and FQDN '$_fqdn' does not resolve here; rendezvous may time out. Set OV2_K8S_NAMESPACE=<ns> or pass a resolvable MASTER_ADDR." >&2
  fi
fi
if [[ "$NNODES" -le 1 ]]; then RDZV="--standalone"; NNODES=1; NODE_RANK=0; else
  RDZV="--nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT"; fi
echo "[ov2-30b-b200] --- rdzv: $RUN_MODE --- master=${MASTER_ADDR:-n/a}:${MASTER_PORT} nnodes=$NNODES node_rank=$NODE_RANK gpus/node=$GPUS_PER_NODE"
WORLD=$(( NPROC * NNODES ))
(( TP >= 1 )) || { echo "[ov2-30b] FATAL: TP must be >=1, got TP=$TP" >&2; exit 1; }
(( WORLD % TP == 0 )) || { echo "[ov2-30b] FATAL: WORLD=$WORLD must be divisible by TP=$TP." >&2; exit 1; }
DP=$(( WORLD / TP ))
# midtrain GBS -> microbatch calc needs GBS % (mbs * DP) == 0. PP/CP are 1 here.
(( MIDTRAIN_GBS % DP == 0 )) || echo "[ov2-30b] WARN: DP=$DP (WORLD=$WORLD / TP=$TP) does not divide GBS=$MIDTRAIN_GBS -> mcore microbatch assert will fire; adjust TP/NNODES or OV2_MIDTRAIN_GBS." >&2
# EP=8 is fixed in the recipe -> mcore requires data parallel size to be a multiple of EP and >= EP.
(( DP >= 8 && DP % 8 == 0 )) || { echo "[ov2-30b] FATAL: EP=8 needs DP=$DP (WORLD=$WORLD / TP=$TP) to be a multiple of 8 and >=8. 1 B200 node (WORLD=8) = EP8 with TP=1; TP=2 needs >=2 B200 nodes." >&2; exit 1; }

# --- in-container env (were docker -e flags) ---
export PYTHONPATH="$REPO/src:$REPO/3rdparty/Megatron-LM:$REPO/aiak_shim${PYTHONPATH:+:$PYTHONPATH}"
export OV2_SKIP_HELPERS="${OV2_SKIP_HELPERS:-1}"   # energon doesn't use helpers_cpp -> skip the C++ index-builder compile
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OV2_MOE_PERMUTE_FUSION="${OV2_MOE_PERMUTE_FUSION:-0}"  # avoid TE Triton MoE-permute wedge (30B-A3B fix)
export OV2_MOE_AUX_LOSS_COEFF="${OV2_MOE_AUX_LOSS_COEFF:-0.01}"  # AIAK midtrain load-balance coeff (HF default 0.001); build_llava_ov2 forces it on the built LLM
export OV2_PACK_FULL_CAUSAL="${OV2_PACK_FULL_CAUSAL:-0}"      # 0=THD block-diagonal (AIAK-faithful); 1=full-causal
export OV2_SEQ_LEN="$SEQ_LEN"                                # recipe reads this at import -> model+dataset+task_encoder
export OV2_MIDTRAIN_GBS="$MIDTRAIN_GBS" OV2_MIDTRAIN_N_SAMPLES="$MIDTRAIN_N_SAMPLES"
# Timing breakdown (opt-in): OV2_TIMING_LOG_LEVEL=1|2 enables the Megatron per-rank (min,max) timing
# block (appended to OVERRIDES below). OV2_TIMING_PRINT_INTERVAL=N prints that block every N iters
# (default 50) so it does not spam every iter; the default-off path is byte-identical (train_utils.py
# gates on this env, defaulting to the original per-log_interval call). forward/backward-compute timers
# also need the setup.py inner-config-timers fix already in this repo -> if REPO is a synced /home copy,
# sync src/megatron/bridge/training/{setup.py,utils/train_utils.py} there too.
export OV2_TIMING_PRINT_INTERVAL="${OV2_TIMING_PRINT_INTERVAL:-50}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_GRAPH_REGISTER="${NCCL_GRAPH_REGISTER:-0}" NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-$HW_NVLS}"  # NVLS: B200/H100 on (intra-node NVSwitch), Ampere off
[[ -n "${NCCL_IB_HCA:-}" ]] && export NCCL_IB_HCA NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
# --- B200 (x86) NCCL tuning: intra-node NVSwitch (NVLink5) + InfiniBand cross-node. NO MNNVL (x86 has no
# multi-node NVLink); cross-node hops go over IB. All env-overridable. ---
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker}"        # bootstrap iface: exclude loopback + docker bridge
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"                              # WARN = quiet (hides the harmless GIN/SHARP plugin-probe INFO). For multi-node bring-up debugging: NCCL_DEBUG=INFO bash ...
export NCCL_MNNVL_ENABLE="${NCCL_MNNVL_ENABLE:-0}"                   # x86 B200: NO multi-node NVLink (cross-node = IB). MNNVL is GB200/NVL72-only -> keep OFF.
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"                       # P2P over NVLink
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-SYS}"
# NCCL_NET_GDR_C2C: Grace<->Blackwell C2C GPUDirect is a GB200-only path; x86 B200 has no Grace C2C -> NOT set.
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-1}"
export NVLINK_DOMAIN_SIZE="${NVLINK_DOMAIN_SIZE:-8}"                # B200: one node NVSwitch = 8 GPUs in one NVLink domain (NOT NVL72)
export NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT:-22}" NCCL_IB_RETRY_CNT="${NCCL_IB_RETRY_CNT:-7}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export UCX_TLS="${UCX_TLS:-tcp}"
# NCCL_ALGO: SAFE set only. CollnetDirect,CollnetChain (IB SHARP) CRASHED this job before with
# "Detected mixed local CollNet device counts across ranks (min 0, max 2)"; Tree/Ring/NVLSTree route over
# NVLink. Re-add CollNet ONLY if your b200_nccl_test passes WITH it: NCCL_ALGO=...,CollnetDirect,CollnetChain.
export NCCL_ALGO="${NCCL_ALGO:-Tree,Ring,NVLSTree}"
# --- no-internet (B200) JIT hygiene: keep Triton/Inductor caches on NODE-LOCAL /tmp (NOT the shared
# /ov2 FS -- a networked JIT cache is slow and races). moe_permute_fusion=0 already avoids the one TE
# Triton kernel that wedged 30B-A3B. If a same-node first-touch compile race ever appears, switch to a
# per-rank dir via a torchrun --no-python wrapper that sets TRITON_CACHE_DIR=/tmp/ov2_triton/r$RANK. ---
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/ov2_triton_cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/ov2_inductor_cache}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

# --- HybridEP topology (ACCEL=2 only): # of EP-communicating ranks sharing one NVLink domain. On a
# single B200 node ALL 8 GPUs are one domain -> = world size. Verify with b200_check.sh topo; override
# (e.g. =4) if EP ranks are split across racks/domains -- a wrong value hurts perf and sometimes
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
OVERRIDES="$OVERRIDES model.tensor_model_parallel_size=$TP model.sequence_parallel=$SP $MOE_CAPACITY_ARGS"
# MoE router in fp32 for 128-expert stability (matches A800). The CLI override is the RELIABLE path:
# the provider also sets it but that field may not survive build_llava_ov2's HF LLM rebuild.
OVERRIDES="$OVERRIDES model.moe_router_dtype=${OV2_ROUTER_DTYPE:-fp32}"
OVERRIDES="$OVERRIDES scheduler.lr_warmup_iters=$WARMUP_ITERS"   # 0.002*train_iters warmup ramp; OV2_WARMUP_ITERS=0 to disable
# LR aligned to AIAK 30B-A3B midtrain: peak 1e-5 -> cosine decay -> 1e-6 (the recipe default is 2e-5 FLAT,
# min_lr==max_lr). lr_decay_style=cosine + lr_decay_iters=train_iters are ALREADY set by the recipe, so
# making min_lr<lr re-enables the cosine ramp-down. Override OV2_LR= / OV2_MIN_LR= (=2e-5 =2e-5 to restore flat).
OVERRIDES="$OVERRIDES optimizer.lr=${OV2_LR:-1e-5} optimizer.min_lr=${OV2_MIN_LR:-1e-6}"
# B200 (192GB HBM): keep the WHOLE optimizer on-GPU as plain fp32 AdamW -- NO CPU offload. This is the
# core B200 simplification vs the A800 launcher (which, to fit 80GB at TP=1, sets optimizer_cpu_offload=true
# + use_precision_aware_optimizer=true). Forcing both OFF here is explicit/defensive: the recipe already
# defaults them off for the midtrain AdamW branch, so this is a functional no-op that GUARANTEES B200 never
# walks the offload path -- so the offload-zero bug class (CPU master never seeded from ckpt -> first step
# zeros weights -> iter-3 forward NaN; NVIDIA/Megatron-LM #1842/#1872/#1986) structurally cannot occur here.
OVERRIDES="$OVERRIDES optimizer.optimizer_cpu_offload=false optimizer.use_precision_aware_optimizer=false"
# dataloader workers/rank. Recipe default for midtrain is 2 (llava_next samples ~10x larger -> a big
# buffer/many workers pressure HOST RAM). 16 is safe here: the 85M-sample data has plenty of shards (so
# no shard-starvation) and Grace has ~1TB RAM; lower via OV2_NUM_WORKERS= if host memory gets tight.
OVERRIDES="$OVERRIDES dataset.num_workers=${OV2_NUM_WORKERS:-16}"
# c10d rendezvous AND the 30B ckpt-load determine_global_metadata all_gather both run on this PG timeout
# (default 10 min = 600000ms -> jobs died there on slow/flaky master-name resolution + on the big-ckpt
# all_gather). 60 min is a safety margin; the FQDN fix above is the real rendezvous cure.
OVERRIDES="$OVERRIDES dist.distributed_timeout_minutes=${OV2_DIST_TIMEOUT_MIN:-100}"
# TensorBoard logdir: the recipe default is $CWD/nemo_experiments, and $CWD = $REPO. If REPO is a
# read-only baked-in copy (e.g. /opt/...), rank N-1's SummaryWriter does os.makedirs() there and dies
# with PermissionError -> that rank crashes -> the others hit a collective timeout (the crash masquerades
# as a 'collective timeout from rank 7'). Pin tensorboard to the WRITABLE $SAVE so it never touches $REPO.
OVERRIDES="$OVERRIDES logger.tensorboard_dir=$SAVE/tensorboard"
# Per-rank (min,max) timing breakdown: opt-in via OV2_TIMING_LOG_LEVEL=1|2. log_timers_to_tensorboard=false
# is REQUIRED alongside it -- otherwise training_log()'s wandb/mlflow/comet timer-writes call
# elapsed(reset=True) (even when those writers are None) and zero the per-iter timers BEFORE the console
# timers.log() -> the block prints nothing. Default (env unset) appends nothing -> behavior unchanged.
[[ -n "${OV2_TIMING_LOG_LEVEL:-}" ]] && OVERRIDES="$OVERRIDES logger.timing_log_level=$OV2_TIMING_LOG_LEVEL logger.log_timers_to_tensorboard=false"
OVERRIDES="$OVERRIDES mixed_precision=$MIXED_PRECISION"
[[ "$DISABLE_RECOMPUTE" == "1" ]] && OVERRIDES="$OVERRIDES model.recompute_activations=false model.recompute_granularity=null"
# NB: the dispatcher is wired in ov2_provider.provide() via OV2_FLEX_BACKEND (above) -- setting
# model.moe_token_dispatcher_type here would be a DEAD field (build_llava_ov2 rebuilds the LLM).

# --- NOTE on EP comm-overlap (a ~1.3x step-time lever on exposed EP a2a, per the perf skills): it is
# NOT exposed as a flag here because the OV2 recipe sets cfg.comm_overlap=None, so a CLI override
# (comm_overlap.overlap_moe_expert_parallel_comm=true) would crash (attr-on-None). Enabling it needs a
# RECIPE change: build a CommOverlapConfig in ov2.py AND re-validate the grad path (the prior OV2 run
# flagged EP/DDP grad-overlap as fragile here). Prereqs once wired: PyTorch>=2.6 (else silent hang),
# selective (NOT full) recompute, moe_shared_expert_overlap=False (HybridEP sets this automatically). ---

# --- OPT-IN: Megatron-FSDP (OV2_FSDP=1). Shards params+grads+optim across the DATA-parallel dim. With
# PP/CP fixed at 1 here, DP = WORLD / TP. FSDP only helps when MODEL-STATE (not activation) memory is
# the limit; OV2's 30B-A3B long-sequence pain is usually activation-bound -> FSDP will NOT fix that
# (use recompute/CP/offload instead).
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
  echo "[ov2-30b-b200] OV2_FSDP=1: Megatron-FSDP ON (world=$WORLD dp=$DP tp=$TP nnodes=$NNODES). FSDP only helps \
when model-state memory dominates; if activation memory dominates it only adds overhead. fsdp_dtensor ckpt is \
ONE-WAY; torch_dist INIT_CKPT will mismatch -> convert it to fsdp_dtensor or point INIT_CKPT at an fsdp ckpt." >&2
fi

mkdir -p "$SAVE"; cd "$REPO"
# --- Muon resume-topology guard (Pass-6): distributed Muon (LayerWise) shards momentum across the DP
# axis and forces the ckpt replica_id DP-coord to 0 ("fixed DP usage only") with NO reshard guard, so
# resuming a Muon ckpt at a DIFFERENT world size (e.g. A800 16-rank -> B200 8-rank) SILENTLY loads
# mismatched momentum (weights are fine; optimizer state corrupts) with no error. AdamW reshards fine
# -> guard Muon only. DP=WORLD/TP here (PP/CP=1). Marker lives in $SAVE so it travels with the ckpt dir.
_is_muon=0; [[ "$RECIPE" == *stage2* && "${OV2_STAGE2_ADAMW:-0}" != "1" ]] && _is_muon=1
_wf="$SAVE/.ov2_train_world"
_has_ckpt=0; { [[ -f "$SAVE/latest_checkpointed_iteration.txt" ]] || compgen -G "$SAVE/iter_*" >/dev/null 2>&1; } && _has_ckpt=1
if [[ "$_is_muon" == "1" && "$_has_ckpt" == "1" && -f "$_wf" ]]; then
  _saved_world="$(cat "$_wf" 2>/dev/null || echo "")"
  if [[ -n "$_saved_world" && "$_saved_world" != "$WORLD" ]]; then
    if [[ "${OV2_ALLOW_DP_RESHARD:-0}" == "1" ]]; then
      echo "[ov2-30b] WARN: Muon ckpt in $SAVE saved at WORLD=$_saved_world, resuming at WORLD=$WORLD -> DP-sharded momentum MISMATCH; OV2_ALLOW_DP_RESHARD=1 set, continuing (optimizer state WILL be wrong)." >&2
    else
      echo "[ov2-30b] FATAL: distributed-Muon ckpt in $SAVE was saved at WORLD=$_saved_world but you are resuming at WORLD=$WORLD. Muon momentum is DP-sharded with NO reshard support -> resuming silently loads MISMATCHED optimizer state. Resume at WORLD=$_saved_world, OR set OV2_ALLOW_DP_RESHARD=1 to override (momentum garbage; expect a loss/grad-norm bump)." >&2
      exit 1
    fi
  fi
fi
[[ "$NODE_RANK" -eq 0 ]] && { echo "$WORLD" > "$_wf" 2>/dev/null || true; }
echo "[ov2-30b-b200] in-container | hw=$HWNAME(cc=$_cc) repo=$REPO recipe=$RECIPE accel=$ACCEL mp=$MIXED_PRECISION flex=${OV2_FLEX_BACKEND:-alltoall} recompute_off=$DISABLE_RECOMPUTE recompute_full=$OV2_RECOMPUTE_FULL peak=${MFU_PEAK_TFLOPS}TF nproc=$NPROC world=$WORLD dp=$DP tp=$TP sp=$SP seq=$SEQ_LEN gbs=$MIDTRAIN_GBS iters=$ITERS warmup=$WARMUP_ITERS lr=${OV2_LR:-1e-5}->${OV2_MIN_LR:-1e-6} router_dtype=${OV2_ROUTER_DTYPE:-fp32} permute_fusion=$OV2_MOE_PERMUTE_FUSION aux_loss=$OV2_MOE_AUX_LOSS_COEFF moe_capacity=$MOE_CAPACITY_FACTOR pad_to_capacity=$MOE_PAD_TO_CAPACITY node_rank=$NODE_RANK nnodes=$NNODES"
# shellcheck disable=SC2086  # $RDZV and $OVERRIDES must word-split into separate args
python -m torch.distributed.run $RDZV --nproc_per_node="$NPROC" scripts/training/run_recipe.py \
  --recipe "$RECIPE" --dataset vlm-energon --step_func ov2_step \
  $OVERRIDES ${EXTRA_ARGS:-} 2>&1 | tee "$SAVE/train_node${NODE_RANK}.log"
