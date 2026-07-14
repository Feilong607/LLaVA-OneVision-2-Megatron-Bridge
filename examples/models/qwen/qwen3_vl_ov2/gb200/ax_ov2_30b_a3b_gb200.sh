#!/usr/bin/env bash
# =============================================================================
# OV2-30B-A3B (Qwen3-30B-A3B MoE) midtrain - GB200/Blackwell - IN-CONTAINER launcher.
# Run this INSIDE the training container: it assembles env + run_recipe overrides and execs torchrun.
#
# HARDWARE: GB200 (sm_100) ONLY -- NPROC=4, MFU peak, NVLS, fp8 HW, /datasets paths hardwired.
#   All values stay env-overridable.
#
# ACCEL MODES (ACCEL=0|1|2|3; default 2):
#   0  bf16 baseline + alltoall + full recompute (AIAK parity)
#   1  MXFP8 expert/matmul GEMMs + alltoall dispatch
#   2  bf16 + HybridEP flex dispatcher (NVL72 topology-aware)
#   3  MXFP8 + HybridEP -- NVIDIA's measured-optimal GB200 combo (dispatch stays bf16; mcore
#      hardcodes fp8_dispatch=False and HybridEP pads internally for fp8 GEMM alignment).
#      UNVALIDATED on OV2 -- A/B loss vs 1/2 before trusting.
# =============================================================================
set -euo pipefail
# REPO: env override wins; else auto-detected below from this script's location. Don't hardcode a path.
REPO="${REPO:-}"

# Auto-detect repo root from this script's location.
_SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$({ __d="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; while [[ "$__d" != "/" && ! -d "$__d/src/megatron/bridge" ]]; do __d="$(dirname "$__d")"; done; echo "$__d"; })}"
[[ -d "$REPO/src/megatron/bridge" ]] || { echo "FATAL: OV2 fork root not found from ${BASH_SOURCE[0]} (no src/megatron/bridge above it). Set REPO=/path/to/LLaVA-OneVision-2-Megatron-Bridge" >&2; exit 1; }
bash "$REPO/3rdparty/apply_megatron_patch.sh"   # apply OV2 mcore submodule patch (apply_rotary_fn hook); idempotent. FAIL LOUD -- a missing hook -> cryptic build error.
RECIPE="${RECIPE:-ov2_30b_a3b_p16m33_midtrain}"  # must be a p16m33 recipe (the /datasets ckpt is p16m33; a merge2 recipe -> vision-config mismatch).
# Tunable constants (mirror the A800 midtrain launcher); the recipe reads the exported OV2_* values below.
MIDTRAIN_GBS="${OV2_MIDTRAIN_GBS:-384}"                  # global batch size; override with OV2_MIDTRAIN_GBS=
MIDTRAIN_N_SAMPLES="${OV2_MIDTRAIN_N_SAMPLES:-8000000}"  # LLaVA-Next 780k default
ITERS="${ITERS:-$(( (MIDTRAIN_N_SAMPLES + MIDTRAIN_GBS - 1) / MIDTRAIN_GBS ))}"
# LR warmup = 0.002 * train_iters; gentle ramp 0->peak then constant. Override OV2_WARMUP_ITERS=.
WARMUP_ITERS="${OV2_WARMUP_ITERS:-$(( ITERS * 2 / 1000 ))}"
# Floor the AUTO-DERIVED value to 1 (a 0-iter auto-warmup is a degenerate ramp), but honor an EXPLICIT
# OV2_WARMUP_ITERS=0 -- that is the documented way to run flat-LR-from-iter-1 (see the OV2_WARMUP_ITERS
# comment on the scheduler.lr_warmup_iters override below).
if [ -z "${OV2_WARMUP_ITERS:-}" ] && [ "$WARMUP_ITERS" -lt 1 ]; then WARMUP_ITERS=1; fi
LOG_EVERY="${LOG_EVERY:-1}"; SAVE_EVERY="${SAVE_EVERY:-2000}"

# --- HARDWARE: GB200 (sm_100) ONLY; per-HW values hardwired, all env-overridable. ---
HWNAME=gb200; _cc=100; HW_NPROC=4; PEAK_BF16=2250; PEAK_FP8=4500; HW_NVLS=1; HW_FP8=1
NPROC="${NPROC:-$HW_NPROC}"        # GPUs/node (GB200=4)
TP="${TP:-1}"                      # GB200 2-node world=8 needs TP=1 so DP=8 can satisfy EP=8. TP=2 needs more nodes.
if [[ "$TP" -gt 1 ]]; then SP=true; else SP=false; fi
SEQ_LEN="${OV2_SEQ_LEN:-10192}"    # seed85m offline-packed length; matches A800 launcher
# MoE token capacity. GB200 has enough memory -> default to no token dropping. If tight, try
# MOE_CAPACITY_FACTOR=1.0 MOE_PAD_TO_CAPACITY=true (or 1.25/1.5 with pad as a middle ground).
MOE_CAPACITY_FACTOR="${MOE_CAPACITY_FACTOR:-none}"
MOE_PAD_TO_CAPACITY="${MOE_PAD_TO_CAPACITY:-false}"
MOE_CAPACITY_ARGS=""
if [[ -n "$MOE_CAPACITY_FACTOR" && "$MOE_CAPACITY_FACTOR" != "none" && "$MOE_CAPACITY_FACTOR" != "None" && "$MOE_CAPACITY_FACTOR" != "-1" ]]; then
  MOE_CAPACITY_ARGS="model.moe_expert_capacity_factor=$MOE_CAPACITY_FACTOR model.moe_pad_expert_input_to_capacity=$MOE_PAD_TO_CAPACITY"
fi

# --- CARD PATHS (GB200); per-path defaults, all env-overridable. ---
OV2_LLM_HF_30B="${OV2_LLM_HF_30B:-/datasets/qwen-models-ea5jyi/Qwen3-30B-A3B-Instruct-2507}"
OV2_HF_PROC_30B="${OV2_HF_PROC_30B:-/datasets/llava-ov2-30b-a3b-m9lvdn/auto_model}"            # bundled processor
OV2_HF_PROC_30B_P16M33="${OV2_HF_PROC_30B_P16M33:-/datasets/llava-ov2-30b-a3b-m9lvdn/auto_model}"   # bundled processor (p16m33 recipe)
OV2_PRETRAIN_ROOT="${OV2_PRETRAIN_ROOT:-/datasets/llava/11May}"      # processor root (stage_0 skipped; bundled auto_model above)
DATA_PATH="${DATA_PATH:-$REPO/examples/models/qwen/qwen3_vl_ov2/gb200/mid_training_seed85m.yaml}"   # /datasets/llava/11May data
INIT_CKPT="${INIT_CKPT:-/datasets/llava-ov2-30b-a3b-m9lvdn}"   # trained ckpt to resume (has iter_0001000 + auto_model)
SAVE="${SAVE:-/home/ftan0055/ckpts_video_sft/ov2_30b_a3b_gb200}"     # output dir (override with SAVE=)
OV2_SKIP_BASE_STITCH="${OV2_SKIP_BASE_STITCH:-1}"   # mid-train from stage2 -> skip the stage_0 stitch
OV2_HF_PROC_30B="${OV2_HF_PROC_30B:-$OV2_PRETRAIN_ROOT/llava_onevision2/llava_onevision2_30b_a3b/auto_model}"
OV2_HF_PROC_30B_P16M33="${OV2_HF_PROC_30B_P16M33:-$OV2_PRETRAIN_ROOT/llava_onevision2/llava_onevision2_30b_a3b_p16_m33/auto_model}"   # p16m33 processor (patch16/merge3)
export OV2_LLM_HF_30B OV2_PRETRAIN_ROOT OV2_SKIP_BASE_STITCH OV2_HF_PROC_30B OV2_HF_PROC_30B_P16M33
export OV2_INIT_CKPT="$INIT_CKPT"   # recipe guard verifies this exists before skipping the stitch

# --- OPTIMIZER on the MoE backbone: AdamW (validated) or distributed Muon (OV2_MIDTRAIN_MUON=1, default ON).
# Muon on the unfrozen EP8 experts is unvalidated (a prior A800 run NaN'd at iter-2); GB200's 192GB removes the
# A800 blockers, so the iter-2 NaN is the open question. Set OV2_MIDTRAIN_MUON=0 for the validated AdamW path. ---
export OV2_MIDTRAIN_MUON="${OV2_MIDTRAIN_MUON:-1}"
ACCEL="${ACCEL:-2}"
if [[ "$ACCEL" == "1" ]]; then          # Phase-2a: MXFP8 + alltoall (GB200 fp8 HW)
  MIXED_PRECISION="${MIXED_PRECISION:-bf16_with_mxfp8_mixed}"
  DISABLE_RECOMPUTE="${DISABLE_RECOMPUTE:-1}"; OV2_RECOMPUTE_FULL="${OV2_RECOMPUTE_FULL:-0}"
  FLEX_BACKEND="${FLEX_BACKEND:-}"                       # default alltoall (validated fp8 lane); ACCEL=3 for MXFP8+HybridEP
elif [[ "$ACCEL" == "2" ]]; then        # Phase-2b: bf16 + HybridEP (best bf16 config on NVL72)
  MIXED_PRECISION="${MIXED_PRECISION:-bf16_mixed}"   # registry key is 'bf16_mixed' (plain 'bf16' -> ValueError)
  # Recompute ON (selective MoE) by DEFAULT for HybridEP -- unlike alltoall, HybridEP holds a PERSISTENT
  # nvshmem symmetric buffer sized to HYBRID_EP_MAX_TOKENS_PER_RANK (10240) tok/rank, and the THD-padding
  # patch pads every dispatch to that target; together with Muon's fp32 momentum on the unfrozen 30B this
  # OOMs at 192GB with recompute OFF. Selective MoE recompute is the cheapest lever back under budget.
  # DISABLE_RECOMPUTE=1 to force recompute off if you have headroom (smaller seq / AdamW / capacity factor).
  DISABLE_RECOMPUTE="${DISABLE_RECOMPUTE:-0}"; OV2_RECOMPUTE_FULL="${OV2_RECOMPUTE_FULL:-0}"; OV2_RECOMPUTE_MOE="${OV2_RECOMPUTE_MOE:-1}"
  FLEX_BACKEND="${FLEX_BACKEND:-hybridep}"
elif [[ "$ACCEL" == "3" ]]; then        # Phase-2c: MXFP8 + HybridEP -- NVIDIA's measured-optimal GB200 combo
  # (QWEN3_VL_30B_A3B_PRETRAIN_CONFIG_GB200_FP8_MX pairs hybridep with mxfp8). The dispatch/combine
  # itself stays bf16 -- mcore hardcodes fp8_dispatch=False (fused_a2a.py) and HybridEP pads
  # internally for fp8 GEMM alignment -- only the GEMMs run MXFP8. UNVALIDATED on OV2: A/B loss
  # vs ACCEL=1/2 before trusting.
  MIXED_PRECISION="${MIXED_PRECISION:-bf16_with_mxfp8_mixed}"
  # Same HybridEP memory pressure as ACCEL=2 -> selective MoE recompute ON by default (see ACCEL=2 note).
  DISABLE_RECOMPUTE="${DISABLE_RECOMPUTE:-0}"; OV2_RECOMPUTE_FULL="${OV2_RECOMPUTE_FULL:-0}"; OV2_RECOMPUTE_MOE="${OV2_RECOMPUTE_MOE:-1}"
  FLEX_BACKEND="${FLEX_BACKEND:-hybridep}"
else                                    # Phase-1: bf16 baseline
  MIXED_PRECISION="${MIXED_PRECISION:-bf16_mixed}"   # registry key is 'bf16_mixed' (plain 'bf16' -> ValueError)
  # recompute OFF by default on GB200 (192GB) -> faster AND numerically identical to AIAK full/uniform/1.
  DISABLE_RECOMPUTE="${DISABLE_RECOMPUTE:-0}"; OV2_RECOMPUTE_FULL="${OV2_RECOMPUTE_FULL:-0}"; OV2_RECOMPUTE_MOE="${OV2_RECOMPUTE_MOE:-1}"   # selective recompute (core_attn + MoE layer); OV2_RECOMPUTE_FULL=1 for AIAK full parity; DISABLE_RECOMPUTE=1 to turn off.
  FLEX_BACKEND="${FLEX_BACKEND:-}"
fi
OV2_RECOMPUTE_MOE="${OV2_RECOMPUTE_MOE:-0}"   # selective MoE-layer recompute when =1; only active when recompute ON and OV2_RECOMPUTE_FULL=0

# --- MXFP8 lane knobs (keyed on the PRECISION, not the ACCEL mode, so ACCEL=1 and ACCEL=3 share one
# source of truth; bf16 lanes take the else). MXFP8 aligns the token/M dim to 32 (1x32 block scaling):
# dense/attention via ov2_step's packed-seq pad; grouped-GEMM experts via TEGroupedMLP's
# `quantization_padding` submodule, created in __init__ ONLY when config.fp8 is set at BUILD time ->
# ov2_provider wires fp8 on the LLM provider pre-build (fp8_fields). Requires transformer_engine >= 2.14.0.
# OV2_MOE_ROUTER_PAD_FP8=0 (default): experts pad expert-side (alltoall) / hybridep pads internally.
if [[ "$MIXED_PRECISION" == *mxfp8* || "$MIXED_PRECISION" == *fp8* || "$MIXED_PRECISION" == *fp4* ]]; then
  MFU_PEAK_TFLOPS="${MFU_PEAK_TFLOPS:-$PEAK_FP8}"        # fp8 tensor-core peak
  export OV2_MOE_ROUTER_PAD_FP8="${OV2_MOE_ROUTER_PAD_FP8:-0}"
else
  MFU_PEAK_TFLOPS="${MFU_PEAK_TFLOPS:-$PEAK_BF16}"
fi
export OV2_RECOMPUTE_FULL OV2_RECOMPUTE_MOE MFU_PEAK_TFLOPS
# OV2_FLEX_BACKEND is read by ov2_provider.provide() to wire the runtime dispatcher (the cfg.model field is dead).
export OV2_FLEX_BACKEND="$FLEX_BACKEND"

# --- HybridEP runtime gates + tuning (keyed on the DISPATCHER, not the ACCEL mode, so ACCEL=2 and
# ACCEL=3 share them; all env-overridable). Both gates are needed or the first nvshmem kernel dies. ---
if [[ "$FLEX_BACKEND" == "hybridep" ]]; then
  #  (0) HARD PREREQ -- THE ACCEL=2/3 crash gate. HybridEP's JIT static_asserts per-rank token count % 64
  #      == 0 AND identical across EP ranks, but OV2's THD-packed batches give ragged counts (e.g. 8176).
  #      The fused_a2a.py padding patch (apply_megatron_patch.sh patch 3) pads to a uniform 64-multiple.
  #      Without it the first dispatch dies with an async cudaErrorIllegalAddress in hybrid_ep_backend.cuh
  #      (cudaMemsetAsync preprocessing_tmp) -- exactly the observed crash. apply_megatron_patch.sh ran at
  #      startup; verify the marker landed in the fused_a2a.py PYTHONPATH will import, and fail LOUD (clear
  #      cause) instead of the cryptic runtime crash.
  _fa2a="$REPO/3rdparty/Megatron-LM/megatron/core/transformer/moe/fused_a2a.py"
  grep -q "_HYBRID_EP_PAD_INFO" "$_fa2a" 2>/dev/null || {
    echo "[ov2-30b] FATAL: HybridEP (FLEX_BACKEND=hybridep, ACCEL=2/3) needs the fused_a2a.py THD-padding patch, but it is NOT applied ($_fa2a lacks _HYBRID_EP_PAD_INFO). Fix: bash \"$REPO/3rdparty/apply_megatron_patch.sh\" (applies patch 3). Without it the first MoE dispatch crashes with cudaErrorIllegalAddress. Or use the validated alltoall lane: ACCEL=1 (MXFP8) / ACCEL=0 (bf16)." >&2
    exit 1; }
  #  (1) GB200 nvshmem symmetric-heap uses CUDA VMM which is broken on this platform -> disable it. =1 is
  #      the EMPIRICALLY-VERIFIED working value for ACCEL=2 on this cluster (commit 63e1085 ran clean with
  #      it); leaving VMM on (=0) fails here. Env-gated so a different fabric can flip it.
  export NVSHMEM_DISABLE_CUDA_VMM="${NVSHMEM_DISABLE_CUDA_VMM:-1}"
  #  (2) HybridEP JIT needs MAX_NUM_OF_TOKENS_PER_RANK % 64 == 0 and identical across EP ranks. Derived
  #      from SEQ_LEN (round up to 64) so the cap tracks the seq you run. An explicit override still wins.
  _hep_cap=$(( (SEQ_LEN + 63) / 64 * 64 ))
  export HYBRID_EP_MAX_TOKENS_PER_RANK="${HYBRID_EP_MAX_TOKENS_PER_RANK:-$_hep_cap}"
  #  (2b) GUARD: a cap below round64(SEQ_LEN) -> EP ranks pad to different targets -> allgather-timeout hang.
  (( HYBRID_EP_MAX_TOKENS_PER_RANK >= _hep_cap )) || {
    echo "[ov2-30b] FATAL: HYBRID_EP_MAX_TOKENS_PER_RANK=$HYBRID_EP_MAX_TOKENS_PER_RANK < round64(SEQ_LEN=$SEQ_LEN)=$_hep_cap -> EP ranks would pad to different targets -> HybridEP allgather-timeout hang. Set it >= $_hep_cap, or lower OV2_SEQ_LEN." >&2; exit 1; }
  #  (2c) GUARD: the HybridEP JIT also requires the cap % 64 == 0 (line-115 invariant). The derived
  #       default is always a multiple of 64, but an explicit large-enough-but-not-%64 override would
  #       pass (2b) and then crash the first nvshmem kernel -- reject it here instead.
  (( HYBRID_EP_MAX_TOKENS_PER_RANK % 64 == 0 )) || {
    echo "[ov2-30b] FATAL: HYBRID_EP_MAX_TOKENS_PER_RANK=$HYBRID_EP_MAX_TOKENS_PER_RANK is not a multiple of 64 (HybridEP JIT asserts % 64 == 0). Round it up to a multiple of 64." >&2; exit 1; }
  #  (3) comm-kernel SM count + (4) unfused-combine chunk size: OPT-IN ONLY (NOT default-on). NVIDIA's
  #      GB200 perf preset forces OV2_HYBRIDEP_NUM_SMS=32 and NUM_OF_TOKENS_PER_CHUNK_COMBINE_API=128, but
  #      the latter is a workaround for the UNLANDED Megatron-LM PR #4089 -- on our pinned (older) deep_ep
  #      it can mis-size the HybridEP preprocessing_tmp buffer -> `cudaErrorIllegalAddress` in
  #      hybrid_ep_backend.cuh's cudaMemsetAsync (observed on GB200). Default = UNSET = deep_ep's own
  #      internal values (the known-good path that ran at ~4300 tok/s). Sweep only when validated:
  #      OV2_HYBRIDEP_NUM_SMS=16|24|32  and/or  NUM_OF_TOKENS_PER_CHUNK_COMBINE_API=128.
  [[ -n "${OV2_HYBRIDEP_NUM_SMS:-}" ]] && export OV2_HYBRIDEP_NUM_SMS
  [[ -n "${NUM_OF_TOKENS_PER_CHUNK_COMBINE_API:-}" ]] && export NUM_OF_TOKENS_PER_CHUNK_COMBINE_API
fi

# --- rendezvous: auto-detect master/worker (no hardcoded IPs -> survives pod reschedules). Priority:
#   (1) operator-injected env (PyTorchJob/Run:AI PET_* + MASTER_ADDR/WORLD_SIZE); (2) manual LIST_IP; (3) single-node.
#   EP8 spans 2 nodes (4+4) -> MoE all-to-all crosses the node boundary; needs NVLink5/NVL72 or IB. ---
GPUS_PER_NODE="$NPROC"
if [[ -n "${PET_NNODES:-}" || ( -n "${MASTER_ADDR:-}" && -n "${WORLD_SIZE:-}" ) ]]; then
  # (1) K8s auto: prefer the PET_* the operator injects; fall back to WORLD_SIZE/RANK arithmetic.
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
  # Match against ALL of this host's IPs (hostname -I lists every iface), not just the first -- on a
  # multi-homed GB200 node the fabric IP is often not enumerated first, so a first-IP-only match would
  # miss a genuinely-listed node and abort. Hostname is also matched (for LIST_IP written with names).
  read -ra _my_ips <<< "$(hostname -I 2>/dev/null)"; CURRENT_HOST="$(hostname)"; NODE_RANK=-1
  for i in "${!list_ip[@]}"; do
    if [[ "${list_ip[$i]}" == "$CURRENT_HOST" ]]; then NODE_RANK=$i && break; fi
    for _ip in "${_my_ips[@]}"; do [[ "${list_ip[$i]}" == "$_ip" ]] && NODE_RANK=$i && break 2; done
  done
  CURRENT_IP="${_my_ips[0]:-}"   # kept for the diagnostic message below
  [[ "$NODE_RANK" -eq -1 ]] && { echo "[ov2-30b] ERROR: this host IP($CURRENT_IP)/name($CURRENT_HOST) not in LIST_IP (${list_ip[*]})" >&2; exit 1; }
  RUN_MODE="multi-node (manual LIST_IP)"
else
  # (3) single-node test
  NNODES=1; NODE_RANK=0; MASTER_ADDR=127.0.0.1; MASTER_PORT="${MASTER_PORT:-26047}"
  RUN_MODE="single-node TEST"
fi
# --- k8s DNS hardening: the operator injects MASTER_ADDR as a SHORT pod name that intermittently fails to
# resolve from workers -> rendezvous times out. Resolution order: (1) if the FQDN resolves, prefer it (most
# reliable); (2) else if the SHORT name already resolves, use it as-is -- NO warning (the common Run:AI case,
# where /etc/resolv.conf search domains resolve the short name); (3) else warn. Namespace via POD_NAMESPACE ->
# OV2_K8S_NAMESPACE -> the pod's OWN serviceaccount namespace (portable, no hardcoded cluster ns). ---
if [[ "$NNODES" -gt 1 && -n "${MASTER_ADDR:-}" && "$MASTER_ADDR" != *.* && "$MASTER_ADDR" != "127.0.0.1" ]]; then
  _sa_ns_file="/var/run/secrets/kubernetes.io/serviceaccount/namespace"
  _ns="${POD_NAMESPACE:-${OV2_K8S_NAMESPACE:-$([ -r "$_sa_ns_file" ] && cat "$_sa_ns_file" 2>/dev/null || echo "")}}"
  _fqdn=""; [[ -n "$_ns" ]] && _fqdn="${MASTER_ADDR}.${_ns}.svc.cluster.local"
  if [[ -n "$_fqdn" ]] && getent hosts "$_fqdn" >/dev/null 2>&1; then
    echo "[ov2-30b-gb200] rdzv: short MASTER_ADDR '$MASTER_ADDR' -> FQDN '$_fqdn' (avoids gai-error rendezvous timeout)" >&2
    MASTER_ADDR="$_fqdn"
  elif getent hosts "$MASTER_ADDR" >/dev/null 2>&1; then
    echo "[ov2-30b-gb200] rdzv: short MASTER_ADDR '$MASTER_ADDR' resolves as-is (no FQDN swap needed)." >&2
  else
    echo "[ov2-30b-gb200] WARN: MASTER_ADDR='$MASTER_ADDR' does not resolve here (tried FQDN in ns='${_ns:-<unknown>}'); rendezvous may time out. Set OV2_K8S_NAMESPACE=<ns> or pass a resolvable MASTER_ADDR." >&2
  fi
fi
if [[ "$NNODES" -le 1 ]]; then RDZV="--standalone"; NNODES=1; NODE_RANK=0; else
  RDZV="--nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT"; fi
echo "[ov2-30b-gb200] --- rdzv: $RUN_MODE --- master=${MASTER_ADDR:-n/a}:${MASTER_PORT} nnodes=$NNODES node_rank=$NODE_RANK gpus/node=$GPUS_PER_NODE"
WORLD=$(( NPROC * NNODES ))
(( TP >= 1 )) || { echo "[ov2-30b] FATAL: TP must be >=1, got TP=$TP" >&2; exit 1; }
(( WORLD % TP == 0 )) || { echo "[ov2-30b] FATAL: WORLD=$WORLD must be divisible by TP=$TP." >&2; exit 1; }
DP=$(( WORLD / TP ))
# GBS % DP == 0 required (mbs=1, PP=CP=1).
(( MIDTRAIN_GBS % DP == 0 )) || { echo "[ov2-30b] FATAL: GBS=$MIDTRAIN_GBS not divisible by DP=$DP (WORLD=$WORLD / TP=$TP; mbs fixed at 1, PP=CP=1) -> mcore num_microbatches assert would fire with a generic message. GBS % DP = $((MIDTRAIN_GBS % DP)). Adjust OV2_MIDTRAIN_GBS / TP / NNODES so GBS % DP == 0." >&2; exit 1; }
# EP=8 is fixed in the recipe -> DP must be a multiple of EP and >= EP.
(( DP >= 8 && DP % 8 == 0 )) || { echo "[ov2-30b] FATAL: EP=8 needs DP=$DP (WORLD=$WORLD / TP=$TP) to be a multiple of 8 and >=8. For 2 GB200 nodes (WORLD=8) keep TP=1; TP=2 needs >=4 GB200 nodes." >&2; exit 1; }

# --- in-container env ---
# Offline packages (GB200 has no network) that are not pip-installed -- e.g. emerging_optimizers for
# distributed Muon. Two ways to make them importable, both committed WITHOUT any username/env path:
#   1. AUTO: drop or symlink the unpacked wheels into "$REPO/pylibs" -> picked up on EVERY node (it is
#      under the auto-detected repo root, so no env var, no per-pod setup). One-time on the shared FS:
#      ln -s /your/pylibs "$REPO/pylibs"
#   2. ENV: OV2_EXTRA_PYLIBS=/abs/path/to/pylibs  (set it in the job/Run:AI env so all pods inherit it).
# Both are folded in AFTER the repo paths so the fork's megatron/bridge always win.
[[ -d "$REPO/pylibs" ]] && OV2_EXTRA_PYLIBS="$REPO/pylibs${OV2_EXTRA_PYLIBS:+:$OV2_EXTRA_PYLIBS}"
export PYTHONPATH="$REPO/_verify_stubs:$REPO/src:$REPO/3rdparty/Megatron-LM:$REPO/aiak_shim${OV2_EXTRA_PYLIBS:+:$OV2_EXTRA_PYLIBS}${PYTHONPATH:+:$PYTHONPATH}"  # _verify_stubs FIRST: sitecustomize stubs must load before transformers->boto3
# HybridEP (ACCEL=2) needs deep_ep, whose .so requires a symbol present ONLY in the pip nvidia-nvshmem lib
# (not CUDA's bundled libnvshmem_host.so.3). Prepend the pip nvshmem ONLY when it exists (no-op otherwise).
_nvshmem_lib="${OV2_NVSHMEM_LIB:-/usr/local/lib/python3.12/dist-packages/nvidia/nvshmem/lib}"
[[ -e "$_nvshmem_lib/libnvshmem_host.so.3" ]] && export LD_LIBRARY_PATH="$_nvshmem_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OV2_SKIP_HELPERS="${OV2_SKIP_HELPERS:-1}"   # energon doesn't use helpers_cpp -> skip the C++ index-builder compile
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"   # HF tokenizer Rust threads x forked energon workers -> warn/deadlock (StepFun parity)
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"                   # tee'd logs keep their tail on crash (offline GB200 debugging)
# TE LayerNorm SM margin: reserve SMs so LayerNorm kernels don't starve CONCURRENT comm kernels --
# matters whenever comm overlaps compute (OV2_EP_OVERLAP=1, grad-reduce overlap). StepFun pairs 20
# with CUDA_DEVICE_MAX_CONNECTIONS=32. Env-overridable; =0 restores the old all-SM behavior.
export NVTE_FWD_LAYERNORM_SM_MARGIN="${NVTE_FWD_LAYERNORM_SM_MARGIN:-20}"
export NVTE_BWD_LAYERNORM_SM_MARGIN="${NVTE_BWD_LAYERNORM_SM_MARGIN:-20}"
export OV2_MOE_PERMUTE_FUSION="${OV2_MOE_PERMUTE_FUSION:-0}"  # avoid TE Triton MoE-permute wedge (30B-A3B fix)
export OV2_MOE_AUX_LOSS_COEFF="${OV2_MOE_AUX_LOSS_COEFF:-0.01}"  # AIAK midtrain load-balance coeff
export OV2_PACK_FULL_CAUSAL="${OV2_PACK_FULL_CAUSAL:-0}"      # 0=THD block-diagonal (AIAK-faithful); 1=full-causal
export OV2_SEQ_LEN="$SEQ_LEN"                                # recipe reads this at import -> model+dataset+task_encoder
export OV2_MIDTRAIN_GBS="$MIDTRAIN_GBS" OV2_MIDTRAIN_N_SAMPLES="$MIDTRAIN_N_SAMPLES"
export OV2_PARALLEL_SHARD_ITERS="${OV2_PARALLEL_SHARD_ITERS:-1}"  # energon per-worker concurrent open shards (default 16 chokes WekaFS)
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_GRAPH_REGISTER="${NCCL_GRAPH_REGISTER:-0}" NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-$HW_NVLS}"  # NVLS on for GB200
[[ -n "${NCCL_IB_HCA:-}" ]] && export NCCL_IB_HCA NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
# --- GB200 cross-node NCCL tuning (MNNVL + NVLink); all env-overridable. ---
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker}"        # bootstrap iface: exclude loopback + docker bridge
export NCCL_DEBUG="${OV2_NCCL_DEBUG:-WARN}"                              # WARN = quiet; OV2_NCCL_DEBUG=INFO for bring-up debugging
export NCCL_MNNVL_ENABLE="${NCCL_MNNVL_ENABLE:-1}"                   # NVL72 Multi-Node NVLink (cross-node GPU NVLink)
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"                       # P2P over NVLink
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-SYS}"
export NCCL_NET_GDR_C2C="${NCCL_NET_GDR_C2C:-1}"                     # Grace<->Blackwell C2C GPUDirect
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-1}"
export NVLINK_DOMAIN_SIZE="${NVLINK_DOMAIN_SIZE:-72}"               # NVL72 = 72 GPUs in one NVLink domain
export NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT:-22}" NCCL_IB_RETRY_CNT="${NCCL_IB_RETRY_CNT:-7}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export UCX_TLS="${UCX_TLS:-tcp}"
# NCCL_ALGO: SAFE set only -- CollNet (IB SHARP) crashed this job before ("mixed local CollNet device counts");
# Tree/Ring/NVLSTree route over NVLink. Re-add CollNet only if your gb200_nccl_test passes with it.
export NCCL_ALGO="${NCCL_ALGO:-Tree,Ring,NVLSTree}"
# --- JIT hygiene: keep Triton/Inductor caches on node-local /tmp (not the shared FS). ---
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/ov2_triton_cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/ov2_inductor_cache}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

# --- HybridEP topology (# of EP ranks of the EP=8 group sharing one NVLink domain). DEFAULT 8 -- the
# value the verified-good ACCEL=2 run (commit 63e1085) used, i.e. the 2 GB200 machines behave as ONE
# NVLink domain (NVL72 MNNVL: NCCL_MNNVL_ENABLE=1 / NVLINK_DOMAIN_SIZE=72 above). NOTE: the first-dispatch
# crash was NOT this value -- it was the missing fused_a2a.py THD-padding patch (gate (0) above); domain is
# a separate, transport-only axis. mcore asserts EP(8) % value == 0. If your 2 machines are genuinely
# SEPARATE 4-GPU NVLink domains (verify: gb200_check.sh topo does NOT show NV* across all 8), override
# NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=4 so deep_ep uses the intra-machine NVLink + inter-machine IB
# path. ---
if [[ "$FLEX_BACKEND" == "hybridep" ]]; then
  export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN="${NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN:-8}"
  (( 8 % NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN == 0 )) || {
    echo "[ov2-30b] FATAL: NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=$NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN must divide EP=8 (use 1/2/4/8)." >&2; exit 1; }
fi
# --- HybridEP fabric diagnostic (once per node): echo the resolved HybridEP env so an illegal-address /
# allgather-timeout is debuggable straight from the log. Expected values: cuda_vmm=off,
# ranks_per_nvlink_domain=8 (matches 63e1085; set =4 if the 2 machines are separate NVLink domains). The
# THD-padding patch (gate 0) is the crash fix; this line just aids post-patch transport debugging. ---
if [[ "$FLEX_BACKEND" == "hybridep" ]]; then
  _imex="absent"; compgen -G "/dev/nvidia-caps-imex-channels/*" >/dev/null 2>&1 && _imex="present"
  _pipnvsh="absent"; [[ -e "$_nvshmem_lib/libnvshmem_host.so.3" ]] && _pipnvsh="present"
  _vmm="on"; [[ "${NVSHMEM_DISABLE_CUDA_VMM:-1}" == "1" ]] && _vmm="off"
  echo "[ov2-30b-gb200] hybridep-fabric: cuda_vmm=$_vmm ranks_per_nvlink_domain=${NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN:-auto} mnnvl=${NCCL_MNNVL_ENABLE:-?} imex_dev=$_imex pip_nvshmem=$_pipnvsh max_tok_per_rank=${HYBRID_EP_MAX_TOKENS_PER_RANK:-?}" >&2
fi
# EP comm-overlap (OV2_EP_OVERLAP=1) requires CUDA_DEVICE_MAX_CONNECTIONS>=32; couple them so the lever
# actually engages. Default path (overlap off) keeps the historical 1.
if [[ "${OV2_EP_OVERLAP:-0}" == "1" ]]; then
  export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-32}"
else
  export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
fi

# --- run_recipe.py CLI overrides ---
OVERRIDES="dataset.path=$DATA_PATH"
[[ "$INIT_CKPT" != "null" && -n "$INIT_CKPT" ]] && OVERRIDES="$OVERRIDES checkpoint.pretrained_checkpoint=$INIT_CKPT"
OVERRIDES="$OVERRIDES checkpoint.save=$SAVE checkpoint.load=$SAVE dataset.dataloader_save=$SAVE"
OVERRIDES="$OVERRIDES checkpoint.save_interval=$SAVE_EVERY train.train_iters=$ITERS validation.eval_iters=0 logger.log_interval=$LOG_EVERY logger.timing_log_level=${OV2_TIMING_LOG_LEVEL:-2} train.micro_batch_size=1"   # packing REQUIRES mbs=1 (model asserts batch==1)
OVERRIDES="$OVERRIDES model.tensor_model_parallel_size=$TP model.sequence_parallel=$SP $MOE_CAPACITY_ARGS"
# MoE router in fp32 for 128-expert stability (matches A800). The CLI override is the reliable path (the
# provider also sets it but that field may not survive build_llava_ov2's HF LLM rebuild).
OVERRIDES="$OVERRIDES model.moe_router_dtype=${OV2_ROUTER_DTYPE:-fp32}"
OVERRIDES="$OVERRIDES scheduler.lr_warmup_iters=$WARMUP_ITERS"   # OV2_WARMUP_ITERS=0 to disable
# LR aligned to AIAK 30B-A3B midtrain: peak 1e-5 -> cosine decay -> 1e-6. Override OV2_LR= / OV2_MIN_LR= (=2e-5 =2e-5 to restore flat).
OVERRIDES="$OVERRIDES optimizer.lr=${OV2_LR:-1e-5} optimizer.min_lr=${OV2_MIN_LR:-1e-6}"
# GB200 (192GB): keep the WHOLE optimizer on-GPU as fp32 AdamW -- NO CPU offload. Forced OFF defensively (the
# recipe already defaults them off) so GB200 never walks the offload path -> the offload-zero bug class
# (CPU master never seeded from ckpt -> first step zeros weights -> NaN; Megatron-LM #1842/#1872/#1986) can't occur.
OVERRIDES="$OVERRIDES optimizer.optimizer_cpu_offload=false optimizer.use_precision_aware_optimizer=false"
# --- MUON STABILITY (applies only when OV2_MIDTRAIN_MUON=1) -----------------------------------------------
# Recipe midtrain-Muon defaults (spectral / 0.15 / wd 0.01) are the config that NaN'd at iter-2 on A800. The
# only Bridge config proven on TRAINABLE EP8 experts is deepseek_v4's: unit_rms_norm + 0.2 + wd 0.1.
#   MUON_STABLE=0 (default): keep recipe values -> reproduces the known NaN.
#   MUON_STABLE=1          : DeepSeek-V4-proven trio -> best chance to NOT NaN.
# Each knob is individually overridable (OV2_MUON_SCALE_MODE / OV2_MUON_EXTRA_SCALE / OV2_MUON_WD). wd is set on
# BOTH optimizer AND scheduler start/end (the scheduler clobbers optimizer.weight_decay every iter otherwise).
if [[ "${OV2_MIDTRAIN_MUON:-0}" == "1" ]]; then
  [[ "${OV2_FSDP:-0}" == "1" ]] && { echo "[ov2-30b-gb200] FATAL: OV2_FSDP=1 is incompatible with Muon (Muon forces use_distributed_optimizer=False; FSDP shards optim state). Unset one." >&2; exit 1; }
  # Preflight: distributed Muon (mcore get_megatron_muon_optimizer) needs the 'emerging_optimizers' pip
  # package. It ships in the mbridge:qwen35-muon image but NOT the base GB200 image, so without it the run
  # crashes DEEP -- after full 16-rank NCCL init -- with a cryptic ImportError. Probe HERE (PYTHONPATH,
  # incl. OV2_EXTRA_PYLIBS, is already set above) and fail loud EARLY with the real fixes.
  if ! python -c "import emerging_optimizers" >/dev/null 2>&1; then
    echo "[ov2-30b-gb200] FATAL: OV2_MIDTRAIN_MUON=1 but 'emerging_optimizers' is not importable (mcore's distributed Muon requires it)." >&2
    echo "  Fix (pick one):" >&2
    echo "    - point PYTHONPATH at an offline-extracted copy:  OV2_EXTRA_PYLIBS=/path/to/pylibs bash \$0" >&2
    echo "    - install it (needs network / local wheel):        pip install emerging-optimizers" >&2
    echo "    - or use the validated AdamW path (no package):    OV2_MIDTRAIN_MUON=0 bash \$0" >&2
    exit 1
  fi
  if [[ "${MUON_STABLE:-0}" == "1" ]]; then
    OV2_MUON_SCALE_MODE="${OV2_MUON_SCALE_MODE:-unit_rms_norm}"
    OV2_MUON_EXTRA_SCALE="${OV2_MUON_EXTRA_SCALE:-0.2}"
    OV2_MUON_WD="${OV2_MUON_WD:-0.1}"
  fi
  [[ -n "${OV2_MUON_SCALE_MODE:-}" ]] && OVERRIDES="$OVERRIDES optimizer.muon_scale_mode=$OV2_MUON_SCALE_MODE"
  [[ -n "${OV2_MUON_EXTRA_SCALE:-}" ]] && OVERRIDES="$OVERRIDES optimizer.muon_extra_scale_factor=$OV2_MUON_EXTRA_SCALE"
  [[ -n "${OV2_MUON_WD:-}" ]] && OVERRIDES="$OVERRIDES optimizer.weight_decay=$OV2_MUON_WD scheduler.start_weight_decay=$OV2_MUON_WD scheduler.end_weight_decay=$OV2_MUON_WD"
  echo "[ov2-30b-gb200] MUON ENABLED: scale_mode=${OV2_MUON_SCALE_MODE:-spectral(recipe)} extra_scale=${OV2_MUON_EXTRA_SCALE:-0.15(recipe)} wd=${OV2_MUON_WD:-0.01(recipe)} stable=${MUON_STABLE:-0} -- watch iter-1->3 grad-norm/NaN." >&2
  echo "[ov2-30b-gb200] MUON resume CAUTION: checkpoint.load=$SAVE -- Muon cannot cross-optimizer-resume from an AdamW ckpt (KeyError on momentum). Use a FRESH SAVE (e.g. SAVE=${SAVE}_muon) or a Muon-saved ckpt." >&2
fi
# ---------------------------------------------------------------------------------------------------------
# dataloader workers/rank. 16 is safe here (the 85M-sample data has plenty of shards, Grace has ~1TB RAM);
# lower via OV2_NUM_WORKERS= if host memory gets tight.
OVERRIDES="$OVERRIDES dataset.num_workers=${OV2_NUM_WORKERS:-16}"
# c10d rendezvous + the 30B ckpt-load all_gather both run on this PG timeout (default 10min -> jobs died there).
# 60min is a safety margin; the FQDN fix above is the real rendezvous cure.
OVERRIDES="$OVERRIDES dist.distributed_timeout_minutes=${OV2_DIST_TIMEOUT_MIN:-100}"
# Pin tensorboard to the WRITABLE $SAVE: the recipe default $CWD=$REPO may be read-only -> rank N-1's
# makedirs -> PermissionError -> the crash masquerades as a collective timeout on another rank.
OVERRIDES="$OVERRIDES logger.tensorboard_dir=$SAVE/tensorboard"
OVERRIDES="$OVERRIDES mixed_precision=$MIXED_PRECISION"
[[ "$DISABLE_RECOMPUTE" == "1" ]] && OVERRIDES="$OVERRIDES model.recompute_activations=false model.recompute_granularity=null"
# NB: the dispatcher is wired in ov2_provider.provide() via OV2_FLEX_BACKEND; model.moe_token_dispatcher_type here would be dead.

# --- OPT-IN: EP comm-overlap (~1.3x on exposed EP a2a) is NOT a flag here -- the OV2 recipe sets
# cfg.comm_overlap=None, so a CLI override would crash (attr-on-None). Enabling it needs a recipe change
# (build a CommOverlapConfig in ov2.py) + re-validating the grad path. ---

# --- OPT-IN: Megatron-FSDP (OV2_FSDP=1). Shards params+grads+optim across DP. Only helps when MODEL-STATE
# (not activation) memory is the limit; OV2's long-sequence pain is activation-bound -> FSDP won't fix that.
# fsdp_dtensor ckpts are a ONE-WAY door (not loadable by the torch_dist recipe or convert/ tools). ---
if [[ "${OV2_FSDP:-0}" == "1" ]]; then
  export CUDA_DEVICE_MAX_CONNECTIONS=32        # Megatron-FSDP asserts != 1; set 32 (a bare `unset` risks the consumer's '1' default re-tripping the assert)
  OVERRIDES="$OVERRIDES dist.use_megatron_fsdp=true ddp.use_megatron_fsdp=true"
  OVERRIDES="$OVERRIDES ddp.data_parallel_sharding_strategy=optim_grads_params ddp.average_in_collective=false"
  OVERRIDES="$OVERRIDES checkpoint.ckpt_format=fsdp_dtensor"   # FSDP forces this; see ckpt caveat below
  # CKPT CAVEAT: the default torch_dist INIT_CKPT will MISMATCH under fsdp_dtensor -- convert it first, or
  # point INIT_CKPT/SAVE at an fsdp_dtensor ckpt. fsdp_dtensor saves are NOT loadable back by torch_dist (one-way).
  echo "[ov2-30b-gb200] OV2_FSDP=1: Megatron-FSDP ON (world=$WORLD dp=$DP tp=$TP nnodes=$NNODES). FSDP only helps \
when model-state memory dominates; if activation memory dominates it only adds overhead. fsdp_dtensor ckpt is \
ONE-WAY; torch_dist INIT_CKPT will mismatch -> convert it to fsdp_dtensor or point INIT_CKPT at an fsdp ckpt." >&2
fi

mkdir -p "$SAVE"; cd "$REPO"
# NOTE: the old Muon resume-topology guard was removed -- distributed Muon now supports DP-reshard, so
# resuming a Muon ckpt at a different WORLD size no longer corrupts the sharded momentum (guard obsolete).
echo "[ov2-30b-gb200] in-container | hw=$HWNAME(cc=$_cc) repo=$REPO recipe=$RECIPE accel=$ACCEL mp=$MIXED_PRECISION flex=${OV2_FLEX_BACKEND:-alltoall} recompute_off=$DISABLE_RECOMPUTE recompute_full=$OV2_RECOMPUTE_FULL recompute_moe=$OV2_RECOMPUTE_MOE peak=${MFU_PEAK_TFLOPS}TF nproc=$NPROC world=$WORLD dp=$DP tp=$TP sp=$SP seq=$SEQ_LEN gbs=$MIDTRAIN_GBS iters=$ITERS warmup=$WARMUP_ITERS lr=${OV2_LR:-1e-5}->${OV2_MIN_LR:-1e-6} router_dtype=${OV2_ROUTER_DTYPE:-fp32} permute_fusion=$OV2_MOE_PERMUTE_FUSION aux_loss=$OV2_MOE_AUX_LOSS_COEFF moe_capacity=$MOE_CAPACITY_FACTOR pad_to_capacity=$MOE_PAD_TO_CAPACITY ep_overlap=${OV2_EP_OVERLAP:-0} ep_delay_wgrad=${OV2_EP_DELAY_WGRAD:-0} hybridep_num_sms=${OV2_HYBRIDEP_NUM_SMS:-default} hybridep_permute=${OV2_HYBRIDEP_PERMUTE_FUSION:-0} router_fusion=${OV2_MOE_ROUTER_FUSION:-0} router_pad_fp8=${OV2_MOE_ROUTER_PAD_FP8:-0} shared_overlap=${OV2_MOE_SHARED_EXPERT_OVERLAP:-0} freeze_vision=${OV2_FREEZE_VISION:-recipe-default} node_rank=$NODE_RANK nnodes=$NNODES"
# --- per-rank NUMA binding (OPT-IN, OV2_NUMA_BIND=1): binds each rank's CPU threads + memory to its
# GPU's local Grace socket via numa_bind_wrapper.sh (upstream Megatron-Bridge #4630 mechanism; NVIDIA
# rates CPU affinity ~+15% on GB200 where host overhead co-limits). Default OFF -> the exec line below
# is byte-identical to before. The wrapper soft-falls-back to unbound if resolution fails. ---
_numa_args=()
if [[ "${OV2_NUMA_BIND:-0}" == "1" ]]; then
  if command -v numactl >/dev/null 2>&1 && [[ -f "$_SELF/numa_bind_wrapper.sh" ]]; then
    _numa_args=(--no-python bash "$_SELF/numa_bind_wrapper.sh")
    echo "[ov2-30b-gb200] NUMA binding ON: per-rank numactl cpunodebind+membind via numa_bind_wrapper.sh (OV2_NUMA_BIND=1)"
  else
    echo "[ov2-30b-gb200] WARN: OV2_NUMA_BIND=1 but numactl or numa_bind_wrapper.sh missing -> running unbound" >&2
  fi
fi
# shellcheck disable=SC2086  # $RDZV and $OVERRIDES must word-split into separate args
python -m torch.distributed.run $RDZV --nproc_per_node="$NPROC" ${_numa_args[@]+"${_numa_args[@]}"} scripts/training/run_recipe.py \
  --recipe "$RECIPE" --dataset vlm-energon --step_func ov2_step \
  $OVERRIDES ${EXTRA_ARGS:-} 2>&1 | tee "$SAVE/train_node${NODE_RANK}.log"
