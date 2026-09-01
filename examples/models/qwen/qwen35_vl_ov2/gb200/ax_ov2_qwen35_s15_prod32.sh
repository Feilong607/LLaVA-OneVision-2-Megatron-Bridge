#!/usr/bin/env bash
# =============================================================================
# Qwen3.5-35B-A3B stage-1.5 (seed85m mid-train @ seq 10192) — 32-GPU PRODUCTION.
#
# Every hyperparameter is baked so the workload form needs only Args + one env
# (OV2_K8S_NAMESPACE) — the ten-env-vars-on-both-sides form is how two bring-up
# runs died with an unreadable exit 1. Config is exactly what the bring-up smoke
# validated, with production budget/save policy:
#
#   TP=2 -> DP=16, EP=8 (recipe), GBS 256, seq 10192, ACCEL=0 (bf16+alltoall),
#   recompute full, Muon (AIAK parity; muon_split_qkv=false for trainable vision),
#   length-aligned batching window 16, INIT = staged stage-2 iter_0006000,
#   8M samples -> 31250 iters, save every 2000, LR 1e-5 -> 1e-6 (recipe/launcher
#   default for this stage; the 2e-5 line belongs to stage-3, not s1.5).
#
# TP=2 is not a preference: Muon's layer-wise full optimizer states do not fit at
# TP=1 on the 35B (measured ~188GB/192GB, OOM in the first fla autotune).
#
# Restart-safe: checkpoint.load == SAVE (the base launcher sets it), so a
# re-submitted job resumes from the newest save in SAVE. Muon cannot resume from
# an AdamW checkpoint — do not point this at an AdamW SAVE.
#
# Debug probes (OV2_NAN_DEBUG / OV2_LAYER_NAN_PROBE / OV2_*_DUMP) are deliberately
# OFF: each does host-synchronising finite checks per layer per microbatch. mcore's
# rerun_state_machine still raises loudly on a NaN loss, and grad-norm lands in the
# iteration lines, so a numerical problem is still visible.
#
# Workload form: Distributed/PyTorch, image
#   mv0004-pytorch-feilong-nemo-qwen35-fla@sha256:6f62b6a6...  (fla 0.5.0),
# gb200-nvl72-nodes, Workers=7 (8 pods x 4 GPU), Command bash (both sides),
# Args = this file's absolute path (both sides), env OV2_K8S_NAMESPACE=runai-mv0004
# (both sides). Optional overrides: SAVE, INIT_CKPT, OV2_MIDTRAIN_N_SAMPLES, TP,
# OV2_MIDTRAIN_MUON, OV2_LENGTH_SORT_WINDOW, SAVE_EVERY.
# =============================================================================
set -euo pipefail

_PD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_PD_BASE="$_PD_DIR/ax_ov2_qwen35_35b_a3b_gb200.sh"
_PD_TAG="$(hostname | sed -E 's/-(master|worker)-[0-9]+$//')"
LOG="$HOME/train_logs/prod_qwen35_s15_${_PD_TAG}_$(hostname).log"
mkdir -p "$HOME/train_logs"

_die() { echo "[qwen35-s15-prod] FATAL: $*" | tee -a "$LOG" >&2; exit 1; }
[[ -f "$_PD_BASE" ]] || _die "base launcher missing: $_PD_BASE"

# ── preflight: assets, before 32 GPUs spin ───────────────────────────────────
_PD_POOL="${OV2_STAGE4_POOL:-/datasets/feilong-stage4-datasets}"
INIT_CKPT="${INIT_CKPT:-$_PD_POOL/35b/ov2_qwen35_35b_a3b_p16m33_stage2_muon_v2/iter_0006000}"
[[ "$(basename "$INIT_CKPT")" == iter_* ]] || _die "INIT_CKPT must point AT an iter_* subdir (tracker=6094 trap): $INIT_CKPT"
[[ -f "$INIT_CKPT/.metadata" || -f "$INIT_CKPT/metadata.json" ]] || _die "INIT_CKPT has no torch_dist metadata: $INIT_CKPT"

_pick() {  # _pick VAR file candidates...
  local _var="$1" _file="$2" _c; shift 2
  for _c in "$@"; do
    [[ -f "$_c/$_file" ]] && { eval "$_var=\"\$_c\""; return 0; }
  done
  _die "$_var not found ($_file missing in all of: $*). Stage the asset or set $_var explicitly."
}
# Bare VLM dir stays LAST for llm_hf: its config.json exists but routes AutoBridge to the
# VLM architecture, which is what the '-text' extract exists to avoid.
[[ -n "${OV2_LLM_HF_QWEN35:-}" ]] || _pick OV2_LLM_HF_QWEN35 config.json \
  "$_PD_POOL/35b/Qwen3.5-35B-A3B-text" "$HOME/Qwen3.5-35B-A3B-text" \
  "/datasets/llava/11May/Qwen3.5-35B-A3B-text" "$_PD_POOL/35b/Qwen3.5-35B-A3B"
[[ -n "${OV2_HF_PROC_QWEN35_P16M33:-}" ]] || _pick OV2_HF_PROC_QWEN35_P16M33 preprocessor_config.json \
  "$_PD_POOL/35b/llava_onevision2_qwen35_35b_a3b_p16_m33/auto_model" \
  "$_PD_POOL/35b/auto_model" \
  "/datasets/llava/11May/llava_onevision2/llava_onevision2_qwen35_35b_a3b_p16_m33/auto_model" \
  "$HOME/qwen35_p16m33_auto_model"
export OV2_LLM_HF_QWEN35 OV2_HF_PROC_QWEN35_P16M33
[[ "$OV2_LLM_HF_QWEN35" == *"-text" ]] || echo "[qwen35-s15-prod] WARN: llm_hf=$OV2_LLM_HF_QWEN35 is not a '-text' extract" | tee -a "$LOG" >&2

_PD_YAML="$_PD_DIR/../../qwen3_vl_ov2/gb200/mid_training_seed85m.yaml"
[[ -f "$_PD_YAML" ]] || _die "seed85m yaml missing: $_PD_YAML"
_first_ds="$(grep -m1 'path:' "$_PD_YAML" | awk '{print $2}')"
[[ -d "$_first_ds" ]] || _die "seed85m shard dir not mounted: $_first_ds"

# ── production config (see header) ───────────────────────────────────────────
# TP=4 -> DP=8 (== EP8). Measured 2026-09-01: TP=2 + Muon does NOT survive its FIRST optimizer
# step on the 35B. Muon is layer-wise (use_distributed_optimizer=False), so its full states only
# materialize at that step, taking the pod from ~119GB (forward-only, which is all the earlier
# short smokes ever reached) to 188.4/192GB — then NCCL, whose buffers are cudaMalloc'd OUTSIDE
# the torch pool, failed with `ncclUnhandledCudaError: Call to CUDA function failed` on the last
# ranks. TP=4 halves weights/main-grads/Muon states again (~77GB -> ~39GB static) and is the
# shape 30B stage-3 runs in production (TP4 + Muon, 11.5k+ iters).
export TP="${TP:-4}"
export OV2_MIDTRAIN_GBS="${OV2_MIDTRAIN_GBS:-256}"               # 30B same-stage production GBS
export OV2_MIDTRAIN_N_SAMPLES="${OV2_MIDTRAIN_N_SAMPLES:-8000000}"   # seed85m budget -> 31250 iters
export SAVE_EVERY="${SAVE_EVERY:-2000}"
export ACCEL="${ACCEL:-0}"                                       # HybridEP/MXFP8 unvalidated on GDN+MTP
export OV2_RECOMPUTE_FULL="${OV2_RECOMPUTE_FULL:-1}"
export OV2_LENGTH_SORT_WINDOW="${OV2_LENGTH_SORT_WINDOW:-16}"    # EP straggler counter-measure
export RECIPE="ov2_qwen35_35b_a3b_midtrain"
export SAVE="${SAVE:-$HOME/ckpts_video_sft/ov2_qwen35_s15_seed85m_muon}"
export INIT_CKPT
export OV2_MIDTRAIN_MUON="${OV2_MIDTRAIN_MUON:-1}"
if [[ "$OV2_MIDTRAIN_MUON" == "1" ]]; then
  # Required for the trainable vision fused-QKV layout (the base launcher only appends it for
  # stage2 recipes; midtrain trains vision too).
  export EXTRA_ARGS="${EXTRA_ARGS:-} optimizer.muon_split_qkv=false"
fi
mkdir -p "$SAVE"

# Peak-memory sampler for this pod (dies with the script).
( _peak=0
  while sleep 30; do
    _m="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)"
    [[ "$_m" =~ ^[0-9]+$ ]] && (( _m > _peak )) && { _peak=$_m; echo "$_peak" >"$LOG.peak"; }
  done ) &

echo "[qwen35-s15-prod] tp=$TP gbs=$OV2_MIDTRAIN_GBS n_samples=$OV2_MIDTRAIN_N_SAMPLES muon=$OV2_MIDTRAIN_MUON sort_window=$OV2_LENGTH_SORT_WINDOW accel=$ACCEL save_every=$SAVE_EVERY init=$INIT_CKPT save=$SAVE" | tee -a "$LOG"
python3 -c "import fla; print('[qwen35-s15-prod] fla', getattr(fla,'__version__','?'), fla.__file__)" 2>&1 | tee -a "$LOG" || true
echo "[qwen35-s15-prod] watch: grep -E 'iteration +[0-9]+/' \$HOME/train_logs/prod_qwen35_s15_*worker-6*.log | tail -5   (iteration lines print on the LAST rank's pod)" | tee -a "$LOG"

set +e
bash "$_PD_BASE" 2>&1 | tee -a "$LOG"
_rc=${PIPESTATUS[0]}
set -e
echo "[qwen35-s15-prod] rc=$_rc pod_peak_mem_mib=$(cat "$LOG.peak" 2>/dev/null || echo '?')" | tee -a "$LOG"
# PyTorchJob reads the MASTER pod's exit code as the job verdict.
if [[ "$(hostname)" == *-master-* ]]; then exit "$_rc"; fi
exit 0
