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
#   SELECTIVE recompute (core_attn + moe) with the vision tower NOT recomputed,
#   Muon (AIAK parity; muon_split_qkv=false for trainable vision), length-aligned
#   batching window 16, INIT = staged stage-2 iter_0006000, 8M samples -> 31250
#   iters, save every 2000, LR 1e-5 -> 1e-6 (recipe/launcher default for this
#   stage; the 2e-5 line belongs to stage-3, not s1.5).
#
# Both TP and the recompute lane are MEASURED choices, not preferences — see the
# numbers in the config block below. Baseline to beat: 108-120 s/iter at 567-687
# tokens/s/GPU (TP=2, full recompute), against 30B stage-3's ~2700.
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
# gb200-nvl72-nodes, Command bash (both sides), Args = this file's absolute path
# (both sides), env OV2_K8S_NAMESPACE=runai-mv0004 (both sides).
#   32 GPU: Workers=7  (8 pods)  -> TP2/DP16, 16 microbatches/rank
#   64 GPU: Workers=15 (16 pods) -> TP2/DP32,  8 microbatches/rank   <- just change Workers
# KEEP TP=2 when scaling out (vs TP4/TP8). Per-GPU LLM compute is total/WORLD either way, but the vision
# tower and adapter are REPLICATED per rank (built TP=1), so tower work per GPU scales with microbatches
# per rank = GBS/DP: higher TP means MORE tower work per GPU, plus doubled step and EP a2a counts. TP8
# additionally splits a TP group across pods (4 GPU/pod). Extra GPUs belong in DP, not TP. Whether TP=1
# beats TP=2 at steady state is UNMEASURED (smoke arms ab-tp1-*); TP=2 is the incumbent.
# 64 GPU = the ENTIRE project quota: stop the export workspace and any eval first, or the gang never
# assembles (that mistake cost 13 h of idle GPUs on 2026-09-01).
# Optional overrides: SAVE, INIT_CKPT, OV2_MIDTRAIN_N_SAMPLES, TP, OV2_MIDTRAIN_MUON,
# OV2_LENGTH_SORT_WINDOW, SAVE_EVERY, OV2_KEEP_CKPTS.
# =============================================================================
set -euo pipefail

# Knobs may also arrive as positional KEY=VALUE args. The workload form's env section can be locked
# by an admin policy (observed 2026-09-03: policy-injected AWS_* rows, "+ ENVIRONMENT VARIABLE"
# rejected), while Args stays editable — so
#     bash <this script> OV2_LENGTH_SORT_KEY=patches TP=2
# is exactly equivalent to setting those in the environment. Exported here, BEFORE any default
# below is read. Anything that is not NAME=value is a typo; fail loud rather than silently run the
# default configuration under an experiment's job name.
for _kv in "$@"; do
  if [[ "$_kv" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]]; then
    export "$_kv"
  else
    echo "FATAL: positional arg '$_kv' is not KEY=VALUE" >&2; exit 1
  fi
done

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
# TP=2 -> DP=16. Early-iteration readings (GBS 256, seq 10192, Muon, FULL recompute) — NOT steady state
# (iteration time keeps falling until ~iter 30-50; TP=1 was killed after iteration 2):
#     TP=2  iter 2-4: 108-120 s/iter   live 26.7G / peak-live 44.3G / reserved 56G
#     TP=1  iter 1: 374 s (vs TP=2 469), iter 2: 176 s   live 51.8G / peak-live 82.1G / reserved 99-102G
# Current production lane (TP=2, selective recompute, tower not recomputed): 63-77 s/iter,
#     874-1082 tokens/s/GPU, live 30.7G / peak-live 92.7G / reserved 125-142G.
# The former "TP=1 is 1.6x slower ... GC-threshold churn" reading of those numbers is WITHDRAWN: one
# iteration-2 sample is not a throughput measurement, the GC threshold was inert (never armed) and
# 99-102 GiB sat below it anyway. TP=1 doubles per-rank model state and runs ETP=1/SP off; TP=2 runs
# ETP=2 (experts tensor-sharded). Settle it with the ab-tp1 smoke pairs. TP=4 is unnecessary — the earlier
# "TP=2 does not fit" conclusion came from the allocator fragmentation bug (see the base launcher's
# PYTORCH_CUDA_ALLOC_CONF note), not from real memory pressure: live peaks at 44 GB of 189.5.
export TP="${TP:-2}"
export OV2_MIDTRAIN_GBS="${OV2_MIDTRAIN_GBS:-256}"               # 30B same-stage production GBS
export OV2_MIDTRAIN_N_SAMPLES="${OV2_MIDTRAIN_N_SAMPLES:-8000000}"   # seed85m budget -> 31250 iters
# SAVE_EVERY is in ITERATIONS; the wall-clock it maps to depends on the current rate (2000 iters is
# ~39 h at the 32-GPU rate, ~20 h at 64 GPUs). Everything here is preemptible, so this interval
# bounds how much work a preemption destroys — tighten it (e.g. 800) if preemptions become frequent.
export SAVE_EVERY="${SAVE_EVERY:-2000}"
export ACCEL="${ACCEL:-0}"                                       # HybridEP/MXFP8 unvalidated on GDN+MTP
# Recompute: spend the headroom the allocator fix returned (peak-live 44 GB of 189.5, ~37 GB of the
# card is non-torch) on throughput. full/uniform/1 recomputes EVERY layer — literally a second
# forward pass; selective keeps that only for core_attn (memory-heavy, compute-light) plus the MoE
# layer (the biggest activation consumer), which is the combination 30B stage-3 runs in production.
# Vision recompute goes off too: the tower re-runs 63k patches through 24 layers for no memory reason
# now. Fall back to OV2_VISION_RECOMPUTE=1 first if memory gets tight — it costs the most memory and
# buys the least speed of the two.
export OV2_RECOMPUTE_FULL="${OV2_RECOMPUTE_FULL:-0}"
export OV2_RECOMPUTE_MOE="${OV2_RECOMPUTE_MOE:-1}"
export OV2_VISION_RECOMPUTE="${OV2_VISION_RECOMPUTE:-0}"
# garbage_collection_threshold is INERT unless OV2_CUDA_MEM_FRACTION arms it (see the base launcher's
# allocator note); 0.8 is kept for parity. If armed, 0.8 x device = ~151 GiB is above the measured
# 92.7 GiB peak-live of this lane.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-garbage_collection_threshold:0.8}"
# Cap torch's CUDA pool so the threshold above is actually consulted and NCCL keeps its ~37 GB out-of-pool
# headroom. MEASURED need (2026-09-04, smoke ab-sortkey-a6, 30 iters, this lane): max_allocated 92.7 G but
# max_reserved 165.7 G and a per-pod nvidia-smi peak of 187.5 of 189.5 GB — reserved creeps with the bin
# sequence (production read 125-142 G at iteration 33), so an unbounded pool will eventually hand NCCL an
# OOM. 0.8 x 189.5 = ~151 G; live peaks at ~93 G, so the cap costs nothing until fragmentation would have.
export OV2_CUDA_MEM_FRACTION="${OV2_CUDA_MEM_FRACTION:-0.8}"
export OV2_MEM_PROBE="${OV2_MEM_PROBE:-8}"                       # allocated-vs-reserved telemetry (one line / 8 forwards)
# Throughput telemetry, on by default because it is the one measurement that separates "the vision
# tower dominates" from "the LLM dominates", and it costs one cuda-event pair per 8 forwards. It also
# prints patches_per_token, which is the number that decides whether this line is genuinely slower
# than the 30B reference or merely processing far more patches per LLM token.
export OV2_PHASE_TIMER="${OV2_PHASE_TIMER:-8}"
# Length-aligned batching window: derived in the base launcher as GBS/DP (= one iteration per rank; 16 at
# TP=2, 8 at TP=1). A fixed 16 at TP=1 spanned two iterations and made them alternate light/heavy
# (ab-tp1-hep, 2026-09-04). OV2_LENGTH_SORT_KEY=patches measured 0% vs tokens (ab-sortkey-a6).
export RECIPE="ov2_qwen35_35b_a3b_midtrain"
export SAVE="${SAVE:-$HOME/ckpts_video_sft/ov2_qwen35_s15_seed85m_muon}"
export INIT_CKPT
export OV2_MIDTRAIN_MUON="${OV2_MIDTRAIN_MUON:-1}"
# Bound the checkpoint directory. mcore's most_recent_k defaults to -1 = keep EVERY save; a 35B model
# with non-distributed Muon states is hundreds of GB per save, and 31250/SAVE_EVERY of them would
# fill the filesystem (and a full filesystem kills the job at the next save).
# ⚠️ This DELETES older saves: export or copy any milestone you want to evaluate BEFORE it ages out
# (the 30B line had to keep iter_16000 as its best-scoring checkpoint — a blind "keep latest 2" would
# have destroyed it). OV2_KEEP_CKPTS raises the window.
export EXTRA_ARGS="${EXTRA_ARGS:-} checkpoint.most_recent_k=${OV2_KEEP_CKPTS:-4}"
if [[ "$OV2_MIDTRAIN_MUON" == "1" ]]; then
  # Required for the trainable vision fused-QKV layout (the base launcher only appends it for
  # stage2 recipes; midtrain trains vision too).
  export EXTRA_ARGS="$EXTRA_ARGS optimizer.muon_split_qkv=false"
fi
mkdir -p "$SAVE"

# Peak-memory sampler for this pod (dies with the script).
( _peak=0
  while sleep 30; do
    _m="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)"
    [[ "$_m" =~ ^[0-9]+$ ]] && (( _m > _peak )) && { _peak=$_m; echo "$_peak" >"$LOG.peak"; }
  done ) &

echo "[qwen35-s15-prod] tp=$TP gbs=$OV2_MIDTRAIN_GBS n_samples=$OV2_MIDTRAIN_N_SAMPLES muon=$OV2_MIDTRAIN_MUON sort_window=$OV2_LENGTH_SORT_WINDOW accel=$ACCEL save_every=$SAVE_EVERY recompute_full=$OV2_RECOMPUTE_FULL recompute_moe=$OV2_RECOMPUTE_MOE vision_recompute=$OV2_VISION_RECOMPUTE alloc=$PYTORCH_CUDA_ALLOC_CONF mem_probe=$OV2_MEM_PROBE init=$INIT_CKPT save=$SAVE" | tee -a "$LOG"
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
