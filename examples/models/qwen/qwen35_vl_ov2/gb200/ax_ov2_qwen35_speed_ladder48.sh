#!/usr/bin/env bash
# =============================================================================
# Qwen3.5-35B-A3B merged video stage — 48-GPU SPEED LADDER (A -> S -> T in one workload, no extra variables).
#
# Runs ax_ov2_qwen35_merged64k_smoke.sh three times on the same 12 pods, 80 iterations per leg (the
# first ~60 are warm-up on this line; the summary averages the LAST 20 Step Time lines), on the blend
# that passed the 09-13 memory smoke (stage3_img38_video62_maveric.yaml, seq 73728, TP4/ETP2 full
# recompute WITH the MTP head: max_allocated 115.2 GiB of a 161.9 cap, 0.6% packs dropped, 0 NaN).
#
# The merged video stage is specified (BRINGUP §13) to match the 30B s2/s3 line: LM + MoE-aux objective with
# NO MTP head, Muon lr 1e-5->1e-6 / wd 0 / extra 0.2 / beta2 0.99, selective recompute (attn+moe), TP4
# HybridEP, seq 73728, 4 microbatches per rank. Every leg here is therefore built WITHOUT the MTP block
# (OV2_MTP_LAYERS=0, fork 3018a1c0). The 09-13 smoke still carried the MTP layer + its 248k-vocab head,
# which the §12.1 model prices at ~22 GiB (TP4) / ~45 GiB (TP2) of fp32 logits, so 115.2 is an upper bound.
#
# Predicted for the no-MTP build (§12.1, calibrated on 115.2; cap 161.9):
#   TP4 full recompute        ~94    (was 115.2 with MTP)
#   TP4 selective attn+moe    ~156   the specified production recompute -- fits only barely, if at all
#   TP2 full, alltoall        ~128   (was ~173 with MTP); HybridEP cap 36864 > 21824 so TP2 runs ACCEL=0 only
#   OV2_CE_FUSION is NOT a memory lever: mcore's fused path still calls calculate_logits_max -> .float()
#   and saves fp32 exp_logits for backward (fused_cross_entropy.py / cross_entropy.py:28); it only batches
#   the TP all-reduces and jit-fuses elementwise ops. So it stays OFF here (same numerics as production).
#
# Round 1 (this file): three legs, CE fusion off, no MTP, same 80 iters, same per-rank work (GBS = 4 x DP):
#   A  TP4/ETP2 GBS48 full recompute,     ACCEL=2 HybridEP   baseline; also confirms a mtp.*-carrying ckpt
#                                                             loads into the no-MTP model
#   S  TP4/ETP2 GBS48 selective attn+moe, ACCEL=2 HybridEP   = the §13 production recompute: does it fit?
#   T  TP2/ETP2 GBS96 full recompute,     ACCEL=0 alltoall   does TP2 fit without MTP, is it faster than A?
#   A leg that does not fit dies as a clean torch OOM at step 1 (~15 min). No leg is gated on another.
#   Throughput is compared as samples/s (GBS / mean Step Time), never as s/iter: T's GBS is 2x A's.
#   speedup(X vs A) = (GBS_X / X_s) / (48 / A_s); for T that is 2 * A_s / T_s.
# Loss table: first 5 lm loss of A vs the 09-13 smoke (same data/seed, WITH MTP). Iteration 1 is comparable only with matched inputs, masks and trunk weights; later
# steps may drift (different gradients, different LR-decay length). Descriptive only.
#
# Coordination: fresh workload names only. All 12 exact pod identities rendezvous before
# launch and after each leg. Each pod records both the smoke shell status and torchrun
# status from its local log. Only the master writes the final summary/exit status.
# Barrier evidence is retained: deleting it while another pod is polling is unsafe.
# Exit: 0 = all three legs passed, 1 = a candidate failed, 2 = summary failure, 3 = coordination failure.
#
# Workload form: Distributed/PyTorch, qwen35-fla image, gb200-nvl72-nodes, Workers=11 (12 pods x 4 GPU),
# Command (both sides) = bash, Args (both sides) = this file's absolute path. Nothing else.
# Read-out (code-sync workspace):  cat ~/train_logs/smoke_speed_ladder_<job>.txt
# =============================================================================
set -uo pipefail

_L_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_L_SMOKE="$_L_DIR/ax_ov2_qwen35_merged64k_smoke.sh"
_L_HOST="$(hostname)"
_L_TAG="$(sed -E 's/-(master|worker)-[0-9]+$//' <<<"$_L_HOST")"
_L_IS_MASTER=0; [[ "$_L_HOST" == *-master-0 ]] && _L_IS_MASTER=1
_L_OUT="$HOME/train_logs/smoke_speed_ladder_${_L_TAG}.txt"
mkdir -p "$HOME/train_logs"
_say() { echo "[speed-ladder] $*" | tee -a "$HOME/train_logs/smoke_speed_ladder_${_L_TAG}_${_L_HOST}.log" >&2; }

# ---- constants shared by all legs (validated 09-13; the launcher applies workers=2/buffer=16 at seq>=32768) ----
export OV2_K8S_NAMESPACE=runai-mv0004
export OV2_SEQ_LEN=73728   # ACCEL is per leg
export OV2_MEM_PROBE=4 OV2_CUDA_MEM_FRACTION=0.88 OV2_LENGTH_SORT_WINDOW=4
export OV2_CE_FUSION="${OV2_CE_FUSION:-false}"   # see header: not a memory lever
export OV2_MTP_LAYERS="${OV2_MTP_LAYERS:-0}"          # no MTP block/head = the §13 objective (set 1 to A/B the old build)
export OV2_MIDTRAIN_MUON=1 OV2_LR=1e-5 OV2_MIN_LR=1e-6 OV2_MOE_AUX_LOSS_COEFF=0.01
# Apply after the shared midtrain defaults (beta2=.95, wd=.01, extra_scale=.15).
# Scheduler weight decay must match optimizer weight decay on every step.
export EXTRA_ARGS="${EXTRA_ARGS:-} optimizer.muon_scale_mode=spectral optimizer.muon_extra_scale_factor=0.2 optimizer.adam_beta2=0.99 optimizer.weight_decay=0 scheduler.start_weight_decay=0 scheduler.end_weight_decay=0 model.freeze_language_model=false model.freeze_vision_model=false model.freeze_adapter=false"
export DATA_PATH=stage3_img38_video62_maveric.yaml
export OV2_LLM_HF_QWEN35="${OV2_LLM_HF_QWEN35:-$HOME/Qwen3.5-35B-A3B-text}"
export OV2_HF_PROC_QWEN35_P16M33="${OV2_HF_PROC_QWEN35_P16M33:-$HOME/qwen35_p16m33_auto_model}"
_ITERS="${OV2_LADDER_ITERS:-80}"
# Reference logs for the loss table: the 09-13 48-GPU memory smoke (leg A's config but WITH MTP; 20-iteration run, different LR-decay length).
_REF_GLOB="${OV2_LADDER_REF_GLOB:-$HOME/train_logs/smoke_qwen35_merged64k_q35-img38-smoke48-0913-2_*.log}"

[[ -f "$_L_SMOKE" ]] || { _say "FATAL: missing $_L_SMOKE"; exit 1; }
[[ -f "$OV2_LLM_HF_QWEN35/config.json" ]] || { _say "FATAL: missing $OV2_LLM_HF_QWEN35/config.json"; exit 1; }
[[ -f "$OV2_HF_PROC_QWEN35_P16M33/preprocessor_config.json" ]] || { _say "FATAL: missing processor at $OV2_HF_PROC_QWEN35_P16M33"; exit 1; }

_result_of() { echo "$HOME/train_logs/smoke_qwen35_merged64k_result_${_L_TAG}-$1.txt"; }
_NPODS="${PET_NNODES:-${OV2_LADDER_NPODS:-12}}"
[[ "$_NPODS" == 12 && "${NPROC:-4}" == 4 ]] || { _say "FATAL: this ladder requires 12 pods x 4 GPUs"; exit 3; }
[[ "$_ITERS" =~ ^[1-9][0-9]*$ ]] || { _say "FATAL: invalid OV2_LADDER_ITERS"; exit 3; }
_EXPECTED=("${_L_TAG}-master-0")
for (( _i=0; _i<11; _i++ )); do _EXPECTED+=("${_L_TAG}-worker-${_i}"); done
_case_rank=""
for (( _i=0; _i<12; _i++ )); do [[ "${_EXPECTED[$_i]}" != "$_L_HOST" ]] || _case_rank=$_i; done
[[ -n "$_case_rank" && "${PET_NODE_RANK:-$_case_rank}" == "$_case_rank" ]] || { _say "FATAL: invalid/conflicting pod identity"; exit 3; }
_BAR_DIR="$HOME/train_logs/.ladder_barrier_${_L_TAG}"
mkdir -p "$_BAR_DIR" || exit 3
mkdir "$_BAR_DIR/join.${_L_HOST}" 2>/dev/null || { _say "FATAL: existing attempt; use a NEW workload name"; exit 3; }
_FINISHED=0
_abort_on_exit() {
  local rc=$?
  trap - EXIT
  if (( rc != 0 && _FINISHED == 0 )); then printf '%s\n' "$rc" > "$_BAR_DIR/abort.${_L_HOST}"; fi
  exit "$rc"
}
trap _abort_on_exit EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
_barrier() {
  local leg="$1" n host dl=$(( $(date +%s) + 90 * 60 ))
  : > "$_BAR_DIR/${leg}.${_L_HOST}"
  while :; do
    if compgen -G "$_BAR_DIR/abort.*" > /dev/null; then _say "FATAL: peer aborted; inspect $_BAR_DIR"; exit 3; fi
    n=0
    for host in "${_EXPECTED[@]}"; do [[ ! -f "$_BAR_DIR/${leg}.${host}" ]] || n=$((n+1)); done
    (( n == 12 )) && { _say "barrier $leg: 12/12 pods done"; return 0; }
    (( $(date +%s) < dl )) || { _say "FATAL: barrier $leg timeout ($n/12)"; exit 3; }
    sleep 1
  done
}
if [[ -e "$_L_OUT" || -e "$(_result_of A)" || -e "$(_result_of S)" || -e "$(_result_of T)" || -d "$HOME/ckpts_video_sft/_smoke_qwen35_merged64k/${_L_TAG}-A" || -d "$HOME/ckpts_video_sft/_smoke_qwen35_merged64k/${_L_TAG}-S" || -d "$HOME/ckpts_video_sft/_smoke_qwen35_merged64k/${_L_TAG}-T" ]]; then
  _say "FATAL: old outputs exist; use a NEW workload name"; exit 3
fi
_barrier joined
_leg_logs()  { echo "$HOME"/train_logs/smoke_qwen35_merged64k_"${_L_TAG}-$1"_*.log; }

# _leg NAME TP GBS ACCEL  (ETP fixed at 2: 48 % (2*8) == 0 for both TP4 and TP2; recompute env exported by the caller)
_leg() {
  local name="$1" tp="$2" gbs="$3" accel="$4" r
  r="$(_result_of "$name")"
  _barrier "prepare_${name}"
  _say "==== leg $name: TP=$tp ETP=2 GBS=$gbs ACCEL=$accel iters=$_ITERS full=$OV2_RECOMPUTE_FULL moe=$OV2_RECOMPUTE_MOE vision=$OV2_VISION_RECOMPUTE ce_fusion=$OV2_CE_FUSION mtp_layers=$OV2_MTP_LAYERS lr=$OV2_LR->$OV2_MIN_LR wd=0 muon_extra=.2 beta2=.99 ===="
  local smoke_rc raw_rc local_log
  OV2_SMOKE_LEG="$name" TP="$tp" OV2_ETP=2 OV2_MIDTRAIN_GBS="$gbs" GBS="$gbs" ACCEL="$accel" ITERS="$_ITERS" OV2_MIDTRAIN_N_SAMPLES=$(( gbs * _ITERS )) bash "$_L_SMOKE"
  smoke_rc=$?
  local_log="$HOME/train_logs/smoke_qwen35_merged64k_${_L_TAG}-${name}_${_L_HOST}.log"
  raw_rc=$(sed -n 's/.*\[qwen35-smoke\] rc=\([0-9][0-9]*\) pod_peak_mem_mib=.*/\1/p' "$local_log" 2>/dev/null | tail -1)
  printf '%s %s\n' "$smoke_rc" "${raw_rc:-unknown}" > "$_BAR_DIR/status_${name}.${_L_HOST}.tmp"
  mv "$_BAR_DIR/status_${name}.${_L_HOST}.tmp" "$_BAR_DIR/status_${name}.${_L_HOST}" || exit 3
  _say "leg $name: shell rc=$smoke_rc torchrun rc=${raw_rc:-unknown}"
  local dl=$(( $(date +%s) + 120 ))
  while [[ ! -f "$r" ]] && (( $(date +%s) < dl )); do sleep 5; done
  _barrier "$name"
}

_passed() {
  local file="$1" leg host
  leg="${file%.txt}"; leg="${leg##*-}"
  for host in "${_EXPECTED[@]}"; do
    grep -qx '0 0' "$_BAR_DIR/status_${leg}.${host}" 2>/dev/null || return 1
  done
  [[ -f "$file" ]] && grep -q '^VERDICT: PASS' "$file" 2>/dev/null || return 1
  local n
  n=$(sed -n 's/^iters: n=\([0-9][0-9]*\).*/\1/p' "$file" | head -1)
  [[ "$n" =~ ^[0-9]+$ ]] && (( n >= _ITERS - 3 && n > 0 ))
}
_max_alloc() { sed -n 's/.*max_allocated=\([0-9.]*\).*/\1/p' "$1" 2>/dev/null | head -1; }
_le()        { python3 -c "import sys; sys.exit(0 if float('$1') <= $2 else 1)" 2>/dev/null; }
# Mean of the last 20 "Step Time : Xs" lines (rank 0 prints them -> master log); bare number, "" if none.
_mean_step() {
  grep -h 'Step Time' "$HOME"/train_logs/smoke_qwen35_merged64k_"${_L_TAG}-$1"_*master*.log 2>/dev/null \
    | tail -20 | awk '{v=$0; sub(/^.*Step Time[[:space:]]*:[[:space:]]*/,"",v); sub(/s.*$/,"",v); if (v ~ /^[0-9]+([.][0-9]+)?$/ && v>0) {s+=v; n++}} END {if (n==20) printf "%.2f", s/n}'
}
# _tput LEG GBS -> "GBS / mean_s = X samples/s (last N steps)"
_tput() {
  local m; m="$(_mean_step "$1")"
  [[ -n "$m" ]] || { echo "n/a"; return; }
  python3 -c "m=float('$m'); g=$2; print(f'{g} samples / {m:.1f} s = {g/m:.3f} samples/s  (mean of last 20 Step Time)')"
}
# First 5 "lm loss" values from a set of logs (megatron iteration lines, last rank).
_first_losses() { cat "$@" 2>/dev/null | grep -o 'lm loss: [0-9.eE+-]*' | head -5 | awk '{print $3}' | tr '\n' ' ' | sed 's/ *$//'; }

# ---------------- A: TP4 full recompute, HybridEP (steady-state baseline) ----------------
export OV2_RECOMPUTE_FULL=1 OV2_RECOMPUTE_MOE=0 OV2_VISION_RECOMPUTE=1
_leg A 4 48 2
_A_RES="$(_result_of A)"; _A_MEM="$(_max_alloc "$_A_RES")"
_say "A: passed=$(_passed "$_A_RES" && echo yes || echo no) max_allocated=${_A_MEM:-?} GiB; $(_tput A 48)"

# ---------------- S: TP4 selective attn+moe, HybridEP (the §13 production recompute: does it fit?) ----------------
export OV2_RECOMPUTE_FULL=0 OV2_RECOMPUTE_MOE=1 OV2_VISION_RECOMPUTE=1
_leg S 4 48 2
_S_RES="$(_result_of S)"; _S_MEM="$(_max_alloc "$_S_RES")"
_say "S: passed=$(_passed "$_S_RES" && echo yes || echo no) max_allocated=${_S_MEM:-?} GiB; $(_tput S 48)"

# ---------------- T: TP2 full recompute, alltoall (fits without MTP? faster?) ----------------
export OV2_RECOMPUTE_FULL=1 OV2_RECOMPUTE_MOE=0 OV2_VISION_RECOMPUTE=1
_leg T 2 96 0
_T_RES="$(_result_of T)"; _T_MEM="$(_max_alloc "$_T_RES")"
_say "T: passed=$(_passed "$_T_RES" && echo yes || echo no) max_allocated=${_T_MEM:-?} GiB; $(_tput T 96)"

_RC=0; _passed "$_A_RES" && _passed "$_S_RES" && _passed "$_T_RES" || _RC=1

# ---------------- summary (master only; workers just mirror the exit code) ----------------
if (( _L_IS_MASTER )); then
  _tmp="$_L_OUT.$$"
  {
    echo "qwen35 merged-stage 48-GPU speed ladder — job $_L_TAG — $(date '+%F %T') — blend $DATA_PATH seq $OV2_SEQ_LEN ce_fusion $OV2_CE_FUSION mtp_layers $OV2_MTP_LAYERS iters/leg $_ITERS (per-rank microbatches equal: GBS = 4 x DP)"
    echo "reference (09-13 memory smoke, WITH MTP, 20 iters, different LR-decay length): TP4 full recompute max_allocated 115.2 GiB, ~100 s/iter during warm-up"
    echo "optimizer: Muon spectral, lr=$OV2_LR->$OV2_MIN_LR, extra_scale=0.2, adam_beta2=0.99, optimizer/scheduler weight_decay=0; all components trainable"
    echo
    echo "---- throughput (samples/s = GBS / mean of last 20 Step Time; s/iter is NOT comparable across legs) ----"
    echo "A: $(_tput A 48)"
    echo "S: $(_tput S 48)"
    echo "T: $(_tput T 96)"
    _ma="$(_mean_step A)"; _ms="$(_mean_step S)"; _mt="$(_mean_step T)"
    if _passed "$_A_RES" && _passed "$_S_RES" && [[ -n "$_ma" && -n "$_ms" ]]; then
      python3 -c "a=float('$_ma'); x=float('$_ms'); print(f'S vs A speedup = (48/{x:.1f}) / (48/{a:.1f}) = A_s/S_s = {a/x:.3f}x')"
    else
      echo "S vs A speedup: n/a (a leg failed, ran too few steps, or has no valid timing)"
    fi
    if _passed "$_A_RES" && _passed "$_T_RES" && [[ -n "$_ma" && -n "$_mt" ]]; then
      python3 -c "a=float('$_ma'); t=float('$_mt'); print(f'T vs A speedup = (96/{t:.1f}) / (48/{a:.1f}) = 2*A_s/T_s = {2*a/t:.3f}x')"
    else
      echo "T vs A speedup: n/a (a leg failed, ran too few steps, or has no valid timing)"
    fi
    echo
    echo "---- descriptive loss comparison: first 5 lm loss, 09-13 reference (WITH MTP, 20-step schedule) vs leg A (no MTP, 80-step); input identity is not verified; descriptive only ----"
    _ref="$(_first_losses $_REF_GLOB)"; _a="$(_first_losses $(_leg_logs A))"
    echo "ref: ${_ref:-n/a}"; echo "A:   ${_a:-n/a}"
    python3 - "$_ref" "$_a" <<'PY'
import sys
r, a = (list(map(float, s.split())) for s in sys.argv[1:3]) if all(sys.argv[1:3]) else ([], [])
n = min(len(r), len(a))
print(f"iter-1 |diff| = {abs(r[0]-a[0]):.4f} (only meaningful with matched input IDs/labels/masks; not an acceptance threshold); max |diff| over first {n} = {max(abs(x-y) for x,y in zip(r[:n],a[:n])):.4f} (later drift from the removed MTP gradient and the LR-decay length is expected; descriptive only, NOT numerical acceptance)" if n else "iter-1 |diff|: n/a (missing series)")
PY
    for leg in A S T; do
      echo; echo "---- leg $leg ----"
      case "$leg" in
        A) echo "TP4/ETP2 GBS48 (DP12), full recompute, ACCEL=2 HybridEP, no MTP (OV2_MTP_LAYERS=$OV2_MTP_LAYERS)";;
        S) echo "TP4/ETP2 GBS48 (DP12), selective attn+moe = the §13 production recompute, ACCEL=2 HybridEP, no MTP";;
        T) echo "TP2/ETP2 GBS96 (DP24), full recompute, ACCEL=0 alltoall, no MTP  [per-rank work = 4 microbatches, same as A]";;
      esac
      r="$(_result_of "$leg")"
      if [[ -f "$r" ]]; then cat "$r"; else echo "no RESULT file (leg died before the verdict -- OOM/FATAL: see its logs)"; fi
    done
  } > "$_tmp" && mv -f "$_tmp" "$_L_OUT" || { _say "FATAL: could not write summary $_L_OUT"; exit 2; }
  printf '%s\n' "$_RC" > "$_BAR_DIR/final_rc.tmp"
  mv "$_BAR_DIR/final_rc.tmp" "$_BAR_DIR/final_rc" || exit 2
  _say "summary -> $_L_OUT"; cat "$_L_OUT"
else
  _dl=$(( $(date +%s) + 300 ))
  while [[ ! -f "$_BAR_DIR/final_rc" ]]; do
    if compgen -G "$_BAR_DIR/abort.*" > /dev/null; then _say "FATAL: summary writer failed"; exit 2; fi
    (( $(date +%s) < _dl )) || { _say "FATAL: summary timeout"; exit 2; }
    sleep 1
  done
fi
_RC=$(cat "$_BAR_DIR/final_rc")
[[ "$_RC" == 0 || "$_RC" == 1 ]] || exit 2
_barrier summary_read
_FINISHED=1
# Keep barrier/status evidence; a new attempt must use a fresh workload name.
exit "$_RC"
