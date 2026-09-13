#!/usr/bin/env bash
# =============================================================================
# Qwen3.5-35B-A3B merged video stage — 48-GPU SPEED LADDER (A -> B -> C in one workload, no Args).
#
# Runs ax_ov2_qwen35_merged64k_smoke.sh up to three times on the same 12 pods, 80 iterations per
# leg (the first ~60 are warm-up on this line; the summary averages the LAST 20 Step Time lines),
# on the blend that passed the 09-13 memory smoke (stage3_img38_video62_maveric.yaml, seq 73728,
# TP4/ETP2 full recompute: max_allocated 115.2 GiB of a 161.9 cap, 0.6% packs dropped, 0 NaN).
#
# Why these legs (BRINGUP §12.1 memory model, calibrated on that 115.2 measurement, cap 161.9):
#   TP4 full recompute           ~116  measured 115.2    <- the only config known to fit
#   TP4 selective (attn+moe)     ~178  OOM by ~16        <- the s1.5 production setting does NOT fit at 64k
#   TP2 full, alltoall           ~173  OOM, and HybridEP cap 36864 > 21824 so TP2 cannot use ACCEL=2
#   fp32 logits x2 heads (LM+MTP) x 248k vocab = ~45 GiB of the TP4 budget -> OV2_CE_FUSION=true is
#   the one lever that turns "selective fits" from no to yes (~178-45 = ~133). Fused CE through the
#   MTP head is UNVALIDATED, so leg A exists to check it numerically before leg B relies on it.
#
# Round 1 (this file, decided 09-13): two legs, both with CE fusion, same 80 iters, so their steady-state
# Step Time is apples-to-apples:
#   A  TP4/ETP2 GBS48 full recompute, ACCEL=2 (HybridEP)   (~70 GiB)  numerics check + fusion-only speedup
#   T  TP2/ETP2 GBS96 full recompute, ACCEL=0 (alltoall)   (~83 GiB: 173 - 90 of fp32 logits at TP2)
#      TP2 has NO measurement on this backbone at 64k: whether it fits and whether halving TP comm beats
#      HybridEP is exactly what this leg answers. T is not gated on A (its memory question is independent).
# Round 2 (later): B = TP4 selective attn+moe + fusion (~133), C = attn-only; gated on A's numerics.
#
# A is compared against the reference run (same blend/seed/GBS 48, CE fusion OFF): first 5 "lm loss"
# values side by side + max abs diff in the summary (>0.1 = fused CE / MTP-head path suspect; then both
# legs' loss lines are suspect, the memory/speed numbers still stand).
#
# Coordination: all 12 pods run this script. torchrun already makes every leg start together and the
# smoke makes every pod wait for the shared RESULT before leaving a leg, so legs are lock-stepped.
# Only the master pod deletes stale RESULTs and writes the summary (no concurrent writers); workers
# read the same shared files. Exit code: 0 = both legs PASS, 1 = a leg did not pass, 2 = summary failed.
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
export OV2_CE_FUSION=true
export DATA_PATH=stage3_img38_video62_maveric.yaml
export OV2_LLM_HF_QWEN35="${OV2_LLM_HF_QWEN35:-$HOME/Qwen3.5-35B-A3B-text}"
export OV2_HF_PROC_QWEN35_P16M33="${OV2_HF_PROC_QWEN35_P16M33:-$HOME/qwen35_p16m33_auto_model}"
_ITERS="${OV2_LADDER_ITERS:-80}"
# Reference logs for the fusion numerics table: the 09-13 48-GPU memory smoke (CE fusion off, same blend/GBS/seed).
_REF_GLOB="${OV2_LADDER_REF_GLOB:-$HOME/train_logs/smoke_qwen35_merged64k_q35-img38-smoke48-0913-2_*.log}"

[[ -f "$_L_SMOKE" ]] || { _say "FATAL: missing $_L_SMOKE"; exit 1; }
[[ -f "$OV2_LLM_HF_QWEN35/config.json" ]] || { _say "FATAL: missing $OV2_LLM_HF_QWEN35/config.json"; exit 1; }
[[ -f "$OV2_HF_PROC_QWEN35_P16M33/preprocessor_config.json" ]] || { _say "FATAL: missing processor at $OV2_HF_PROC_QWEN35_P16M33"; exit 1; }

_result_of() { echo "$HOME/train_logs/smoke_qwen35_merged64k_result_${_L_TAG}-$1.txt"; }
_leg_logs()  { echo "$HOME"/train_logs/smoke_qwen35_merged64k_"${_L_TAG}-$1"_*.log; }

# _leg NAME TP GBS ACCEL  (ETP fixed at 2: 48 % (2*8) == 0 for both TP4 and TP2; recompute env exported by the caller)
_leg() {
  local name="$1" tp="$2" gbs="$3" accel="$4" r
  r="$(_result_of "$name")"
  if (( _L_IS_MASTER )); then rm -f "$r" 2>/dev/null || true; else sleep 5; fi   # master clears, workers give it a head start
  _say "==== leg $name: TP=$tp ETP=2 GBS=$gbs ACCEL=$accel iters=$_ITERS full=$OV2_RECOMPUTE_FULL moe=$OV2_RECOMPUTE_MOE vision=$OV2_VISION_RECOMPUTE ce_fusion=$OV2_CE_FUSION ===="
  OV2_SMOKE_LEG="$name" TP="$tp" OV2_ETP=2 GBS="$gbs" ACCEL="$accel" OV2_MIDTRAIN_N_SAMPLES=$(( gbs * _ITERS )) bash "$_L_SMOKE"
  _say "leg $name: smoke returned rc=$? (it exits 0 after the shared-result wait; PASS/FAIL is in the RESULT)"
  local dl=$(( $(date +%s) + 120 ))
  while [[ ! -f "$r" ]] && (( $(date +%s) < dl )); do sleep 5; done
}

_passed()    { [[ -f "$1" ]] && grep -q '^VERDICT: PASS' "$1" 2>/dev/null; }
_max_alloc() { sed -n 's/.*max_allocated=\([0-9.]*\).*/\1/p' "$1" 2>/dev/null | head -1; }
_le()        { python3 -c "import sys; sys.exit(0 if float('$1') <= $2 else 1)" 2>/dev/null; }
# Mean of the last 20 "Step Time : Xs" lines (rank 0 prints them -> master log).
_tail_step() {
  grep -h 'Step Time' "$HOME"/train_logs/smoke_qwen35_merged64k_"${_L_TAG}-$1"_*master*.log 2>/dev/null \
    | tail -20 | awk '{gsub("s","",$4); s+=$4; n++} END {if (n) printf "%.1f s/iter over last %d steps", s/n, n; else print "n/a"}'
}
# First 5 "lm loss" values from a set of logs (megatron iteration lines, last rank).
_first_losses() { cat "$@" 2>/dev/null | grep -o 'lm loss: [0-9.E+-]*' | head -5 | awk '{print $3}' | tr '\n' ' ' | sed 's/ *$//'; }

# ---------------- A: TP4 full recompute + CE fusion, HybridEP (numerics check) ----------------
export OV2_RECOMPUTE_FULL=1 OV2_RECOMPUTE_MOE=0 OV2_VISION_RECOMPUTE=1
_leg A 4 48 2
_A_RES="$(_result_of A)"; _A_MEM="$(_max_alloc "$_A_RES")"
_say "A: passed=$(_passed "$_A_RES" && echo yes || echo no) max_allocated=${_A_MEM:-?} GiB; $(_tail_step A)"

# ---------------- T: TP2 full recompute + CE fusion, alltoall (fits? faster?) ----------------
export OV2_RECOMPUTE_FULL=1 OV2_RECOMPUTE_MOE=0 OV2_VISION_RECOMPUTE=1
_leg T 2 96 0
_T_RES="$(_result_of T)"; _T_MEM="$(_max_alloc "$_T_RES")"
_say "T: passed=$(_passed "$_T_RES" && echo yes || echo no) max_allocated=${_T_MEM:-?} GiB; $(_tail_step T)"

_RC=0; _passed "$_A_RES" && _passed "$_T_RES" || _RC=1

# ---------------- summary (master only; workers just mirror the exit code) ----------------
if (( _L_IS_MASTER )); then
  _tmp="$_L_OUT.$$"
  {
    echo "qwen35 merged-stage 48-GPU speed ladder — job $_L_TAG — $(date '+%F %T') — blend $DATA_PATH seq $OV2_SEQ_LEN ce_fusion $OV2_CE_FUSION iters/leg $_ITERS (per-rank microbatches equal: GBS = 4 x DP)"
    echo "reference (CE fusion off, 09-13 memory smoke): TP4/ETP2 full recompute max_allocated 115.2 GiB, ~100 s/iter during warm-up"
    echo
    echo "---- fused-CE numerics: first 5 lm loss, reference vs leg A ----"
    _ref="$(_first_losses $_REF_GLOB)"; _a="$(_first_losses $(_leg_logs A))"
    echo "ref: ${_ref:-n/a}"; echo "A:   ${_a:-n/a}"
    python3 - "$_ref" "$_a" <<'PY'
import sys
r, a = (list(map(float, s.split())) for s in sys.argv[1:3]) if all(sys.argv[1:3]) else ([], [])
n = min(len(r), len(a))
print(f"max |diff| over first {n}: {max(abs(x-y) for x,y in zip(r[:n],a[:n])):.4f}  (bf16 same-batch noise ~1e-2; >0.1 = fused CE/MTP path suspect)" if n else "max |diff|: n/a (missing series)")
PY
    for leg in A T; do
      echo; echo "---- leg $leg ----"
      case "$leg" in
        A) echo "TP4/ETP2 GBS48 (DP12), full recompute + CE fusion, ACCEL=2 HybridEP";;
        T) echo "TP2/ETP2 GBS96 (DP24), full recompute + CE fusion, ACCEL=0 alltoall  [per-rank work = 4 microbatches, same as A]";;
      esac
      r="$(_result_of "$leg")"
      if [[ -f "$r" ]]; then echo "steady-state: $(_tail_step "$leg")"; cat "$r"; else echo "no RESULT file (leg died before the verdict -- OOM/FATAL: see its logs)"; fi
    done
  } > "$_tmp" && mv -f "$_tmp" "$_L_OUT" || { _say "FATAL: could not write summary $_L_OUT"; exit 2; }
  _say "summary -> $_L_OUT"; cat "$_L_OUT"
else
  _dl=$(( $(date +%s) + 300 )); while [[ ! -f "$_L_OUT" ]] && (( $(date +%s) < _dl )); do sleep 10; done
fi
exit "$_RC"
