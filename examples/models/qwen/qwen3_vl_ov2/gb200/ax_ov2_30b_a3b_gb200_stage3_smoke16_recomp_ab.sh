#!/usr/bin/env bash
# =============================================================================
# STAGE-3 16-GPU recompute A/B smoke — one workload, both arms, one verdict.
#
# Question: what does OV2_RECOMPUTE_MOE=0 buy on the production topology, and
# does it fit in memory? The running 32-GPU stage-3 job has all three recompute
# layers on and 36-39GiB free — too thin a margin to flip the knob blind
# (the MoE-activation increase is the same order of magnitude).
#
# Arms (run sequentially by every pod, ~20 iters each):
#   a = control: OV2_RECOMPUTE_MOE=1 (production behavior)
#   b = OV2_RECOMPUTE_MOE=0
# Both at TP=4, which REQUIRES 8 pods / 32 GPUs (Workers=7): EP8 with ETP=TP=4
# needs world%32==0 — the stage-3 launcher FATALs on DP%8 at 16 GPUs (first
# submission died exactly there). At 8 pods this is the production topology
# exactly, so peak-memory readings transfer 1:1; still judge by the A/B ratio.
#
# Output: ~/train_logs/smoke_s3_recomp_ab_result_<workload>.txt with p50 iter
# time per arm, speedup %, per-pod peak memory, and an APPLY / DO-NOT-APPLY
# verdict. Per-pod logs: ~/train_logs/smoke_s3_recomp_ab_{a,b}_<hostname>.log.
#
# Iteration lines print on the LAST global rank only, so exactly one pod can
# compute the timing comparison; it writes the shared result file and every
# other pod (master included) waits for it before exiting, so the PyTorchJob
# "master exited => job complete" semantics cannot tear the gang down while
# the verdict is still being written (the B15 lesson).
# =============================================================================
set -euo pipefail

_AB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_AB_BASE="$_AB_DIR/ax_ov2_30b_a3b_gb200_stage3.sh"
_AB_ROOT="$HOME/ckpts_video_sft/_smoke_s3_recomp_ab"
_AB_TAG="$(hostname | sed -E 's/-(master|worker)-[0-9]+$//')"
RESULT="$HOME/train_logs/smoke_s3_recomp_ab_result_${_AB_TAG}.txt"
mkdir -p "$HOME/train_logs"

# smoke16-equivalent scale, production TP. MUON=0 matches smoke16; both arms
# share it, so the ratio is unaffected.
export TP="${TP:-4}"
export OV2_MIDTRAIN_MUON="${OV2_MIDTRAIN_MUON:-0}"
export OV2_TOTAL_SAMPLES="${OV2_TOTAL_SAMPLES:-640}"   # -> 20 iters @ GBS32
export OV2_EPOCHS="${OV2_EPOCHS:-1}"
export SAVE_EVERY="${SAVE_EVERY:-100000}"              # no interval saves

_AB_RC_A="-"; _AB_RC_B="-"; _AB_PEAK_A="?"; _AB_PEAK_B="?"

for _arm in a b; do
  if [[ "$_arm" == "a" ]]; then _moe=1; else _moe=0; fi
  _log="$HOME/train_logs/smoke_s3_recomp_ab_${_arm}_$(hostname).log"
  _save="$_AB_ROOT/$_arm"
  # Fresh SAVE per arm per attempt: stage-3 resumes from checkpoint.load=SAVE,
  # so a leftover iter_0000020 would make the arm run zero iterations.
  [[ "$_save" == "$HOME/ckpts_video_sft/_smoke_s3_recomp_ab/"* ]] && rm -rf "$_save"
  mkdir -p "$_save"
  # Peak-memory sampler for this pod's 4 GPUs; dies with the arm.
  (
    _peak=0
    while sleep 5; do
      _m="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)"
      [[ "$_m" =~ ^[0-9]+$ ]] && (( _m > _peak )) && { _peak=$_m; echo "$_peak" >"$_log.peak"; }
    done
  ) &
  _mon=$!
  echo "[ab] arm=$_arm OV2_RECOMPUTE_MOE=$_moe tp=$TP save=$_save $(date +%Y-%m-%dT%H:%M:%S)" | tee -a "$_log"
  set +e
  SAVE="$_save" OV2_RECOMPUTE_MOE="$_moe" bash "$_AB_BASE" 2>&1 | tee -a "$_log"
  _rc=${PIPESTATUS[0]}
  set -e
  kill "$_mon" 2>/dev/null || true
  wait "$_mon" 2>/dev/null || true
  _peak="$(cat "$_log.peak" 2>/dev/null || echo '?')"
  if [[ "$_arm" == "a" ]]; then _AB_RC_A=$_rc; _AB_PEAK_A=$_peak; else _AB_RC_B=$_rc; _AB_PEAK_B=$_peak; fi
  echo "[ab] arm=$_arm rc=$_rc pod_peak_mem_mib=$_peak" | tee -a "$_log"
  # Arm a is the baseline: if the CONTROL cannot run, the A/B is meaningless.
  if [[ "$_arm" == "a" && "$_rc" != "0" ]]; then
    echo "[ab] FATAL: control arm failed (rc=$_rc) — no baseline, aborting before arm b" | tee -a "$_log"
    exit 1
  fi
done

# ── verdict: written by the one pod whose logs carry iteration lines ─────────
python3 - "$HOME/train_logs/smoke_s3_recomp_ab_a_$(hostname).log" \
          "$HOME/train_logs/smoke_s3_recomp_ab_b_$(hostname).log" \
          "$RESULT" "$_AB_RC_A" "$_AB_RC_B" "$_AB_PEAK_A" "$_AB_PEAK_B" <<'PYEOF'
import os
import re
import statistics as st
import sys

log_a, log_b, result, rc_a, rc_b, peak_a, peak_b = sys.argv[1:8]

def iters(path):
    try:
        xs = [float(x) for x in re.findall(
            r"elapsed time per iteration \(ms\): ([\d.]+)", open(path, errors="replace").read())]
    except OSError:
        return []
    return xs[3:]  # drop warmup

a, b = iters(log_a), iters(log_b)
if not a:
    sys.exit(0)  # not the last-rank pod (or control produced no iterations) — a waiter, not the writer

lines = [f"recompute A/B smoke — {os.path.basename(log_a).rsplit('_', 1)[-1].removesuffix('.log')}"]
p50_a = st.median(a)
lines.append(f"arm a (RECOMPUTE_MOE=1): rc={rc_a} n={len(a)} p50_iter_ms={p50_a:.0f} pod_peak_mib={peak_a}")
if rc_b == "0" and b:
    p50_b = st.median(b)
    speedup = (p50_a - p50_b) / p50_a * 100
    lines.append(f"arm b (RECOMPUTE_MOE=0): rc={rc_b} n={len(b)} p50_iter_ms={p50_b:.0f} pod_peak_mib={peak_b}")
    lines.append(f"speedup: {speedup:.1f}%")
    if speedup >= 8:
        lines.append("VERDICT: APPLY — restart production with OV2_RECOMPUTE_MOE=0 right after a checkpoint lands")
    elif speedup >= 5:
        lines.append("VERDICT: MARGINAL — apply only if a restart is happening anyway")
    else:
        lines.append("VERDICT: DO NOT APPLY — gain under 5%, not worth a restart")
else:
    lines.append(f"arm b (RECOMPUTE_MOE=0): rc={rc_b} FAILED (OOM likely) pod_peak_mib={peak_b}")
    lines.append("VERDICT: DO NOT APPLY — arm b did not survive at DP4; production DP8 is roomier but unproven")
lines.append("note: 8 pods = production topology; still judge by the ratio, not absolutes.")
lines.append("note: peaks are per-pod — check every pod: grep pod_peak ~/train_logs/smoke_s3_recomp_ab_*.log")
tmp = result + ".tmp"
with open(tmp, "w") as f:
    f.write("\n".join(lines) + "\n")
os.replace(tmp, result)
print("\n".join(lines))
PYEOF

# Every pod holds for the verdict so the writer is never reaped mid-write.
_dl=$(( $(date +%s) + 900 ))
while [[ ! -f "$RESULT" ]]; do
  (( $(date +%s) >= _dl )) && { echo "[ab] no result after 15min — check per-pod logs"; break; }
  sleep 10
done
[[ -f "$RESULT" ]] && { echo "[ab] ---- $RESULT ----"; cat "$RESULT"; }
exit 0
