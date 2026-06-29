#!/usr/bin/env bash
# =============================================================================
# OV2-30B-A3B MoE throughput benchmark on GB200 — A/B sweep over the perf knobs.
# Thin wrapper around ax_ov2_30b_a3b_gb200.sh: short timed run per preset (no save —
# SAVE_EVERY=2000 > BENCH_ITERS), parses steady-state fwd/bwd/iter/MFU + the
# per-component (min,max)-across-ranks spread, appends a row to a CSV.
#
# DIAGNOSIS of MFU≈2.9% (DATA RULED OUT — user confirmed num_workers=16 is best on
# the /datasets GB200; the dataloader is NOT the bottleneck). iter(20.86s)≈fwd(9.2s)+
# bwd(11.5s) with ~6 effective TFLOP/s ⇒ the GPUs STALL in the MoE EP all-to-all, not
# compute. Prime cause: EP8 is split 4+4 across two GB200 nodes and the DEFAULT
# dispatcher is plain `alltoall` (ov2_provider.py:368, ACCEL=0) → every MoE dispatch
# +combine crosses the inter-node fabric. Bridge has NO separate a2a timer — the a2a
# is folded INTO forward-compute/backward-compute, so it looks like "slow compute"
# (fwd≈bwd≈10s with tiny FLOP/s = the collective-bound MoE signature). Fix order:
#   1) HybridEP topology-aware dispatcher  -> ACCEL=2  (intra-NVLink-domain tokens skip the fabric)
#      + domain size = 4 for a 2-node 4+4 split (launcher default 8 is wrong here; can hurt CORRECTNESS)
#   2) fused MoE permute (grouped-GEMM is on, permute is unfused)  -> OV2_MOE_PERMUTE_FUSION=1
#   3) recompute-off (192GB should fit)  -> DISABLE_RECOMPUTE=1   (ACCEL=2 already does this)
#   4) capacity-pad the a2a payload       -> MOE_CAPACITY_FACTOR=1.0 MOE_PAD_TO_CAPACITY=true
#   (fp8 = ACCEL=1 is a SEPARATE path: MXFP8 + alltoall; ACCEL=1 + HybridEP is a HARD ERROR — never stack)
#
# STEP 0 — confirm 2.9% is REAL: the banner prints `peak=...TF`. Via this launcher it is
# 2250 (GB200 bf16) so 2.9% is real; if you ran run_recipe.py directly the default peak 312
# would make true MFU ~20% (a denominator artifact). Trust the raw fwd/bwd/iter seconds.
#
# Usage — run the SAME cmd on BOTH GB200 nodes (node_rank auto from LIST_IP):
#   NPROC=4 LIST_IP="<ip0> <ip1>" bash ov2_bench_gb200.sh baseline
#   ... hybridep | hybridep8 | permute | norecompute | capacity | fp8 | best
# Results accumulate in $BENCH_CSV; only node_rank 0 prints/saves metrics.
# Before trusting `hybridep` (domain=4) verify the NVLink layout: bash gb200/gb200_check.sh
#   (or: nvidia-smi -q | grep -i fabric).  Do NOT edit files — every lever is an env override.
# =============================================================================
set -uo pipefail
PRESET="${1:?usage: ov2_bench_gb200.sh baseline|hybridep|hybridep8|permute|norecompute|capacity|fp8|best}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="${LAUNCHER:-$HERE/ax_ov2_30b_a3b_gb200.sh}"
[[ -f "$LAUNCHER" ]] || { echo "FATAL: launcher not found at $LAUNCHER (set LAUNCHER=)"; exit 1; }

BENCH_ITERS="${BENCH_ITERS:-30}"                  # ~10 warmup (incl. Triton JIT) + 20 measured
BENCH_CSV="${BENCH_CSV:-$HERE/ov2_bench_gb200.csv}"
LOGDIR="${LOGDIR:-$HOME/ov2_bench_logs}"; mkdir -p "$LOGDIR"
LOG="$LOGDIR/bench_${PRESET}.log"

# Common bench env: short run, frequent log, full per-component timing, NO save (SAVE_EVERY 2000 > ITERS).
# num_workers is left at the launcher default (16 — user's confirmed best); data is NOT varied.
export ITERS="$BENCH_ITERS" LOG_EVERY=1 OV2_TIMING_LOG_LEVEL=2 OV2_WARMUP_ITERS="${OV2_WARMUP_ITERS:-1}"

case "$PRESET" in
  baseline)    PENV=() ;;                                                                       # A0: current default (alltoall + recompute-on + bf16) — the 2.9% anchor
  hybridep)    PENV=(ACCEL=2 NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=4) ;;                      # FIX 1: topology-aware dispatcher, domain=4 (correct for 2-node 4+4); ACCEL=2 also turns recompute OFF
  hybridep8)   PENV=(ACCEL=2 NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=8) ;;                      # compare: launcher-default domain=8 (likely mis-set for 2 nodes) — expect WORSE than hybridep
  permute)     PENV=(OV2_MOE_PERMUTE_FUSION=1) ;;                                                # FIX 3: fuse MoE permute on the alltoall baseline (watch first 2 iters for a Triton wedge → revert if a rank hangs)
  norecompute) PENV=(DISABLE_RECOMPUTE=1) ;;                                                     # FIX 4: drop ALL recompute on alltoall baseline (core_attn+moe → none; OOM ⇒ keep on)
  recompute_moe_off) PENV=(OV2_RECOMPUTE_MOE=0) ;;                                                # FIX 4b: drop only MoE from the recompute set (keep core_attn) — cuts bwd recompute FLOPs, less OOM-risk than full norecompute (llava_ov2.py:534-537; GB200 default is core_attn+moe)
  ep_overlap)  PENV=(OV2_EP_OVERLAP=1 CUDA_DEVICE_MAX_CONNECTIONS=32) ;;                           # FIX 5 (env-gated recipe wiring, default OFF): overlap the inter-node EP all-to-all behind expert-FFN compute on the alltoall baseline (~1.3x on exposed a2a). ⚠ MUST diff loss + grad-norm vs baseline — the 2-node MIMO grad-finalize path is fragile (same reason cuda_graph is off)
  capacity)    PENV=(MOE_CAPACITY_FACTOR=1.0 MOE_PAD_TO_CAPACITY=true) ;;                        # bound the a2a payload (drops tokens — speed knob, watch loss)
  fp8)         PENV=(ACCEL=1) ;;                                                                 # MXFP8 + alltoall (peak 4500); SEPARATE from hybridep (ACCEL=1+hybridep is a hard error)
  best)        PENV=(ACCEL=2 NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=4 OV2_MOE_PERMUTE_FUSION=1) ;;  # hybridep(d4) + fused-permute + recompute-off (do NOT add ACCEL=1)
  *) echo "unknown preset '$PRESET'"; exit 1 ;;
esac

echo "[bench] preset=$PRESET  iters=$BENCH_ITERS  log=$LOG"
echo "[bench] env: ${PENV[*]:-<launcher defaults>}"
env "${PENV[@]}" bash "$LAUNCHER" 2>&1 | tee "$LOG"

# --- parse steady-state metrics (rank0 only prints them) ---
python3 - "$LOG" "$PRESET" "$BENCH_CSV" <<'PY'
import os, re, sys, statistics
log, preset, csv = sys.argv[1], sys.argv[2], sys.argv[3]
lines = open(log, errors="ignore").read().splitlines()
fwd=[]; bwd=[]; it=[]; tok=[]; mfu=[]
for l in lines:
    m = re.search(r"forward=([0-9.]+)ms.*?backward=([0-9.]+)ms.*?iter=([0-9.]+)ms.*?tokens/s/GPU=([0-9.]+)", l)
    if m:
        fwd.append(float(m[1])); bwd.append(float(m[2])); it.append(float(m[3])); tok.append(float(m[4]))
    mm = re.search(r"MFU[:= ]+([0-9.]+)", l)
    if mm: mfu.append(float(mm[1]))
def med(x, skip=10):
    x = x[skip:] if len(x) > skip else x
    return statistics.median(x) if x else float("nan")
n = len(it)
peak = next((l.split("peak=")[1].split()[0] for l in lines if "peak=" in l), "?")
if n == 0:
    print(f"[bench] no metric lines parsed (normal on node_rank>0; check node 0's log). banner peak={peak}")
    sys.exit(0)
row = f"{preset},{med(fwd):.0f},{med(bwd):.0f},{med(it):.0f},{med(tok):.0f},{med(mfu):.2f},{peak},{n}"
new = not os.path.exists(csv)
with open(csv, "a") as f:
    if new: f.write("preset,fwd_ms,bwd_ms,iter_ms,tok/s/GPU,MFU%,peakTF,iters\n")
    f.write(row + "\n")
print(f"\n[bench] STEADY-STATE (median of measured iters, {n} logged, banner peak={peak} TF):")
print("   %-12s fwd=%.0fms bwd=%.0fms iter=%.0fms tok/s/GPU=%.0f MFU=%.2f%%" % (preset, med(fwd), med(bwd), med(it), med(tok), med(mfu)))
# collective-imbalance tell: forward/backward-compute (min,max) across ranks at level 2
strag = [l for l in lines if ("min" in l and "max" in l) and ("forward" in l or "backward" in l or "batch" in l or "all-reduce" in l or "all-gather" in l)]
if strag:
    print("\n[bench] per-component (min,max)-across-ranks (LARGE fwd/bwd spread = one EP rank waiting on the inter-node a2a = collective-bound; small spread = real per-rank work):")
    for l in strag[-12:]:
        print("   ", l.strip()[:175])
PY
echo
echo "[bench] comparison so far ($BENCH_CSV):"
column -t -s, "$BENCH_CSV" 2>/dev/null || cat "$BENCH_CSV"
