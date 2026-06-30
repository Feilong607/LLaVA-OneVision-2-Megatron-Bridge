#!/usr/bin/env bash
# =============================================================================
# OV2-30B-A3B GB200 VERIFICATION — correctness tests + throughput speed-alignment.
# Run IN-CONTAINER on GB200 (you are already in docker; this only execs pytest /
# the bench, no `docker run` wrapper).
#
# MODES:
#   tests : OV2 Bridge unit (10) + functional (1) tests. Single GPU. Run on ONE node.
#   speed : throughput bench (baseline vs best). EP8 -> run on BOTH nodes (LIST_IP).
#   all   : tests, then speed.
#
# USAGE:
#   bash gb200_verify.sh tests
#   NPROC=4 LIST_IP="<ip0> <ip1>" bash gb200_verify.sh speed
#   NPROC=4 LIST_IP="<ip0> <ip1>" bash gb200_verify.sh all
#
# SPEED ALIGNMENT — the PRIMARY verdict is the baseline/best per-iteration SPEEDUP
# (the trustworthy metric: for THD-packed OV2 the MFU%%/tok-s counter is padded-seq /
# not packing-aware, ~7x off — see ov2_bench_gb200.sh header). Aligned if
# speedup >= OV2_TARGET_SPEEDUP (default 1.5x). MFU%%/tok-s are ADVISORY and only
# checked when you set OV2_TARGET_MFU / OV2_TARGET_TOKS (>0). Override the compared
# presets with BENCH_PRESETS="baseline best".
# =============================================================================
set -uo pipefail
MODE="${1:-all}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$({ d="$HERE"; while [[ "$d" != "/" && ! -d "$d/src/megatron/bridge" ]]; do d="$(dirname "$d")"; done; echo "$d"; })}"
[[ -d "$REPO/src/megatron/bridge" ]] || { echo "FATAL: OV2 repo root not found from $HERE (set REPO=)"; exit 1; }

BENCH="$HERE/ov2_bench_gb200.sh"
BENCH_CSV="${BENCH_CSV:-$HERE/ov2_bench_gb200.csv}"
BENCH_PRESETS="${BENCH_PRESETS:-baseline best}"
OV2_TARGET_SPEEDUP="${OV2_TARGET_SPEEDUP:-1.5}"   # PRIMARY verdict: best/baseline iter-ms speedup (trustworthy for THD-packed)
OV2_TARGET_MFU="${OV2_TARGET_MFU:-0}"             # ADVISORY only (0=skip); OV2 MFU%% is padded-seq/not-THD-aware
OV2_TARGET_TOKS="${OV2_TARGET_TOKS:-0}"           # ADVISORY only (0=skip)

run_tests() {
  echo "================ [verify] OV2 Bridge correctness tests ================"
  export PYTHONPATH="$REPO/src:$REPO/3rdparty/Megatron-LM:$REPO/aiak_shim${PYTHONPATH:+:$PYTHONPATH}"
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OV2_SKIP_HELPERS=1
  # Clean off-repo cwd: the recipe's experiment dir / any stale conftest artifact can't interfere,
  # and pytest still discovers the repo conftest by walking up from the test files.
  cd /tmp
  local U="$REPO/tests/unit_tests/models/ov2" F="$REPO/tests/functional_tests/models/ov2"
  echo "[verify] unit: $U"
  echo "[verify] functional: $F   (set RUN_FUNCTIONAL=0 to skip)"
  local targets="$U"
  [[ "${RUN_FUNCTIONAL:-1}" == "1" ]] && targets="$U $F"
  python -m pytest $targets -v --no-header -p no:cacheprovider
  local rc=$?
  echo "[verify] pytest exit=$rc  ($([ $rc -eq 0 ] && echo PASS || echo FAIL))"
  return $rc
}

run_speed() {
  [[ -f "$BENCH" ]] || { echo "FATAL: bench not found at $BENCH"; return 1; }
  echo "================ [verify] OV2 GB200 throughput bench ================"
  echo "[verify] presets: $BENCH_PRESETS  (run this on BOTH nodes; metrics print on node_rank 0)"
  for p in $BENCH_PRESETS; do
    echo "---- bench preset: $p ----"
    bash "$BENCH" "$p" || echo "[verify] WARN: preset '$p' returned nonzero (inspect its log)"
  done
  [[ -f "$BENCH_CSV" ]] || { echo "[verify] no CSV at $BENCH_CSV (normal on node_rank>0; alignment prints on node 0)"; return 0; }
  python3 - "$BENCH_CSV" "$OV2_TARGET_SPEEDUP" "$OV2_TARGET_MFU" "$OV2_TARGET_TOKS" "$BENCH_PRESETS" <<'PY'
import csv, sys
csvf, tsp, tmfu, ttok, presets = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5].split()
rows = {}
with open(csvf) as f:
    for r in csv.DictReader(f):
        rows[r["preset"]] = r   # last row per preset wins
def g(p,k):
    try: return float(rows[p][k])
    except Exception: return None
print("\n================ SPEED ALIGNMENT ================")
print(f"{'preset':<14}{'fwd_ms':>9}{'bwd_ms':>9}{'iter_ms':>9}{'tok/s/GPU':>11}{'MFU%':>8}")
for p in presets:
    if p in rows:
        r=rows[p]; print(f"{p:<14}{r['fwd_ms']:>9}{r['bwd_ms']:>9}{r['iter_ms']:>9}{r['tok/s/GPU']:>11}{r['MFU%']:>8}")
b, best = g("baseline","iter_ms"), g("best","iter_ms")
key = "best" if "best" in rows else presets[-1]
mfu, tok = g(key,"MFU%"), g(key,"tok/s/GPU")
# PRIMARY verdict = baseline/best iter-ms speedup (trustworthy). MFU%/tok = advisory (padded-seq artifact).
sp = (b/best) if (b and best) else None
if sp is not None:
    print(f"\nspeedup (baseline/best iter-ms): {sp:.2f}x   ({b:.0f}ms -> {best:.0f}ms)   [target >= {tsp}x]")
else:
    print(f"\n[verify] need BOTH 'baseline' and 'best' iter_ms rows for the speedup verdict (have: {sorted(rows)})")
if tmfu>0 and mfu is not None:
    print(f"  (advisory) MFU={mfu}% vs {tmfu}%  [{'OK' if mfu>=tmfu else 'below'}]  -- MFU%% is padded-seq/not-THD-aware; trust the speedup")
if ttok>0 and tok is not None:
    print(f"  (advisory) tok/s/GPU={tok} vs {ttok:.0f}  [{'OK' if tok>=ttok else 'below'}]")
if sp is None:
    print("\nVERDICT: INCONCLUSIVE -- missing baseline/best iter_ms (see above; check each preset's bench log).")
elif sp >= tsp:
    print(f"\nVERDICT: ALIGNED -- speedup {sp:.2f}x >= target {tsp}x.")
else:
    print(f"\nVERDICT: GAP -- speedup {sp:.2f}x < target {tsp}x. Check the per-component (min,max)-across-ranks lines in\n"
          "        the bench log: a large fwd/bwd spread = one EP rank waiting on the inter-node all-to-all =\n"
          "        still collective-bound (try ACCEL=2 HybridEP, NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN matched to the rack).")
PY
}

echo "[gb200_verify] mode=$MODE repo=$REPO"
case "$MODE" in
  tests) run_tests ;;
  speed) run_speed ;;
  all)   run_tests; trc=$?; echo; run_speed; echo; echo "[gb200_verify] tests=$([ ${trc:-1} -eq 0 ] && echo PASS || echo FAIL); see SPEED ALIGNMENT above." ;;
  *) echo "usage: gb200_verify.sh tests|speed|all"; exit 1 ;;
esac
