#!/usr/bin/env bash
# =============================================================================
# Qwen3.5-35B-A3B stage-1.5 (seed85m mid-train @ seq 10192) smoke — 32 GPU.
#
# 30-iteration smoke for the qwen3.5 s1.5 line (bring-up AND throughput A/B): full-model SFT
# (midtrain recipe) on the seed85m_video20m_47m5p_packed pool at its offline
# pack length 10192, initialized from the staged stage-2 checkpoint. Wraps the
# base launcher with what the bare launcher lacks:
#   * asset preflight — fail LOUD with the missing path BEFORE GPUs spin,
#   * baked-in smoke env — nothing to (forget to) type into per-side workload
#     env fields; a missing env on one side is how the bare-launcher submit
#     died with a log-less exit 1,
#   * a persisted per-hostname tee — gang teardown deletes pod UI logs, so the
#     base launcher's pre-torchrun FATALs were unreadable; here EVERY line
#     (preflight + launcher guards + training) lands on WekaFS,
#   * a peak-memory sampler and a written verdict.
#
# Verdict numbers a real s1.5 launch needs: iter p50/p90, dropped-pack count
# at seq 10192 (packs were made with the Qwen2.5-VL tokenizer; under the 3.5
# tokenizer some may exceed seq_length -> SkipSample = biased data loss), fb
# straggler spread, per-pod peak memory (recompute-full fit -> whether the
# DISABLE_RECOMPUTE=1 speed lever is affordable).
#
# Topology (NPROC=4/pod; EP=8 fixed in the recipe): 32 GPU = 8 pods
# (Workers=7): TP=2 -> DP=16. TP=2 is the INCUMBENT, not a measured winner: the
# only TP=1 sample is one iteration-2 reading (176 s, run killed) against TP=2
# iterations 2-4 (108-120 s) — none of it steady state, and TP=1's iteration 1
# was FASTER (374 vs 469 s). Structurally TP=1 does half the replicated tower
# work and half the a2a count per GPU; its cost is memory (2x per-rank state).
# The ab-tp1 arms below settle it. GBS=256 = the 30B same-stage production value.
#
# Workload form: Distributed/PyTorch, image feilong-nemo, gb200-nvl72-nodes,
# Command bash (both sides), Args = this file's absolute path (both sides),
# env OV2_K8S_NAMESPACE=runai-mv0004 (both sides). Optional env: TP, GBS
# (OV2_MIDTRAIN_GBS), ACCEL, OV2_MIDTRAIN_MUON, OV2_RECOMPUTE_FULL,
# OV2_LENGTH_SORT_WINDOW.
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

_SM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_SM_BASE="$_SM_DIR/ax_ov2_qwen35_35b_a3b_gb200.sh"
_SM_TAG="$(hostname | sed -E 's/-(master|worker)-[0-9]+$//')"
_SM_ROOT="$HOME/ckpts_video_sft/_smoke_qwen35_s15"
SAVE_DIR="$_SM_ROOT/$_SM_TAG"
RESULT="$HOME/train_logs/smoke_qwen35_s15_result_${_SM_TAG}.txt"
LOG="$HOME/train_logs/smoke_qwen35_s15_${_SM_TAG}_$(hostname).log"
mkdir -p "$HOME/train_logs" "$SAVE_DIR"
# Stale-state guard (the B16 lesson applies to training smokes too): a REUSED workload name
# resurrects the previous run's verdict file, and the hold-for-verdict loop then short-circuits —
# workers exit 0 instantly against a stale FAIL/PASS. Clear this tag's verdict + peak marker at
# startup (all pods start minutes before any verdict write, so the rm cannot race a fresh one).
# Prefer a fresh workload name anyway; this makes reuse safe rather than silently poisonous.
rm -f "$RESULT" "$LOG.peak"

# ── preflight: every asset the recipe will touch, checked before GPUs spin ───
_die() { echo "[qwen35-s15-smoke] FATAL: $*" | tee -a "$LOG" >&2; exit 1; }

[[ -f "$_SM_BASE" ]] || _die "base launcher missing: $_SM_BASE"

# INIT = staged stage-2 checkpoint. Must point AT the iter_* subdir (the
# tracker says 6094 but only iter_0006000/ exists — BRINGUP §1.4 trap).
INIT_CKPT="${INIT_CKPT:-/datasets/feilong-stage4-datasets/35b/ov2_qwen35_35b_a3b_p16m33_stage2_muon_v2/iter_0006000}"
[[ "$(basename "$INIT_CKPT")" == iter_* ]] || _die "INIT_CKPT must point AT an iter_* subdir (tracker=6094 trap): $INIT_CKPT"
[[ -f "$INIT_CKPT/.metadata" || -f "$INIT_CKPT/metadata.json" ]] || _die "INIT_CKPT has no torch_dist metadata: $INIT_CKPT"

# LLM HF (config source) and processor: prefer the stage-4 mount, fall back to
# the recipe's legacy default root — the two known staging gaps on MAVERIC.
_SM_POOL="${OV2_STAGE4_POOL:-/datasets/feilong-stage4-datasets}"
_pick() {  # _pick VAR file candidates...
  local _var="$1" _file="$2" _c; shift 2
  for _c in "$@"; do
    [[ -f "$_c/$_file" ]] && { eval "$_var=\"\$_c\""; return 0; }
  done
  _die "$_var not found ($_file missing in all of: $*). Stage the asset or set $_var explicitly."
}
# Candidate order: staged official copies first, then the on-cluster generated/assembled fallbacks
# ($HOME/Qwen3.5-35B-A3B-text = extract_qwen35_text.py --weights output; $HOME/qwen35_p16m33_auto_model
# = 30B p16m33 image-processor half + Qwen3.5 tokenizer half, verified image_pad=248056 / merge 3).
# The bare VLM dir stays LAST for llm_hf: its config.json exists but routes AutoBridge to the VLM
# architecture (the -text extract exists precisely to avoid that) — the WARN below fires on it.
[[ -n "${OV2_LLM_HF_QWEN35:-}" ]] || _pick OV2_LLM_HF_QWEN35 config.json \
  "$_SM_POOL/35b/Qwen3.5-35B-A3B-text" "$HOME/Qwen3.5-35B-A3B-text" \
  "/datasets/llava/11May/Qwen3.5-35B-A3B-text" "$_SM_POOL/35b/Qwen3.5-35B-A3B"
[[ -n "${OV2_HF_PROC_QWEN35_P16M33:-}" ]] || _pick OV2_HF_PROC_QWEN35_P16M33 preprocessor_config.json \
  "$_SM_POOL/35b/llava_onevision2_qwen35_35b_a3b_p16_m33/auto_model" \
  "$_SM_POOL/35b/auto_model" \
  "/datasets/llava/11May/llava_onevision2/llava_onevision2_qwen35_35b_a3b_p16_m33/auto_model" \
  "$HOME/qwen35_p16m33_auto_model"
export OV2_LLM_HF_QWEN35 OV2_HF_PROC_QWEN35_P16M33
[[ "$OV2_LLM_HF_QWEN35" == *"-text" ]] || echo "[qwen35-s15-smoke] WARN: llm_hf=$OV2_LLM_HF_QWEN35 is not a '-text' extract; if the build dies routing the VLM config, run tools/extract_qwen35_text.py --weights first." | tee -a "$LOG" >&2

# Data = the committed seed85m blend the base launcher already defaults to;
# verify the yaml and its first shard dir exist on this mount.
_SM_YAML="$_SM_DIR/../../qwen3_vl_ov2/gb200/mid_training_seed85m.yaml"
[[ -f "$_SM_YAML" ]] || _die "seed85m yaml missing: $_SM_YAML"
_first_ds="$(grep -m1 'path:' "$_SM_YAML" | awk '{print $2}')"
[[ -d "$_first_ds" ]] || _die "seed85m shard dir not mounted: $_first_ds (from $_SM_YAML)"

# ── scale + knobs (see header) ────────────────────────────────────────────────
# TP=2 -> DP=16 (32 GPU). TP=1 vs TP=2 steady-state throughput is UNMEASURED (one iteration-2 sample
# each way; see the ab-tp1 arms). TP=1 doubles per-rank model state (measured allocated 51.8 vs 26.7 GiB)
# and runs SP off / ETP=1. The "GC-threshold churn" explanation once written here was wrong: the
# threshold is inert without OV2_CUDA_MEM_FRACTION and 99-102 GiB was below it anyway. 256 % 16 == 0.
export TP="${TP:-2}"
export OV2_MIDTRAIN_GBS="${OV2_MIDTRAIN_GBS:-${GBS:-256}}"   # 30B same-stage production GBS
# 30 iterations by default: iteration 1 is JIT/autotune (~470 s measured) and per-iteration time keeps
# easing for a few tens of iterations (TB on the production lane: ~100 s at iter 2 -> ~65 s by iter
# 30-50), so the verdict drops the first OV2_SMOKE_WARM (default 10) and reports p50/p90 over the rest,
# plus a last-20 p50. For a decision between arms that may differ by <10% (e.g. TP=1 vs TP=2) give BOTH
# arms OV2_MIDTRAIN_N_SAMPLES=15360 (60 iters) and read the last-20 p50.
export OV2_MIDTRAIN_N_SAMPLES="${OV2_MIDTRAIN_N_SAMPLES:-$(( OV2_MIDTRAIN_GBS * 30 ))}"  # -> 30 iters
export OV2_SMOKE_WARM="${OV2_SMOKE_WARM:-10}"
# SAVE_EVERY=0 disables BOTH interval saves and the end-of-run save (train.py skips the final save
# when save_interval == 0, and the interval check is truthiness-guarded, so no modulo-by-zero).
# A 35B+Muon save is hundreds of GB and several minutes; a throughput smoke has no use for one.
export SAVE_EVERY="${SAVE_EVERY:-0}"
export ACCEL="${ACCEL:-0}"                     # bf16 + alltoall (HybridEP/MXFP8 unvalidated on GDN+MTP)
# Recompute / allocator / telemetry MUST track ax_ov2_qwen35_s15_prod32.sh, or every A/B run here
# is measured against a baseline production no longer uses. Production values as of 2026-09-02
# (measured: 63-77 s/iter, 874-1082 tokens/s/GPU, peak-live 92.7 G, reserved 125-142 G):
export OV2_RECOMPUTE_FULL="${OV2_RECOMPUTE_FULL:-0}"
export OV2_RECOMPUTE_MOE="${OV2_RECOMPUTE_MOE:-1}"
export OV2_VISION_RECOMPUTE="${OV2_VISION_RECOMPUTE:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-garbage_collection_threshold:0.8}"
export OV2_CUDA_MEM_FRACTION="${OV2_CUDA_MEM_FRACTION:-0.8}"   # arms the threshold; tracks prod (see prod32 note)
export OV2_MEM_PROBE="${OV2_MEM_PROBE:-8}"
export OV2_PHASE_TIMER="${OV2_PHASE_TIMER:-8}"
# A/B usage: submit ONE knob per job, everything else default, job name = the knob. Candidates:
#   OV2_LENGTH_SORT_KEY=patches | OV2_FEAT_CHECK_EVERY=8 | OV2_RECOMPUTE_MOE=0 |
#   OV2_MOE_PERMUTE_FUSION=1 | ACCEL=2 | OV2_CE_FUSION=true | OV2_MIDTRAIN_GBS=64 (diagnostic only)
# TP arms — run as PAIRS, 60 iters each, fresh job names:
#   ab-tp1-full : TP=1 OV2_RECOMPUTE_FULL=1 OV2_VISION_RECOMPUTE=1 OV2_MIDTRAIN_N_SAMPLES=15360
#   ab-tp2-full : TP=2 OV2_RECOMPUTE_FULL=1 OV2_VISION_RECOMPUTE=1 OV2_MIDTRAIN_N_SAMPLES=15360
#   (both known to fit: the lane the withdrawn "1.6x" claim was made on, this time to steady state) then
#   ab-tp1-sel  : TP=1 OV2_VISION_RECOMPUTE=1 OV2_MIDTRAIN_N_SAMPLES=15360   (selective; MARGINAL fit,
#                 predicted reserved 137-177 GiB — abort if max_reserved > 150)
#   ab-base-vis : TP=2 OV2_VISION_RECOMPUTE=1 OV2_MIDTRAIN_N_SAMPLES=15360
# The verdict file prints the knob set alongside p50 iter time / tokens/s / memory / phase split,
# so two verdict files ARE the A/B table.
export RECIPE="ov2_qwen35_35b_a3b_midtrain"
export SAVE="$SAVE_DIR" INIT_CKPT
export OV2_DIST_TIMEOUT_MIN="${OV2_DIST_TIMEOUT_MIN:-60}"   # smoke fails fast, not 300min

# Muon ON — the s1.5 line's required optimizer (AIAK parity; 30B midtrain+Muon+EP8 has 11.5k+
# production iterations, 30B stage3 runs TP4+Muon, and qwen3.5 s1.5 has 800+ clean iterations).
# muon_split_qkv=false is required for the trainable vision fused-QKV layout.
# OV2_MIDTRAIN_MUON=0 falls back to the recipe's AdamW auto-route (memory A/B lever).
export OV2_MIDTRAIN_MUON="${OV2_MIDTRAIN_MUON:-1}"
if [[ "$OV2_MIDTRAIN_MUON" == "1" ]]; then
  export EXTRA_ARGS="${EXTRA_ARGS:-} optimizer.muon_split_qkv=false"
fi
# Length-aligned batching (default 16 = 4 steps' worth of bins per rank): the
# counter-measure to the measured EP per-layer straggler on the 30B line. The
# seed85m bins share a nominal pack length but vary in fill, so alignment
# still matters. Set 0 to A/B against the unsorted path.
export OV2_LENGTH_SORT_WINDOW="${OV2_LENGTH_SORT_WINDOW:-16}"
# HARNESS-ONLY WORKAROUND (2026-09-04): with the production sort key (tokens) the harness's bin
# sequence hits a deterministic device-side assert (fetch_and_cast, rank 10, before forward #8 —
# ab-base twice, on two racks), while the SAME 16 bins in `patches` order run clean and at identical
# throughput (ab-sortkey-a6: last-3 iters 63-65 s = production). So A/B arms default to `patches` here
# so that every arm gets past iteration 1; comparability is unaffected (measured 0% effect). Production
# keeps `tokens`. Revert to tokens once ab-dbg (CUDA_LAUNCH_BLOCKING=1) has named the faulty op.
export OV2_LENGTH_SORT_KEY="${OV2_LENGTH_SORT_KEY:-patches}"

# Peak-memory sampler for this pod (dies with the script).
( _peak=0
  while sleep 5; do
    _m="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)"
    [[ "$_m" =~ ^[0-9]+$ ]] && (( _m > _peak )) && { _peak=$_m; echo "$_peak" >"$LOG.peak"; }
  done ) &
_mon=$!

_knobs="tp=$TP gbs=$OV2_MIDTRAIN_GBS accel=$ACCEL recompute_full=$OV2_RECOMPUTE_FULL recompute_moe=$OV2_RECOMPUTE_MOE vision_recompute=$OV2_VISION_RECOMPUTE alloc=$PYTORCH_CUDA_ALLOC_CONF sort_window=$OV2_LENGTH_SORT_WINDOW sort_key=${OV2_LENGTH_SORT_KEY:-tokens} feat_check_every=${OV2_FEAT_CHECK_EVERY:-1} permute_fusion=${OV2_MOE_PERMUTE_FUSION:-0} muon=$OV2_MIDTRAIN_MUON"
echo "[qwen35-s15-smoke] launch: $_knobs n_samples=$OV2_MIDTRAIN_N_SAMPLES init=$INIT_CKPT" | tee -a "$LOG"
# Which fla will the run execute? Print version+path into the persisted log — the NaN case turned
# on exactly this question (image fla 0.4.2 vs the qwen35-fla image's isolated /opt/ov2-fla), and
# the training logs otherwise never say. PYTHONPATH here matches what the base launcher composes
# minus repo paths, which do not carry fla.
python3 -c "import fla; print('[qwen35-s15-smoke] fla', getattr(fla,'__version__','?'), fla.__file__)" 2>&1 | tee -a "$LOG" || true
set +e
bash "$_SM_BASE" 2>&1 | tee -a "$LOG"
_rc=${PIPESTATUS[0]}
set -e
kill "$_mon" 2>/dev/null || true; wait "$_mon" 2>/dev/null || true
_peak="$(cat "$LOG.peak" 2>/dev/null || echo '?')"
echo "[qwen35-s15-smoke] rc=$_rc pod_peak_mem_mib=$_peak" | tee -a "$LOG"

# ── verdict: written by the pod whose log carries iteration lines ────────────
python3 - "$LOG" "$RESULT" "$_rc" "$_peak" "$TP" "$OV2_MIDTRAIN_GBS" \
          "$OV2_MIDTRAIN_MUON" "$OV2_LENGTH_SORT_WINDOW" "$_knobs" <<'PYEOF'
import os
import re
import statistics as st
import sys

log, result, rc, peak, tp, gbs, muon, sortw, knobs = sys.argv[1:10]
text = open(log, errors="replace").read()
sort_on = "length-sorted batching ON" in text
WARM = int(os.environ.get("OV2_SMOKE_WARM", "10") or 10)  # iteration time keeps easing for tens of iters
its = [float(x) for x in re.findall(r"elapsed time per iteration \(ms\): ([\d.]+)", text)][WARM:]
tps = [float(x) for x in re.findall(r"tokens/s/GPU: ([\d.]+)", text)][WARM:]
tf = [float(x) for x in re.findall(r"TFLOP/s/GPU\)?: ([\d.]+)", text)][WARM:]
fb = [(float(a), float(b)) for a, b in re.findall(r"forward-backward[ .]*:? *\(([\d.]+), ([\d.]+)\)", text)][WARM:]
mem = [(float(a), float(r)) for a, r in re.findall(r"max_allocated=([\d.]+)G reserved=([\d.]+)G", text)]
ph = [(int(sh), float(ppt)) for sh, ppt in re.findall(r"prefix_share=(\d+)% .*?patches_per_token=([\d.]+)", text)]
skips = sum(1 for ln in text.splitlines() if "exceed seq_length" in ln or "Skipping this pack" in ln)
nans = len(re.findall(r"skipping batch|found NaN|nan detected", text, re.I))
if not its and rc == "0":
    sys.exit(0)  # healthy waiter pod (iteration lines print on the last rank only)

lines = [f"qwen35 s1.5 (seed85m@10192) smoke — {knobs} "
         f"(sort engaged: {'yes' if sort_on else 'NO — check recipe log'})"]
if its:
    s = sorted(its)
    q = lambda p: s[min(len(s) - 1, int(p * len(s)))]
    lines.append(f"iters (after dropping first {WARM}): n={len(its)} p50={q(0.5):.0f}ms p90={q(0.9):.0f}ms max={s[-1]:.0f}ms")
    if len(its) >= 20:
        lines.append(f"iters last-20 p50={st.median(its[-20:]):.0f}ms  (use this for close A/B calls)")
if tps:
    lines.append(f"tokens/s/GPU: p50={st.median(tps):.0f}  (baseline 2026-09-02 prod defaults ~1000; full-recompute ~620)")
if tf:
    lines.append(f"tflops/gpu (reported; omits the vision tower): p50={st.median(tf):.1f}")
if mem:
    lines.append(f"torch memory: max_allocated={max(a for a, _ in mem):.1f}G max_reserved={max(r for _, r in mem):.1f}G "
                 f"(card 189.5G, ~37G non-torch; reserved >150G = OOM territory)")
if ph:
    lines.append(f"phase split: prefix(vision+adapter) share p50={st.median(sh for sh, _ in ph):.0f}% "
                 f"patches_per_llm_token p50={st.median(p for _, p in ph):.1f}")
if fb:
    ratio = st.median(hi / lo for lo, hi in fb if lo > 0)
    lines.append(f"fb straggler spread (max/min across ranks) p50={ratio:.2f}")
lines.append(f"pod_peak_mem_mib={peak} (per-pod; grep pod_peak ~/train_logs/smoke_qwen35_s15_*.log)")
lines.append(f"dropped packs (seq_length exceeded): {skips}")
if skips:
    lines.append(f"SEQ VERDICT: {skips} packs skipped at seq 10192 — the Qwen2.5-VL-tokenizer packs "
                 "overflow under the 3.5 tokenizer: BIASED data loss. Decide: tolerate the rate, raise "
                 "OV2_SEQ_LEN slightly, or repack seed85m with the 3.5 tokenizer.")
else:
    lines.append("SEQ VERDICT: no skips in this sample — keep watching the counter over a longer run.")
lines.append(f"nan/skipped-batch lines: {nans}")
lines.append("VERDICT: " + ("PASS — config viable, ready for a scaled launch decision" if rc == "0" and its and not nans
              else f"FAIL — rc={rc}, inspect this log's tail and train_node*.log under the smoke SAVE dir"))
tmp = result + ".tmp"
with open(tmp, "w") as f:
    f.write("\n".join(lines) + "\n")
os.replace(tmp, result)
print("\n".join(lines))
PYEOF

# Every pod holds for the verdict so the writer is never reaped by the
# PyTorchJob master-exit teardown (the B15 lesson).
_dl=$(( $(date +%s) + 900 ))
while [[ ! -f "$RESULT" ]]; do
  (( $(date +%s) >= _dl )) && { echo "[qwen35-s15-smoke] no result after 15min — check per-pod logs"; break; }
  sleep 10
done
[[ -f "$RESULT" ]] && { echo "[qwen35-s15-smoke] ---- $RESULT ----"; cat "$RESULT"; }
# PyTorchJob takes the MASTER pod's exit code as the job verdict: the master exits with the
# real launcher rc so a failed smoke shows Failed in the UI (a blanket exit 0 masked the OOM
# run as Completed). Workers still exit 0 — they only hold for the verdict file.
if [[ "$(hostname)" == *-master-* ]]; then exit "${_rc:-0}"; fi
exit 0
