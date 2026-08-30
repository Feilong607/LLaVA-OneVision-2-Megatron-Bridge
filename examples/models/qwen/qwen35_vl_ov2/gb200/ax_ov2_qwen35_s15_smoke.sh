#!/usr/bin/env bash
# =============================================================================
# Qwen3.5-35B-A3B stage-1.5 (seed85m mid-train @ seq 10192) smoke — 32 GPU.
#
# ~20-iteration bring-up smoke for the qwen3.5 s1.5 line: full-model SFT
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
# (Workers=7): TP=2 -> DP=16. Muon (required for AIAK parity) does not fit at
# TP=1 on the 35B — measured ~188GB/192GB pod peak, OOM in the first triton
# autotune — and TP=2 also gives the s4 line its first TP>1 validation on the
# GDN hybrid. GBS=256 = the 30B same-stage production value (256 % 16 == 0;
# the bare default 16 fails the launcher's GBS%DP guard at this world size).
#
# Workload form: Distributed/PyTorch, image feilong-nemo, gb200-nvl72-nodes,
# Command bash (both sides), Args = this file's absolute path (both sides),
# env OV2_K8S_NAMESPACE=runai-mv0004 (both sides). Optional env: TP, GBS
# (OV2_MIDTRAIN_GBS), ACCEL, OV2_MIDTRAIN_MUON, OV2_RECOMPUTE_FULL,
# OV2_LENGTH_SORT_WINDOW.
# =============================================================================
set -euo pipefail

_SM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_SM_BASE="$_SM_DIR/ax_ov2_qwen35_35b_a3b_gb200.sh"
_SM_TAG="$(hostname | sed -E 's/-(master|worker)-[0-9]+$//')"
_SM_ROOT="$HOME/ckpts_video_sft/_smoke_qwen35_s15"
SAVE_DIR="$_SM_ROOT/$_SM_TAG"
RESULT="$HOME/train_logs/smoke_qwen35_s15_result_${_SM_TAG}.txt"
LOG="$HOME/train_logs/smoke_qwen35_s15_${_SM_TAG}_$(hostname).log"
mkdir -p "$HOME/train_logs" "$SAVE_DIR"

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
# TP=2 -> DP=16 (32 GPU). TP=1 does NOT fit Muon on the 35B: measured ~188GB/192GB pod peak,
# OOM in the first triton autotune (see the Muon note below). TP=2 halves weights/grads/Muon
# states (~105GB -> ~52GB static) and doubles as the first TP>1 validation on the GDN hybrid
# (the s4 line plans TP4). GBS 256 % DP16 == 0; DP16 % EP8 == 0.
export TP="${TP:-2}"
export OV2_MIDTRAIN_GBS="${OV2_MIDTRAIN_GBS:-${GBS:-256}}"   # 30B same-stage production GBS
export OV2_MIDTRAIN_N_SAMPLES="${OV2_MIDTRAIN_N_SAMPLES:-$(( OV2_MIDTRAIN_GBS * 20 ))}"  # -> 20 iters
export SAVE_EVERY="${SAVE_EVERY:-100000}"      # no interval saves; the end-of-run save lands in scratch
export ACCEL="${ACCEL:-0}"                     # bf16 + alltoall (HybridEP/MXFP8 unvalidated on GDN+MTP)
export OV2_RECOMPUTE_FULL="${OV2_RECOMPUTE_FULL:-1}"    # fit first; speed levers after the memory verdict
export RECIPE="ov2_qwen35_35b_a3b_midtrain"
export SAVE="$SAVE_DIR" INIT_CKPT
export OV2_DIST_TIMEOUT_MIN="${OV2_DIST_TIMEOUT_MIN:-60}"   # smoke fails fast, not 300min

# Muon ON — the s1.5 line's required optimizer (AIAK parity; 30B midtrain+Muon+EP8 has 11.5k+
# production iterations, and 30B stage3 runs TP4+Muon). Muon's layer-wise full states are why
# the default TP above is 2, not 1: at TP=1 the measured pod peak hit ~188GB of 192GB and the
# first fla/GDN triton autotune benchmark died with "Triton Error [CUDA]: out of memory"
# (2026-08-30 smoke). muon_split_qkv=false is required for the trainable vision fused-QKV
# layout. OV2_MIDTRAIN_MUON=0 falls back to the recipe's AdamW auto-route (memory A/B lever).
export OV2_MIDTRAIN_MUON="${OV2_MIDTRAIN_MUON:-1}"
if [[ "$OV2_MIDTRAIN_MUON" == "1" ]]; then
  export EXTRA_ARGS="${EXTRA_ARGS:-} optimizer.muon_split_qkv=false"
fi
# Length-aligned batching (default 16 = 4 steps' worth of bins per rank): the
# counter-measure to the measured EP per-layer straggler on the 30B line. The
# seed85m bins share a nominal pack length but vary in fill, so alignment
# still matters. Set 0 to A/B against the unsorted path.
export OV2_LENGTH_SORT_WINDOW="${OV2_LENGTH_SORT_WINDOW:-16}"

# Peak-memory sampler for this pod (dies with the script).
( _peak=0
  while sleep 5; do
    _m="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)"
    [[ "$_m" =~ ^[0-9]+$ ]] && (( _m > _peak )) && { _peak=$_m; echo "$_peak" >"$LOG.peak"; }
  done ) &
_mon=$!

echo "[qwen35-s15-smoke] launch: tp=$TP gbs=$OV2_MIDTRAIN_GBS n_samples=$OV2_MIDTRAIN_N_SAMPLES accel=$ACCEL recompute_full=$OV2_RECOMPUTE_FULL muon=$OV2_MIDTRAIN_MUON sort_window=$OV2_LENGTH_SORT_WINDOW init=$INIT_CKPT" | tee -a "$LOG"
set +e
bash "$_SM_BASE" 2>&1 | tee -a "$LOG"
_rc=${PIPESTATUS[0]}
set -e
kill "$_mon" 2>/dev/null || true; wait "$_mon" 2>/dev/null || true
_peak="$(cat "$LOG.peak" 2>/dev/null || echo '?')"
echo "[qwen35-s15-smoke] rc=$_rc pod_peak_mem_mib=$_peak" | tee -a "$LOG"

# ── verdict: written by the pod whose log carries iteration lines ────────────
python3 - "$LOG" "$RESULT" "$_rc" "$_peak" "$TP" "$OV2_MIDTRAIN_GBS" \
          "$OV2_MIDTRAIN_MUON" "$OV2_LENGTH_SORT_WINDOW" <<'PYEOF'
import os
import re
import statistics as st
import sys

log, result, rc, peak, tp, gbs, muon, sortw = sys.argv[1:9]
text = open(log, errors="replace").read()
sort_on = "length-sorted batching ON" in text
its = [float(x) for x in re.findall(r"elapsed time per iteration \(ms\): ([\d.]+)", text)][3:]
tf = [float(x) for x in re.findall(r"TFLOP/s/GPU\)?: ([\d.]+)", text)][3:]
fb = [(float(a), float(b)) for a, b in re.findall(r"forward-backward[ .]*:? *\(([\d.]+), ([\d.]+)\)", text)][3:]
skips = sum(1 for ln in text.splitlines() if "exceed seq_length" in ln or "Skipping this pack" in ln)
nans = len(re.findall(r"skipping batch|found NaN|nan detected", text, re.I))
if not its and rc == "0":
    sys.exit(0)  # healthy waiter pod (iteration lines print on the last rank only)

lines = [f"qwen35 s1.5 (seed85m@10192) smoke — TP={tp} gbs={gbs} muon={muon} "
         f"sort_window={sortw} (engaged: {'yes' if sort_on else 'NO — check recipe log'})"]
if its:
    s = sorted(its)
    q = lambda p: s[min(len(s) - 1, int(p * len(s)))]
    lines.append(f"iters: n={len(its)} p50={q(0.5):.0f}ms p90={q(0.9):.0f}ms max={s[-1]:.0f}ms")
if tf:
    lines.append(f"tflops/gpu: p50={st.median(tf):.1f}")
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
