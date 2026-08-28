#!/usr/bin/env bash
# =============================================================================
# Qwen3.5-35B-A3B merged-stage (stage2+3 in one) 64k smoke — 32 or 64 GPUs.
#
# Bring-up smoke for the MAVERIC stage-4 line: full-model SFT (midtrain recipe,
# LLM+vision+adapter all trainable) on the /datasets/feilong-stage4-datasets
# pool, initialized from the staged stage-2 checkpoint. ~20 iterations, scratch
# SAVE, and a written verdict with the four numbers a real launch needs:
# iteration time, dropped-pack rate at the chosen SEQ_LEN, DP straggler spread,
# and per-pod peak memory.
#
# Topology (NPROC=4/pod; EP=8 fixed; guards: WORLD%TP==0, DP%8==0, GBS%DP==0):
#   32 GPU = 8 pods  (Workers=7):  TP=4->DP=8 (default) | TP=2->DP=16 | TP=1->DP=32
#   64 GPU = 16 pods (Workers=15): TP=4->DP=16          | TP=2->DP=32 | TP=1 needs GBS>=64
#
# Defaults deliberately conservative for the GDN+MTP hybrid:
#   ACCEL=0 (bf16+alltoall; HybridEP/MXFP8 are UNVALIDATED on this backbone),
#   OV2_RECOMPUTE_FULL=1 (fit first; flip to 0 [+OV2_RECOMPUTE_MOE] in later
#   speed smokes once memory headroom is known), AdamW via the recipe's
#   midtrain auto-route (do NOT set OV2_MIDTRAIN_MUON=1 — Muon deadlocks EP
#   backward on trainable 256-expert MoE), SEQ_LEN=65536 per the merged-stage
#   plan — the verdict counts SkipSample'd packs and says whether 65536 holds
#   or the pool needs 73728 like the 30B stage-3 packs did.
#
# Known trap encoded here (BRINGUP §1.4): the staged stage-2 ckpt's tracker
# says iter 6094 but only iter_0006000/ exists — INIT must point AT the iter
# subdirectory, never at the checkpoint root.
#
# Blend: SMOKE-ONLY equal-weight yaml auto-generated over every
# <pack>/<part>/webdataset under the stage-4 mount (95 expected). Real blend
# weights are a recipe decision — do not reuse the generated yaml for a run.
#
# Workload form: Distributed/PyTorch, image feilong-nemo, gb200-nvl72-nodes,
# Command bash (both sides), Args = this file's absolute path (both sides),
# env OV2_K8S_NAMESPACE=runai-mv0004 (both sides). Optional env: TP, SEQ_LEN
# (OV2_SEQ_LEN), GBS (OV2_MIDTRAIN_GBS), ACCEL, OV2_RECOMPUTE_FULL.
# =============================================================================
set -euo pipefail

_SM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_SM_BASE="$_SM_DIR/ax_ov2_qwen35_35b_a3b_gb200.sh"
_SM_POOL="${OV2_STAGE4_POOL:-/datasets/feilong-stage4-datasets}"
_SM_TAG="$(hostname | sed -E 's/-(master|worker)-[0-9]+$//')"
_SM_ROOT="$HOME/ckpts_video_sft/_smoke_qwen35_merged64k"
SAVE_DIR="$_SM_ROOT/$_SM_TAG"
RESULT="$HOME/train_logs/smoke_qwen35_merged64k_result_${_SM_TAG}.txt"
LOG="$HOME/train_logs/smoke_qwen35_merged64k_${_SM_TAG}_$(hostname).log"
mkdir -p "$HOME/train_logs"

# ── preflight: every asset the recipe will touch, checked before GPUs spin ───
_die() { echo "[qwen35-smoke] FATAL: $*" | tee -a "$LOG" >&2; exit 1; }

INIT_CKPT="${INIT_CKPT:-$_SM_POOL/35b/ov2_qwen35_35b_a3b_p16m33_stage2_muon_v2/iter_0006000}"
[[ "$(basename "$INIT_CKPT")" == iter_* ]] || _die "INIT_CKPT must point AT an iter_* subdir (tracker=6094 trap, BRINGUP §1.4): $INIT_CKPT"
[[ -f "$INIT_CKPT/.metadata" || -f "$INIT_CKPT/metadata.json" ]] || _die "INIT_CKPT has no torch_dist metadata: $INIT_CKPT"

# LLM HF (config source) and processor: prefer the stage-4 mount, fall back to
# the recipe's legacy default root. FATAL with candidates listed — these two
# are the known staging gap on MAVERIC (§1.4 lists only base + stage-2 ckpt).
_pick() {  # _pick VAR file candidates...
  local _var="$1" _file="$2" _c; shift 2
  for _c in "$@"; do
    [[ -f "$_c/$_file" ]] && { eval "$_var=\"\$_c\""; return 0; }
  done
  _die "$_var not found ($_file missing in all of: $*). Stage the asset or set $_var explicitly."
}
[[ -n "${OV2_LLM_HF_QWEN35:-}" ]] || _pick OV2_LLM_HF_QWEN35 config.json \
  "$_SM_POOL/35b/Qwen3.5-35B-A3B-text" "$_SM_POOL/35b/Qwen3.5-35B-A3B" \
  "/datasets/llava/11May/Qwen3.5-35B-A3B-text"
[[ -n "${OV2_HF_PROC_QWEN35_P16M33:-}" ]] || _pick OV2_HF_PROC_QWEN35_P16M33 preprocessor_config.json \
  "$_SM_POOL/35b/llava_onevision2_qwen35_35b_a3b_p16_m33/auto_model" \
  "$_SM_POOL/35b/auto_model" \
  "/datasets/llava/11May/llava_onevision2/llava_onevision2_qwen35_35b_a3b_p16_m33/auto_model"
export OV2_LLM_HF_QWEN35 OV2_HF_PROC_QWEN35_P16M33
[[ "$OV2_LLM_HF_QWEN35" == *"-text" ]] || echo "[qwen35-smoke] WARN: llm_hf=$OV2_LLM_HF_QWEN35 is not a '-text' extract; if the build dies routing the VLM config, run tools/extract_qwen35_text.py --weights first." | tee -a "$LOG" >&2

# ── SMOKE-ONLY blend: every part of the stage-4 pool, equal weight ───────────
mkdir -p "$SAVE_DIR"
_SM_YAML="$SAVE_DIR/smoke_blend_equal.yaml"
{
  echo "# SMOKE-ONLY equal-weight blend over $_SM_POOL — real weights are a recipe decision."
  echo "__module__: megatron.energon"
  echo "__class__: Metadataset"
  echo "splits:"
  echo "  train:"
  echo "    datasets:"
  find "$_SM_POOL" -mindepth 3 -maxdepth 3 -type d -name webdataset 2>/dev/null | sort | while read -r d; do
    printf '      - weight: 1\n        path: %s\n        subflavors: {augmentation: false}\n' "$d"
  done
} > "$_SM_YAML"
_n_ds="$(grep -c "path:" "$_SM_YAML" || true)"
(( _n_ds > 0 )) || _die "no <pack>/<part>/webdataset dirs found under $_SM_POOL"
echo "[qwen35-smoke] blend: $_n_ds datasets (expected 95) -> $_SM_YAML" | tee -a "$LOG"

# ── scale + knobs (see header) ────────────────────────────────────────────────
export TP="${TP:-4}"
export OV2_SEQ_LEN="${OV2_SEQ_LEN:-${SEQ_LEN:-65536}}"
export OV2_MIDTRAIN_GBS="${OV2_MIDTRAIN_GBS:-${GBS:-32}}"
export OV2_MIDTRAIN_N_SAMPLES="${OV2_MIDTRAIN_N_SAMPLES:-$(( OV2_MIDTRAIN_GBS * 20 ))}"  # -> 20 iters
export SAVE_EVERY="${SAVE_EVERY:-100000}"      # no interval saves; post-loop final save lands in scratch
export ACCEL="${ACCEL:-0}"
export OV2_RECOMPUTE_FULL="${OV2_RECOMPUTE_FULL:-1}"
export RECIPE="ov2_qwen35_35b_a3b_midtrain"
export DATA_PATH="$_SM_YAML" SAVE="$SAVE_DIR" INIT_CKPT
export OV2_DIST_TIMEOUT_MIN="${OV2_DIST_TIMEOUT_MIN:-60}"   # smoke fails fast, not 300min

# Peak-memory sampler for this pod (dies with the script).
( _peak=0
  while sleep 5; do
    _m="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)"
    [[ "$_m" =~ ^[0-9]+$ ]] && (( _m > _peak )) && { _peak=$_m; echo "$_peak" >"$LOG.peak"; }
  done ) &
_mon=$!

echo "[qwen35-smoke] launch: tp=$TP seq=$OV2_SEQ_LEN gbs=$OV2_MIDTRAIN_GBS accel=$ACCEL recompute_full=$OV2_RECOMPUTE_FULL init=$INIT_CKPT" | tee -a "$LOG"
set +e
bash "$_SM_BASE" 2>&1 | tee -a "$LOG"
_rc=${PIPESTATUS[0]}
set -e
kill "$_mon" 2>/dev/null || true; wait "$_mon" 2>/dev/null || true
_peak="$(cat "$LOG.peak" 2>/dev/null || echo '?')"
echo "[qwen35-smoke] rc=$_rc pod_peak_mem_mib=$_peak" | tee -a "$LOG"

# ── verdict: written by the pod whose log carries iteration lines ────────────
python3 - "$LOG" "$RESULT" "$_rc" "$_peak" "$TP" "$OV2_SEQ_LEN" "$OV2_MIDTRAIN_GBS" "$_n_ds" <<'PYEOF'
import os
import re
import statistics as st
import sys

log, result, rc, peak, tp, seq, gbs, n_ds = sys.argv[1:9]
text = open(log, errors="replace").read()
its = [float(x) for x in re.findall(r"elapsed time per iteration \(ms\): ([\d.]+)", text)][3:]
tf = [float(x) for x in re.findall(r"TFLOP/s/GPU\)?: ([\d.]+)", text)][3:]
fb = [(float(a), float(b)) for a, b in re.findall(r"forward-backward[ .]*:? *\(([\d.]+), ([\d.]+)\)", text)][3:]
skips = sum(1 for ln in text.splitlines() if "exceed seq_length" in ln or "Skipping this pack" in ln)
nans = len(re.findall(r"skipping batch|found NaN|nan detected", text, re.I))
if not its and rc == "0":
    sys.exit(0)  # healthy waiter pod (iteration lines print on the last rank only)

lines = [f"qwen35 merged-stage 64k smoke — TP={tp} seq={seq} gbs={gbs} datasets={n_ds}"]
if its:
    s = sorted(its)
    q = lambda p: s[min(len(s) - 1, int(p * len(s)))]
    lines.append(f"iters: n={len(its)} p50={q(0.5):.0f}ms p90={q(0.9):.0f}ms max={s[-1]:.0f}ms")
if tf:
    lines.append(f"tflops/gpu: p50={st.median(tf):.1f}")
if fb:
    ratio = st.median(hi / lo for lo, hi in fb if lo > 0)
    lines.append(f"fb straggler spread (max/min across ranks) p50={ratio:.2f}")
lines.append(f"pod_peak_mem_mib={peak} (per-pod; grep pod_peak ~/train_logs/smoke_qwen35_merged64k_*.log)")
lines.append(f"dropped packs (seq_length exceeded): {skips}")
if skips:
    lines.append(f"SEQ VERDICT: {skips} packs skipped at seq={seq} — biased data loss; rerun with SEQ_LEN=73728 (30B stage-3 precedent) or repack.")
else:
    lines.append(f"SEQ VERDICT: no skips at seq={seq} in this sample — keep watching the counter over a longer run.")
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
  (( $(date +%s) >= _dl )) && { echo "[qwen35-smoke] no result after 15min — check per-pod logs"; break; }
  sleep 10
done
[[ -f "$RESULT" ]] && { echo "[qwen35-smoke] ---- $RESULT ----"; cat "$RESULT"; }
exit 0
