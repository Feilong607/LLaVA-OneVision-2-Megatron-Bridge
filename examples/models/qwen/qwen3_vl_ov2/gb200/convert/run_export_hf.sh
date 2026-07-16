#!/usr/bin/env bash
# =============================================================================
# Export a TRAINED OV2-30B-A3B GB200 checkpoint (EP8/TP1/PP1) -> HuggingFace VLM, via the registered bridge.
#
# WHY THIS WRAPPER: convert.sh's gb200 CKPTA/CFG DEFAULTS point at the /datasets *pretrained skeleton*, NOT
# your trained run. Running `convert.sh export` without overriding CKPTA silently converts the WRONG (and,
# on this cluster, bad-format) checkpoint. This wrapper sets CKPTA to YOUR trained ckpt + a valid dispatch
# CFG, then runs convert.sh 'export' -- which loads the mcore ckpt at EP8 through the bridge, writes the HF
# VLM, and runs do_fixup (repairs use_patch_position_encoding + preprocessor patch/merge in the HF skeleton
# -- the two SILENT bugs that corrupt image features if skipped). Muon vs AdamW is irrelevant to export
# (only model weights are read, never optimizer state).
#
# Paths default under $HOME (per-machine; no username committed). Override any via env.
#
# EP8 export needs world==8. On GB200 (4 GPU/node) run the SAME command on BOTH nodes:
#   NPROC=4 LIST_IP="<ip0> <ip1>" bash .../convert/run_export_hf.sh [CKPT_DIR]
# Single 4-GPU node (EP4, UNVALIDATED -> always VERIFY=1 afterwards): OV2_EP=4 NPROC=4 bash .../run_export_hf.sh
# Add allclose roundtrip verification (export + HF->mcore->HF allclose; needs the 8 GPUs again): VERIFY=1
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The TRAINED checkpoint to export: parent dir holding iter_* + latest_checkpointed_iteration.txt.
# Arg 1 wins, else $CKPT_DIR, else the training launcher's $HOME SAVE convention.
CKPT_DIR="${1:-${CKPT_DIR:-$HOME/ckpts_video_sft/ov2_30b_a3b_gb200}}"
[[ -f "$CKPT_DIR/latest_checkpointed_iteration.txt" ]] || {
  echo "FATAL: '$CKPT_DIR' is not a trained ckpt root (no latest_checkpointed_iteration.txt)." >&2
  echo "  Pass it explicitly:  bash run_export_hf.sh /path/to/your/ckpt_dir   (or set CKPT_DIR=)" >&2
  exit 1; }

# Dispatch config skeleton (HF auto_model with architectures for AutoBridge). Prefer the $HOME copy, else
# the /datasets mount. convert.sh's ensure_dispatch_cfg sets architectures if the skeleton ships null.
_cfg="$HOME/llava-ov2-30b-a3b-m9lvdn/auto_model"
[[ -d "$_cfg" ]] || _cfg="/datasets/llava-ov2-30b-a3b-m9lvdn/auto_model"

export CKPTA="${CKPTA:-$CKPT_DIR}"                                        # <-- THE key override: your trained ckpt
export CFG="${CFG:-$_cfg}"
export HF_OUT="${HF_OUT:-$HOME/ov2_hf_export/$(basename "$CKPT_DIR")_hf}"  # off-repo HF output (30-58G)
export WORK="${WORK:-$HOME/_ov2_convert}"                                 # off-repo scratch (cfg_dispatch)

echo "[run-export-hf] trained ckpt : $CKPTA  (latest iter $(cat "$CKPT_DIR/latest_checkpointed_iteration.txt" 2>/dev/null || echo '?'))"
echo "[run-export-hf] dispatch cfg : $CFG"
echo "[run-export-hf] HF output    : $HF_OUT"
echo "[run-export-hf] mode         : $([[ "${VERIFY:-0}" == 1 ]] && echo '30b (export + roundtrip allclose)' || echo 'export')"

exec bash "$HERE/convert.sh" "$([[ "${VERIFY:-0}" == 1 ]] && echo 30b || echo export)"
