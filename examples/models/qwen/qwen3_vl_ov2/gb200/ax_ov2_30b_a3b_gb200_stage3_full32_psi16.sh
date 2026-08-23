#!/usr/bin/env bash
# =============================================================================
# STAGE-3 32-GPU A/B wrapper: OV2_PARALLEL_SHARD_ITERS=16 (launcher default is
# 1 — "energon default 16 chokes WekaFS"). Purpose: retrain from the same INIT
# with ONLY this knob changed and compare the loss curve + iteration time
# against the psi=1 production run. parallel_shard_iters changes how energon
# parallelizes shard reading, which can reorder the sample stream — the loss
# trajectory difference, if any, is the thing under test.
#
# Everything else inherits the launcher's baked defaults (TP4/DP8/EP8, GBS 32,
# seq 73728, Muon, 617482 samples -> 19297 iters, INIT = trained stage-2 root).
# SAVE is a SEPARATE directory so the two runs keep independent checkpoints
# and restart semantics; the tee log is likewise namespaced (psi16).
#
# Watch items vs the psi=1 twin: elapsed-time-per-iteration spikes / dataloader
# stalls (the WekaFS choke the default guards against), lm-loss overlay over
# the first few thousand iters, exceed-seq count must stay 0.
# =============================================================================
set -euo pipefail
export HOME="${HOME:-/home/ftan0055}"
# This launcher is the psi=16 A/B lane. Do not inherit a stale psi=1 value
# from a copied Run:ai workload, which would contaminate the comparison.
export OV2_PARALLEL_SHARD_ITERS=16
export SAVE="${SAVE:-$HOME/ckpts_video_sft/ov2_30b_a3b_stage3_img10_gbs32_psi16}"
mkdir -p "$HOME/train_logs"
bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ax_ov2_30b_a3b_gb200_stage3.sh" 2>&1 \
  | tee -a "$HOME/train_logs/stage3_psi16_32_$(hostname).log"
