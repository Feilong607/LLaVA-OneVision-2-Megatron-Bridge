#!/usr/bin/env bash
# =============================================================================
# STAGE-3 32-GPU PRODUCTION wrapper (8 nodes x 4 GPU) - launcher + persisted
# per-hostname tee (gang teardown deletes pods and their UI logs; the WekaFS
# log is what post-mortems and the worker-6 iteration lines are read from).
#
# Every hyperparameter is the launcher's baked default: TP4/DP8/EP8, HybridEP,
# GBS 32, 617482 samples -> 19297 iters, Muon lr 2e-5 -> 1e-6, save every 2000
# + auto final save at 19297. INIT = trained stage-2 ckpt root (tracker picks
# the latest); SAVE = ~/ckpts_video_sft/ov2_30b_a3b_stage3_img10_gbs32, so any
# restart resumes from the newest stage-3 save automatically.
#
# First-hour watch items: iters 1-3 grad-norm/NaN (Muon on MoE midtrain is
# unvalidated in this fork - fall back to OV2_MIDTRAIN_MUON=0 + fresh SAVE if
# it hangs or NaNs), memory peak at ~iter 50 (<170GB before enabling the
# OV2_RECOMPUTE_MOE=0 speed lever at a save point), exceed-seq count == 0.
# =============================================================================
set -euo pipefail
export HOME="${HOME:-/home/ftan0055}"
mkdir -p "$HOME/train_logs"
bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ax_ov2_30b_a3b_gb200_stage3.sh" 2>&1 \
  | tee -a "$HOME/train_logs/stage3_32_$(hostname).log"
