#!/usr/bin/env bash
# =============================================================================
# STAGE-3 16-GPU SMOKE (4 nodes x 4 GPU) - thin wrapper over the stage-3
# launcher. Purpose: prove the stage-3 assets run end-to-end BEFORE the 32-GPU
# production submit, using only 16 free GPUs.
#
# Why TP=2 (the only 16-GPU shape):
#   the launcher enforces DP >= 8 && DP % 8 == 0 (EP=8 is fixed by the ckpt
#   sharding). 16 GPU @ TP4 -> DP=4 FATAL; @ TP2 -> DP=8 OK. At TP2 the
#   HybridEP guard auto-falls back to AllToAll (ceil(73728/2)=36864 > 21824
#   tokens/rank) - the WARN it prints is EXPECTED here.
# Why AdamW (OV2_MIDTRAIN_MUON=0):
#   TP2 doubles per-GPU weight/optimizer shards; TP2+Muon OOMed in the 48-GPU
#   study. AdamW is the validated midtrain path (the packed64k 19k-iter run).
#
# What this smoke DOES validate:
#   stage3_mix_img10.yaml (paths/weights readable), packed-seq admission at
#   OV2_SEQ_LEN=73728 (grep 'exceed seq_length' MUST be 0), INIT weights-only
#   load from the trained stage-2 ckpt (torch_dist TP re-shard), EP8 dispatch,
#   full-model-unfrozen fwd/bwd, finite loss, one checkpoint save.
# What it does NOT validate (covered by the first ~50 iters of the real run):
#   TP4 memory profile under Muon+midtrain (never run in this fork), the
#   Muon-on-MoE-midtrain path itself (a2a risk, ov2.py:509-512), HybridEP.
#
# PASS checklist (worker logs land in the SAVE dir's tee or stdout):
#   [1] banner shows world=16 dp=8 tp=2 muon=0 iters=20, plus the HybridEP
#       fallback WARN;   [2] 'exceed seq_length' count == 0;
#   [3] loss finite, no NaN, grad-norm sane across 20 iters;
#   [4] final save at iter 20 completes.
# If INIT load dies on expert-shard shapes: TP2-loading-a-TP4-save needs a
# conversion - abandon the 16-GPU smoke and smoke directly at 32 GPU instead.
# SAVE is a throwaway - delete it after the checklist passes.
# =============================================================================
set -euo pipefail
export HOME="${HOME:-/home/ftan0055}"

export TP="${TP:-2}"
export OV2_MIDTRAIN_MUON="${OV2_MIDTRAIN_MUON:-0}"
export OV2_TOTAL_SAMPLES="${OV2_TOTAL_SAMPLES:-640}"   # -> ITERS=20 @ GBS32
export OV2_EPOCHS="${OV2_EPOCHS:-1}"
export SAVE_EVERY="${SAVE_EVERY:-100000}"              # no interval saves; auto final save at 20
export SAVE="${SAVE:-$HOME/ckpts_video_sft/_smoke_stage3_16gpu}"

exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ax_ov2_30b_a3b_gb200_stage3.sh"
