#!/usr/bin/env bash
# =============================================================================
# OV2-30B-A3B · MID-TRAIN (full model) · 2 NODES (A100-22 + A100-26, 16 GPU, EP8/DP2)
# Thin wrapper over ax_ov2_30b_a3b_midtrain.sh defaulting LIST_IP to the 22+26 pair.
# Run the SAME command on BOTH nodes (node_rank auto-detected). Chains from a trained stage-2 via
# INIT_CKPT (default ckpts_video_sft/ov2_30b_a3b_stage2). NOTE: full-model 30B is memory-heavy — if
# EP8/2-node OOMs, raise nodes or TP. Mid-train auto-uses AdamW for the MoE backbone (no env needed).
# =============================================================================
export LIST_IP="${LIST_IP:-172.16.5.22 172.16.5.26}"
exec "$(dirname "$(readlink -f "$0")")/ax_ov2_30b_a3b_midtrain.sh" "$@"
