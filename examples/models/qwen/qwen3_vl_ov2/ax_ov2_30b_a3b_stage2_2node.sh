#!/usr/bin/env bash
# =============================================================================
# OV2-30B-A3B · Stage-2 SFT (vit+adapter, Muon) · 2 NODES (A100-22 + A100-26, 16 GPU, EP8/DP2)
# Thin convenience wrapper over ax_ov2_30b_a3b_stage2.sh that defaults LIST_IP to the 22+26 pair.
# Run the SAME command on BOTH nodes — node_rank auto-detected. Chains from a trained stage-1 via
# INIT_CKPT (defaults to ckpts_video_sft/ov2_30b_a3b_stage1). Override LIST_IP / INIT_CKPT / ITERS etc.
#   on A100-22:  bash ax_ov2_30b_a3b_stage2_2node.sh
#   on A100-26:  bash ax_ov2_30b_a3b_stage2_2node.sh
# Readable loss on the LAST node -> train_node1.log.
# =============================================================================
export LIST_IP="${LIST_IP:-172.16.5.22 172.16.5.26}"
exec "$(dirname "$(readlink -f "$0")")/ax_ov2_30b_a3b_stage2.sh" "$@"
