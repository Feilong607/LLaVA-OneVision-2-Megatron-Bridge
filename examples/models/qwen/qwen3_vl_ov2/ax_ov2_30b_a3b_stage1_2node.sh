#!/usr/bin/env bash
# =============================================================================
# OV2-30B-A3B · Stage-1 alignment · 2 NODES (A100-22 + A100-26, 16 GPU, EP8/DP2)
# Thin convenience wrapper over ax_ov2_30b_a3b_stage1.sh that defaults LIST_IP to the 22+26 pair.
# Run the SAME command on BOTH nodes — node_rank is auto-detected from each host's IP.
#   on A100-22:  bash ax_ov2_30b_a3b_stage1_2node.sh
#   on A100-26:  bash ax_ov2_30b_a3b_stage1_2node.sh
# Override the node pair with LIST_IP="ip0 ip1"; all other env (ITERS, SAVE, SAVE_EVERY, ...) passes
# through to the base script. Clean per-node logs: node0->train_node0.log, node1->train_node1.log
# (the readable iteration|lm loss lines are on the LAST node, train_node1.log).
# =============================================================================
export LIST_IP="${LIST_IP:-172.16.5.22 172.16.5.26}"
exec "$(dirname "$(readlink -f "$0")")/ax_ov2_30b_a3b_stage1.sh" "$@"
