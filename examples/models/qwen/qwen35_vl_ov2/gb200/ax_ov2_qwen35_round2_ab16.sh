#!/usr/bin/env bash
# =============================================================================
# Qwen3.5-35B-A3B merged video stage — 16-GPU ROUND-2 A/B (A -> S -> H in ONE workload, no variables).
#
# Workload form: Distributed/PyTorch, qwen35-fla image, gb200-nvl72-nodes, **Workers=3 (4 pods x 4 GPU)**,
#   Command (both sides) = bash
#   Args    (both sides) = this file's absolute path
# Nothing else. Every constant below is baked; each one can still be overridden from the Args if a
# follow-up needs it, but the default run needs no Args beyond the path.
# Read-out (code-sync workspace): cat ~/train_logs/smoke_speed_ladder_<job>.txt
#
# Why this exists (BRINGUP 13.23). The 48-GPU production run q35-img38-a7-48gpu-psi16-5 is healthy but
# tight: each pod's GPU0/GPU3 carry ~53 GiB that NO process created (cumem ledger: net cuMemCreate
# outstanding 121.6 GiB = torch reserved 114.8 + nccl 4.99 + hybrid_ep 1.86), leaving r32 with 2 GiB of
# device memory free. 16 GPUs do NOT reproduce that level (unattr 0.45 GiB on the same per-rank work),
# so the ownership question belongs to a 48-GPU run. What 16 free GPUs CAN do is price and size the only
# decision that would follow from it:
#
#   A  TP4/ETP2 GBS48 full recompute, ACCEL=2 HybridEP   same-session baseline (step time, torch peak)
#   S  TP4/ETP2 GBS48 selective attn+moe, ACCEL=2        THE PRIZE: how much torch memory the production
#                                                        recompute needs. On 48 GPUs rank 32's torch budget
#                                                        is 184 - 68.2 (non-torch) = ~115.8 GiB and it
#                                                        already uses 114.8, so S's peak says whether
#                                                        reclaiming the 53 GiB would buy anything at all.
#                                                        An OOM at the 0.88 cap is itself the answer (>161.9).
#   H  TP4/ETP2 GBS48 full recompute, ACCEL=0 alltoall   THE PRICE: step-time cost of dropping HybridEP,
#                                                        the only lever big enough to free that memory.
#
# Transferability of the S number: per-rank work is identical at 16 and 48 GPUs (4 microbatches/rank,
# TP4/ETP2, seq 73728) and Muon forces use_distributed_optimizer=False, so optimizer state is replicated,
# not sharded by DP -- per-rank memory does not depend on the DP degree.
#
# Runtime: 80 iterations per leg (the ladder averages the last 20), ~60 s/iter at 16 GPUs => ~1.4 h per leg,
# ~4 h for all three. The ladder runs the legs serially behind its own per-leg gang barrier; it writes only
# to ~/train_logs and ~/ckpts_video_sft/_smoke_qwen35_merged64k/<job>-<leg>, never to a production SAVE.
# Exit code is the ladder's: 0 = every selected leg passed, 1 = a leg failed (an S OOM lands here), 2/3 = infra.
# =============================================================================
set -uo pipefail

_R_HOST="$(hostname)"
_R_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_R_LADDER="$_R_DIR/ax_ov2_qwen35_speed_ladder48.sh"
_R_LOG="$HOME/train_logs/round2_ab16_$(sed -E 's/-(master|worker)-[0-9]+$//' <<<"$_R_HOST")_${_R_HOST}.log"
mkdir -p "$HOME/train_logs"
_say() { echo "[round2-ab16] $*" | tee -a "$_R_LOG" >&2; }
_die() { echo "[round2-ab16] FATAL: $*" | tee -a "$_R_LOG" >&2; exit 3; }

[[ -f "$_R_LADDER" ]] || _die "missing $_R_LADDER (stale checkout? run: cd ~/bridge-export && git pull)"

# 16-GPU form only. The ladder itself accepts 4 or 12 pods; this file is the 4-pod experiment, and a
# 12-pod launch of it would silently spend the production shard on a smoke.
# The ladder resolves pods as PET_NNODES, else OV2_LADDER_NPODS, else 12 -- so pin its fallback to 4 here,
# or an unset PET_NNODES would make the ladder wait at a 12-pod barrier that only 4 pods can ever join.
export OV2_LADDER_NPODS=4
_R_NPODS="${PET_NNODES:-4}"
[[ "$_R_NPODS" == 4 ]] || _die "this is the 16-GPU (4-pod) experiment, got PET_NNODES=$_R_NPODS; for 48 GPUs call the ladder directly"

# ---- baked constants (all overridable from Args, all defaulted to the intended run) ----
export OV2_LADDER_LEGS="${OV2_LADDER_LEGS:-A,S,H}"      # H needs fork de8bb31d; A,S alone work on older vendors
export OV2_LADDER_ITERS="${OV2_LADDER_ITERS:-80}"       # ladder averages the LAST 20; ~60 first are warm-up
export OV2_MEM_PROBE_DEVICE="${OV2_MEM_PROBE_DEVICE:-1}"          # dev_used/proc_used/unattr per sample
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"  # = production since -4
export OV2_PARALLEL_SHARD_ITERS="${OV2_PARALLEL_SHARD_ITERS:-16}" # = production (psi16)
export OV2_ENERGON_DROP_YIELDED="${OV2_ENERGON_DROP_YIELDED:-1}"  # = production (worker-leak fix)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"          # = production since -3; older 16-GPU runs used 1, so
                                                        # compare step time against THIS run's A leg, not those
export OV2_K8S_NAMESPACE="${OV2_K8S_NAMESPACE:-runai-mv0004}"
export OV2_LLM_HF_QWEN35="${OV2_LLM_HF_QWEN35:-$HOME/Qwen3.5-35B-A3B-text}"
export OV2_HF_PROC_QWEN35_P16M33="${OV2_HF_PROC_QWEN35_P16M33:-$HOME/qwen35_p16m33_auto_model}"

_say "host=$_R_HOST pods=$_R_NPODS legs=$OV2_LADDER_LEGS iters=$OV2_LADDER_ITERS alloc=$PYTORCH_CUDA_ALLOC_CONF psi=$OV2_PARALLEL_SHARD_ITERS drop_yielded=$OV2_ENERGON_DROP_YIELDED omp=$OMP_NUM_THREADS mem_probe_device=$OV2_MEM_PROBE_DEVICE"
_say "ladder -> $_R_LADDER"

bash "$_R_LADDER"
_rc=$?
_say "ladder exit rc=$_rc; summary: cat ~/train_logs/smoke_speed_ladder_\$(hostname | sed -E 's/-(master|worker)-[0-9]+\$//').txt"
exit "$_rc"
