#!/usr/bin/env bash
# =============================================================================
# OV2-30B-A3B GB200 -- personal "base" entry point (per-machine convenience wrapper).
#
# WHY THIS EXISTS: the portable launcher (ax_ov2_30b_a3b_gb200.sh) auto-detects REPO from its own
# location so it runs UNCHANGED on GB200, A100/A800 (/ov2), and CI. To skip setting env every run
# WITHOUT hardcoding a machine-specific path into the committed launcher -- a hardcode both breaks the
# A-card + CI auto-detect (see that script's REPO comment) AND is forbidden by AGENTS.md ("NEVER commit
# environment-specific paths / account names / usernames") -- this wrapper loads your PERSONAL,
# UNCOMMITTED overrides from ~/.ov2_local.sh, then execs the portable launcher.
#
# ONE-TIME SETUP (kept in your home dir, NOT committed -- put REPO + any secrets/paths here):
#   echo 'export REPO=/home/<you>/LLaVA-OneVision-2-Megatron-Bridge' > ~/.ov2_local.sh
#   # e.g. also: export SAVE=... INIT_CKPT=... HF_HOME=... WANDB_API_KEY=...
#
# THEN JUST RUN (all env + extra args pass straight through to the real launcher):
#   bash examples/models/qwen/qwen3_vl_ov2/gb200/ax_ov2_30b_a3b_gb200_base.sh
#   MOE_CAPACITY_FACTOR=1.5 MOE_PAD_TO_CAPACITY=true ACCEL=1 bash .../ax_ov2_30b_a3b_gb200_base.sh
# =============================================================================
set -euo pipefail

# Load personal per-machine overrides (REPO, SAVE, tokens, ...). Override the location with OV2_LOCAL_ENV=.
_ov2_local="${OV2_LOCAL_ENV:-$HOME/.ov2_local.sh}"
if [[ -f "$_ov2_local" ]]; then
  echo "[ov2-base] sourcing local overrides: $_ov2_local" >&2
  # shellcheck disable=SC1090
  source "$_ov2_local"
else
  echo "[ov2-base] no local override file ($_ov2_local) -- relying on \$REPO env + auto-detect. To pin REPO once:" >&2
  echo "[ov2-base]   echo 'export REPO=/path/to/LLaVA-OneVision-2-Megatron-Bridge' > $_ov2_local" >&2
fi

# Exec the portable launcher sitting next to this wrapper, forwarding all args.
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ax_ov2_30b_a3b_gb200.sh" "$@"
