#!/usr/bin/env bash
# Re-apply the OV2.1 mcore edits that are carried as patches because the 3rdparty/Megatron-LM submodule
# remote is NVIDIA upstream (un-pushable). A fresh `clone --recurse-submodules` checks out the clean
# pinned SHA WITHOUT these edits. Each patch is guarded INDEPENDENTLY (grep marker), so re-running on a
# checkout that has only the older patch applies just the missing one. Run ONCE after clone; idempotent.
#
# Patch 1 (megatron_lm_ov2.patch): attention.py SelfAttentionSubmodules.apply_rotary_fn hook + custom-
#   rotary dispatch (else OV2 build crashes: "unexpected keyword argument 'apply_rotary_fn'"), and
#   nvrx.py assert->return-False graceful degrade.
# Patch 2 (megatron_lm_ov2_ep_overlap.patch): combined_1f1b.py relax the isinstance-GPTModel assert to
#   also accept MIMO VLM wrappers exposing build_schedule_plan (LlavaOnevision2 delegates to its inner
#   GPTModel) -- required for EP a2a comm-overlap (OV2_EP_OVERLAP=1); without it the combined-1F1B
#   schedule asserts at the first step.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
M="$HERE/Megatron-LM"

# --- patch 1: rotary hook + nvrx degrade ---
P1="$HERE/megatron_lm_ov2.patch"
AT="$M/megatron/core/transformer/attention.py"
[ -f "$P1" ] || { echo "[megatron-patch] patch not found: $P1"; exit 1; }
if grep -q 'apply_rotary_fn' "$AT" 2>/dev/null; then
  echo "[megatron-patch] rotary/nvrx already applied (apply_rotary_fn present)"
else
  git -C "$M" apply "$P1"
  n=$(grep -c 'apply_rotary_fn' "$AT" || true)
  if [ "${n:-0}" -ge 3 ]; then echo "[megatron-patch] rotary/nvrx applied OK (apply_rotary_fn x$n)"; else echo "[megatron-patch] rotary/nvrx FAILED (x${n:-0}, expected >=3)"; exit 1; fi
fi

# --- patch 2: EP-overlap GPTModel-assert relaxation (combined_1f1b.py) ---
P2="$HERE/megatron_lm_ov2_ep_overlap.patch"
CF="$M/megatron/core/pipeline_parallel/combined_1f1b.py"
[ -f "$P2" ] || { echo "[megatron-patch] patch not found: $P2"; exit 1; }
if grep -q 'OV2 EP-overlap patch' "$CF" 2>/dev/null; then
  echo "[megatron-patch] ep-overlap already applied (OV2 EP-overlap patch present)"
else
  git -C "$M" apply "$P2"
  if grep -q 'OV2 EP-overlap patch' "$CF"; then echo "[megatron-patch] ep-overlap applied OK"; else echo "[megatron-patch] ep-overlap FAILED (marker missing after apply)"; exit 1; fi
fi
