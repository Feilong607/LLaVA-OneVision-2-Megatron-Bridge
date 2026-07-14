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

# --- guard: UNINITIALIZED submodule (clone without --recurse-submodules). With an empty
# 3rdparty/Megatron-LM, `git -C "$M" apply` resolves against the OUTER repo, SILENTLY SKIPS every
# file in the patch, and returns 0 (verified: `apply --verbose` prints "Skipped patch '...'" x2) --
# the old flow then died later with a misleading "FAILED (x0, expected >=3)". Fail HERE with the
# real cause. This fail-loud matters: the launchers put $REPO/3rdparty/Megatron-LM on PYTHONPATH,
# so continuing with an empty submodule would silently import the container's UNPATCHED mcore and
# crash much deeper with "unexpected keyword argument 'apply_rotary_fn'". ---
AT="$M/megatron/core/transformer/attention.py"
[ -f "$AT" ] || {
  echo "[megatron-patch] FATAL: Megatron-LM submodule not initialized ($M has no megatron/core tree)." >&2
  echo "  Fix:  git -C \"$HERE/..\" submodule update --init --depth 1 3rdparty/Megatron-LM" >&2
  echo "  (or re-clone with: git clone --recurse-submodules ...)" >&2
  exit 1
}

# --- patch 1: rotary hook + nvrx degrade ---
P1="$HERE/megatron_lm_ov2.patch"
[ -f "$P1" ] || { echo "[megatron-patch] patch not found: $P1"; exit 1; }
if grep -q 'apply_rotary_fn' "$AT" 2>/dev/null; then
  echo "[megatron-patch] rotary/nvrx already applied (apply_rotary_fn present)"
else
  git -C "$M" apply "$P1"
  n=$(grep -c 'apply_rotary_fn' "$AT" || true)
  if [ "${n:-0}" -ge 3 ]; then echo "[megatron-patch] rotary/nvrx applied OK (apply_rotary_fn x$n)"; else echo "[megatron-patch] rotary/nvrx FAILED (x${n:-0}, expected >=3)"; exit 1; fi
fi

# --- patch 2: EP-overlap GPTModel-assert relaxation (combined_1f1b.py) ---
# NON-FATAL on failure: this patch only matters for the OPT-IN OV2_EP_OVERLAP=1 path. A context
# mismatch (e.g. a checkout whose mcore SHA differs from the pin) must not block the default
# ACCEL=0/1/2/3 training lanes -- if the patch is missing and someone sets OV2_EP_OVERLAP=1,
# mcore's own combined_1f1b assert fails with a clear "only GPTModel is supported" message.
P2="$HERE/megatron_lm_ov2_ep_overlap.patch"
CF="$M/megatron/core/pipeline_parallel/combined_1f1b.py"
if [ ! -f "$P2" ]; then
  echo "[megatron-patch] WARN: ep-overlap patch not found: $P2 (OV2_EP_OVERLAP=1 will not work)" >&2
elif grep -q 'OV2 EP-overlap patch' "$CF" 2>/dev/null; then
  echo "[megatron-patch] ep-overlap already applied (OV2 EP-overlap patch present)"
elif git -C "$M" apply "$P2" 2>/dev/null && grep -q 'OV2 EP-overlap patch' "$CF"; then
  echo "[megatron-patch] ep-overlap applied OK"
else
  echo "[megatron-patch] WARN: ep-overlap patch did NOT apply (mcore context mismatch?). Training lanes are unaffected; OV2_EP_OVERLAP=1 will fail with mcore's 'only GPTModel is supported' assert until the patch applies." >&2
fi
