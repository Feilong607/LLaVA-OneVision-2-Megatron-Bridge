#!/usr/bin/env bash
# Re-apply the OV2.1 mcore edits that are carried as patches because the 3rdparty/Megatron-LM submodule
# remote is NVIDIA upstream (un-pushable). A fresh `clone --recurse-submodules` checks out the clean
# pinned SHA WITHOUT these edits. Each patch is guarded INDEPENDENTLY (grep marker), so re-running on a
# checkout that has only the older patch applies just the missing one. Run ONCE after clone; idempotent.
#
# Patch 1 (megatron_lm_ov2.patch): attention.py SelfAttentionSubmodules.apply_rotary_fn hook + custom-
#   rotary dispatch (else OV2 build crashes: "unexpected keyword argument 'apply_rotary_fn'"), and
#   nvrx.py assert->return-False graceful degrade.  HARD dependency -> fatal on failure.
# Patch 2 (megatron_lm_ov2_ep_overlap.patch): combined_1f1b.py relax the isinstance-GPTModel assert to
#   also accept MIMO VLM wrappers exposing build_schedule_plan (LlavaOnevision2 delegates to its inner
#   GPTModel) -- required only for the OPT-IN EP a2a comm-overlap (OV2_EP_OVERLAP=1) -> non-fatal.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
M="$HERE/Megatron-LM"

# --- guard: UNINITIALIZED submodule (clone without --recurse-submodules -> empty dir). Fail with the
# real cause. Load-bearing: the launchers put $REPO/3rdparty/Megatron-LM on PYTHONPATH, so continuing
# with an empty submodule would silently import the container's UNPATCHED mcore and crash much deeper. ---
AT="$M/megatron/core/transformer/attention.py"
[ -f "$AT" ] || {
  echo "[megatron-patch] FATAL: Megatron-LM submodule not initialized ($M has no megatron/core tree)." >&2
  echo "  Fix:  git -C \"$HERE/..\" submodule update --init --depth 1 3rdparty/Megatron-LM" >&2
  echo "  (or re-clone with: git clone --recurse-submodules ...)" >&2
  exit 1
}

# Robust patch application. The deploy may COPY the submodule files WITHOUT its .git (e.g. an image
# build that filters 3rdparty/Megatron-LM/.git). Then $M is a plain subdir of the OUTER repo, so
# `git -C "$M" apply` resolves the patch paths against the outer root and SILENTLY SKIPS every file
# (exit 0) -- the failure the old flow misreported as "rotary/nvrx FAILED (x0)". So we pick the apply
# method by whether $M is its OWN git work-tree, and ALWAYS verify the marker afterwards:
#   - marker already present            -> skip (idempotent)
#   - $M is its own git top-level       -> `git apply` (the tested submodule path)
#   - else (files copied, no .git)      -> plain `patch -p1` (git-independent)
#   - marker still absent after apply   -> fail (fatal for patch 1, warn for the opt-in patch 2)
_apply_one() {   # $1=patch  $2=file-to-check  $3=marker  $4=label  $5=fatal(1|0)
  local pf="$1" cf="$2" marker="$3" label="$4" fatal="$5" _top
  if [ ! -f "$pf" ]; then
    if [ "$fatal" = "1" ]; then echo "[megatron-patch] FATAL: patch not found: $pf" >&2; exit 1; fi
    echo "[megatron-patch] WARN: patch not found: $pf ($label skipped)" >&2; return 0
  fi
  if grep -q "$marker" "$cf" 2>/dev/null; then
    echo "[megatron-patch] $label already applied ($marker present)"; return 0
  fi
  _top="$(git -C "$M" rev-parse --show-toplevel 2>/dev/null || true)"
  if [ -n "$_top" ] && [ "$_top" -ef "$M" ]; then
    git -C "$M" apply "$pf" 2>/dev/null || true          # $M is its own git repo -> tested path
  elif command -v patch >/dev/null 2>&1; then
    patch -p1 -N -s -r /dev/null -d "$M" < "$pf" >/dev/null 2>&1 || true   # files-only -> git-independent
  fi
  if grep -q "$marker" "$cf" 2>/dev/null; then
    echo "[megatron-patch] $label applied OK"; return 0
  fi
  # did not take
  if [ "$fatal" = "1" ]; then
    echo "[megatron-patch] FATAL: $label did not apply ($cf still lacks '$marker')." >&2
    echo "  $M git-toplevel='${_top:-<none>}' (expected $M); if this is not the submodule's own repo," >&2
    echo "  git apply targets the outer repo and skips. Fix: git submodule update --init 3rdparty/Megatron-LM" >&2
    echo "  (or ensure 'patch' is installed for the files-only fallback)." >&2
    exit 1
  fi
  echo "[megatron-patch] WARN: $label did not apply ($cf lacks '$marker'). Training lanes are unaffected; OV2_EP_OVERLAP=1 will hit mcore's 'only GPTModel is supported' assert until it applies." >&2
  return 0
}

_apply_one "$HERE/megatron_lm_ov2.patch"            "$AT"                                                  "apply_rotary_fn"      "rotary/nvrx" 1
_apply_one "$HERE/megatron_lm_ov2_ep_overlap.patch" "$M/megatron/core/pipeline_parallel/combined_1f1b.py" "OV2 EP-overlap patch" "ep-overlap"  0
