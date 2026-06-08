#!/usr/bin/env bash
# Re-apply the OV2.1 mcore edits that are carried as a patch because the 3rdparty/Megatron-LM submodule
# remote is NVIDIA upstream (un-pushable). A fresh `clone --recurse-submodules` checks out the clean
# pinned SHA WITHOUT these edits, so the OV2 model build crashes:
#   TypeError: SelfAttentionSubmodules.__init__() got an unexpected keyword argument 'apply_rotary_fn'
# (layer_spec.py passes apply_rotary_fn=apply_rotary_pos_emb_vision into mcore SelfAttentionSubmodules).
# The patch adds: attention.py SelfAttentionSubmodules.apply_rotary_fn hook + custom-rotary dispatch,
# and nvrx.py assert->return-False graceful degrade. Run ONCE after clone. Idempotent (no-op if applied).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
M="$HERE/Megatron-LM"; P="$HERE/megatron_lm_ov2.patch"
AT="$M/megatron/core/transformer/attention.py"
[ -f "$P" ] || { echo "[megatron-patch] patch not found: $P"; exit 1; }
if grep -q 'apply_rotary_fn' "$AT" 2>/dev/null; then
  echo "[megatron-patch] already applied (apply_rotary_fn present)"; exit 0
fi
git -C "$M" apply "$P"
n=$(grep -c 'apply_rotary_fn' "$AT" || true)
if [ "${n:-0}" -ge 3 ]; then echo "[megatron-patch] applied OK (apply_rotary_fn x$n)"; else echo "[megatron-patch] FAILED (x${n:-0}, expected >=3)"; exit 1; fi
