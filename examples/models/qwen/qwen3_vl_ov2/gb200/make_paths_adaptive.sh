#!/usr/bin/env bash
# =============================================================================
# make_paths_adaptive.sh — make every gb200/*.sh self-adaptive.
#
#   * REPO  -> "marker walk": each script walks UP from its own location to the
#     dir holding src/megatron/bridge, so PYTHONPATH ALWAYS points at the fork the
#     script lives in (never the container's /opt/Megatron-Bridge). Fail-loud guard
#     if not found -> can never silently fall back to /opt again.
#   * data dirs -> machine-aware: /ov2 on A100, $HOME elsewhere (e.g. GB200).
#
# Idempotent (safe to re-run). Save this file INSIDE .../qwen3_vl_ov2/gb200/ and run:
#     bash make_paths_adaptive.sh
# =============================================================================
set -uo pipefail
shopt -s nullglob
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[make_paths_adaptive] target dir: $SELF"

python3 - "$SELF" "$(basename "${BASH_SOURCE[0]}")" <<'PYEOF'
import re, pathlib, sys
gb200 = pathlib.Path(sys.argv[1]); selfname = sys.argv[2]
M = "src/megatron/bridge"
BLK = ('REPO="${REPO:-$({ __d="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; '
       'while [[ "$__d" != "/" && ! -d "$__d/src/megatron/bridge" ]]; do __d="$(dirname "$__d")"; done; '
       'echo "$__d"; })}"\n'
       '[[ -d "$REPO/src/megatron/bridge" ]] || { echo "FATAL: OV2 fork root not found from '
       '${BASH_SOURCE[0]}. Set REPO=/path/to/LLaVA-OneVision-2-Megatron-Bridge" >&2; exit 1; }')
rx = re.compile(r'^REPO="\$\{REPO:-.*\}"[ \t]*$', re.M)
ch = []
for f in sorted(gb200.rglob("*.sh")):
    if f.name == selfname:
        continue
    s = o = f.read_text()
    s = rx.sub(lambda m: m.group(0) if M in m.group(0) else BLK, s, count=1)
    s = s.replace('OV2_PRETRAIN_ROOT="${OV2_PRETRAIN_ROOT:-/ov2/pretrain_models}"',
                  'OV2_PRETRAIN_ROOT="${OV2_PRETRAIN_ROOT:-$([[ -d /ov2/pretrain_models ]] && echo /ov2/pretrain_models || echo "$HOME/pretrain_models")}"')
    s = s.replace(':-/ov2/feilong/gb200/', ':-$([[ -d /ov2/feilong ]] && echo /ov2/feilong/gb200 || echo "$HOME/ov2")/')
    if s != o:
        f.write_text(s); ch.append(f.name)
print("scripts CHANGED:", ch or "(none -- already adaptive)")
PYEOF

echo "[make_paths_adaptive] bash -n syntax check:"
for f in "$SELF"/*.sh "$SELF"/convert/*.sh; do
  bash -n "$f" && echo "  OK   ${f##*/}" || echo "  FAIL ${f##*/}"
done

echo "[make_paths_adaptive] resolved REPO from this location:"
REPO="$({ __d="$SELF"; while [[ "$__d" != "/" && ! -d "$__d/src/megatron/bridge" ]]; do __d="$(dirname "$__d")"; done; echo "$__d"; })"
if [[ -d "$REPO/src/megatron/bridge" ]]; then
  echo "  REPO=$REPO"
  echo "  -> PYTHONPATH will point HERE (the fork), not /opt/Megatron-Bridge"
else
  echo "  WARNING: fork root not found from $SELF (no src/megatron/bridge above it)"
fi
echo "[make_paths_adaptive] done."
