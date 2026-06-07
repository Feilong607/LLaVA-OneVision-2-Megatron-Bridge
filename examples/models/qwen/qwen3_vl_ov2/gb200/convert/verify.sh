#!/usr/bin/env bash
# =============================================================================
# OV2 before/after weight-consistency check (CPU only -- no GPU, no torchrun).
# Compares two torch_dist checkpoints tensor-by-tensor on their GLOBAL (unsharded)
# values, so it works across a parallelism change (EP8 source vs EP4/EP8 output).
#
#   # quick structural + representative-value check (seconds-minutes):
#   A=/path/src_ckpt B=/path/out_ckpt bash convert/verify.sh
#
#   # exhaustive: every model tensor, bit-exact (the real "转化前后一致性" gate):
#   A=/path/src_ckpt B=/path/out_ckpt VALUES=full bash convert/verify.sh
#
#   # allow tolerance / also diff optimizer state:
#   A=... B=... ATOL=1e-6 EXTRA="--include-optim" bash convert/verify.sh
#
# Runs in the SAME container/python as training; if already inside the container,
# set IN_CONTAINER=1 to skip the docker wrapper.
# =============================================================================
set -euo pipefail
REPO="${REPO:-$({ __d="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; while [[ "$__d" != "/" && ! -d "$__d/src/megatron/bridge" ]]; do __d="$(dirname "$__d")"; done; echo "$__d"; })}"
[[ -d "$REPO/src/megatron/bridge" ]] || { echo "FATAL: OV2 fork root not found from ${BASH_SOURCE[0]} (no src/megatron/bridge above it). Set REPO=/path/to/LLaVA-OneVision-2-Megatron-Bridge" >&2; exit 1; }
IMAGE="${IMAGE:-mbridge:qwen35}"
: "${A:?set A=<checkpoint dir before/source>}"
: "${B:?set B=<checkpoint dir after/converted>}"
VALUES="${VALUES:-sample}"
ATOL="${ATOL:-0}"
RTOL="${RTOL:-0}"
CHUNK_GB="${CHUNK_GB:-4}"
EXTRA="${EXTRA:-}"
SCRIPT="$REPO/examples/models/qwen/qwen3_vl_ov2/gb200/convert/verify_consistency.py"

run() {
  PYTHONPATH="$REPO/src:$REPO/3rdparty/Megatron-LM:$REPO/aiak_shim${PYTHONPATH:+:$PYTHONPATH}" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python3 "$SCRIPT" --a "$A" --b "$B" --values "$VALUES" --atol "$ATOL" --rtol "$RTOL" \
    --chunk-gb "$CHUNK_GB" $EXTRA
}

echo "[verify] A=$A"
echo "[verify] B=$B  (values=$VALUES atol=$ATOL rtol=$RTOL)"
if [[ "${IN_CONTAINER:-0}" == "1" ]]; then
  run
else
  # Mount /ov2 (repo + ckpts) plus any A/B path that lives outside it (e.g. /vlm, /data), bind
  # dir-over-itself so the in-container paths match the host paths the python script receives.
  MOUNTS=(-v /ov2:/ov2); declare -A seen=(["/ov2"]=1)
  for d in "$A" "$B"; do
    rp="$(realpath "$d" 2>/dev/null || echo "$d")"
    case "$rp" in /ov2|/ov2/*) continue;; esac
    [[ -z "${seen[$rp]:-}" ]] && { MOUNTS+=(-v "$rp":"$rp"); seen[$rp]=1; }
  done
  # CPU only on purpose: no --gpus, so it never contends with training GPUs.
  docker run --rm --ipc=host "${MOUNTS[@]}" -e A="$A" -e B="$B" \
    -e VALUES="$VALUES" -e ATOL="$ATOL" -e RTOL="$RTOL" -e CHUNK_GB="$CHUNK_GB" \
    -e EXTRA="$EXTRA" -e IN_CONTAINER=1 -e REPO="$REPO" "$IMAGE" \
    bash "$REPO/examples/models/qwen/qwen3_vl_ov2/gb200/convert/verify.sh"
fi
