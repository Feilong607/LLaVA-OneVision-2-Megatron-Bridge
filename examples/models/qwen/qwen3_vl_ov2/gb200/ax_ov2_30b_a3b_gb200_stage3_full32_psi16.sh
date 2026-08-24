#!/usr/bin/env bash
# =============================================================================
# STAGE-3 32-GPU controlled A/B lane: parallel_shard_iters=16.
#
# The production psi=1 run is the control. This wrapper deliberately changes
# only parallel_shard_iters and isolates every writable namespace. It resolves
# the exact Stage-2 iteration used by the control from its durable log, then
# validates Megatron torch_dist's completion marker before allocating GPUs. A
# different checkpoint is not a valid A/B control and is never chosen silently.
#
# Read-only preflight from a code-sync workspace:
#   OV2_PSI16_PREFLIGHT_ONLY=1 bash "$0"
# If the production log is unavailable/ambiguous, explicitly provide the exact
# control checkpoint:
#   OV2_PSI16_INIT_CKPT=.../iter_0004000 OV2_PSI16_PREFLIGHT_ONLY=1 bash "$0"
# =============================================================================
set -euo pipefail

_psi16_die() {
  echo "[psi16-ab] FATAL: $*" >&2
  exit 1
}

_psi16_valid_torch_dist() {
  local checkpoint_dir="$1"
  [[ -d "$checkpoint_dir" && -s "$checkpoint_dir/metadata.json" && -s "$checkpoint_dir/common.pt" ]] || return 1
  python -c '
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    metadata = json.load(stream)
if not isinstance(metadata.get("sharded_backend"), str) or not metadata["sharded_backend"]:
    raise ValueError("metadata.json has no sharded_backend")
if int(metadata.get("sharded_backend_version", 1)) < 1:
    raise ValueError("invalid sharded_backend_version")
' "$checkpoint_dir/metadata.json" >/dev/null 2>&1
}

_psi16_sha256() {
  python -c '
import hashlib
import sys

digest = hashlib.sha256()
with open(sys.argv[1], "rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
print(digest.hexdigest())
' "$1"
}

export HOME="${HOME:-/home/ftan0055}"
readonly _PSI16_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly _PSI16_BASE_LAUNCHER="$_PSI16_SCRIPT_DIR/ax_ov2_30b_a3b_gb200_stage3.sh"
readonly _PSI16_DATA_PATH="$_PSI16_SCRIPT_DIR/stage3_mix_img10.yaml"
readonly _PSI16_INIT_ROOT="$HOME/ckpts_video_sft/ov2_30b_a3b_stage2mix_v3_gbs32"
readonly _PSI1_SAVE="$HOME/ckpts_video_sft/ov2_30b_a3b_stage3_img10_gbs32"
readonly _PSI16_SAVE="$HOME/ckpts_video_sft/ov2_30b_a3b_stage3_img10_gbs32_psi16_v2"

[[ -f "$_PSI16_BASE_LAUNCHER" ]] || _psi16_die "base Stage-3 launcher missing: $_PSI16_BASE_LAUNCHER"
[[ -f "$_PSI16_DATA_PATH" ]] || _psi16_die "Stage-3 dataset YAML missing: $_PSI16_DATA_PATH"

# Do not inherit copied-workload paths. A stale SAVE can resume/write the
# production run before anyone notices that the A/B lane is contaminated.
if [[ -n "${SAVE:-}" && "$SAVE" != "$_PSI16_SAVE" ]]; then
  _psi16_die "refusing inherited SAVE=$SAVE; psi16 SAVE must be $_PSI16_SAVE"
fi
if [[ -n "${DATA_PATH:-}" && "$DATA_PATH" != "$_PSI16_DATA_PATH" ]]; then
  _psi16_die "refusing inherited DATA_PATH=$DATA_PATH; expected $_PSI16_DATA_PATH"
fi
if [[ -n "${INIT_CKPT:-}" && -z "${OV2_PSI16_INIT_CKPT:-}" && "$INIT_CKPT" != "$_PSI16_INIT_ROOT" ]]; then
  _psi16_die "refusing inherited INIT_CKPT=$INIT_CKPT; use OV2_PSI16_INIT_CKPT=<exact iter_*> explicitly"
fi

export OV2_PARALLEL_SHARD_ITERS=16
export SAVE="$_PSI16_SAVE"
export DATA_PATH="$_PSI16_DATA_PATH"

_psi16_save_real="$(python -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$SAVE")"
_psi1_save_real="$(python -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$_PSI1_SAVE")"
[[ "$_psi16_save_real" != "$_psi1_save_real" ]] || _psi16_die "psi16 SAVE resolves to the production SAVE"

# Prefer an explicit direct iteration. Otherwise recover the control run's
# actual Stage-2 iteration from durable psi=1 logs. Multiple values mean the
# glob contains multiple control runs and must be disambiguated explicitly.
_psi16_init="${OV2_PSI16_INIT_CKPT:-}"
if [[ -z "$_psi16_init" ]]; then
  _psi16_control_iters=()
  while IFS= read -r _psi16_control_iter; do
    [[ -n "$_psi16_control_iter" ]] && _psi16_control_iters+=("$_psi16_control_iter")
  done < <(
    grep -hF "loading distributed checkpoint from $_PSI16_INIT_ROOT at iteration " \
      "$HOME"/train_logs/stage3_32_*.log 2>/dev/null \
      | sed -nE 's/.* at iteration ([0-9]+).*/\1/p' \
      | sort -nu
  )
  if (( ${#_psi16_control_iters[@]} != 1 )); then
    _psi16_valid_list=""
    for _psi16_candidate in "$_PSI16_INIT_ROOT"/iter_*; do
      if _psi16_valid_torch_dist "$_psi16_candidate"; then
        _psi16_valid_list+=" $_psi16_candidate"
      fi
    done
    _psi16_die "could not infer one psi=1 INIT iteration (found: ${_psi16_control_iters[*]:-none}). Set OV2_PSI16_INIT_CKPT to the exact control iter. Valid candidates:${_psi16_valid_list:- none}"
  fi
  printf -v _psi16_init '%s/iter_%07d' "$_PSI16_INIT_ROOT" "${_psi16_control_iters[0]}"
fi

[[ "$(basename "$_psi16_init")" == iter_* ]] || _psi16_die "INIT must point directly to an iter_* directory, got $_psi16_init"
_psi16_valid_torch_dist "$_psi16_init" || _psi16_die "incomplete/non-torch_dist INIT: $_psi16_init (requires non-empty common.pt and valid final metadata.json)"
export INIT_CKPT="$_psi16_init"

# The control's archived launcher and YAML are evidence that all settings other
# than psi are identical. Fail rather than compare across code or data changes.
if [[ "${OV2_PSI16_SKIP_CONTROL_ASSET_CHECK:-0}" != "1" ]]; then
  [[ -f "$_PSI1_SAVE/$(basename "$_PSI16_BASE_LAUNCHER")" ]] || _psi16_die "control launcher archive missing under $_PSI1_SAVE"
  [[ -f "$_PSI1_SAVE/$(basename "$_PSI16_DATA_PATH")" ]] || _psi16_die "control dataset archive missing under $_PSI1_SAVE"
  cmp -s "$_PSI16_BASE_LAUNCHER" "$_PSI1_SAVE/$(basename "$_PSI16_BASE_LAUNCHER")" \
    || _psi16_die "base launcher differs from the psi=1 archived launcher"
  cmp -s "$_PSI16_DATA_PATH" "$_PSI1_SAVE/$(basename "$_PSI16_DATA_PATH")" \
    || _psi16_die "dataset YAML differs from the psi=1 archived YAML"
fi

echo "[psi16-ab] PASS: parallel_shard_iters=$OV2_PARALLEL_SHARD_ITERS"
echo "[psi16-ab] PASS: init=$INIT_CKPT"
echo "[psi16-ab] PASS: save=$SAVE (control=$_PSI1_SAVE)"
echo "[psi16-ab] PASS: base_launcher_sha256=$(_psi16_sha256 "$_PSI16_BASE_LAUNCHER")"
echo "[psi16-ab] PASS: dataset_sha256=$(_psi16_sha256 "$_PSI16_DATA_PATH")"

if [[ "${OV2_PSI16_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "[psi16-ab] PREFLIGHT ONLY: no training process started and no SAVE files written"
  exit 0
fi

mkdir -p "$SAVE" "$HOME/train_logs"
_psi16_manifest="$SAVE/psi16_ab_manifest.txt"
_psi16_manifest_tmp="$_psi16_manifest.$(hostname).$$.tmp"
_psi16_repo="$(git -C "$_PSI16_SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
_psi16_git_head="$(git -C "${_psi16_repo:-$_PSI16_SCRIPT_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
{
  echo "parallel_shard_iters=$OV2_PARALLEL_SHARD_ITERS"
  echo "init_ckpt=$INIT_CKPT"
  echo "save=$SAVE"
  echo "control_save=$_PSI1_SAVE"
  echo "git_head=$_psi16_git_head"
  echo "base_launcher_sha256=$(_psi16_sha256 "$_PSI16_BASE_LAUNCHER")"
  echo "dataset_sha256=$(_psi16_sha256 "$_PSI16_DATA_PATH")"
  echo "init_metadata_sha256=$(_psi16_sha256 "$INIT_CKPT/metadata.json")"
} >"$_psi16_manifest_tmp"

if [[ -f "$_psi16_manifest" ]]; then
  cmp -s "$_psi16_manifest_tmp" "$_psi16_manifest" \
    || _psi16_die "existing psi16 manifest differs; use a fresh SAVE namespace instead of mixing runs"
elif ! ln "$_psi16_manifest_tmp" "$_psi16_manifest" 2>/dev/null; then
  cmp -s "$_psi16_manifest_tmp" "$_psi16_manifest" \
    || _psi16_die "another pod created a different psi16 manifest"
fi
rm -f "$_psi16_manifest_tmp"

bash "$_PSI16_BASE_LAUNCHER" 2>&1 \
  | tee -a "$HOME/train_logs/stage3_psi16_v2_32_$(hostname).log"
