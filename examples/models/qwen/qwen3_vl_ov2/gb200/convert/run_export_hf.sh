#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Export a trained OV2-30B-A3B torch_dist checkpoint to a complete HuggingFace VLM.
# This is the GB200 launch wrapper; convert.sh remains the backend because it performs
# AutoBridge export, tokenizer/processor copying, HF skeleton fixups, and optional roundtrip verification.

set -euo pipefail

NPROC="${NPROC:-4}"
EP="${OV2_EP:-8}"
VERIFY="${VERIFY:-0}"

_HOME="${HOME:-}"
[[ -n "$_HOME" ]] || _HOME="$(getent passwd "$(id -un 2>/dev/null)" 2>/dev/null | cut -d: -f6)"
[[ -n "$_HOME" ]] || _HOME="/home/$(id -un 2>/dev/null)"

# Positional arg wins, followed by SRC, the legacy CKPT_DIR alias, and finally the GB200 training default.
SRC="${1:-${SRC:-${CKPT_DIR:-$_HOME/ckpts_video_sft/ov2_30b_a3b_gb200}}}"
OUT="${OUT:-${HF_OUT:-$_HOME/ov2_hf_export/$(basename "${SRC%/}")_hf}}"
WORK="${WORK:-$_HOME/_ov2_convert}"
LOGDIR="${LOGDIR:-$_HOME/export_hf_logs}"

REPO="${REPO:-$({ __d="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; while [[ "$__d" != "/" && ! -d "$__d/src/megatron/bridge" ]]; do __d="$(dirname "$__d")"; done; echo "$__d"; })}"
[[ -d "$REPO/src/megatron/bridge" ]] || REPO="$_HOME/LLaVA-OneVision-2-Megatron-Bridge"
[[ -d "$REPO/src/megatron/bridge" ]] || { echo "[export-hf] FATAL: OV2 fork root not found (set REPO=)" >&2; exit 1; }

CONVDIR="$REPO/examples/models/qwen/qwen3_vl_ov2/gb200/convert"
DRIVER="$CONVDIR/convert.sh"
WORKER="$REPO/examples/models/qwen/qwen3_vl_ov2/gb200/ov2_30b_export_ep8.py"
[[ -f "$DRIVER" ]] || { echo "[export-hf] FATAL: $DRIVER not found" >&2; exit 1; }
[[ -f "$WORKER" ]] || { echo "[export-hf] FATAL: $WORKER not found" >&2; exit 1; }

# Accept either a checkpoint root or a pinned iter_* directory, and fail before reserving all GPUs.
if [[ -f "$SRC/.metadata" ]]; then
  ITER="$(basename "$SRC")"
  ITER="${ITER//[^0-9]/}"
  [[ -n "$ITER" ]] && ITER=$((10#$ITER)) || ITER="pinned"
elif [[ -f "$SRC/latest_checkpointed_iteration.txt" ]]; then
  ITER="$(tr -d '\r\n\t ' < "$SRC/latest_checkpointed_iteration.txt")"
  [[ "$ITER" =~ ^[0-9]+$ ]] || { echo "[export-hf] FATAL: $SRC tracker='$ITER' is not a numeric iteration" >&2; exit 1; }
  ITER_DIR="$SRC/$(printf 'iter_%07d' "$ITER")"
  [[ -f "$ITER_DIR/.metadata" ]] || { echo "[export-hf] FATAL: $ITER_DIR/.metadata not found" >&2; exit 1; }
else
  echo "[export-hf] FATAL: $SRC is neither a checkpoint root nor an iter_* directory" >&2
  exit 1
fi
[[ -z "${EXPECT_ITER:-}" || "$ITER" == "$EXPECT_ITER" ]] || { echo "[export-hf] FATAL: iter=$ITER != EXPECT_ITER=$EXPECT_ITER" >&2; exit 1; }
[[ "$SRC" != "$OUT" ]] || { echo "[export-hf] FATAL: source and output paths are identical: $SRC" >&2; exit 1; }
[[ "$NPROC" =~ ^[1-9][0-9]*$ && "$EP" =~ ^[1-9][0-9]*$ ]] || { echo "[export-hf] FATAL: NPROC and OV2_EP must be positive integers" >&2; exit 1; }
[[ "$VERIFY" == "0" || "$VERIFY" == "1" ]] || { echo "[export-hf] FATAL: VERIFY must be 0 or 1" >&2; exit 1; }

# Dispatch skeleton: prefer a home-staged copy, then the GB200 /datasets mount.
_cfg="$_HOME/llava-ov2-30b-a3b-m9lvdn/auto_model"
[[ -d "$_cfg" ]] || _cfg="/datasets/llava-ov2-30b-a3b-m9lvdn/auto_model"
CFG="${CFG:-$_cfg}"
[[ -f "$CFG/config.json" ]] || { echo "[export-hf] FATAL: dispatch config missing: $CFG/config.json" >&2; exit 1; }

export CKPT_DIR="$SRC" CKPTA="$SRC" HF_OUT="$OUT" WORK CFG NPROC OV2_EP="$EP"
export PYTHONPATH="$REPO/_verify_stubs:$REPO/src:$REPO/3rdparty/Megatron-LM:$REPO/aiak_shim${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export OV2_MOE_PERMUTE_FUSION="${OV2_MOE_PERMUTE_FUSION:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-1}"
export NCCL_MNNVL_ENABLE="${NCCL_MNNVL_ENABLE:-0}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}" PYTHONUNBUFFERED=1

# Mirror convert.sh's rendezvous priorities so bad world-size requests fail before torchrun starts.
GPUS_PER_NODE="$NPROC"
if [[ "${FORCE_STANDALONE:-0}" == "1" ]]; then
  NNODES=1
  NODE_RANK=0
  RUN_MODE="single-node (FORCE_STANDALONE)"
elif [[ -n "${PET_NNODES:-}" || ( -n "${MASTER_ADDR:-}" && -n "${WORLD_SIZE:-}" ) ]]; then
  NNODES="${PET_NNODES:-$(( WORLD_SIZE / GPUS_PER_NODE ))}"
  NODE_RANK="${PET_NODE_RANK:-$(( ${RANK:-0} / GPUS_PER_NODE ))}"
  RUN_MODE="multi-node (K8s auto-detected)"
elif [[ -n "${LIST_IP:-}" ]]; then
  read -ra list_ip <<< "$LIST_IP"
  (( ${#list_ip[@]} >= 1 )) || { echo "[export-hf] FATAL: LIST_IP is empty" >&2; exit 1; }
  NNODES=${#list_ip[@]}
  read -ra _my_ips <<< "$(hostname -I 2>/dev/null)"
  CURRENT_HOST="$(hostname)"
  NODE_RANK=-1
  for i in "${!list_ip[@]}"; do
    if [[ "${list_ip[$i]}" == "$CURRENT_HOST" ]]; then NODE_RANK=$i && break; fi
    for _ip in "${_my_ips[@]}"; do [[ "${list_ip[$i]}" == "$_ip" ]] && NODE_RANK=$i && break 2; done
  done
  [[ "$NODE_RANK" -ge 0 ]] || { echo "[export-hf] FATAL: this host is not present in LIST_IP (${list_ip[*]})" >&2; exit 1; }
  RUN_MODE="multi-node (manual LIST_IP)"
else
  NNODES=1
  NODE_RANK=0
  RUN_MODE="single-node"
fi
WORLD=$((NPROC * NNODES))
(( WORLD == EP )) || { echo "[export-hf] FATAL: export requires world=$WORLD to equal OV2_EP=$EP (GB200 EP8 needs 2 nodes x 4 GPUs)" >&2; exit 1; }

MODE="export"
[[ "$VERIFY" == "1" ]] && MODE="30b"
mkdir -p "$WORK" "$OUT" "$LOGDIR"

echo "[export-hf] rdzv=$RUN_MODE nnodes=$NNODES node_rank=$NODE_RANK nproc=$NPROC world=$WORLD ep=$EP"
echo "[export-hf] $SRC (iter=$ITER) -> $OUT"
echo "[export-hf] cfg=$CFG mode=$MODE work=$WORK"

cd "$REPO"
bash "$DRIVER" "$MODE" 2>&1 | tee "$LOGDIR/export_hf_ep${EP}_iter${ITER}_node${NODE_RANK}.log"
echo "[export-hf] DONE: $OUT"
