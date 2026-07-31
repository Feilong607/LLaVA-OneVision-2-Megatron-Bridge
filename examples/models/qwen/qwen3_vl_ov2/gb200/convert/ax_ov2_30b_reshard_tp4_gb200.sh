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

set -euo pipefail

TP="${TP:-4}"; EP="${EP:-8}"; ETP="${ETP:-1}"; NPROC="${NPROC:-4}"
SRC="${SRC:-$HOME/ckpts_video_sft/ov2_30b_a3b_gb200}"
OUT="${OUT:-${SRC%/}_tp${TP}_ep${EP}_etp${ETP}}"
BACKBONE="${BACKBONE:-qwen3-30b-a3b-p16m33}"
LOGDIR="${LOGDIR:-$HOME/reshard_logs}"; mkdir -p "$LOGDIR"

REPO="${REPO:-$({ __d="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; while [[ "$__d" != "/" && ! -d "$__d/src/megatron/bridge" ]]; do __d="$(dirname "$__d")"; done; echo "$__d"; })}"
[[ -d "$REPO/src/megatron/bridge" ]] || REPO="$HOME/LLaVA-OneVision-2-Megatron-Bridge"
[[ -d "$REPO/src/megatron/bridge" ]] || { echo "[reshard] FATAL: OV2 fork root not found (set REPO=)" >&2; exit 1; }
CONVDIR="$REPO/examples/models/qwen/qwen3_vl_ov2/gb200/convert"
WORKER="$CONVDIR/convert_ov2_checkpoint.py"
[[ -f "$WORKER" ]] || { echo "[reshard] FATAL: $WORKER not found" >&2; exit 1; }

if [[ -f "$SRC/.metadata" ]]; then
  ITER="$(basename "$SRC")"; ITER="${ITER//[^0-9]/}"
  [[ -n "$ITER" ]] && ITER=$((10#$ITER)) || ITER="pinned"
elif [[ -f "$SRC/latest_checkpointed_iteration.txt" ]]; then
  ITER="$(cat "$SRC/latest_checkpointed_iteration.txt")"; ITER="${ITER//[$'\r\n\t ']/}"
  [[ "$ITER" =~ ^[0-9]+$ ]] || { echo "[reshard] FATAL: $SRC tracker='$ITER' 不是数字 iter" >&2; exit 1; }
  [[ -f "$SRC/$(printf 'iter_%07d' "$ITER")/.metadata" ]] || { echo "[reshard] FATAL: $SRC/$(printf 'iter_%07d' "$ITER") 缺 .metadata" >&2; exit 1; }
else
  echo "[reshard] FATAL: $SRC 既不是 ckpt 根目录也不是 iter_ 目录" >&2; exit 1
fi
[[ -z "${EXPECT_ITER:-}" || "$ITER" == "$EXPECT_ITER" ]] || { echo "[reshard] FATAL: iter=$ITER != EXPECT_ITER=$EXPECT_ITER" >&2; exit 1; }
[[ -e "$OUT" ]] && { echo "[reshard] FATAL: 输出已存在: $OUT" >&2; exit 1; }

export PYTHONPATH="$REPO/src:$REPO/3rdparty/Megatron-LM:$REPO/aiak_shim${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OV2_MOE_PERMUTE_FUSION="${OV2_MOE_PERMUTE_FUSION:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OV2_LLM_HF_30B="${OV2_LLM_HF_30B:-/datasets/qwen-models-ea5jyi/Qwen3-30B-A3B-Instruct-2507}"
export OV2_CONVERT_TOKENIZER="${OV2_CONVERT_TOKENIZER:-/datasets/llava-ov2-30b-a3b-m9lvdn/auto_model}"

export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-1}"
export NCCL_MNNVL_ENABLE="${NCCL_MNNVL_ENABLE:-0}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}" PYTHONUNBUFFERED=1

GPUS_PER_NODE="$NPROC"
if [[ "${FORCE_STANDALONE:-0}" == 1 ]]; then
  NNODES=1; NODE_RANK=0; MASTER_ADDR=127.0.0.1; MASTER_PORT="${MASTER_PORT:-26049}"
  RUN_MODE="single-node (FORCE_STANDALONE)"
elif [[ -n "${PET_NNODES:-}" || ( -n "${MASTER_ADDR:-}" && -n "${WORLD_SIZE:-}" ) ]]; then
  NNODES="${PET_NNODES:-$(( WORLD_SIZE / GPUS_PER_NODE ))}"
  (( NNODES >= 1 )) || { echo "[reshard] FATAL: WORLD_SIZE=${WORLD_SIZE:-?} < NPROC=$NPROC" >&2; exit 1; }
  NODE_RANK="${PET_NODE_RANK:-}"
  if [[ -z "$NODE_RANK" ]]; then
    if (( ${RANK:-0} < NNODES )); then NODE_RANK="${RANK:-0}"; else NODE_RANK=$(( ${RANK:-0} / GPUS_PER_NODE )); fi
  fi
  MASTER_ADDR="${MASTER_ADDR:-${PET_MASTER_ADDR:-}}"
  MASTER_PORT="${MASTER_PORT:-${PET_MASTER_PORT:-26049}}"
  [[ "$NNODES" -gt 1 && -z "${MASTER_ADDR}" ]] && { echo "[reshard] FATAL: multi-node 但没有 MASTER_ADDR/PET_MASTER_ADDR" >&2; exit 1; }
  RUN_MODE="multi-node (K8s auto-detected)"
elif [[ -n "${LIST_IP:-}" ]]; then
  read -ra list_ip <<< "$LIST_IP"
  (( ${#list_ip[@]} >= 1 )) || { echo "[reshard] FATAL: LIST_IP 为空" >&2; exit 1; }
  NNODES=${#list_ip[@]}; MASTER_ADDR="${list_ip[0]}"; MASTER_PORT="${MASTER_PORT:-26049}"
  read -ra _my_ips <<< "$(hostname -I 2>/dev/null)"; CURRENT_HOST="$(hostname)"; NODE_RANK=-1
  for i in "${!list_ip[@]}"; do
    if [[ "${list_ip[$i]}" == "$CURRENT_HOST" ]]; then NODE_RANK=$i && break; fi
    for _ip in "${_my_ips[@]}"; do [[ "${list_ip[$i]}" == "$_ip" ]] && NODE_RANK=$i && break 2; done
  done
  [[ "$NODE_RANK" -eq -1 ]] && { echo "[reshard] ERROR: 本机 IP(${_my_ips[0]:-?})/名($CURRENT_HOST) 不在 LIST_IP (${list_ip[*]})" >&2; exit 1; }
  RUN_MODE="multi-node (manual LIST_IP)"
else
  NNODES=1; NODE_RANK=0; MASTER_ADDR=127.0.0.1; MASTER_PORT="${MASTER_PORT:-26049}"
  RUN_MODE="single-node"
fi

if [[ "$NNODES" -gt 1 && -n "${MASTER_ADDR:-}" && "$MASTER_ADDR" != *.* && "$MASTER_ADDR" != "127.0.0.1" ]]; then
  _sa_ns_file="/var/run/secrets/kubernetes.io/serviceaccount/namespace"
  _ns="${POD_NAMESPACE:-${OV2_K8S_NAMESPACE:-$([ -r "$_sa_ns_file" ] && cat "$_sa_ns_file" 2>/dev/null || echo "")}}"
  _fqdn=""; [[ -n "$_ns" ]] && _fqdn="${MASTER_ADDR}.${_ns}.svc.cluster.local"
  if [[ -n "$_fqdn" ]] && getent hosts "$_fqdn" >/dev/null 2>&1; then
    echo "[reshard] rdzv: short MASTER_ADDR '$MASTER_ADDR' -> FQDN '$_fqdn'" >&2
    MASTER_ADDR="$_fqdn"
  elif getent hosts "$MASTER_ADDR" >/dev/null 2>&1; then
    echo "[reshard] rdzv: MASTER_ADDR '$MASTER_ADDR' 可直接解析" >&2
  else
    echo "[reshard] WARN: MASTER_ADDR='$MASTER_ADDR' 解析不了, rendezvous 可能超时" >&2
  fi
fi
if [[ "$NNODES" -le 1 ]]; then RDZV="--standalone --nnodes=1"; NNODES=1; NODE_RANK=0; else RDZV="--nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT"; fi
WORLD=$(( NPROC * NNODES ))

(( WORLD >= EP && WORLD % EP == 0 )) || { echo "[reshard] FATAL: world=$WORLD 必须 >=EP=$EP 且整除" >&2; exit 1; }
(( WORLD % TP == 0 )) || { echo "[reshard] FATAL: world=$WORLD 必须整除 TP=$TP" >&2; exit 1; }
ETP_EFF=$(( ETP > 0 ? ETP : TP ))
(( WORLD % (ETP_EFF * EP) == 0 )) || { echo "[reshard] FATAL: world=$WORLD 不整除专家网格 $((ETP_EFF*EP))" >&2; exit 1; }

echo "[reshard] rdzv=$RUN_MODE master=${MASTER_ADDR}:${MASTER_PORT} nnodes=$NNODES node_rank=$NODE_RANK world=$WORLD"
echo "[reshard] $SRC (iter=$ITER) -> $OUT (TP$TP/EP$EP/ETP$ETP)"

cd "$REPO"
# shellcheck disable=SC2086  # RDZV intentionally expands into multiple torchrun arguments.
python -m torch.distributed.run $RDZV --nproc_per_node="$NPROC" "$WORKER" reshard --backbone "$BACKBONE" --src "$SRC" --out "$OUT" --tp "$TP" --ep "$EP" --etp "$ETP" 2>&1 | tee "$LOGDIR/reshard_tp${TP}_iter${ITER}_node${NODE_RANK}.log"

if [[ "$NODE_RANK" == 0 ]]; then
  IN_CONTAINER=1 A="$SRC" B="$OUT" VALUES="${VALUES:-full}" bash "$CONVDIR/verify.sh" 2>&1 | tee "$LOGDIR/verify_tp${TP}_iter${ITER}.log"
  echo "[reshard] DONE: $OUT (iter_0000000; TP=$TP EP=$EP ETP=$ETP)"
else
  echo "[reshard] node_rank=$NODE_RANK reshard 完成; 校验由 node0 执行"
fi
