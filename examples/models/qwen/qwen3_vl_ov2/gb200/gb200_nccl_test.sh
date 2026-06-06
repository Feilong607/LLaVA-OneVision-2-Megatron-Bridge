#!/usr/bin/env bash
# =============================================================================
# GB200 NCCL collective benchmark (all_reduce + all_to_all). In-container.
#   single node (intra-node NVLink):   NPROC=4 bash .../gb200_nccl_test.sh
#   multi-node  (NVL72 cross-node):    LIST_IP="<ip0> <ip1>" bash .../gb200_nccl_test.sh   (run on EACH node)
# Set NCCL_DEBUG=INFO to dump the topology/transport NCCL chose (confirms NVLink/NVLS vs IB).
# =============================================================================
set -uo pipefail
REPO="${REPO:-/ov2/feilong/gb200/Megatron-Bridge}"
NPROC="${NPROC:-4}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
PROBE="$REPO/examples/models/qwen/qwen3_vl_ov2/gb200/gb200_nccl_probe.py"

if [[ -n "${LIST_IP:-}" ]]; then read -ra list_ip <<< "$LIST_IP"; else list_ip=(); fi
NN=${#list_ip[@]}
if [[ "$NN" -le 1 ]]; then
  RDZV="--standalone"
else
  MASTER_ADDR="${list_ip[0]}"; MASTER_PORT="${MASTER_PORT:-26048}"
  CUR="$(hostname -I | awk '{print $1}')"; NR=-1
  for i in "${!list_ip[@]}"; do [[ "${list_ip[$i]}" == "$CUR" ]] && NR=$i && break; done
  [[ "$NR" -eq -1 ]] && { echo "ERROR: $CUR not in LIST_IP (${list_ip[*]})" >&2; exit 1; }
  RDZV="--nnodes=$NN --node_rank=$NR --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT"
fi
echo "[gb200-nccl] nnodes=${NN:-1} nproc_per_node=$NPROC NVLS=$NCCL_NVLS_ENABLE debug=$NCCL_DEBUG"
python -m torch.distributed.run $RDZV --nproc_per_node="$NPROC" "$PROBE"
