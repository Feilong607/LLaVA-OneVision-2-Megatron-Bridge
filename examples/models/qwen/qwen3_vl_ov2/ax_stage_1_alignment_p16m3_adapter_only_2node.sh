#!/usr/bin/env bash
# =============================================================================
# LLaVA-OneVision-2.1 4B (p16m33) · Stage-1 Alignment · adapter-only · TWO-NODE
# 16 GPU = A100-22 + A100-26. Bridge-native (run_recipe.py) + torchrun multi-node rendezvous.
#
# RUN THE SAME SCRIPT ON BOTH NODES (start within ~a couple minutes of each other). Each node
# auto-detects its node_rank from its own IP vs LIST_IP (node-0 = rendezvous master), mirroring
# the AIAK LIST_IP/NODE_RANK pattern and the stage-2 2-node launcher.
#
# Recipe/optimizer/data identical to the single-node ax_stage_1_alignment_p16m3_adapter_only.sh:
#   AdamW(0.9,0.99,eps1e-5,wd0) · lr 2e-5 -> cosine -> 1e-6 (warmup-frac 0.002) · clip 1.0 ·
#   gbs 256 (16 GPU -> 16 microbatches/GPU) · blip_laion_cc_sbu_558k (MultiMixQASample, non-packed) ·
#   token-weighted loss · bf16 · 2181 steps (1 epoch / 558k) · FREEZE LLM + vision, TRAIN m33 adapter.
#   Adam (NOT Muon) -> image is mbridge:qwen35 (no emerging_optimizers needed).
# Init: OV2.1 base mcore ckpt (LLM + vision) via the provider's stitch pre_wrap_hook; adapter trained
#   fresh. NO pretrained_checkpoint (stage-1 is the FIRST stage; nothing upstream to chain from).
#
# PREREQUISITES:
#   1. recipes/ov2/ov2.py defines `ov2_1_stage1_adapter_only_config` (registered in __init__).
#   2. mbridge:qwen35 present on BOTH nodes.
#   3. SAVE on shared NFS so all 16 DP ranks write the torch_dist + dataloader state to one dir.
# =============================================================================
set -euo pipefail
REPO="${REPO:-/ov2/feilong/gb200/Megatron-Bridge}"
IMAGE="${IMAGE:-mbridge:qwen35}"             # Adam adapter-only; Muon (emerging_optimizers) NOT needed
NPROC="${NPROC:-8}"                          # GPUs per node
DATA_PATH="${DATA_PATH:-/vlm/data/blip_laion_cc_sbu_558k_wds}"
SAVE="${SAVE:-/ov2/feilong/gb200/ckpts_video_sft/llava_ov2_4b_stage1_blip_2node}"   # mirror single-node layout
ITERS="${ITERS:-2181}"          # 1 epoch over 558k @ gbs 256
LOG_EVERY="${LOG_EVERY:-1}"; SAVE_EVERY="${SAVE_EVERY:-500}"
MASTER_PORT="${MASTER_PORT:-26017}"

# ---- NCCL / InfiniBand tuning (AIAK multi-node reference for this cluster; same as stage-2 2-node) ----
# IB data NICs are mlx5_1..mlx5_8 (GPU-attached); mlx5_0 is management and must NOT be used by NCCL.
# All overridable; if inter-node NCCL hangs at init, set NCCL_IB_DISABLE=1 to fall back to eth0 sockets.
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"          # bootstrap/control plane (avoid docker0)
NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_1}"
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-2}"             # GPUDirect RDMA
NCCL_IB_QPS_PER_CONNECTION="${NCCL_IB_QPS_PER_CONNECTION:-4}"
NCCL_IB_TC="${NCCL_IB_TC:-160}"
NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT:-22}"
NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-1}"
NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-8}"
NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-16}"
NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"

# ---- node list: node-0 first = rendezvous master. Override: LIST_IP="ip0 ip1 ...". ----
if [[ -n "${LIST_IP:-}" ]]; then read -ra list_ip <<< "$LIST_IP"; else list_ip=(172.16.5.22 172.16.5.26); fi
NNODES=${#list_ip[@]}
MASTER_ADDR="${list_ip[0]}"
CURRENT_IP="$(hostname -I | awk '{print $1}')"
NODE_RANK=-1
for i in "${!list_ip[@]}"; do
  if [[ "${list_ip[$i]}" == "$CURRENT_IP" ]]; then NODE_RANK=$i; break; fi
done
if [[ "$NODE_RANK" -eq -1 ]]; then
  echo "[ov2-s1-2node] ERROR: this host IP ($CURRENT_IP) is not in LIST_IP (${list_ip[*]}); run on a listed node." >&2
  exit 1
fi
echo "[ov2-s1-2node] NNODES=$NNODES node_rank=$NODE_RANK master=$MASTER_ADDR:$MASTER_PORT current=$CURRENT_IP world=$((NNODES*NPROC))"
mkdir -p "$SAVE"
docker rm -f ov2_s1_2node 2>/dev/null || true
# --network=host: torchrun rendezvous + cross-node NCCL use the host net (bootstrap pinned to eth0
# via NCCL_SOCKET_IFNAME; allreduce over IB mlx5_1..8). --cap-add IPC_LOCK + --ulimit memlock=-1 are
# required for IB memory pinning. CUDA_DEVICE_MAX_CONNECTIONS=1 per Megatron. All NCCL/IB values are
# the AIAK multi-node reference for this cluster (overridable via env).
# --privileged: NVML init on these hosts; --gpus all (the daemon rejects --gpus "device=...").
docker run -d --name ov2_s1_2node --privileged --cap-add IPC_LOCK --gpus all --network=host --ipc=host --shm-size=32g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e PYTHONPATH="$REPO/src:$REPO/3rdparty/Megatron-LM:$REPO/aiak_shim" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e OMP_NUM_THREADS=8 -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
  -e NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME" -e GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}" \
  -e NCCL_IB_DISABLE="$NCCL_IB_DISABLE" -e NCCL_IB_HCA="$NCCL_IB_HCA" -e NCCL_IB_GID_INDEX="$NCCL_IB_GID_INDEX" \
  -e NCCL_NET_GDR_LEVEL="$NCCL_NET_GDR_LEVEL" -e NCCL_IB_QPS_PER_CONNECTION="$NCCL_IB_QPS_PER_CONNECTION" \
  -e NCCL_IB_TC="$NCCL_IB_TC" -e NCCL_IB_TIMEOUT="$NCCL_IB_TIMEOUT" -e NCCL_CROSS_NIC="$NCCL_CROSS_NIC" \
  -e NCCL_MIN_NCHANNELS="$NCCL_MIN_NCHANNELS" -e NCCL_MAX_NCHANNELS="$NCCL_MAX_NCHANNELS" -e NCCL_TIMEOUT="$NCCL_TIMEOUT" \
  -v /ov2:/ov2 -v /vlm:/vlm -w "$REPO" \
  "$IMAGE" bash -lc "python -m torch.distributed.run \
     --nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
     --nproc_per_node=$NPROC scripts/training/run_recipe.py \
     --recipe ov2_1_stage1_adapter_only_config --dataset vlm-energon --step_func ov2_step \
     dataset.path=$DATA_PATH \
     checkpoint.save=$SAVE checkpoint.load=$SAVE \
     dataset.dataloader_save=$SAVE/dataloader logger.tensorboard_dir=$SAVE/tensorboard \
     checkpoint.save_interval=$SAVE_EVERY train.train_iters=$ITERS \
     validation.eval_iters=0 logger.log_interval=$LOG_EVERY \
     > $SAVE/train_node${NODE_RANK}.log 2>&1"
echo "[ov2-s1-2node] launched node $NODE_RANK/$((NNODES-1)) ($NPROC GPUs) -> tail -f $SAVE/train_node${NODE_RANK}.log"
echo "[ov2-s1-2node] NOW RUN THE SAME SCRIPT ON THE OTHER NODE(S): ${list_ip[*]}"
echo "[ov2-s1-2node] NOTE: Megatron prints 'iteration | lm loss' on the LAST rank -> see train_node$((NNODES-1)).log (node ${list_ip[$((NNODES-1))]}); lower-rank nodes show only 'Step Time' lines."
