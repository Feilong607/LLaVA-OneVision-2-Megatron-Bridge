#!/usr/bin/env bash
# =============================================================================
# LLaVA-OneVision-2.1 4B (p16m33) · Stage-2 SFT · true vit + adapter (Muon) · TWO-NODE
# 16 GPU = A100-22 + A100-26. Bridge-native (run_recipe.py) + torchrun multi-node rendezvous.
#
# RUN THE SAME SCRIPT ON BOTH NODES (start within ~a couple minutes of each other). Each node
# auto-detects its node_rank from its own IP vs LIST_IP (node-0 = rendezvous master), mirroring
# the AIAK date0523 LIST_IP/NODE_RANK pattern. Bridge's own examples are single-node torchrun
# (examples/.../qwen3_vl/sft.sh: `python -m torch.distributed.run --nproc_per_node=8`); this adds
# the standard --nnodes/--node_rank/--master_addr/--master_port for 2 nodes.
#
# Recipe/optimizer/data identical to the single-node ax_stage_2_alignment_p16m3_muon_vit_adapter.sh:
#   Muon(momentum0.95, ns-steps5, matched-adamw-rms0.2)+AdamW(0.9,0.99,eps1e-5) · lr 2e-5 CONSTANT ·
#   clip1 · wd0 · gbs 128 (16 GPU -> 8 microbatches/GPU) · LLaVA-Next 780k (MultiMixQASample) ·
#   bf16 · 6094 steps · freeze LLM, TRAIN vision_model + m33 adapter.
# Init: trained stage-1 ckpt MODEL-ONLY via checkpoint.pretrained_checkpoint (INIT_CKPT, shared NFS).
#
# PREREQUISITES:
#   1. recipes/ov2/ov2.py defines `ov2_1_stage2_vit_adapter_muon_config` (registered in __init__).
#   2. mbridge:qwen35-muon present on BOTH nodes.
#   3. INIT_CKPT = a trained stage-1 output dir on shared /ov2 (run the stage-1 script first).
#   4. SAVE on shared NFS so all 16 DP ranks write the torch_dist + dataloader state to one dir.
# =============================================================================
set -euo pipefail
REPO="${REPO:-/ov2/feilong/gb200/Megatron-Bridge}"
IMAGE="${IMAGE:-mbridge:qwen35-muon}"
NPROC="${NPROC:-8}"                          # GPUs per node
DATA_PATH="${DATA_PATH:-/vlm/data/llava_next_full_mega}"
INIT_CKPT="${INIT_CKPT:-/ov2/feilong/gb200/results/ov2_1_stage1_native}"   # trained stage-1 (model-only load)
SAVE="${SAVE:-/ov2/feilong/gb200/ckpts_video_sft/llava_ov2_4b_stage2_2node}"   # mirror stage-1 layout
ITERS="${ITERS:-6094}"          # 1 epoch over 780k @ gbs 128
LOG_EVERY="${LOG_EVERY:-1}"; SAVE_EVERY="${SAVE_EVERY:-500}"
MASTER_PORT="${MASTER_PORT:-26016}"

# ---- NCCL / InfiniBand tuning (from the AIAK multi-node reference for this cluster) ----
# The IB data NICs are mlx5_1..mlx5_8 (GPU-attached); mlx5_0 is the management NIC and must NOT be
# used for NCCL. All overridable; if inter-node NCCL hangs at init, set NCCL_IB_DISABLE=1 to fall
# back to eth0 sockets (slower but no IB fabric dependency).
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
  echo "[ov2-2node] ERROR: this host IP ($CURRENT_IP) is not in LIST_IP (${list_ip[*]}); run on a listed node." >&2
  exit 1
fi
echo "[ov2-2node] NNODES=$NNODES node_rank=$NODE_RANK master=$MASTER_ADDR:$MASTER_PORT current=$CURRENT_IP world=$((NNODES*NPROC))"
# Fail fast if INIT_CKPT is set but missing (stage-2 inits from a TRAINED stage-1 ckpt).
# Use INIT_CKPT=null to skip the pretrained load (from-scratch / stitch-base smoke).
if [[ "$INIT_CKPT" != "null" && -n "$INIT_CKPT" && ! -e "$INIT_CKPT" ]]; then
  echo "[ov2-2node] ERROR: INIT_CKPT=$INIT_CKPT not found. Run stage-1 first and set INIT_CKPT to its output dir, or set INIT_CKPT=null to skip the pretrained load." >&2
  exit 1
fi
mkdir -p "$SAVE"
docker rm -f ov2_s2_2node 2>/dev/null || true
# --network=host: torchrun rendezvous + cross-node NCCL use the host net (bootstrap pinned to eth0
# via NCCL_SOCKET_IFNAME; allreduce over IB mlx5_1..8). --cap-add IPC_LOCK + --ulimit memlock=-1 are
# required for IB memory pinning. CUDA_DEVICE_MAX_CONNECTIONS=1 per Megatron. All NCCL/IB values are
# the AIAK multi-node reference for this cluster (overridable via env).
# --privileged: NVML init on these hosts; --gpus all (the daemon rejects --gpus "device=...").
docker run -d --name ov2_s2_2node --privileged --cap-add IPC_LOCK --gpus all --network=host --ipc=host --shm-size=32g \
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
     --recipe ov2_1_stage2_vit_adapter_muon_config --dataset vlm-energon --step_func ov2_step \
     dataset.path=$DATA_PATH \
     checkpoint.save=$SAVE checkpoint.load=$SAVE checkpoint.pretrained_checkpoint=$INIT_CKPT \
     dataset.dataloader_save=$SAVE/dataloader logger.tensorboard_dir=$SAVE/tensorboard \
     checkpoint.save_interval=$SAVE_EVERY train.train_iters=$ITERS \
     validation.eval_iters=0 logger.log_interval=$LOG_EVERY \
     > $SAVE/train_node${NODE_RANK}.log 2>&1"
echo "[ov2-2node] launched node $NODE_RANK/$((NNODES-1)) ($NPROC GPUs) -> tail -f $SAVE/train_node${NODE_RANK}.log"
echo "[ov2-2node] NOW RUN THE SAME SCRIPT ON THE OTHER NODE(S): ${list_ip[*]}"
echo "[ov2-2node] NOTE: Megatron prints 'iteration | lm loss' on the LAST rank -> see train_node$((NNODES-1)).log (node ${list_ip[$((NNODES-1))]}); lower-rank nodes show only 'Step Time' lines."
