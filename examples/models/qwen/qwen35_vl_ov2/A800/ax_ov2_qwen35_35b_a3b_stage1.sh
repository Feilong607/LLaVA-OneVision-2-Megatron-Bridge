#!/usr/bin/env bash
# =============================================================================
# OV2 · Qwen3.5-35B-A3B (qwen3_5_moe_text: GatedDeltaNet hybrid + 256-expert MoE + MTP)
#       + OneVision p16m33 encoder + merge3 adapter · STAGE-1 alignment
# Adapter-only (LLM + vision FROZEN), AdamW, EP8. Bridge-native: run_recipe.py + ov2_qwen35_35b_a3b_stage1.
# Single node (8 GPU, --standalone) OR multi-node (set LIST_IP). Parallel to qwen3_vl_ov2/A800 -- the
# Qwen3.5 stack is kept fully separate from the Qwen3-30B stack (do NOT cross recipes/ckpts).
#
# GPU PASSTHROUGH (this box): A100-18 / A800-80GB docker `--gpus` hook is BROKEN (new containers get
# NVML/error-100, no CUDA). Verified-working pattern here = `--privileged -v /dev:/dev`. Override with
# GPU_ARGS="--gpus all" on a box whose hook works.
#
# >>> DEPENDENCIES still to build (this launcher already targets them by name) <<<
#   [ ] recipe   ov2_qwen35_35b_a3b_stage1   -> src/megatron/bridge/recipes/ov2/ov2_qwen35.py
#   [ ] WEIGHTS  Qwen3.5-35B-A3B-text re-keyed safetensors:
#                  python qwen35_vl_ov2/tools/extract_qwen35_text.py --weights
#   [ ] INIT     stage_0 combined base (qwen3.5 text LLM + OneVision p16m33 tower + fresh merge3 adapter,
#                  torch_dist) -> built via a qwen35_vl_ov2 convert step
#   [x] PROC     qwen3.5 p16m33 processor (Qwen3.5 tokenizer, image_token_id=248056) -- BUILT (auto_model)
# GATES already PASSED: LLM build (35.51B, GDN30+MTP) + OneVisionEncoder stitch (vision303M+adapter104M).
# =============================================================================
set -euo pipefail

REPO="${REPO:-/ov2/feilong/gb200/Megatron-Bridge}"
bash "$REPO/3rdparty/apply_megatron_patch.sh"   # OV2 mcore patch (apply_rotary_fn) -- REQUIRED by the OneVision vision tower (FAIL LOUD: missing -> "unexpected keyword argument 'apply_rotary_fn'")
IMAGE="${IMAGE:-mbridge:qwen35}"
DATA_PATH="${DATA_PATH:-/vlm/data/blip_laion_cc_sbu_558k_wds}"   # 558k alignment data (verify mount on this box)
SAVE="${SAVE:-/ov2/feilong/gb200/ckpts_video_sft/ov2_qwen35_35b_a3b_p16m33_stage1}"
# stage_0 combined base: qwen3.5 text LLM + p16m33 OneVision tower + fresh merge3 adapter (torch_dist).
# Loaded via pretrained_checkpoint with OV2_SKIP_BASE_STITCH=1 (skip the AIAK .pt release/mp_rank stitch).
INIT="${INIT:-/ov2/pretrain_models/llava_onevision2/llava_onevision2_qwen35_35b_a3b_p16_m33/stage_0_tp1_pp1_ep8}"
# qwen3.5 p16m33 processor (image/video processor + Qwen3.5 tokenizer, image_token_id=248056).
OV2_HF_PROC_QWEN35_P16M33="${OV2_HF_PROC_QWEN35_P16M33:-/ov2/pretrain_models/llava_onevision2/llava_onevision2_qwen35_35b_a3b_p16_m33/auto_model}"
NPROC="${NPROC:-8}"
ITERS="${ITERS:-2181}"                          # 1 epoch over 558k @ gbs 256
LOG_EVERY="${LOG_EVERY:-1}"; SAVE_EVERY="${SAVE_EVERY:-250}"
# A100-18/A800 docker --gpus hook is broken -> privileged + raw /dev + the host NVML lib. NCCL needs
# libnvidia-ml.so.1 for multi-GPU topology; without it -> ncclSystemError "Failed to open libnvidia-ml.so.1".
# The broken --gpus toolkit ships 0-byte stub driver libs in the image; mount the REAL host driver libs
# (libcuda for cuCtxSetCurrent / NCCL / Triton-PTX-JIT) over them. Globs the host driver version dynamically.
_NVLIBS=""
for _f in /usr/lib/x86_64-linux-gnu/libcuda.so.*.* /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.*.* /usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.*.* /usr/lib/x86_64-linux-gnu/libnvidia-nvvm.so.*.* /usr/lib/x86_64-linux-gnu/libcudadebugger.so.*.*; do
  [[ -s "$_f" ]] && _NVLIBS="$_NVLIBS -v $_f:$_f:ro"
done
GPU_ARGS="${GPU_ARGS:---privileged -v /dev:/dev $_NVLIBS}"   # A100-18 workaround; set GPU_ARGS='--gpus all' where the hook works

# node list: node-0 first = rendezvous master. Single node => leave LIST_IP unset (uses --standalone).
if [[ -n "${LIST_IP:-}" ]]; then read -ra list_ip <<< "$LIST_IP"; else list_ip=(); fi
NNODES=${#list_ip[@]}
if [[ "$NNODES" -le 1 ]]; then
  RDZV="--standalone"
else
  MASTER_ADDR="${list_ip[0]}"; MASTER_PORT="${MASTER_PORT:-26041}"
  CURRENT_IP="$(hostname -I | awk '{print $1}')"; NODE_RANK=-1
  for i in "${!list_ip[@]}"; do [[ "${list_ip[$i]}" == "$CURRENT_IP" ]] && NODE_RANK=$i && break; done
  [[ "$NODE_RANK" -eq -1 ]] && { echo "ERROR: $CURRENT_IP not in LIST_IP (${list_ip[*]})"; exit 1; }
  RDZV="--nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT"
fi

# multi-node IB fabric (mlx5_1..8 GPU-attached; mlx5_0 is mgmt). Only used when LIST_IP is set.
NCCL_ENV="-e NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0} \
  -e NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8} \
  -e NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3} -e NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}"

# stage-1 freezes LLM+vision, but backprop still traverses all 40 LLM layers to reach the adapter,
# so at the packed seq length the attention activations need recompute to fit 80GB. SELECTIVE core_attn
# by default; OV2_RECOMPUTE_FULL=1 forces full-layer recompute. (moe_permute_fusion stays OFF.)
RECOMPUTE_FLAG=""; [[ "${RECOMPUTE:-1}" == "1" ]] && RECOMPUTE_FLAG="model.recompute_activations=true"
mkdir -p "$SAVE"; docker rm -f ov2_qwen35_s1 2>/dev/null || true
echo "[ov2-qwen35-stage1] nnodes=${NNODES:-1} save=$SAVE repo=$REPO image=$IMAGE gpu_args='$GPU_ARGS'"
# shellcheck disable=SC2086
docker run -d --name ov2_qwen35_s1 $GPU_ARGS --network=host --ipc=host --shm-size=32g \
  --ulimit memlock=-1 --ulimit stack=67108864 $NCCL_ENV \
  -e PYTHONPATH="$REPO/src:$REPO/3rdparty/Megatron-LM:$REPO/aiak_shim" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e OMP_NUM_THREADS=8 \
  -e OV2_SKIP_BASE_STITCH=1 -e OV2_INIT_CKPT="$INIT" \
  -e OV2_HF_PROC_QWEN35_P16M33="$OV2_HF_PROC_QWEN35_P16M33" \
  -e OV2_MOE_PERMUTE_FUSION="${OV2_MOE_PERMUTE_FUSION:-0}" \
  -e OV2_MTP_LOSS_SCALE="${OV2_MTP_LOSS_SCALE:-}" \
  -e OV2_FREEZE_VISION="${OV2_FREEZE_VISION:-}" \
  -e OV2_RECOMPUTE_FULL="${OV2_RECOMPUTE_FULL:-0}" \
  -v /ov2:/ov2 -v /vlm:/vlm -w "$REPO" "$IMAGE" bash -lc "
    bash 3rdparty/apply_megatron_patch.sh;
    python -m torch.distributed.run $RDZV --nproc_per_node=$NPROC scripts/training/run_recipe.py \
      --recipe ov2_qwen35_35b_a3b_stage1 --dataset vlm-energon --step_func ov2_step \
      dataset.path=$DATA_PATH $RECOMPUTE_FLAG \
      checkpoint.pretrained_checkpoint=$INIT \
      checkpoint.save=$SAVE checkpoint.load=$SAVE dataset.dataloader_save=$SAVE \
      checkpoint.save_interval=$SAVE_EVERY train.train_iters=$ITERS \
      validation.eval_iters=0 logger.log_interval=$LOG_EVERY \
      > $SAVE/train_node${NODE_RANK:-0}.log 2>&1"
echo "[ov2-qwen35-stage1] launched -> tail -f $SAVE/train_node*.log"
