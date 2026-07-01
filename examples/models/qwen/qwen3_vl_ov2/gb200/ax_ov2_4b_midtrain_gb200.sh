#!/usr/bin/env bash
# =============================================================================
# OV2.1 4B (Qwen3-4B dense) · MID-TRAIN · FULL model (LLM + vision + adapter)
# GB200/Blackwell · IN-CONTAINER launcher (the dense sibling of ax_ov2_30b_a3b_gb200.sh).
# Run this INSIDE the training container (you are ALREADY in docker on GB200) -- it only assembles env +
# run_recipe overrides and execs torchrun; NO `docker run` wrapper.
#
# HW auto-detected (GB200 sm_100 = 4 GPU/node | A100/A800 sm_80 = 8 GPU/node); override HW=gb200|a100.
# The SAME script runs on A-cards too -- only NPROC, the path profile, and the recompute default switch.
#
# 4B is DENSE (no MoE) -> NO EP / HybridEP / fp8-expert / ACCEL modes; dense midtrain keeps Muon (the recipe
# default; the is_moe->AdamW auto-route does NOT fire for 4B). MEMORY: 4B+Muon at TP1 holds Muon's ~56GB
# UNSHARDED fp32 state (Muon can't CPU-offload). GB200 192GB fits it WITHOUT recompute -> recompute OFF by
# default (faster); A100/A800 80GB needs OV2_RECOMPUTE_FULL=1 (auto-on below). Both env-overridable.
#
# GB200 PATHS: stage the 4B base+processor+init ckpt on the box and set OV2_PRETRAIN_ROOT / INIT_CKPT / DATA_PATH
# (the recipe's qwen3-4b backbone reads OV2_PRETRAIN_ROOT; its stage-1 adapter default is an /vlm A100 path,
# so on GB200 use a staged stage-2 INIT_CKPT + OV2_SKIP_BASE_STITCH=1 instead of the base stitch).
# =============================================================================
set -euo pipefail
_SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$({ __d="$_SELF"; while [[ "$__d" != "/" && ! -d "$__d/src/megatron/bridge" ]]; do __d="$(dirname "$__d")"; done; echo "$__d"; })}"
[[ -d "$REPO/src/megatron/bridge" ]] || { echo "FATAL: OV2 repo root not found from $_SELF (set REPO=)" >&2; exit 1; }
bash "$REPO/3rdparty/apply_megatron_patch.sh"   # OV2 mcore patch (apply_rotary_fn hook); idempotent, fail-loud
RECIPE="${RECIPE:-ov2_4b_midtrain}"

# --- HW profile (GB200 sm_100 / Hopper sm_90 / Ampere sm_80). Sets NPROC + the recompute default only. ---
HW="${HW:-auto}"
case "$HW" in
  gb200) _cc=100;; h100|hopper) _cc=90;; a100|a800|ampere) _cc=80;;
  auto) _cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')";;
  *) _cc="";;
esac
_cc="${_cc:-80}"; [[ "$_cc" =~ ^[0-9]+$ ]] || _cc=80
if [[ "$_cc" -ge 100 ]]; then HWNAME=gb200; HW_NPROC=4; HW_NVLS=1; else HWNAME=ampere; HW_NPROC=8; HW_NVLS=0; fi
[[ "$_cc" -ge 90 && "$_cc" -lt 100 ]] && { HWNAME=hopper; HW_NPROC=8; HW_NVLS=1; }
NPROC="${NPROC:-$HW_NPROC}"
TP="${TP:-1}"; if [[ "$TP" -gt 1 ]]; then SP=true; else SP=false; fi
SEQ_LEN="${OV2_SEQ_LEN:-10192}"
# recompute: GB200 192GB fits 4B+Muon WITHOUT recompute (fast); A-cards 80GB need full recompute.
if [[ "$HWNAME" == "gb200" ]]; then OV2_RECOMPUTE_FULL="${OV2_RECOMPUTE_FULL:-0}"; else OV2_RECOMPUTE_FULL="${OV2_RECOMPUTE_FULL:-1}"; fi
RECOMPUTE_ARGS=""; [[ "$OV2_RECOMPUTE_FULL" == "1" ]] && RECOMPUTE_ARGS="model.recompute_granularity=full model.recompute_method=uniform model.recompute_num_layers=1"
OV2_VISION_RECOMPUTE="${OV2_VISION_RECOMPUTE:-$OV2_RECOMPUTE_FULL}"

# --- CARD PATH PROFILE: GB200 (/datasets + /home) <-> A100 (/ov2 + /vlm). All env-overridable. ---
if [[ "$HWNAME" == "gb200" || -d /datasets ]]; then
  OV2_PRETRAIN_ROOT="${OV2_PRETRAIN_ROOT:-/datasets/pretrain_models}"   # stage the 4B base/proc under here (Qwen3-4B-Instruct-2507 + lmms-lab/LLaVA-OneVision-2-4B-p16m33[-mcore-tp1-pp1])
  DATA_PATH="${DATA_PATH:-$REPO/examples/models/qwen/qwen3_vl_ov2/gb200/mid_training_seed85m.yaml}"
  INIT_CKPT="${INIT_CKPT:-null}"                        # GB200: set a STAGED 4B stage-2 ckpt (the /vlm stage-1 adapter default is A100-only) + OV2_SKIP_BASE_STITCH=1
  SAVE="${SAVE:-/home/ftan0055/ckpts_video_sft/ov2_4b_p16m33_midtrain}"
  OV2_SKIP_BASE_STITCH="${OV2_SKIP_BASE_STITCH:-1}"     # GB200: skip the /vlm stage-1 stitch -> weights come from INIT_CKPT
else
  OV2_PRETRAIN_ROOT="${OV2_PRETRAIN_ROOT:-/ov2/pretrain_models}"
  DATA_PATH="${DATA_PATH:-$REPO/examples/models/qwen/qwen3_vl_ov2/A800/mid_training_seed85m.yaml}"
  INIT_CKPT="${INIT_CKPT:-null}"                        # A100: null -> base stitch (4B mcore_ckpt + /vlm stage-1 adapter); or set a trained stage-2
  SAVE="${SAVE:-/ov2/feilong/gb200/ckpts_video_sft/ov2_4b_p16m33_midtrain}"
  OV2_SKIP_BASE_STITCH="${OV2_SKIP_BASE_STITCH:-0}"
fi
export OV2_PRETRAIN_ROOT OV2_SKIP_BASE_STITCH
[[ "$INIT_CKPT" != "null" && -n "$INIT_CKPT" ]] && export OV2_INIT_CKPT="$INIT_CKPT"
if [[ "$INIT_CKPT" != "null" && -n "$INIT_CKPT" && ! -e "$INIT_CKPT" ]]; then
  echo "[ov2-4b-gb200] ERROR: INIT_CKPT=$INIT_CKPT not found. Stage a 4B stage-2 ckpt, or (A100) set INIT_CKPT=null for the base stitch." >&2; exit 1
fi
if [[ "$HWNAME" == "gb200" && "$OV2_SKIP_BASE_STITCH" == "1" && ( "$INIT_CKPT" == "null" || -z "$INIT_CKPT" ) ]]; then
  echo "[ov2-4b-gb200] FATAL: GB200 skips the /vlm stage-1 base-stitch (OV2_SKIP_BASE_STITCH=1) but INIT_CKPT is null -> the model would train on RANDOM weights. Set INIT_CKPT=<staged 4B stage-2 ckpt>, or OV2_SKIP_BASE_STITCH=0 with the base staged under OV2_PRETRAIN_ROOT + /vlm." >&2; exit 1
fi

MIDTRAIN_GBS="${OV2_MIDTRAIN_GBS:-16}"
MIDTRAIN_N_SAMPLES="${OV2_MIDTRAIN_N_SAMPLES:-780000}"
ITERS="${ITERS:-$(( (MIDTRAIN_N_SAMPLES + MIDTRAIN_GBS - 1) / MIDTRAIN_GBS ))}"
LOG_EVERY="${LOG_EVERY:-1}"; SAVE_EVERY="${SAVE_EVERY:-500}"; WARMUP_ITERS="${OV2_WARMUP_ITERS:-100}"

# --- rendezvous: standalone (1 node) or multi-node (LIST_IP; node_rank auto). 4B fits 1 node easily. ---
if [[ -n "${LIST_IP:-}" ]]; then
  read -ra list_ip <<< "$LIST_IP"; NNODES=${#list_ip[@]}
  MASTER_ADDR="${list_ip[0]}"; MASTER_PORT="${MASTER_PORT:-26046}"
  CURRENT_IP="$(hostname -I | awk '{print $1}')"; NODE_RANK=-1
  for i in "${!list_ip[@]}"; do [[ "${list_ip[$i]}" == "$CURRENT_IP" ]] && NODE_RANK=$i && break; done
  [[ "$NODE_RANK" -eq -1 ]] && { echo "ERROR: $CURRENT_IP not in LIST_IP (${list_ip[*]})" >&2; exit 1; }
  RDZV="--nnodes=$NNODES --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT"
else
  NNODES=1; NODE_RANK=0; RDZV="--standalone"
fi
WORLD=$(( NPROC * NNODES )); DP=$(( WORLD / TP ))
(( MIDTRAIN_GBS % DP == 0 )) || { echo "[ov2-4b-gb200] FATAL: GBS=$MIDTRAIN_GBS not divisible by DP=$DP (WORLD=$WORLD / TP=$TP; mbs=1). Adjust OV2_MIDTRAIN_GBS / NPROC / NNODES." >&2; exit 1; }

# --- in-container env ---
export PYTHONPATH="$REPO/src:$REPO/3rdparty/Megatron-LM:$REPO/aiak_shim${PYTHONPATH:+:$PYTHONPATH}"
# (4B is dense -> no deep_ep/HybridEP; the pip-nvshmem LD prepend is harmless + kept for parity, no-op if absent)
_nvshmem_lib="${OV2_NVSHMEM_LIB:-/usr/local/lib/python3.12/dist-packages/nvidia/nvshmem/lib}"
[[ -e "$_nvshmem_lib/libnvshmem_host.so.3" ]] && export LD_LIBRARY_PATH="$_nvshmem_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OV2_SKIP_HELPERS="${OV2_SKIP_HELPERS:-1}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OV2_SEQ_LEN="$SEQ_LEN" OV2_MIDTRAIN_GBS="$MIDTRAIN_GBS" OV2_MIDTRAIN_N_SAMPLES="$MIDTRAIN_N_SAMPLES"
export OV2_RECOMPUTE_FULL OV2_VISION_RECOMPUTE OV2_PACK_FULL_CAUSAL="${OV2_PACK_FULL_CAUSAL:-0}"
export OV2_MIDTRAIN_MUON="${OV2_MIDTRAIN_MUON:-0}" OV2_MIDTRAIN_ADAMW="${OV2_MIDTRAIN_ADAMW:-0}"   # dense 4B keeps Muon; OV2_MIDTRAIN_ADAMW=1 for AdamW (distopt-sharded, no recompute needed)
export OV2_FREEZE_LLM="${OV2_FREEZE_LLM:-0}" OV2_FREEZE_VISION="${OV2_FREEZE_VISION:-0}" OV2_FREEZE_ADAPTER="${OV2_FREEZE_ADAPTER:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-$HW_NVLS}"
if [[ "$HWNAME" == "gb200" ]]; then
  export NCCL_MNNVL_ENABLE="${NCCL_MNNVL_ENABLE:-1}" NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}" NCCL_NET_GDR_C2C="${NCCL_NET_GDR_C2C:-1}"
fi
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/ov2_triton_cache}"; mkdir -p "$TRITON_CACHE_DIR"

# --- run_recipe.py overrides (dense: NO MoE capacity / router-dtype / EP) ---
PRELOAD=""; [[ "$INIT_CKPT" != "null" && -n "$INIT_CKPT" ]] && PRELOAD="checkpoint.pretrained_checkpoint=$INIT_CKPT"
OVERRIDES="dataset.path=$DATA_PATH $PRELOAD"
OVERRIDES="$OVERRIDES checkpoint.save=$SAVE checkpoint.load=$SAVE dataset.dataloader_save=$SAVE"
OVERRIDES="$OVERRIDES checkpoint.save_interval=$SAVE_EVERY train.train_iters=$ITERS validation.eval_iters=0 logger.log_interval=$LOG_EVERY train.micro_batch_size=1"
OVERRIDES="$OVERRIDES model.tensor_model_parallel_size=$TP model.sequence_parallel=$SP $RECOMPUTE_ARGS"
OVERRIDES="$OVERRIDES scheduler.lr_warmup_iters=$WARMUP_ITERS logger.tensorboard_dir=$SAVE/tensorboard"

mkdir -p "$SAVE"
echo "[ov2-4b-gb200] hw=$HWNAME(cc=$_cc) repo=$REPO recipe=$RECIPE nnodes=$NNODES nproc=$NPROC world=$WORLD dp=$DP tp=$TP seq=$SEQ_LEN gbs=$MIDTRAIN_GBS iters=$ITERS recompute_full=$OV2_RECOMPUTE_FULL init=$INIT_CKPT save=$SAVE (dense, Muon)"
# shellcheck disable=SC2086
python -m torch.distributed.run $RDZV --nproc_per_node="$NPROC" scripts/training/run_recipe.py \
  --recipe "$RECIPE" --dataset vlm-energon --step_func ov2_step \
  $OVERRIDES ${EXTRA_ARGS:-} 2>&1 | tee "$SAVE/train_node${NODE_RANK}.log"
echo "[ov2-4b-gb200] done -> $SAVE/train_node${NODE_RANK}.log"
