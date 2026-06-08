#!/usr/bin/env bash
# =============================================================================
# OV2-30B-A3B · GB200 · SINGLE-NODE QUICK-START SMOKE (4 GPU) · IN-CONTAINER
# Fast sanity check on ONE GB200 node before the real 2-4 node run. Validates:
# container is Blackwell-capable, recipe builds, energon data loads, fwd+bwd runs
# (no NaN, grad-norm sane), MFU prints, a checkpoint saves. ~20 iters, tiny config.
#
# *** EP CAVEAT — READ THIS ***
# 30B-A3B's VERIFIED MoE config is EP=8 (needs 8 GPU). One node = 4 GPU can only do EP=4,
# which OV2 has NOT validated (ov2_provider: "only EP=8/ETP=1 verified"), and the base ckpt
# is EP8 so it must reshard EP8->EP4. If this smoke errors on MoE / expert load / stitch,
# that is an EP4 limitation, NOT your GB200 setup.
# >>> The REPRESENTATIVE smoke is 2 nodes (8 GPU, EP8) on the MAIN script:
#       ITERS=20 SAVE_EVERY=10 LIST_IP="<ip0> <ip1>" bash ax_ov2_30b_a3b_gb200.sh
# This single-node script is only the lightest "does the container/code/data work" check.
# =============================================================================
set -euo pipefail

REPO="${REPO:-$({ __d="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; while [[ "$__d" != "/" && ! -d "$__d/src/megatron/bridge" ]]; do __d="$(dirname "$__d")"; done; echo "$__d"; })}"
[[ -d "$REPO/src/megatron/bridge" ]] || { echo "FATAL: OV2 fork root not found from ${BASH_SOURCE[0]} (no src/megatron/bridge above it). Set REPO=/path/to/LLaVA-OneVision-2-Megatron-Bridge" >&2; exit 1; }
RECIPE="${RECIPE:-ov2_35b_a3b_midtrain}"
INIT_CKPT="${INIT_CKPT:-null}"     # smoke: skip the trained-stage load (base ckpt is still stitched)
# --- CARD PATH PROFILE (smoke): A100 (/ov2) <-> GB200 (/datasets); INIT_CKPT stays null. ---
if [[ "${HWNAME:-}" == "gb200" || -d /datasets/qwen-models-ea5jyi ]]; then
  OV2_LLM_HF_30B="${OV2_LLM_HF_30B:-/datasets/qwen-models-ea5jyi/Qwen3-30B-A3B-Instruct-2507}"
  OV2_PRETRAIN_ROOT="${OV2_PRETRAIN_ROOT:-/datasets/llava/11May}"      # TODO-GB200
  DATA_PATH="${DATA_PATH:-$REPO/examples/models/qwen/qwen3_vl_ov2/gb200/mid_training_seed85m.yaml}"
  SAVE="${SAVE:-/home/ftan0055/ckpts_video_sft/ov2_30b_a3b_gb200_smoke}"
else
  OV2_LLM_HF_30B="${OV2_LLM_HF_30B:-/ov2/pretrain_models/Qwen3-30B-A3B-Instruct-2507}"
  OV2_PRETRAIN_ROOT="${OV2_PRETRAIN_ROOT:-/ov2/pretrain_models}"
  DATA_PATH="${DATA_PATH:-$REPO/examples/models/qwen/qwen3_vl_ov2/mid_training_seed85m.yaml}"
  SAVE="${SAVE:-/ov2/feilong/gb200/ckpts_video_sft/ov2_30b_a3b_gb200_smoke}"
fi
export OV2_LLM_HF_30B OV2_PRETRAIN_ROOT
NPROC="${NPROC:-4}"                # single node = 4 GPU

# --- smoke-sized overrides (tiny + fast + memory-safe on 4 GPU) ---
EP="${EP:-4}"                      # forced: 4 GPU cannot do EP8 (EP4 is UNVALIDATED for OV2)
SEQLEN="${SEQLEN:-4096}"           # vs 32000 real -> cut activations so it fits 4 GPU
GBS="${GBS:-4}"                    # = dense DP (4): 1 sample/rank, no grad-accum
MBS="${MBS:-1}"
ITERS="${ITERS:-20}"; SAVE_EVERY="${SAVE_EVERY:-10}"; LOG_EVERY="${LOG_EVERY:-1}"

# --- env (in-container; bf16 baseline + known 30B-A3B stability fixes) ---
export MFU_PEAK_TFLOPS="${MFU_PEAK_TFLOPS:-2250}"               # GB200 bf16; set 312 on A800
export OV2_RECOMPUTE_FULL="${OV2_RECOMPUTE_FULL:-1}"            # full/uniform/1 recompute (AIAK + mem-safe)
export OV2_MOE_PERMUTE_FUSION="${OV2_MOE_PERMUTE_FUSION:-0}"   # avoid TE Triton MoE-permute wedge
export PYTHONPATH="$REPO/src:$REPO/3rdparty/Megatron-LM:$REPO/aiak_shim${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-1}"

OVERRIDES="dataset.path=$DATA_PATH"
[[ "$INIT_CKPT" != "null" && -n "$INIT_CKPT" ]] && OVERRIDES="$OVERRIDES checkpoint.pretrained_checkpoint=$INIT_CKPT"
OVERRIDES="$OVERRIDES checkpoint.save=$SAVE checkpoint.load=$SAVE dataset.dataloader_save=$SAVE"
OVERRIDES="$OVERRIDES checkpoint.save_interval=$SAVE_EVERY train.train_iters=$ITERS validation.eval_iters=0 logger.log_interval=$LOG_EVERY"
OVERRIDES="$OVERRIDES scheduler.lr_warmup_iters=0 mixed_precision=bf16_mixed"
OVERRIDES="$OVERRIDES model.expert_model_parallel_size=$EP"
OVERRIDES="$OVERRIDES model.seq_length=$SEQLEN dataset.seq_length=$SEQLEN"
OVERRIDES="$OVERRIDES train.global_batch_size=$GBS train.micro_batch_size=$MBS"

mkdir -p "$SAVE"; cd "$REPO"
echo "[ov2-30b-smoke] 1 node x $NPROC GPU | EP=$EP seq=$SEQLEN gbs=$GBS mbs=$MBS iters=$ITERS init=$INIT_CKPT peak=${MFU_PEAK_TFLOPS}"
echo "[ov2-30b-smoke] NOTE EP=$EP is UNVALIDATED for OV2 (verified=EP8). If it dies on MoE/ckpt -> smoke on 2 nodes (EP8): ITERS=20 LIST_IP=\"ip0 ip1\" bash ax_ov2_30b_a3b_gb200.sh"
# shellcheck disable=SC2086  # $OVERRIDES must word-split
python -m torch.distributed.run --standalone --nproc_per_node="$NPROC" scripts/training/run_recipe.py \
  --recipe "$RECIPE" --dataset vlm-energon --step_func ov2_step \
  $OVERRIDES 2>&1 | tee "$SAVE/smoke.log"
