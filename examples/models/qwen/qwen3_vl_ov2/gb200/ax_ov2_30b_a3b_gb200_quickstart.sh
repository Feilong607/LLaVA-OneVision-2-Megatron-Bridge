#!/usr/bin/env bash
# =============================================================================
# OV2-30B-A3B (Qwen3-30B-A3B MoE + p16m33) · SINGLE-NODE QUICK-START SMOKE · IN-CONTAINER
# Fast one-node sanity check before the real 2-4 node run. AUTO-ADAPTS like the main launcher
# (ax_ov2_30b_a3b_gb200.sh): HW (A100/A800 8-GPU sm_80 | Hopper 8-GPU sm_90 | GB200 4-GPU sm_100)
# -> NPROC / MFU-peak / NVLS auto; paths (A100 /ov2 <-> GB200 /datasets) auto; seq/bs/iters cut
# small so it runs fast + fits one node. Validates: recipe builds, energon data loads, fwd+bwd
# runs (no NaN, grad-norm sane), MFU prints, a checkpoint saves.
#
# *** EP CAVEAT — READ THIS ***  30B-A3B's VERIFIED MoE config is EP=8 (needs 8 GPU).
#   - A100/A800 1 node = 8 GPU -> EP8  (the REAL, validated config) — a faithful smoke.
#   - GB200    1 node = 4 GPU -> EP4  (OV2 only validated EP8; the EP8 ckpt reshards EP8->EP4 via
#     torch_dist). If it dies on MoE/expert/ckpt that's the EP4 limit, NOT your GB200 setup ->
#     run the REPRESENTATIVE EP8 smoke on 2 nodes via the MAIN script:
#       ITERS=20 SAVE_EVERY=10 LIST_IP="<ip0> <ip1>" bash ax_ov2_30b_a3b_gb200.sh
# =============================================================================
set -euo pipefail

REPO="${REPO:-$({ __d="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; while [[ "$__d" != "/" && ! -d "$__d/src/megatron/bridge" ]]; do __d="$(dirname "$__d")"; done; echo "$__d"; })}"
[[ -d "$REPO/src/megatron/bridge" ]] || { echo "FATAL: OV2 fork root not found from ${BASH_SOURCE[0]} (no src/megatron/bridge above it). Set REPO=/path/to/LLaVA-OneVision-2-Megatron-Bridge" >&2; exit 1; }
bash "$REPO/3rdparty/apply_megatron_patch.sh"   # fresh-clone safety: OV2 mcore submodule patch (idempotent)
RECIPE="${RECIPE:-ov2_30b_a3b_p16m33_midtrain}"  # p16m33 (matches main launcher). For frozen-LLM SFT: ov2_30b_a3b_p16m33_stage2
ITERS="${ITERS:-20}"; SAVE_EVERY="${SAVE_EVERY:-10}"; LOG_EVERY="${LOG_EVERY:-1}"

# --- HARDWARE PROFILE (same auto-detect as the main launcher) ---
HW="${HW:-auto}"
case "$HW" in
  gb200) _cc=100;; h100|hopper) _cc=90;; a100|a800|ampere) _cc=80;;
  auto) _cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')";;
  *) _cc="";;
esac
_cc="${_cc:-80}"; [[ "$_cc" =~ ^[0-9]+$ ]] || _cc=80
if   [[ "$_cc" -ge 100 ]]; then HWNAME=gb200;  HW_NPROC=4; PEAK_BF16=2250; HW_NVLS=1
elif [[ "$_cc" -ge 90  ]]; then HWNAME=hopper; HW_NPROC=8; PEAK_BF16=989;  HW_NVLS=1
else                            HWNAME=ampere; HW_NPROC=8; PEAK_BF16=312;  HW_NVLS=0
fi
NPROC="${NPROC:-$HW_NPROC}"        # GB200=4 GPU/node, A100/A800/H100=8 GPU/node

# --- CARD PATH PROFILE (same as main launcher; SAVE -> *_smoke). All env-overridable. ---
if [[ "${HWNAME:-}" == "gb200" || -d /datasets/qwen-models-ea5jyi ]]; then
  OV2_LLM_HF_30B="${OV2_LLM_HF_30B:-/datasets/qwen-models-ea5jyi/Qwen3-30B-A3B-Instruct-2507}"
  OV2_HF_PROC_30B="${OV2_HF_PROC_30B:-/datasets/llava-ov2-30b-a3b-m9lvdn/auto_model}"
  OV2_HF_PROC_30B_P16M33="${OV2_HF_PROC_30B_P16M33:-/datasets/llava-ov2-30b-a3b-m9lvdn/auto_model}"
  OV2_PRETRAIN_ROOT="${OV2_PRETRAIN_ROOT:-/datasets/llava/11May}"
  DATA_PATH="${DATA_PATH:-$REPO/examples/models/qwen/qwen3_vl_ov2/gb200/mid_training_seed85m.yaml}"
  INIT_CKPT="${INIT_CKPT:-/datasets/llava-ov2-30b-a3b-m9lvdn}"   # trained p16m33 ckpt (has iter_/ + auto_model)
  SAVE="${SAVE:-/home/ftan0055/ckpts_video_sft/ov2_30b_a3b_gb200_smoke}"
else
  OV2_LLM_HF_30B="${OV2_LLM_HF_30B:-/ov2/pretrain_models/Qwen3-30B-A3B-Instruct-2507}"
  OV2_PRETRAIN_ROOT="${OV2_PRETRAIN_ROOT:-/ov2/pretrain_models}"
  DATA_PATH="${DATA_PATH:-$REPO/examples/models/qwen/qwen3_vl_ov2/mid_training_seed85m.yaml}"
  INIT_CKPT="${INIT_CKPT:-/ov2/feilong/gb200/ckpts_video_sft/ov2_30b_a3b_stage2}"
  SAVE="${SAVE:-/ov2/feilong/gb200/ckpts_video_sft/ov2_30b_a3b_gb200_smoke}"
fi
OV2_SKIP_BASE_STITCH="${OV2_SKIP_BASE_STITCH:-1}"   # smoke: resume INIT_CKPT (torch_dist) -> skip the AIAK stage_0 stitch
OV2_HF_PROC_30B="${OV2_HF_PROC_30B:-$OV2_PRETRAIN_ROOT/llava_onevision2/llava_onevision2_30b_a3b/auto_model}"
OV2_HF_PROC_30B_P16M33="${OV2_HF_PROC_30B_P16M33:-$OV2_PRETRAIN_ROOT/llava_onevision2/llava_onevision2_30b_a3b_p16_m33/auto_model}"
export OV2_LLM_HF_30B OV2_PRETRAIN_ROOT OV2_SKIP_BASE_STITCH OV2_HF_PROC_30B OV2_HF_PROC_30B_P16M33
export OV2_INIT_CKPT="$INIT_CKPT"   # recipe guard verifies this exists before skipping the stitch

# --- in-container env (same 30B-A3B stability fixes as the main launcher) ---
export PYTHONPATH="$REPO/src:$REPO/3rdparty/Megatron-LM:$REPO/aiak_shim${PYTHONPATH:+:$PYTHONPATH}"
export OV2_SKIP_HELPERS="${OV2_SKIP_HELPERS:-1}"            # energon doesn't use helpers_cpp -> skip the C++ compile
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export OV2_MOE_PERMUTE_FUSION="${OV2_MOE_PERMUTE_FUSION:-0}"  # avoid TE Triton MoE-permute wedge
export OV2_RECOMPUTE_FULL="${OV2_RECOMPUTE_FULL:-1}"         # full recompute (mem-safe on one node)
export MFU_PEAK_TFLOPS="${MFU_PEAK_TFLOPS:-$PEAK_BF16}"      # auto per-HW (GB200 2250 / Ampere 312)
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-$HW_NVLS}"      # NVLS: GB200/Hopper on, Ampere off
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/ov2_triton_cache}"; mkdir -p "$TRITON_CACHE_DIR"

# --- single-node smoke sizing (auto from GPU count) ---
WORLD="$NPROC"
EP="${EP:-$(( WORLD < 8 ? WORLD : 8 ))}"   # A100/A800 1 node -> EP8 (real); GB200 1 node -> EP4
SEQLEN="${SEQLEN:-4096}"                    # vs 32000 real -> cut activations: fast + fits one node
MBS="${MBS:-1}"                            # THD packing REQUIRES mbs=1 (model asserts batch==1)
GBS="${GBS:-$WORLD}"                        # DP=WORLD (TP/PP/CP=1) -> GBS must be a multiple of WORLD; =WORLD: 1 sample/rank, no grad-accum
(( EP > 0 && WORLD % EP == 0 )) || { echo "FATAL: EP=$EP must divide WORLD=$WORLD." >&2; exit 1; }
(( GBS % WORLD == 0 ))          || { echo "FATAL: GBS=$GBS must be a multiple of WORLD=$WORLD (=DP)." >&2; exit 1; }

# --- run_recipe.py overrides (smoke-sized) ---
OVERRIDES="dataset.path=$DATA_PATH"
[[ "$INIT_CKPT" != "null" && -n "$INIT_CKPT" ]] && OVERRIDES="$OVERRIDES checkpoint.pretrained_checkpoint=$INIT_CKPT"
OVERRIDES="$OVERRIDES checkpoint.save=$SAVE checkpoint.load=$SAVE dataset.dataloader_save=$SAVE"
OVERRIDES="$OVERRIDES checkpoint.save_interval=$SAVE_EVERY train.train_iters=$ITERS validation.eval_iters=0 logger.log_interval=$LOG_EVERY"
OVERRIDES="$OVERRIDES scheduler.lr_warmup_iters=0 mixed_precision=bf16_mixed"
OVERRIDES="$OVERRIDES model.expert_model_parallel_size=$EP"
OVERRIDES="$OVERRIDES model.seq_length=$SEQLEN dataset.seq_length=$SEQLEN"
OVERRIDES="$OVERRIDES train.global_batch_size=$GBS train.micro_batch_size=$MBS"

mkdir -p "$SAVE"; cd "$REPO"
echo "[ov2-30b-smoke] hw=$HWNAME(cc=$_cc) | 1 node x $NPROC GPU | recipe=$RECIPE EP=$EP seq=$SEQLEN gbs=$GBS mbs=$MBS iters=$ITERS peak=${MFU_PEAK_TFLOPS}TF init=$INIT_CKPT skip_stitch=$OV2_SKIP_BASE_STITCH"
if (( EP < 8 )); then
  echo "[ov2-30b-smoke] NOTE EP=$EP (<8) is UNVALIDATED for OV2 (verified=EP8). If it dies on MoE/expert/ckpt -> smoke on 2 nodes (EP8): ITERS=20 LIST_IP=\"ip0 ip1\" bash ax_ov2_30b_a3b_gb200.sh"
fi
# shellcheck disable=SC2086  # $OVERRIDES must word-split into separate args
python -m torch.distributed.run --standalone --nproc_per_node="$NPROC" scripts/training/run_recipe.py \
  --recipe "$RECIPE" --dataset vlm-energon --step_func ov2_step \
  $OVERRIDES 2>&1 | tee "$SAVE/smoke.log"
