#!/usr/bin/env bash
# =============================================================================
# OV2 (LLaVA-OneVision-2) VLM inference — bespoke ov2_generate.py driver.
# AUTO-DETECTS platform: GB200 (cc>=10.0) -> 4 GPU/node + /datasets paths; A100/A800 -> /ov2 paths.
# VERIFIED 2026-06-29 (A100 8xGPU, 30B EP8): coherent image-grounded + text-only generation.
#
# WHY NOT examples/conversion/hf_to_megatron_generate_vlm.py (generic VLM entry):
#   it calls model(**forward_args) with Qwen3-VL arg names (pixel_values=...); OV2's forward sig is
#   forward(images, image_grid_thw, input_ids, ...) -> TypeError missing 'images'. ov2_generate.py
#   builds the OV2 model (build_llava_ov2), loads the torch_dist ckpt, and runs the OV2 training-forward
#   with the correct arg names under autocast(bf16) (vision patch_embed is fp32, inputs bf16).
#
# Launch (run the SAME cmd on every node; node_rank auto from LIST_IP):
#   A800/A100 single 8-GPU node:        bash inference.sh
#   GB200 EP8 across 2x4-GPU nodes:     NPROC=4 LIST_IP="<ip0> <ip1>" bash inference.sh
# No-internet boxes: pass a LOCAL IMAGE (IMAGE=/path/to.png); the default URL needs egress.
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$({ d="$HERE"; while [[ "$d" != "/" && ! -d "$d/src/megatron/bridge" ]]; do d="$(dirname "$d")"; done; echo "$d"; })}"
[[ -d "$REPO/src/megatron/bridge" ]] || { echo "FATAL: OV2 fork root not found above $HERE (set REPO=)" >&2; exit 1; }
GEN="${GEN:-$HERE/ov2_generate.py}"
[[ -f "$GEN" ]] || { echo "FATAL: $GEN not found (the OV2 generate driver)" >&2; exit 1; }

# --- platform auto-detect ---
_cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.' || echo 0)"
if [[ "${_cc:-0}" -ge 100 ]]; then
  PLAT=gb200; DEF_NPROC=4
  MCKPT_DEF="${MEGATRON_CKPT:-/datasets/llava-ov2-30b-a3b-m9lvdn/iter_0001000}"
  PROC_DEF="${OV2_HF_PROC_30B:-/datasets/llava-ov2-30b-a3b-m9lvdn/auto_model}"
  LLM_DEF="${OV2_LLM_HF_30B:-/datasets/qwen-models-ea5jyi/Qwen3-30B-A3B-Instruct-2507}"
else
  PLAT=a800; DEF_NPROC=8
  MCKPT_DEF="${MEGATRON_CKPT:-/ov2/feilong/gb200/ckpts_video_sft/ov2_30b_a3b_p16m33_stage2_muon/iter_0006094}"
  PROC_DEF="${OV2_HF_PROC_30B:-/ov2/pretrain_models/llava_onevision2/llava_onevision2_30b_a3b_p16_m33/auto_model}"
  LLM_DEF="${OV2_LLM_HF_30B:-/ov2/pretrain_models/Qwen3-30B-A3B-Instruct-2507}"
fi
NPROC="${NPROC:-$DEF_NPROC}"
BACKBONE="${BACKBONE:-qwen3-30b-a3b-p16m33}"
EP="${EP:-8}"; TP="${TP:-1}"; ETP="${ETP:-1}"          # OV2 verified layout = TP1/EP8 (PP/CP forced 1)
MEGATRON_CKPT="$MCKPT_DEF"
IMAGE="${IMAGE:-https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16/resolve/main/images/table.png}"
PROMPT="${PROMPT:-Describe this image.}"
MAX_NEW="${MAX_NEW:-32}"

# env contract: offline HF; PYTHONPATH incl aiak_shim + _verify_stubs (shims modelopt + diffusers --
# ov2_generate imports recipes.ov2 -> recipes/__init__ -> flux -> `from diffusers import ...`);
# MoE permute fusion OFF; recipe backbone reads OV2_LLM_HF_30B / OV2_HF_PROC_30B from env.
export PYTHONPATH="$REPO/_verify_stubs:$REPO/src:$REPO/3rdparty/Megatron-LM:$REPO/aiak_shim${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TE_EXTRA_STATE_MISSING_CHECK="${TE_EXTRA_STATE_MISSING_CHECK:-1}" OV2_MOE_PERMUTE_FUSION="${OV2_MOE_PERMUTE_FUSION:-0}"
export OV2_HF_PROC_30B="$PROC_DEF" OV2_LLM_HF_30B="$LLM_DEF"

# torchrun rendezvous
if [[ -n "${LIST_IP:-}" ]]; then
  read -ra ip <<< "$LIST_IP"; NN=${#ip[@]}
  MA="${ip[0]}"; MP="${MASTER_PORT:-26061}"; CUR="$(hostname -I | awk '{print $1}')"; NR=-1
  for i in "${!ip[@]}"; do [[ "${ip[$i]}" == "$CUR" ]] && NR=$i && break; done
  [[ "$NR" -eq -1 ]] && { echo "ERROR: $CUR not in LIST_IP (${ip[*]})" >&2; exit 1; }
  RDZV="--nnodes=$NN --node_rank=$NR --master_addr=$MA --master_port=$MP"
else
  RDZV="--standalone --nnodes=1"
fi
cd "$REPO"

echo "[ov2-infer/$PLAT] backbone=$BACKBONE TP=$TP EP=$EP ckpt=$MEGATRON_CKPT image=$IMAGE"
# shellcheck disable=SC2086
python -m torch.distributed.run $RDZV --nproc_per_node="$NPROC" "$GEN" \
    --backbone "$BACKBONE" --megatron_ckpt "$MEGATRON_CKPT" \
    --image "$IMAGE" --prompt "$PROMPT" --max_new_tokens "$MAX_NEW" \
    --tp "$TP" --ep "$EP" --etp "$ETP"

echo "[ov2-infer/$PLAT] text-only prompt"
# shellcheck disable=SC2086
python -m torch.distributed.run $RDZV --nproc_per_node="$NPROC" "$GEN" \
    --backbone "$BACKBONE" --megatron_ckpt "$MEGATRON_CKPT" \
    --prompt "Briefly, what is LLaVA-OneVision?" --max_new_tokens "$MAX_NEW" \
    --tp "$TP" --ep "$EP" --etp "$ETP"
