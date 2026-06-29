#!/usr/bin/env bash
# 30B OV2 new-bridge roundtrip: ckptA -> HF -> ckptB, then verify A==B (GLOBAL tensors).
# Run inside the OV2 container with 8 GPUs (EP8). All paths overridable.
set -euo pipefail
R=/ov2/feilong/gb200/Megatron-Bridge
A=${A:-/ov2/feilong/gb200/ckpts_video_sft/ov2_30b_a3b_p16m33_stage2_muon/iter_0006094}
HF=${HF:-/ov2/feilong/gb200/_rt30b/hf_export}
B=${B:-/ov2/feilong/gb200/_rt30b/ckptB}
HFREF=${HFREF:-/ov2/pretrain_models/llava_onevision2/llava_onevision2_30b_a3b/auto_model}  # 30B cfg (has architectures now)
NPROC=${NPROC:-8}
export PYTHONPATH="$R/_verify_stubs:$R/src:$R/3rdparty/Megatron-LM:$R/aiak_shim"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TE_EXTRA_STATE_MISSING_CHECK=1 OV2_MOE_PERMUTE_FUSION=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
mkdir -p "$(dirname "$HF")"
cd "$R"
echo "[rt30b] === STEP 2: export ckptA -> HF ($A -> $HF) ==="
python -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
    examples/conversion/convert_checkpoints.py export \
    --hf-model "$HFREF" --megatron-path "$A" --hf-path "$HF" --trust-remote-code
echo "[rt30b] === STEP 3: import HF -> ckptB ($HF -> $B) ==="
python -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
    examples/conversion/convert_checkpoints.py import \
    --hf-model "$HF" --megatron-path "$B" --trust-remote-code
echo "[rt30b] === STEP 4: verify A vs B (GLOBAL tensors, CPU) ==="
A="$A" B="$B" bash "$R/examples/models/qwen/qwen3_vl_ov2/gb200/convert/verify.sh" || \
  python "$R/examples/models/qwen/qwen3_vl_ov2/gb200/convert/verify_consistency.py" --a "$A" --b "$B" --values sample
echo "[rt30b] DONE"
