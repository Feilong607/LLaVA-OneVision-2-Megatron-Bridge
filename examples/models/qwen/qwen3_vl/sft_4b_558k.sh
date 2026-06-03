#!/usr/bin/env bash
# Qwen3-VL-4B (dense) full SFT on the LLaVA-Pretrain 558k captioning set, 8x A800-80GB.
#
# Data: the ORIGINAL energon WDS /vlm/data/blip_laion_cc_sbu_558k_wds (unmodified).
#   Its .nv-meta/dataset.yaml references aiak_training_llm.data.multimodal.MultiMixQASample,
#   resolved by the standalone shim under <repo>/aiak_shim (added to PYTHONPATH below).
#
# Run INSIDE the mbridge:qwen35 container, e.g. (detached):
#   R=/ov2/feilong/gb200/Megatron-Bridge
#   docker run -d --name q3vl_558k --gpus all --ipc=host \
#     --ulimit memlock=-1 --ulimit stack=67108864 \
#     -v /ov2:/ov2 -v /vlm:/vlm -w "$R" mbridge:qwen35 \
#     bash examples/models/qwen/qwen3_vl/sft_4b_558k.sh 300
set -euo pipefail

R=/ov2/feilong/gb200/Megatron-Bridge
export PYTHONPATH="$R/src:$R/3rdparty/Megatron-LM:$R/aiak_shim"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8

ITERS=${1:-30}
GBS=${2:-16}
MBS=${3:-1}

DATA=/vlm/data/blip_laion_cc_sbu_558k_wds                              # original WDS, unmodified
PRETRAINED=/ov2/feilong/gb200/models/Qwen3-VL-4B-Instruct-mcore        # convert_checkpoints.py import
SAVE=/ov2/feilong/gb200/results/qwen3_vl_4b_558k

cd "$R"
python -m torch.distributed.run --nproc_per_node=8 scripts/training/run_recipe.py \
    --recipe qwen3_vl_4b_sft_energon_config --dataset vlm-energon --step_func qwen3_vl_step \
    dataset.path="$DATA" \
    checkpoint.pretrained_checkpoint="$PRETRAINED" \
    checkpoint.load="$SAVE" checkpoint.save="$SAVE" checkpoint.save_interval=500 \
    model.tensor_model_parallel_size=2 model.pipeline_model_parallel_size=1 \
    dataset.seq_length=4096 model.seq_length=4096 \
    train.train_iters="$ITERS" train.global_batch_size="$GBS" train.micro_batch_size="$MBS" \
    validation.eval_iters=0 \
    optimizer.lr=0.000005 optimizer.min_lr=0.0000005 scheduler.lr_warmup_iters=5 \
    logger.log_interval=1
