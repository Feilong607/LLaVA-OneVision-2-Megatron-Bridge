#!/bin/bash
R=/ov2/feilong/gb200/Megatron-Bridge
: > /ov2/feilong/gb200/_rt30b/gen.log
docker exec -d -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e PYTHONPATH="$R/_verify_stubs:$R/src:$R/3rdparty/Megatron-LM:$R/aiak_shim" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e TE_EXTRA_STATE_MISSING_CHECK=1 -e OV2_MOE_PERMUTE_FUSION=0 \
  -e OV2_HF_PROC_30B=/ov2/pretrain_models/llava_onevision2/llava_onevision2_30b_a3b_p16_m33/auto_model \
  llava_megatron_container_ax \
  bash -lc "cd $R && nohup python -m torch.distributed.run --standalone --nproc_per_node=8 \
    examples/models/qwen/qwen3_vl_ov2/ov2_generate.py \
    --backbone qwen3-30b-a3b-p16m33 \
    --megatron_ckpt /ov2/feilong/gb200/ckpts_video_sft/ov2_30b_a3b_p16m33_stage2_muon/iter_0006094 \
    --image /ov2/feilong/gb200/_rt30b/test.png \
    --prompt 'Describe this image.' --max_new_tokens 32 --tp 1 --ep 8 --etp 1 \
    > /ov2/feilong/gb200/_rt30b/gen.log 2>&1 &"
echo "ov2_generate launched"
