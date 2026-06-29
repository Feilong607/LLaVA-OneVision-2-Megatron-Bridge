#!/bin/bash
R=/ov2/feilong/gb200/Megatron-Bridge
: > /ov2/feilong/gb200/_rt30b/rt30b_ep8.log
docker exec -d -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e PYTHONPATH="$R/_verify_stubs:$R/src:$R/3rdparty/Megatron-LM:$R/aiak_shim" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e TE_EXTRA_STATE_MISSING_CHECK=1 -e OV2_MOE_PERMUTE_FUSION=0 \
  llava_megatron_container_ax \
  bash -lc "cd $R && nohup python -m torch.distributed.run --standalone --nproc_per_node=8 \
    examples/conversion/hf_megatron_roundtrip_multi_gpu.py \
    --hf-model-id /ov2/feilong/gb200/_rt30b/hf_export \
    --tp 1 --pp 1 --ep 8 --trust-remote-code --not-strict \
    > /ov2/feilong/gb200/_rt30b/rt30b_ep8.log 2>&1 &"
echo "30B EP8 roundtrip launched"
