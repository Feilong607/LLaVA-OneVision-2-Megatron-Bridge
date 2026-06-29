#!/bin/bash
C=/ov2/feilong/gb200/Megatron-Bridge/examples/models/qwen/qwen3_vl_ov2/A800/convert
: > /ov2/feilong/gb200/_rt30b/conv_moved.log
docker exec -d -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 -e NPROC=8 \
  -e HF_OUT=/ov2/feilong/gb200/_rt30b/hf_export_moved \
  llava_megatron_container_ax \
  bash -lc "nohup bash $C/conversion.sh 30b > /ov2/feilong/gb200/_rt30b/conv_moved.log 2>&1 &"
echo "moved conversion.sh 30b launched"
