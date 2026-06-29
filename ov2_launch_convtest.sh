#!/bin/bash
D=/ov2/feilong/gb200/Megatron-Bridge/examples/models/qwen/qwen3_vl_ov2
: > /ov2/feilong/gb200/_rt30b/conv_final.log
docker exec -d -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 -e NPROC=8 \
  -e HF_OUT=/ov2/feilong/gb200/_rt30b/hf_export_v2 \
  llava_megatron_container_ax \
  bash -lc "nohup bash $D/conversion.sh 30b > /ov2/feilong/gb200/_rt30b/conv_final.log 2>&1 &"
echo "conversion.sh 30b launched (end-to-end)"
