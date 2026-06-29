#!/bin/bash
R=/ov2/feilong/gb200/Megatron-Bridge
: > /ov2/feilong/gb200/_rt30b/infer.log
docker exec -d -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e PYTHONPATH="$R/_verify_stubs:$R/src:$R/3rdparty/Megatron-LM:$R/aiak_shim" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e TE_EXTRA_STATE_MISSING_CHECK=1 -e OV2_MOE_PERMUTE_FUSION=0 \
  -e HF_OV2=/ov2/feilong/gb200/_rt30b/infer_hf \
  -e MCORE=/ov2/feilong/gb200/ckpts_video_sft/ov2_30b_a3b_p16m33_stage2_muon/iter_0006094 \
  llava_megatron_container_ax \
  bash -lc "cd $R && PYTHONSTARTUP=<(echo 'import megatron.bridge.models.qwen_vl_ov2.ov2_bridge') \
    nohup python -m torch.distributed.run --standalone --nproc_per_node=8 \
    examples/conversion/hf_to_megatron_generate_vlm.py \
    --hf_model_path /ov2/feilong/gb200/_rt30b/infer_hf \
    --megatron_model_path /ov2/feilong/gb200/ckpts_video_sft/ov2_30b_a3b_p16m33_stage2_muon/iter_0006094 \
    --image_path https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16/resolve/main/images/table.png \
    --prompt 'Describe this image.' --max_new_tokens 32 --ep 8 --trust_remote_code \
    > /ov2/feilong/gb200/_rt30b/infer.log 2>&1 &"
echo "inference launched"
