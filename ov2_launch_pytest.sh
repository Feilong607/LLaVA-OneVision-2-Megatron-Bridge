#!/bin/bash
R=/ov2/feilong/gb200/Megatron-Bridge
: > /ov2/feilong/gb200/_rt30b/pytest_ov2.log
docker exec -d -e CUDA_VISIBLE_DEVICES=0 \
  -e PYTHONPATH="$R/_verify_stubs:$R/src:$R/3rdparty/Megatron-LM:$R/aiak_shim" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e TE_EXTRA_STATE_MISSING_CHECK=1 -e OV2_MOE_PERMUTE_FUSION=0 \
  llava_megatron_container_ax \
  bash -lc "cd $R && nohup python -m pytest \
    tests/unit_tests/models/ov2/test_ov2_bridge.py \
    tests/functional_tests/models/ov2/test_ov2_conversion.py \
    -v -p no:cacheprovider > /ov2/feilong/gb200/_rt30b/pytest_ov2.log 2>&1 &"
echo "pytest (ov2 unit + functional) launched"
