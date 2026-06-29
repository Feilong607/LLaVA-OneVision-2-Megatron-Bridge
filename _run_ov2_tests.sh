#!/bin/bash
# OV2 test runner: A800 --gpus broken -> bind-mount host driver libs + --privileged. GPU 6 (free).
set -o pipefail
cd /ov2/feilong/gb200/Megatron-Bridge
REPO=/ov2/feilong/gb200/Megatron-Bridge
LIBDIR=/usr/lib/x86_64-linux-gnu
LIBS="libcuda.so.550.144.03 libnvidia-ml.so.550.144.03 libnvidia-ptxjitcompiler.so.550.144.03 libnvidia-nvvm.so.550.144.03 libcudadebugger.so.550.144.03"
MOUNTS=""
for L in $LIBS; do MOUNTS="$MOUNTS -v $LIBDIR/$L:$LIBDIR/$L"; done
docker run --rm --privileged -v /dev:/dev -v /ov2:/ov2 $MOUNTS \
  -e CUDA_VISIBLE_DEVICES=6 -e NCCL_IB_DISABLE=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e TE_EXTRA_STATE_MISSING_CHECK=1 \
  -e OV2_MOE_PERMUTE_FUSION=0 -e OV2_SKIP_BASE_STITCH=1 \
  -e PYTHONPATH=$REPO/_verify_stubs:$REPO/src:$REPO/3rdparty/Megatron-LM:$REPO/aiak_shim \
  -w $REPO mbridge:qwen35-muon bash -lc '
    ldconfig 2>/dev/null
    echo "=== cuda visible: $CUDA_VISIBLE_DEVICES ; torch sees GPU? ==="
    python -c "import torch; print(\"cuda.is_available=\", torch.cuda.is_available(), \"ndev=\", torch.cuda.device_count())"
    echo "===================== UNIT: test_ov2_bridge.py ====================="
    python -m pytest -v -p no:cacheprovider tests/unit_tests/models/ov2/test_ov2_bridge.py
    echo "EXIT_UNIT=$?"
    echo "================ FUNCTIONAL: test_ov2_conversion.py ================"
    python -m pytest -v -s -p no:cacheprovider tests/functional_tests/models/ov2/test_ov2_conversion.py
    echo "EXIT_FUNC=$?"
  '
echo "ALL_DONE rc=$?"
