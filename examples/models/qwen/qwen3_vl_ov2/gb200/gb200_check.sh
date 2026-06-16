#!/usr/bin/env bash
# =============================================================================
# GB200 in-container DIAGNOSTIC (single node, NO torchrun). Run before training:
#   bash examples/models/qwen/qwen3_vl_ov2/gb200/gb200_check.sh
# Checks: GPUs/HBM/Blackwell(sm_100), NVLink/NVL72 topology + link status, NCCL env,
# bf16 compute (MFU-peak calibration), P2P NVLink bandwidth, and that the OV2 training
# stack + the packing/rename changes import cleanly in this container.
# =============================================================================
set -uo pipefail
REPO="${REPO:-/ov2/feilong/gb200/Megatron-Bridge}"
export PYTHONPATH="$REPO/src:$REPO/3rdparty/Megatron-LM:$REPO/aiak_shim${PYTHONPATH:+:$PYTHONPATH}"
sec(){ echo; echo "========== $* =========="; }

sec "1. GPUs (name / HBM / driver / compute-cap)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap --format=csv 2>/dev/null || nvidia-smi || echo "nvidia-smi unavailable"

sec "2. NVLink topology matrix (NV# = NVLink links between GPUs; NVL72 should show NV* everywhere)"
nvidia-smi topo -m 2>/dev/null || echo "topo unavailable"

sec "3. NVLink per-link status / speed"
nvidia-smi nvlink --status 2>/dev/null | head -80 || echo "nvlink status unavailable"

sec "4. NCCL / CUDA env"
env | grep -E "^(NCCL|CUDA_DEVICE_MAX|PYTORCH_CUDA|NVLS|OMP_NUM)" | sort || echo "(none set)"

sec "5. torch / capability / bf16 compute / P2P / stack imports"
python - <<'PY'
import time, traceback
def ok(m): print("  [OK]   "+m)
def bad(m): print("  [FAIL] "+m)
try:
    import torch
    print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}  n_gpu={torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        p=torch.cuda.get_device_properties(i)
        print(f"  GPU{i}: {p.name}  cap=({p.major},{p.minor})  HBM={p.total_memory/1e9:.0f}GB")
    cap=torch.cuda.get_device_capability(0)
    print(f"  Blackwell sm_100? {'YES' if cap[0]>=10 else 'NO (not GB200)'}  cap={cap}")
    d=torch.device("cuda",0); n=8192
    a=torch.randn(n,n,device=d,dtype=torch.bfloat16); b=torch.randn(n,n,device=d,dtype=torch.bfloat16)
    for _ in range(10): c=a@b
    torch.cuda.synchronize(); t=time.perf_counter(); it=50
    for _ in range(it): c=a@b
    torch.cuda.synchronize(); dt=(time.perf_counter()-t)/it
    print(f"  bf16 matmul {n}^3 -> {2*n**3/dt/1e12:.0f} TFLOP/s/GPU  (set MFU_PEAK_TFLOPS near this)")
    if torch.cuda.device_count()>=2:
        m=256*1024*1024//2; x0=torch.empty(m,device='cuda:0',dtype=torch.bfloat16); x1=torch.empty(m,device='cuda:1',dtype=torch.bfloat16)
        for _ in range(5): x1.copy_(x0)
        torch.cuda.synchronize(); t=time.perf_counter()
        for _ in range(20): x1.copy_(x0)
        torch.cuda.synchronize(); dt=(time.perf_counter()-t)/20
        print(f"  P2P GPU0->GPU1 copy -> {m*2/dt/1e9:.0f} GB/s (intra-node NVLink)")
except Exception: traceback.print_exc()
for mod in ["transformer_engine","flash_attn","megatron.core","megatron.energon"]:
    try:
        mm=__import__(mod); ok(f"import {mod} ({getattr(mm,'__version__','?')})")
    except Exception as e: bad(f"import {mod}: {e}")
try:
    from aiak_training_llm.data.multimodal import PackedCaptioningSample, MultiMixQASample; ok("aiak_shim PackedCaptioningSample + MultiMixQASample")
except Exception as e: bad(f"aiak_shim samples: {e}")
try:
    from megatron.bridge.models.qwen_vl_ov2.llava_ov2 import build_llava_ov2, LlavaOnevision2; ok("llava_ov2 (renamed) build_llava_ov2 + LlavaOnevision2")
except Exception as e: bad(f"llava_ov2: {e}")
try:
    from megatron.bridge.models.qwen_vl_ov2.llava_ov2_4b import build_llava_ov2 as _b; ok("llava_ov2_4b shim re-export")
except Exception as e: bad(f"llava_ov2_4b shim: {e}")
try:
    from megatron.bridge.recipes.ov2.ov2 import ov2_35b_a3b_midtrain; ok("recipe ov2_35b_a3b_midtrain")
except Exception as e: bad(f"recipe import: {e}")
try:
    from megatron.bridge.models.qwen_vl_ov2.ov2_step import forward_step, get_batch; ok("ov2_step (packed forward)")
except Exception as e: bad(f"ov2_step: {e}")
PY
sec "6. Multi-precision GEMM peaks (calibrate MFU_PEAK_TFLOPS: bf16=Phase1, fp8=Phase2)"
python "$REPO/examples/models/qwen/qwen3_vl_ov2/gb200/gb200_flops.py" 2>/dev/null || echo "flops sweep skipped/failed"

sec "7. Phase-2 (ACCEL=1/2) no-internet preflight: MXFP8 recipe + flex/HybridEP backend import"
python - <<'PY'
def ok(m): print("  [OK]   "+m)
def bad(m): print("  [FAIL] "+m)
def warn(m): print("  [WARN] "+m)
# (a) MXFP8 recipe present (ACCEL=1 sets mixed_precision=bf16_with_mxfp8_mixed -> ov2_provider copies
#     these fp8 fields onto the RUNTIME LLM config; cfg.model alone is dead).
try:
    from megatron.bridge.training.mixed_precision import bf16_with_mxfp8_mixed
    r = bf16_with_mxfp8_mixed()
    ok("MXFP8 recipe: fp8=%s recipe=%s param_gather=%s" % (r.fp8, r.fp8_recipe, getattr(r, "fp8_param_gather", None)))
except Exception as e: bad("MXFP8 recipe import: %s" % e)
# (b) flex dispatcher helper (ACCEL=2 -> ov2_provider calls apply_flex_dispatcher_backend on runtime config).
try:
    from megatron.bridge.training.flex_dispatcher_backend import apply_flex_dispatcher_backend  # noqa
    ok("apply_flex_dispatcher_backend import (HybridEP wired via this in ov2_provider.provide)")
except Exception as e: bad("flex_dispatcher_backend import: %s" % e)
# (c) HybridEP/DeepEP RUNTIME package: MUST import offline for ACCEL=2. A missing pkg crashes BEFORE
#     iter 1 and CANNOT be pip-installed on GB200 (no internet). If [WARN], stay on ACCEL=0/1 (alltoall).
import importlib
try:
    importlib.import_module("deep_ep")
    from deep_ep import HybridEPBuffer  # the actual buffer mcore's HybridEP dispatcher imports
    from megatron.core.transformer.moe.fused_a2a import HAVE_HYBRIDEP
    assert HAVE_HYBRIDEP, "deep_ep imports but mcore HAVE_HYBRIDEP is False (HybridEPBuffer/hybrid_ep_cpp missing)"
    ok("HybridEP fully importable (deep_ep + HybridEPBuffer + HAVE_HYBRIDEP) -> ACCEL=2 usable offline")
except Exception as e:
    warn("runtime pkg 'deep_ep' NOT importable: %s -> ACCEL=2 (HybridEP) WILL FAIL at runtime; "
         "use ACCEL=0/1 (alltoall) unless deep_ep is baked into the container." % e)
PY
echo "  NOTE: after a real run, confirm Phase-2 ENGAGED (it silently no-ops if the provider wiring regresses):"
echo "        grep train log for '[ov2 provider] fp8 wired' (ACCEL=1) and '[ov2 provider] flex dispatcher wired' (ACCEL=2)"

echo; echo "========== gb200_check done =========="
