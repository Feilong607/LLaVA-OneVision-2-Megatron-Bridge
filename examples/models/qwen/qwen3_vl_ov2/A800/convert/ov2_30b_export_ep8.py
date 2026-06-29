# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""EP8 multi-GPU export: OV2 30B mcore ckpt -> HF, via load_megatron_model (collective) + save_hf_pretrained.

NO hardcoded paths -- CFG/CKPTA/HF are REQUIRED from env. The platform-specific paths live in convert.sh
(which sets them per platform via its a800/gb200 auto-detect branches). Run via convert.sh, or directly:
  CFG=<hf_config_dir> CKPTA=<mcore_ckpt_root> HF=<out_dir> torchrun --nproc_per_node=8 ov2_30b_export_ep8.py
"""
import os, torch, torch.distributed as dist
torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))   # THE FIX: per-rank device (else all on GPU0->OOM)
if not dist.is_initialized():
    dist.init_process_group("nccl")
RANK = dist.get_rank()
def log(m):
    if RANK == 0:
        print("[ep8-export] " + m, flush=True)

def _req(name):
    v = os.environ.get(name)
    if not v:
        raise SystemExit(
            f"[ep8-export] env {name} is REQUIRED (convert.sh sets CFG/CKPTA/HF per platform). "
            f"Direct use: CFG=<hf_config_dir> CKPTA=<mcore_ckpt_root> HF=<out_dir> "
            f"torchrun --nproc_per_node=8 ov2_30b_export_ep8.py"
        )
    return v

from megatron.bridge import AutoBridge
CFG  = _req("CFG")    # HF config dir (dispatch-ready p16m33 auto_model; convert.sh's ensure_dispatch_cfg)
CKPT = _req("CKPTA")  # trained mcore torch_dist ckpt root (or iter_/ dir)
HF   = _req("HF")     # output HF dir
EP   = int(os.environ.get("OV2_EP", "8"))   # OV2 verified layout = EP8 (GB200 2x4 nodes still = world 8 = EP8)

log(f"AutoBridge.from_auto_config(CKPT={CKPT}, CFG={CFG})  # config-only bridge: no source-HF-weights lookup")
bridge = AutoBridge.from_auto_config(CKPT, CFG, trust_remote_code=True)
log(f"load_megatron_model({CKPT}) @ EP{EP}/TP1/PP1 (collective)")
model = bridge.load_megatron_model(
    CKPT,
    mp_overrides=dict(tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
                      expert_model_parallel_size=EP, expert_tensor_parallel_size=1),
    wrap_with_ddp=False,
)
# load_megatron_model returns a list (PP/vp stages); save_hf_pretrained expects the list
log("save_hf_pretrained -> " + HF)
bridge.save_hf_pretrained(model, HF)
dist.barrier()
log("EP8 EXPORT DONE")
