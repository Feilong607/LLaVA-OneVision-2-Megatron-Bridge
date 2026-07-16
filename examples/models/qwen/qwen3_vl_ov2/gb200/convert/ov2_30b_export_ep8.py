"""EP multi-GPU export: OV2 30B mcore ckpt -> HF, via load_megatron_model (collective) + save_hf_pretrained.

NO hardcoded paths -- CFG/CKPTA/HF are REQUIRED from env. The platform-specific paths live in convert.sh
(which sets them per platform via its a800/gb200 auto-detect branches). Run via convert.sh, or directly:
  CFG=<hf_config_dir> CKPTA=<mcore_ckpt_root> HF=<out_dir> torchrun --nproc_per_node=8 ov2_30b_export_ep8.py

Observability: every rank prints a flushed stderr line BEFORE cuda/dist init (see _diag). The export used to be
silent until after init_process_group + the heavy megatron.bridge import, so a hang at NCCL init (or ranks that
never arrive) produced ZERO output and was indistinguishable from a crash. The _diag lines pinpoint the exact
stall: 'before init' with no following 'after init' == the process group is waiting for ranks that never show up
(wrong world size / missing peer node); no _diag line at all == torchrun's own agent is still at rendezvous
(the peer node never launched) and the worker never started.
"""
import os
import sys
from datetime import timedelta

import torch
import torch.distributed as dist


def _diag(msg):
    """Emit a flushed, all-rank progress line to stderr (visible before dist init and while a call hangs)."""
    print(f"[ep8-export r{os.environ.get('RANK', '?')}] {msg}", file=sys.stderr, flush=True)


LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
_diag(
    f"worker up: RANK={os.environ.get('RANK')} WORLD_SIZE={os.environ.get('WORLD_SIZE')} "
    f"LOCAL_RANK={LOCAL_RANK} MASTER_ADDR={os.environ.get('MASTER_ADDR')} "
    f"MASTER_PORT={os.environ.get('MASTER_PORT')} OV2_EP={os.environ.get('OV2_EP')} "
    f"visible_gpus={torch.cuda.device_count()}"
)
torch.cuda.set_device(LOCAL_RANK)  # per-rank device (else all ranks land on GPU0 -> OOM)

if not dist.is_initialized():
    _diag("before init_process_group(nccl) -- if this is the LAST line, not all WORLD_SIZE ranks are present")
    _init_kwargs = dict(backend="nccl", timeout=timedelta(minutes=10))
    try:
        # device_id (PyTorch 2.3+) binds the rank to its GPU and forces eager NCCL init -- on GB200/MNNVL this
        # avoids a class of lazy-init barrier hangs. Fall back if the installed torch predates the kwarg.
        dist.init_process_group(device_id=torch.device(f"cuda:{LOCAL_RANK}"), **_init_kwargs)
    except TypeError:
        dist.init_process_group(**_init_kwargs)
    _diag("after init_process_group -- distributed is up")

RANK = dist.get_rank()
WORLD = dist.get_world_size()
_diag(f"rank {RANK}/{WORLD} ready")


def log(m):
    """Print a milestone line from rank 0 only (stdout)."""
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


CFG = _req("CFG")  # HF config dir (dispatch-ready p16m33 auto_model; convert.sh's ensure_dispatch_cfg)
CKPT = _req("CKPTA")  # trained mcore torch_dist ckpt root (or iter_/ dir)
HF = _req("HF")  # output HF dir
EP = int(os.environ.get("OV2_EP", "8"))  # experts split across EP ranks; TP=PP=1 -> EP must equal world size
if EP != WORLD:
    raise SystemExit(
        f"[ep8-export] OV2_EP={EP} must equal the world size ({WORLD}). Layout is TP=1/PP=1/EP=world, so "
        f"nproc_per_node * nnodes must equal OV2_EP. EP8 = 8 GPUs (GB200: 2x4 nodes); single 4-GPU node = "
        f"OV2_EP=4 NPROC=4; single GPU = OV2_EP=1 NPROC=1. (torch_dist reshards the EP8 ckpt to any EP.)"
    )

log(f"AutoBridge.from_auto_config(CKPT={CKPT}, CFG={CFG})  # config-only bridge: no source-HF-weights lookup")
bridge = AutoBridge.from_auto_config(CKPT, CFG, trust_remote_code=True)
log(f"load_megatron_model({CKPT}) @ EP{EP}/TP1/PP1 (collective)")
model = bridge.load_megatron_model(
    CKPT,
    mp_overrides=dict(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=EP,
        expert_tensor_parallel_size=1,
    ),
    wrap_with_ddp=False,
)
# load_megatron_model returns a list (PP/vp stages); save_hf_pretrained expects the list
log("save_hf_pretrained -> " + HF)
bridge.save_hf_pretrained(model, HF)
dist.barrier()
log("EP export DONE")
