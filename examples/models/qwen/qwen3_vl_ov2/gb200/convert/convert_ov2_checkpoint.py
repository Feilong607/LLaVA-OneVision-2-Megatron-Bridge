#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""OV2 3-sibling checkpoint conversion (GB200, in-container, torchrun).

OV2 = language_model (Qwen3 MoE via Bridge AutoBridge) + vision_model (OV2.1 encoder) + adapter.
Produces a *loadable* Bridge torch_dist checkpoint at a TARGET parallelism (TP/PP/EP/ETP),
reusing the VERIFIED build_llava_ov2 stitch builder + load_ov2_mcore_checkpoint, then saving via
Bridge's own ``save_megatron_model`` so the output is byte-compatible with what training writes
(``iter_<N>/`` + "model"-wrapped sharded dict + run_config.yaml + latest_checkpointed_iteration.txt
+ tokenizer/). The result is directly usable as ``pretrained_checkpoint`` (or ``load``).

Supported target parallelism (OV2 = monolithic 3-sibling VLM):
  TP  (tensor)      : >=1  -- LLM shards; vision/adapter shard or replicate; forward does the SP scatter.
  EP  (expert)      : >=1  -- MoE experts; EP8 is the validated layout, EP!=8 works but is UNVALIDATED.
  ETP (expert TP)   : >=1
  PP  (pipeline)    : 1 ONLY -- HARD BLOCKED. OV2 is a single 3-sibling module ("Assumes PP=1"); the
                      vision tower is pinned to PP1 (llava_ov2.py:116) and vision/adapter are built on
                      every rank (no pre_process gating), so PP>1 duplicates them across stages -> an
                      UNUSABLE ckpt. (Same stance as Megatron-MIMO: the vision encoder does not support
                      PP>1; only an LLM could, via heterogeneous per-module parallelism OV2 lacks.)
  CP  (context)     : 1 ONLY -- forward hard-asserts context_parallel_size==1.
For GB200 scaling of 30B-A3B (48L, hidden 2048, 128 experts) the right lever is EP (then TP), not PP.

Modes:
  from_base : assembled AIAK OV2 mcore base (release/mp_rank_00*/model_optim_rng.pt, EP-sharded)
              -> clean Bridge torch_dist (bakes the per-train-start stitch once). TP1/EP8 ONLY
              (the AIAK stitch loader is TP1/EP8-specific); to retarget TP/EP, from_base then reshard.
  reshard   : Bridge OV2 torch_dist ckpt -> torch_dist at a DIFFERENT TP/EP/ETP. dist_checkpointing
              auto-reshards on load (saved ShardedTensors carry global shape + per-rank offsets), so a
              model built at the new TP/EP just reads its slice. NOTE: a GB200 migration that keeps EP=8
              and TP1 does NOT need this -- training auto-reshards the DP dim on load.
  export_hf : OV2 torch_dist -> HF. LLM via AutoBridge.save_hf_pretrained; vision+adapter dumped as
              .pt (OV2 has no HF vision/adapter format). Best-effort, for inference/release.

After any conversion, verify weights survived:  A=<src> B=<out> bash convert/verify.sh
Launch via convert.sh (sets torchrun world >= TP*EP-consistent). OV2 is verified at TP1/PP1/EP8.
"""
import argparse
import os

import torch
import torch.distributed as dist


def _resolve_iter_dir(path):
    """Accept a Bridge ckpt root (latest_checkpointed_iteration.txt) or a direct dist-ckpt dir."""
    if os.path.isfile(os.path.join(path, ".metadata")):
        return path
    tag = os.path.join(path, "latest_checkpointed_iteration.txt")
    if os.path.isfile(tag):
        with open(tag) as f:
            it = f.read().strip()
        cand = os.path.join(path, "iter_{:07d}".format(int(it)))
        if os.path.isfile(os.path.join(cand, ".metadata")):
            return cand
    if os.path.isdir(path):
        iters = sorted(d for d in os.listdir(path) if d.startswith("iter_"))
        for d in reversed(iters):
            if os.path.isfile(os.path.join(path, d, ".metadata")):
                return os.path.join(path, d)
    return path


def _assert_all_loaded(res, where):
    """dist_checkpointing.load + load_state_dict(strict=False) SILENTLY ignore missing keys -> a param
    the source ckpt lacks stays at its (uninitialized) build value and gets saved as garbage. Fail loud."""
    missing = [k for k in getattr(res, "missing_keys", []) if not k.endswith("_extra_state")]
    if missing:
        raise SystemExit(
            "[ov2-convert] {} load left {} model params UNLOADED (would save GARBAGE): {}{} -- "
            "source/target arch mismatch?".format(where, len(missing), missing[:15],
                                                  " ..." if len(missing) > 15 else ""))


def _init(args):
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    # Pre-flight the expert-grid divisibility BEFORE mcore's initialize_model_parallel (which otherwise
    # raises a cryptic "world not divisible by expert_tensor_model_pipeline_parallel size"). mcore
    # defaults expert-TP to TP when etp is unset, so the expert layout needs world % (ETP_eff * EP) == 0.
    world = dist.get_world_size()
    _etp_eff = args.etp if args.etp else args.tp
    if world % (_etp_eff * args.ep) != 0:
        raise SystemExit(
            "[ov2-convert] world_size={} not divisible by expert grid ETP_eff*EP={}*{}={}. With --tp {} "
            "and no --etp, expert-TP defaults to TP -> pass --etp 1 (experts un-TP-sharded, grid 1*EP).".format(
                world, _etp_eff, args.ep, _etp_eff * args.ep, args.tp))
    from megatron.core import parallel_state
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=args.tp,
            pipeline_model_parallel_size=args.pp,
            expert_model_parallel_size=args.ep,
            expert_tensor_parallel_size=(args.etp or None),
        )


def _check_parallelism(args):
    """OV2 supports TP/EP/ETP reshard; PP>1 and CP>1 are architecturally blocked (see module docstring)."""
    world = dist.get_world_size()
    rank = dist.get_rank()
    # --- PP>1: hard block (monolithic VLM; vision tower pinned PP1; vision/adapter built on every rank). ---
    if args.pp != 1:
        raise SystemExit(
            "[ov2-convert] PP>1 is not supported for OV2: it is a single 3-sibling module that assumes "
            "PP==1 (vision tower pinned to PP1, vision/adapter built on every rank -> PP>1 duplicates them "
            "into an unusable ckpt). Scale with EP (then TP) instead. Got PP{}.".format(args.pp)
        )
    # --- world must satisfy BOTH the attention (TP x DP) and expert (EP x EDP) layouts. ---
    if world < args.ep or world % args.ep != 0:
        raise SystemExit(
            "[ov2-convert] world_size={} incompatible with EP={}: need world >= EP and world % EP == 0 "
            "(DP = world/EP must be a whole number). On GB200 (4 GPU/node) EP8 needs >=2 nodes; "
            "set LIST_IP='<ip0> <ip1>' NPROC=4.".format(world, args.ep)
        )
    if world % args.tp != 0:
        raise SystemExit(
            "[ov2-convert] world_size={} not divisible by TP={} (attention needs TP x DP == world).".format(
                world, args.tp)
        )
    # --- from_base uses the AIAK TP1/EP8 stitch loader; retarget TP/EP via a follow-up `reshard`. ---
    if args.mode == "from_base" and (args.ep != 8 or args.tp != 1):
        raise SystemExit(
            "[ov2-convert] from_base requires TP1/EP8 (the AIAK base is TP1/EP8-sharded and the OV2 stitch "
            "loader is verified there). Got TP{}/EP{}. Run from_base at TP1/EP8, then `reshard` to the "
            "target TP/EP.".format(args.tp, args.ep)
        )
    if args.tp != 1 and rank == 0:
        print("[ov2-convert] NOTE: TP={} reshard. OV2 forward implements the TP sequence-parallel scatter, "
              "but TP>1 is less exercised than TP1 -- confirm with convert/verify.sh (compares GLOBAL "
              "tensors, so it catches any TP-sharding error).".format(args.tp), flush=True)
    if args.ep != 8 and rank == 0:
        print("[ov2-convert] WARNING: EP={} != 8 is UNVALIDATED for OV2 MoE; the expert all-to-all / router "
              "layout is only validated at EP8. MUST run convert/verify.sh on the output.".format(args.ep),
              flush=True)


def main():
    ap = argparse.ArgumentParser(description="OV2 3-sibling checkpoint conversion (torch_dist).")
    ap.add_argument("mode", choices=["from_base", "reshard", "export_hf"])
    ap.add_argument("--backbone", default="qwen3-30b-a3b")
    ap.add_argument("--src", required=True, help="source checkpoint dir")
    ap.add_argument("--out", required=True, help="output dir (Bridge torch_dist; or HF dir for export_hf)")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--pp", type=int, default=1)
    ap.add_argument("--ep", type=int, default=8)
    ap.add_argument("--etp", type=int, default=0, help="expert TP (0 -> default/None)")
    ap.add_argument("--no-adapter", action="store_true", help="from_base: leave adapter at init")
    args = ap.parse_args()

    _init(args)
    _check_parallelism(args)
    from megatron.core import dist_checkpointing
    from megatron.bridge.models.qwen_vl_ov2.llava_ov2 import build_llava_ov2
    from megatron.bridge.recipes.ov2.ov2 import _ov2_backbone_paths
    rank = dist.get_rank()
    p = _ov2_backbone_paths(args.backbone)

    def log(m):
        if rank == 0:
            print("[ov2-convert] {}".format(m), flush=True)

    def _build():
        """Structure-only OV2 model at the target parallelism (weights filled by the caller)."""
        return build_llava_ov2(
            llm_hf_path=p["llm_hf"],
            perform_init=False,                      # weights come from the source ckpt
            tensor_model_parallel_size=args.tp,
            pipeline_model_parallel_size=args.pp,
            expert_model_parallel_size=args.ep,
            expert_tensor_parallel_size=(args.etp or None),
            sequence_parallel=False,
            load_llm_weights=False,
            patch_size=p["vision_patch_size"],
            spatial_merge_size=p["vision_spatial_merge_size"],
            vision_hidden_size=p["vision_hidden_size"],
            vision_num_layers=p["vision_num_layers"],
            vision_model_name=p["vision_model_name"],
        )

    def _save_loadable(model):
        """Write a Bridge-loadable torch_dist ckpt (iter_/ + model wrapper + run_config + tokenizer)."""
        from megatron.bridge.training.model_load_save import save_megatron_model
        save_megatron_model([model], args.out, ckpt_format="torch_dist", hf_tokenizer_path=os.environ.get("OV2_CONVERT_TOKENIZER", p["llm_hf"]))
        log("saved Bridge torch_dist (loadable) -> {} (use as pretrained_checkpoint/load)".format(args.out))

    if args.mode == "from_base":
        from megatron.bridge.models.qwen_vl_ov2.llava_ov2 import load_ov2_mcore_checkpoint
        log("{} TP{}/PP{}/EP{}/ETP{} | stitch base: {}".format(
            args.backbone, args.tp, args.pp, args.ep, args.etp or "-", args.src))
        model = _build()
        load_ov2_mcore_checkpoint(model, args.src, load_adapter=not args.no_adapter, load_vision=True)
        _save_loadable(model)

    elif args.mode == "reshard":
        src = _resolve_iter_dir(args.src)
        log("reshard {} -> TP{}/PP{}/EP{}/ETP{} -> {}".format(
            src, args.tp, args.pp, args.ep, args.etp or "-", args.out))
        model = _build()
        loaded = dist_checkpointing.load(model.sharded_state_dict(), src)
        _assert_all_loaded(model.load_state_dict(loaded, strict=False), "reshard")
        _save_loadable(model)

    elif args.mode == "export_hf":
        from megatron.bridge import AutoBridge
        src = _resolve_iter_dir(args.src)
        log("export_hf {} -> {} (LLM->HF, vision/adapter->.pt)".format(src, args.out))
        model = _build()
        loaded = dist_checkpointing.load(model.sharded_state_dict(), src)
        _assert_all_loaded(model.load_state_dict(loaded, strict=False), "export_hf")
        if rank == 0:
            os.makedirs(args.out, exist_ok=True)
        dist.barrier()
        # vision/adapter first: rank0-only, NON-collective -> a later collective LLM-export failure
        # can neither lose these dumps nor deadlock mid-gather.
        if rank == 0:
            torch.save(model.vision_model.state_dict(), os.path.join(args.out, "vision_model.pt"))
            torch.save(model.adapter.state_dict(), os.path.join(args.out, "adapter.pt"))
            log("  vision_model.pt + adapter.pt saved")
        dist.barrier()
        # LLM->HF is COLLECTIVE (all ranks all-gather TP/PP/EP shards + internal barriers): run it on
        # EVERY rank with NO try/except -- swallowing a non-uniform failure would deadlock the job.
        bridge = AutoBridge.from_hf_pretrained(p["llm_hf"])
        bridge.save_hf_pretrained(model.language_model, os.path.join(args.out, "language_model_hf"))
        log("  language_model -> HF ok")

    dist.barrier()
    log("done")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
