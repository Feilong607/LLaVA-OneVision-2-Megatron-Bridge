# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

"""LLaVA-OneVision-2.1 (OV2) SFT recipe with the **Muon** optimizer.

This mirrors the OV2.1 reference stage-2 SFT launcher

    examples/llava_onevision2/quick_start_4b/
        date0528_ax_stage_2_alignment_p16m3_4n_muon_from_full_muon_from_adam_lr2e5.sh

on the Megatron-Bridge framework.

Design (per request):
  * "align the encoder, use Bridge's own LLM" -> we graft the OV2.1 p16m33
    vision tower onto Bridge's native Qwen3-VL-8B language model (exactly what
    the Adam recipe ``qwen3_ov2_8b_sft_config`` already does).
  * Optimizer Adam -> Muon. Bridge / Megatron-Core ship Muon natively
    (``optimizer="dist_muon"`` via ``distributed_muon_with_cosine_annealing``),
    so we only need a NEW ``*_muon`` recipe.

It deliberately does **not** modify the two existing files
(``qwen3_ov2.py`` recipe and ``qwen3_ov2_provider.py``). Instead it imports the
Adam recipe to build the full model + OV2 dataset graph, then swaps the
optimizer/scheduler to distributed Muon and pins the stage-2 hyper-parameters.

Reference -> Bridge alignment (date0528 stage-2):
    --seq-length 32000                -> seq_length=32000
    GBS 128 / MBS 1                   -> global_batch_size=128 / micro_batch_size=1
    NSTEP=ceil(780000/128)=6094       -> train_iters=6094
    --lr 2e-5 --min-lr 2e-5           -> lr=min_lr=2e-5  (== => CONSTANT LR)
    --lr-warmup-fraction 0            -> lr_warmup_iters=0
    --lr-decay-style cosine           -> cosine (flat, since min_lr == lr)
    --weight-decay 0                  -> weight_decay=0.0
    --clip-grad 1.0                   -> clip_grad=1.0
    --optimizer muon                  -> optimizer="dist_muon"
    --muon-momentum 0.95              -> muon_momentum=0.95
    --muon-matched-adamw-rms 0.15     -> muon_matched_adamw_rms=0.15
    --adam-beta1 0.9 --adam-beta2 0.99-> adam_beta1=0.9 / adam_beta2=0.99
"""

from typing import Optional

from megatron.bridge.recipes.qwen_vl.qwen3_ov2 import qwen3_ov2_8b_sft_config
from megatron.bridge.recipes.utils.optimizer_utils import (
    distributed_muon_with_cosine_annealing,
)
from megatron.bridge.training.config import ConfigContainer


# Faithful defaults taken straight from the date0528 stage-2 reference. Every
# value is overridable via **user_kwargs or Hydra-style CLI overrides applied on
# top of the returned ConfigContainer.
_REF = dict(
    seq_length=32000,
    global_batch_size=128,
    micro_batch_size=1,
    train_iters=6094,
    lr=2.0e-5,
    min_lr=2.0e-5,              # == lr  => constant LR (cosine stays flat)
    lr_warmup_iters=0,
    lr_decay_iters=None,        # defaults to train_iters below
    weight_decay=0.0,
    clip_grad=1.0,
    muon_momentum=0.95,
    muon_matched_adamw_rms=0.15,
    adam_beta1=0.9,
    adam_beta2=0.99,
)


def qwen3_ov2_8b_sft_muon_config(
    hf_path: Optional[str] = None,
    **user_kwargs,
) -> ConfigContainer:
    """OV2.1 p16m33 vision tower + Qwen3-VL-8B LLM, OV2 packed WDS SFT, Muon.

    LR schedule is *constant* at 2e-5 (cosine with ``min_lr == lr`` and zero
    warmup), matching the reference ``--lr 2e-5 --min-lr 2e-5
    --lr-warmup-fraction 0``.
    """
    p = dict(_REF)
    p.update(user_kwargs)

    # Build the full model + OV2 dataset graph from the Adam recipe, passing the
    # train/seq knobs through (the Adam recipe reads these from user_kwargs).
    cfg = qwen3_ov2_8b_sft_config(
        hf_path=hf_path,
        seq_length=p["seq_length"],
        global_batch_size=p["global_batch_size"],
        micro_batch_size=p["micro_batch_size"],
        train_iters=p["train_iters"],
    )

    # ---- swap optimizer: Adam -> distributed Muon (dist_muon) ----
    opt_cfg, sched_cfg = distributed_muon_with_cosine_annealing(
        lr=p["lr"],
        min_lr=p["min_lr"],
        lr_warmup_iters=p["lr_warmup_iters"],
        lr_decay_iters=p["lr_decay_iters"] if p["lr_decay_iters"] is not None else p["train_iters"],
        weight_decay=p["weight_decay"],
        adam_beta1=p["adam_beta1"],
        adam_beta2=p["adam_beta2"],
        muon_momentum=p["muon_momentum"],
        muon_matched_adamw_rms=p["muon_matched_adamw_rms"],
        clip_grad=p["clip_grad"],
    )
    cfg.optimizer = opt_cfg
    cfg.scheduler = sched_cfg
    return cfg
