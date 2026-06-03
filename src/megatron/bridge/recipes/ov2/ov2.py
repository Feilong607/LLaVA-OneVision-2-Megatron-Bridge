# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""OV2.1-4B (p16m33) Bridge-native recipes (run via run_recipe.py --step_func ov2_step).

Mirrors the Qwen3-VL native stack (qwen3_vl.py): an EnergonProvider subclass carrying the OV2
task encoder, plus a parameterless recipe builder that assembles the ConfigContainer using
_sft_common_vlm() (NullTokenizer, torch_dist checkpoint save/load, VLM DDP). The legacy OV2 mcore
stitch ckpt loads via the provider's pre_wrap_hook (not Bridge's torch_dist loader).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Optional

import torch

from megatron.bridge.data.energon.energon_provider import EnergonProvider
from megatron.bridge.data.utils import DatasetBuildContext
from megatron.bridge.models.qwen_vl_ov2.ov2_provider import LlavaOnevision2Provider
from megatron.bridge.recipes.common import _sft_common_vlm
from megatron.bridge.recipes.utils.optimizer_utils import (
    distributed_fused_adam_with_cosine_annealing,
    distributed_muon_with_cosine_annealing,
)
from megatron.bridge.training.config import ConfigContainer

# --- OV2.1-4B p16m33 assets ---
OV2_LLM_HF = "/ov2/pretrain_models/Qwen3-4B-Instruct-2507"
OV2_MCORE_CKPT = "/ov2/pretrain_models/lmms-lab/LLaVA-OneVision-2-4B-p16m33-mcore-tp1-pp1"
OV2_HF_PROC = "/ov2/pretrain_models/lmms-lab/LLaVA-OneVision-2-4B-p16m33"
# AIAK date0511 stage-1 (adapter-only) mcore ckpt — the reference stage-1 the dev-clone aligned to.
# Stage-2 inits from this KNOWN-GOOD trained adapter (load_ov2_adapter=True) to remove any doubt
# about the stage-1. Dir holds release/mp_rank_00/model_optim_rng.pt + latest_checkpointed_iteration.txt.
AIAK_STAGE1_CKPT = "/vlm/yinxie/code/OV2/OV2_public_main/checkpoints/date0511-LLaVA-OneVision-2-4B-p16m33-mcore-tp1-pp1-stage1-alignment-adapter-only/ax_stage_1_alignment_p16m3_adapter_only"
N_SAMPLES = 558128
SEQ_LEN = 32000

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class OV2EnergonProvider(EnergonProvider):
    """EnergonProvider carrying the OV2 task encoder. (Base EnergonProvider lacks a `tokenizer`
    field referenced by build_datasets, and `dataloader_save` is read via getattr by the
    checkpointer — add both here.)"""

    tokenizer: Optional[Any] = None
    dataloader_save: Optional[str] = None

    def build_datasets(self, context: DatasetBuildContext):
        from megatron.bridge.data.energon.base_energon_datamodule import EnergonMultiModalDataModule

        assert self.path, "OV2EnergonProvider.path must be set (CLI: dataset.path=<dir>)"
        if self.task_encoder is not None:
            self.task_encoder.seq_len = self.seq_length
            self.task_encoder.seq_length = self.seq_length
        dataset = EnergonMultiModalDataModule(
            path=self.path,
            tokenizer=context.tokenizer if context.tokenizer is not None else self.tokenizer,
            image_processor=self.image_processor,
            seq_length=self.seq_length,
            task_encoder=self.task_encoder,
            micro_batch_size=self.micro_batch_size,
            global_batch_size=self.global_batch_size,
            num_workers=self.num_workers,
            pg_collection=context.pg_collection,
        )
        train_loader = dataset.train_dataloader()
        try:
            val_loader = dataset.val_dataloader()
        except Exception as e:
            logger.warning("OV2EnergonProvider: no usable 'val' split (%s); reusing train loader.", e)
            val_loader = dataset.train_dataloader()
        # Return the EnergonDataloader OBJECTS (not iter()): with dataloader_type="external",
        # _get_iterator wraps them directly as RerunDataIterator(loader), so train_iterator.iterable
        # is the EnergonDataloader and maybe_save_dataloader_state's .iterable.save_state() works.
        return (train_loader, val_loader, val_loader)


def _make_ov2_energon_dataset(
    hf_processor_path: str, seq_length: int, micro_batch_size: int, global_batch_size: int,
    dataloader_save: Optional[str] = None,
) -> OV2EnergonProvider:
    from megatron.bridge.recipes.ov2.data.energon.task_encoder import OV2TaskEncoder

    te = OV2TaskEncoder(hf_processor_path=hf_processor_path, seq_length=seq_length)
    return OV2EnergonProvider(
        path="",                                  # set via CLI: dataset.path=/vlm/data/blip_laion_cc_sbu_558k_wds
        tokenizer=None,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        num_workers=2,
        dataloader_type="external",
        task_encoder=te,
        pack_sequences_in_batch=False,
        dataloader_save=dataloader_save,
    )


def ov2_1_stage1_adapter_only_config() -> ConfigContainer:
    """OV2.1-4B (p16m33) stage-1 adapter-only alignment (Bridge-native).

    Freeze LLM + vision, train the m33 adapter. AIAK stage-1: AdamW(0.9,0.99,eps1e-5,wd0),
    lr 2e-5 -> cosine -> 1e-6, warmup-frac 0.002, clip 1.0, gbs 256, mbs 1, bf16, 1 epoch over 558k.
    Weights load via the provider's stitch pre_wrap_hook (adapter trained fresh).
    """
    cfg = _sft_common_vlm()
    train_iters = math.ceil(N_SAMPLES / 256)      # 2181 @ gbs=256

    # ---- Model provider (built from the Qwen3-4B AutoBridge provider) ----
    cfg.model = LlavaOnevision2Provider.from_llm(
        OV2_LLM_HF,
        ov2_mcore_ckpt_path=OV2_MCORE_CKPT,
        load_ov2_adapter=False,
        load_ov2_vision=True,
        load_llm_weights=False,
        freeze_language_model=True,
        freeze_vision_model=True,
        freeze_adapter=False,                     # train the m33 adapter
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=False,
    )
    cfg.model.seq_length = SEQ_LEN
    cfg.model.pipeline_dtype = None
    cfg.model.register_ov2_ckpt_hook(skip_if_resumable_load=cfg.checkpoint.load)

    # ---- Train / validation ----
    cfg.train.train_iters = train_iters
    cfg.train.global_batch_size = 256
    cfg.train.micro_batch_size = 1
    cfg.validation.eval_iters = 0                 # 558k has no usable val split

    # ---- Optimizer + cosine schedule (AIAK stage-1) ----
    opt_cfg, sched_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=max(1, int(0.002 * train_iters)),
        lr_decay_iters=train_iters,
        max_lr=2e-5,
        min_lr=1e-6,
    )
    cfg.optimizer = opt_cfg
    cfg.scheduler = sched_cfg
    cfg.optimizer.adam_beta1 = 0.9
    cfg.optimizer.adam_beta2 = 0.99
    cfg.optimizer.adam_eps = 1e-5
    cfg.optimizer.weight_decay = 0.0
    cfg.optimizer.clip_grad = 1.0
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # ---- Dataset (Energon; path via CLI) + native dataloader-state save ----
    cfg.dataset = _make_ov2_energon_dataset(
        hf_processor_path=OV2_HF_PROC,
        seq_length=SEQ_LEN,
        micro_batch_size=cfg.train.micro_batch_size,
        global_batch_size=cfg.train.global_batch_size,
        dataloader_save=cfg.checkpoint.save,      # activates maybe_save_dataloader_state
    )

    # ---- Token-weighted loss (AIAK): loss = sum(token losses)/sum(tokens) over DP+grad-accum.
    # calculate_per_token_loss=True makes mcore normalize by the GLOBAL token count (what the
    # standalone trainer did manually with /global_tok); average_in_collective=False sums grads
    # across DP so that normalization is correct. Without these the backward loss is the raw sum
    # -> ~token_count x too-large grads (grad norm ~1e5). ----
    cfg.model.calculate_per_token_loss = True

    # ---- DDP / precision ----
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = False
    cfg.mixed_precision = "bf16_mixed"
    cfg.comm_overlap = None
    # checkpoint.pretrained_checkpoint stays None; checkpoint.load (from _sft_common_vlm)
    # satisfies finetune()'s assert and gives warm resume; cold-start weights come from the hook.
    return cfg


def ov2_1_stage2_vit_adapter_muon_config() -> ConfigContainer:
    """OV2.1-4B (p16m33) stage-2 SFT — true vit + adapter (distributed Muon), Bridge-native.

    Freeze LLM; TRAIN vision_model + m33 adapter. Mirrors AIAK date0523 stage-2:
    Muon(momentum 0.95, ns-steps 5, matched-adamw-rms 0.2) + AdamW(0.9,0.99,eps1e-5) for the
    scalar/1-D params; lr 2e-5 CONSTANT (cosine flat, warmup 0); clip 1.0; wd 0; gbs 128, mbs 1;
    non-packed LLaVA-Next 780k (MultiMixQASample); bf16; 1 epoch (6094 steps).

    Init: a TRAINED stage-1 checkpoint loads MODEL-ONLY via checkpoint.pretrained_checkpoint
    (set by the launcher == AIAK '--load <stage1> --no-load-optim --no-load-rng'). The OV2 mcore
    stitch hook supplies the pretrained LLM + vision base when no stage-1 ckpt is given (smoke); the
    m33 adapter is NOT loaded from the (merge-2) stitch base (load_ov2_adapter=False) — it comes from
    the (merge-3) stage-1 checkpoint, or is trained fresh in a smoke.
    """
    cfg = _sft_common_vlm()
    train_iters = math.ceil(780000 / 128)         # 6094 @ gbs=128 (LLaVA-Next 780k, 1 epoch)

    # ---- Model provider: freeze LLM, TRAIN vision + adapter ----
    cfg.model = LlavaOnevision2Provider.from_llm(
        OV2_LLM_HF,
        ov2_mcore_ckpt_path=AIAK_STAGE1_CKPT,      # init from the AIAK date0511 stage-1 (adapter-only)
        load_ov2_adapter=True,                     # load the TRAINED merge-3 adapter (the point of chaining)
        load_ov2_vision=True,
        load_llm_weights=False,
        freeze_language_model=True,
        freeze_vision_model=False,                 # TRAIN the vision tower
        freeze_adapter=False,                      # TRAIN the m33 adapter
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=False,
    )
    cfg.model.seq_length = SEQ_LEN
    cfg.model.pipeline_dtype = None
    cfg.model.register_ov2_ckpt_hook(skip_if_resumable_load=cfg.checkpoint.load)

    # ---- Train / validation ----
    cfg.train.train_iters = train_iters
    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 1
    cfg.validation.eval_iters = 0

    # ---- Optimizer: distributed Muon, constant LR (AIAK date0523) ----
    opt_cfg, sched_cfg = distributed_muon_with_cosine_annealing(
        max_lr=2e-5,
        min_lr=2e-5,                               # == max_lr => constant LR (cosine stays flat)
        lr_warmup_iters=0,
        lr_decay_iters=train_iters,
        weight_decay=0.0,
        muon_momentum=0.95,
        muon_num_ns_steps=5,
        clip_grad=1.0,
    )
    cfg.optimizer = opt_cfg
    cfg.scheduler = sched_cfg
    cfg.optimizer.clip_grad = 1.0
    cfg.optimizer.muon_matched_adamw_rms = 0.2     # AIAK date0523
    cfg.optimizer.adam_beta1 = 0.9                 # Muon's AdamW for scalar/1-D params
    cfg.optimizer.adam_beta2 = 0.99
    cfg.optimizer.adam_eps = 1e-5
    cfg.optimizer.weight_decay = 0.0
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32
    # Muon ("dist_muon") runs via Megatron's LayerWiseDistributedOptimizer; the MIMO DDP wrapper
    # does NOT tag params for layer-wise buffer routing, so with a distributed optimizer the
    # Adam-vector DistOpt receives a grad-buffer that also holds Muon matrices -> KeyError in
    # distrib_optimizer._build_optimizer_group_ranges. Turn the distributed optimizer OFF (Muon
    # still shards matrix state via the LayerWise legacy path); cheap here since only vision+adapter
    # train. Must set BOTH cfg.optimizer and cfg.ddp (a config hook force-enables both if either True).
    cfg.optimizer.use_distributed_optimizer = False

    # ---- Dataset (Energon; path via CLI; non-packed MultiMixQASample) + dataloader-state save ----
    cfg.dataset = _make_ov2_energon_dataset(
        hf_processor_path=OV2_HF_PROC,
        seq_length=SEQ_LEN,
        micro_batch_size=cfg.train.micro_batch_size,
        global_batch_size=cfg.train.global_batch_size,
        dataloader_save=cfg.checkpoint.save,
    )

    # ---- Token-weighted loss (AIAK), same as stage-1 ----
    cfg.model.calculate_per_token_loss = True

    # ---- DDP / precision ----
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.use_distributed_optimizer = False      # must match cfg.optimizer (see Muon note above)
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = False
    cfg.mixed_precision = "bf16_mixed"
    cfg.comm_overlap = None
    return cfg
