# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""OV2 (LLaVA-OneVision-2) Bridge-native recipes (run via run_recipe.py --step_func ov2_step).

Mirrors the Qwen3-VL native stack (qwen3_vl.py / qwen35_vl.py): a path factory + one
`_ov2_common(backbone, stage, ...)` helper + per-size recipe functions. An EnergonProvider
subclass carries the OV2 task encoder, and the ConfigContainer is assembled with _sft_common_vlm()
(NullTokenizer, torch_dist checkpoint save/load, VLM DDP). The legacy OV2 mcore stitch ckpt loads
via the provider's pre_wrap_hook (not Bridge's torch_dist loader).

3 backbones (all AutoBridge-supported); the OV2 vision encoder + m33 adapter + step are
LLM-agnostic (the adapter auto-sizes to llm_cfg.hidden_size), so only the per-backbone PATHS and a
dense-vs-MoE flag change:

  * qwen3-4b        Qwen3-4B-Instruct-2507  (dense, qwen3,        36L, hidden 2560) — the verified base.
  * qwen3-8b        Qwen3-8B                (dense, qwen3,        36L, hidden 4096).
  * qwen3.5-35b-a3b Qwen3.5-35B-A3B         (MoE,   qwen3_5_moe,  40L, hidden 2048) — EP + MoE provider.

Stage-1 = adapter-only alignment (AdamW, gbs 256, 558k). Stage-2 = vit+adapter SFT (distributed
Muon, gbs 128, LLaVA-Next 780k). Both use the AIAK token-weighted loss (calculate_per_token_loss).

Back-compat: the original 4B function names (ov2_1_stage1_adapter_only_config /
ov2_1_stage2_vit_adapter_muon_config) are kept as aliases of ov2_4b_stage1 / ov2_4b_stage2.
"""
from __future__ import annotations

import logging
import math
import os
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

logger = logging.getLogger(__name__)

# --- Shared OV2 p16m33 training constants (per AIAK) ---
SEQ_LEN = int(os.environ.get("OV2_SEQ_LEN", "32768"))   # seq length. AIAK OV2 default = 32768 (examples/llava_onevision*/quick_start_*/stage_1.5_mid_training_*.sh: SEQ_LEN=${3:-32768}); the old "32000" was wrong. Override via OV2_SEQ_LEN=; single source -> model (cfg.model.seq_length) + dataset + task_encoder stay in sync. (The A800 midtrain launcher overrides to 10192 = seed85m packed length.)
# Each is env-overridable (same single-source pattern as SEQ_LEN) so the launcher script is the one
# tuning surface; defaults are UNCHANGED (the AIAK values). gbs/n_samples feed train_iters=ceil(n/gbs)
# and the LR schedule -- override the recipe-side env (OV2_*) AND the matching script var together.
N_SAMPLES = int(os.environ.get("OV2_N_SAMPLES", "558128"))            # blip_laion_cc_sbu_558k (stage-1 alignment, 1 epoch)
STAGE1_GBS = int(os.environ.get("OV2_STAGE1_GBS", "256"))
STAGE2_GBS = int(os.environ.get("OV2_STAGE2_GBS", "128"))
STAGE2_N_SAMPLES = int(os.environ.get("OV2_STAGE2_N_SAMPLES", "780000"))     # LLaVA-Next 780k (stage-2 SFT, 1 epoch)
MIDTRAIN_GBS = int(os.environ.get("OV2_MIDTRAIN_GBS", "128"))            # AIAK date0528 mid-train (full-model SFT) gbs
MIDTRAIN_N_SAMPLES = int(os.environ.get("OV2_MIDTRAIN_N_SAMPLES", "780000"))   # default LLaVA-Next 780k (date0528,
                              # NON-packed). NOTE: AIAK's true stage_1p5 uses a 47M PACKED corpus
                              # (OFFLINE_PACKED_DATA=1); Bridge has no in-batch packing yet, so we default to
                              # the non-packed 780k. Override via OV2_MIDTRAIN_N_SAMPLES= or CLI train.train_iters.


# =============================================================================
# Per-backbone path factory
# =============================================================================
# Returns the four asset paths each backbone needs:
#   llm_hf       — HF dir for the LLM (AutoBridge reads arch/config + optionally weights from here).
#   mcore_ckpt   — assembled OV2 VLM mcore ckpt (3-sibling: language_model + vision_model + adapter)
#                  used as the stage-1 base stitch (and stage-2 smoke base).
#   hf_proc      — HF processor dir for the Energon task encoder (image processor + tokenizer).
#   stage1_ckpt  — the TRAINED stage-1 (vit+adapter, Muon) ckpt that stage-2 chains from
#                  (== AIAK '--load <stage1> --no-load-optim --no-load-rng').
#
# 4B values are VERIFIED (the running recipe). 8B / 35B-A3B values are the documented dirs; the
# exact mcore-ckpt SUBPATH, processor dir, and a real trained stage-1 for those two are NOT
# verifiable from the local copy and MUST be confirmed on the server (see report).
_OV2_PRETRAIN_ROOT = os.environ.get("OV2_PRETRAIN_ROOT", "/ov2/pretrain_models")

_OV2_BACKBONES: dict[str, dict[str, Any]] = {
    "qwen3-4b": {
        "is_moe": False,
        "llm_hf": f"{_OV2_PRETRAIN_ROOT}/Qwen3-4B-Instruct-2507",
        "mcore_ckpt": f"{_OV2_PRETRAIN_ROOT}/lmms-lab/LLaVA-OneVision-2-4B-p16m33-mcore-tp1-pp1",
        "hf_proc": f"{_OV2_PRETRAIN_ROOT}/lmms-lab/LLaVA-OneVision-2-4B-p16m33",
        # VERIFIED 4B p16m33 vision tower (the validated config; DO NOT change).
        "vision_patch_size": 16,
        "vision_spatial_merge_size": 3,        # adapter merges 1024 * 3^2 = 9216
        "vision_hidden_size": 1024,
        "vision_num_layers": 24,
        "vision_model_name": None,             # 4B uses get_vision_config's default base geometry
        # Stage-1 the AIAK date0523 stage-2 ACTUALLY chains from (its .sh CHECKPOINT_PATH): the
        # date0513 "corrected Muon stage-1 vit-adapter" — trains BOTH vision tower and adapter
        # (Muon), so stage-2 starts from a TRAINED vit (-> aligns ~1.27). NOT date0511 (adapter-only,
        # vit frozen at the OV2 base -> stage-2 plateaus ~1.5). mcore dir: release/mp_rank_00 +
        # latest_checkpointed_iteration.txt.
        "stage1_ckpt": (
            "/vlm/yinxie/code/OV2/OV2_public_main/checkpoints/"
            "date0513-corrected-muon-stage1-vit-adapter/"
            "date0511_ax_stage_1_alignment_p16m3_packed_new16_muon"
        ),
    },
    "qwen3-8b": {
        "is_moe": False,
        "llm_hf": f"{_OV2_PRETRAIN_ROOT}/Qwen3-8B",
        # VERIFIED single-rank tp1_pp1 release (release/mp_rank_00/model_optim_rng.pt) -> the
        # current single-chunk stitch-load handles it directly (like the 4B). text hidden=4096/36L.
        "mcore_ckpt": f"{_OV2_PRETRAIN_ROOT}/llava_onevision2/llava_onevision2_8b/llava_onevision2_8b_stage1.5_mcore_release",
        # 8B's OWN processor (auto-model): a standard Qwen2_5_VLProcessor w/ patch_size=14, merge_size=2
        # — NOT a custom config (the task encoder loads it via AutoProcessor exactly like the 4B one).
        # REQUIRED: the 4B p16m33 processor patchifies pixels at patch16 (pixel_values divisible by
        # 3*16*16=768); feeding those to the 8B patch14 tower crashes at patch_embed reshape
        # ([-1,3,14,14] needs /588). This dir produces patch14 grids that match the 8B tower.
        "hf_proc": f"{_OV2_PRETRAIN_ROOT}/llava_onevision2/llava_onevision2_8b/auto-model",
        # 8B vision tower: patch14 / merge2 (1024-d / 24L, same size class as 4B but patch14+merge2).
        # The 8B ckpt's patch_embed is [1024,3,14,14] — building patch16 was the smoke-test mismatch.
        "vision_patch_size": 14,
        "vision_spatial_merge_size": 2,        # adapter merges 1024 * 2^2 = 4096
        "vision_hidden_size": 1024,
        "vision_num_layers": 24,
        "vision_model_name": None,             # base 1024/24 geometry; only patch/merge differ from 4B
        "stage1_ckpt": None,   # no verified trained stage-1 yet; set via CLI for stage-2 chaining.
    },
    # The A3B-MoE OV2 backbone. NAMING: the user calls this "qwen3.5-35b-a3b", but the only existing
    # OV2-MoE ckpt (llava_onevision2_30b_a3b) is built on Qwen3-30B-A3B, NOT Qwen3.5-35B-A3B — proven
    # by the ckpt's own auto_model/config.json: text model_type=qwen3_moe, num_experts=128,
    # moe_intermediate_size=768, num_hidden_layers=48, hidden=2048; and by the EP8 shard
    # (router=[128,2048], 16 local_experts/shard, expert fc1=[1536,2048] => ffn 768). Qwen3.5-35B-A3B
    # is a DIFFERENT base (num_experts=256, moe_intermediate=512) with NO OV2 ckpt, so pairing it with
    # this ckpt cannot load. We target the loadable+validatable asset (Qwen3-30B-A3B). To build OV2 on
    # the literal Qwen3.5-35B-A3B instead, take its LLM from HF (load_llm_weights=True) + stitch ONLY
    # the OV2 vision tower from this ckpt's vision_model + fresh adapter (no aligned ckpt -> stage-1).
    "qwen3-30b-a3b": {
        "is_moe": True,
        "llm_hf": os.environ.get("OV2_LLM_HF_30B", f"{_OV2_PRETRAIN_ROOT}/Qwen3-30B-A3B-Instruct-2507"),   # qwen3_moe: 128 experts, ffn 768, 48L
        # OV2-30B-A3B mcore ckpt is EP8-SHARDED (release/mp_rank_00_000 .. _007). Each shard replicates
        # non-expert + vision_model(291) + adapter(6) and carries its own 16 local_experts (128/8). The
        # EP-aware branch in load_ov2_mcore_checkpoint loads THIS rank's shard; build with EP=8. The
        # experts are PER-EXPERT (SequentialMLP keys) in the ckpt; the BUILT model keeps moe_grouped_gemm=True
        # (TE-grouped) and load_ov2_mcore_checkpoint remaps per-expert -> grouped at load time.
        # Other EP layouts available: stage_0_tp1_pp2_ep8 / stage_0_tp1_pp2_ep4_16_32.
        "mcore_ckpt": f"{_OV2_PRETRAIN_ROOT}/llava_onevision2/llava_onevision2_30b_a3b/stage_0_tp1_pp1_ep8",
        # OWN processor (auto_model, NOTE underscore): standard Qwen2_5_VLProcessor, patch14 / merge2
        # (same reason as 8B; the 4B p16m33 processor patchifies at patch16 and crashes a patch14 tower).
        "hf_proc": os.environ.get("OV2_HF_PROC_30B", f"{_OV2_PRETRAIN_ROOT}/llava_onevision2/llava_onevision2_30b_a3b/auto_model"),
        # Vision tower = patch14 / merge2 / hidden 1024 / 24L — IDENTICAL size class to 8B (NOT a larger
        # 1664/48 "vision-2b"). CONFIRMED by the ckpt config (vision hidden_size=1024, num_hidden_layers
        # =24) and the 291 vision keys/shard (~12/layer x 24). adapter merges 1024 * 2^2 = 4096.
        "vision_patch_size": 14,
        "vision_spatial_merge_size": 2,
        "vision_hidden_size": 1024,
        "vision_num_layers": 24,
        "vision_model_name": None,
        "stage1_ckpt": None,   # no verified trained stage-1 yet; set via CLI for stage-2 chaining.
    },
    # p16m33 variant of the 30B-A3B backbone: SAME Qwen3-30B-A3B MoE LLM, vision tower is patch16 /
    # merge3 (OV2.1 p16m33 OneVisionEncoder); adapter merges 1024*3^2=9216 -> 2048. Vision weights are
    # baked from the standalone HF OneVisionEncoder (onevision_encoder_patch16_0507-tf57) into the
    # combined mcore stitch at build time via A800/convert from_base --vision_hf. hf_proc = p16m33 proc.
    "qwen3-30b-a3b-p16m33": {
        "is_moe": True,
        "llm_hf": os.environ.get("OV2_LLM_HF_30B", f"{_OV2_PRETRAIN_ROOT}/Qwen3-30B-A3B-Instruct-2507"),
        "mcore_ckpt": os.environ.get(
            "OV2_MCORE_30B_P16M33",
            f"{_OV2_PRETRAIN_ROOT}/llava_onevision2/llava_onevision2_30b_a3b_p16_m33/stage_0_tp1_pp1_ep8",
        ),
        "hf_proc": os.environ.get(
            "OV2_HF_PROC_30B_P16M33",
            f"{_OV2_PRETRAIN_ROOT}/llava_onevision2/llava_onevision2_30b_a3b_p16_m33/auto_model",
        ),
        "vision_patch_size": 16,
        "vision_spatial_merge_size": 3,
        "vision_hidden_size": 1024,
        "vision_num_layers": 24,
        "vision_model_name": None,
        "stage1_ckpt": None,
    },
}


def _ov2_backbone_paths(backbone: str) -> dict[str, Any]:
    """Return the per-backbone config dict for a known backbone key.

    Keys: is_moe, llm_hf, mcore_ckpt, hf_proc, stage1_ckpt, and the per-backbone vision-tower
    geometry (vision_patch_size, vision_spatial_merge_size, vision_hidden_size, vision_num_layers,
    vision_model_name).
    """
    key = backbone.strip().lower()
    if key not in _OV2_BACKBONES:
        raise ValueError(
            f"Unknown OV2 backbone {backbone!r}; expected one of {sorted(_OV2_BACKBONES)}"
        )
    return dict(_OV2_BACKBONES[key])


# =============================================================================
# Energon dataset provider (carries the OV2 task encoder)
# =============================================================================
@dataclass(kw_only=True)
class OV2EnergonProvider(EnergonProvider):
    """EnergonProvider carrying the OV2 task encoder. (Base EnergonProvider lacks a `tokenizer`
    field referenced by build_datasets, and `dataloader_save` is read via getattr by the
    checkpointer — add both here.)"""

    tokenizer: Optional[Any] = None
    dataloader_save: Optional[str] = None
    # Energon's default shuffle_buffer_size is only 100 — far too small for 558k samples sharded
    # across many DP ranks (each rank reads few tar shards; a 100-sample window barely mixes across
    # them). The adapter-only stage-1 is hyper-sensitive to early-batch diversity, so at 16 ranks the
    # tiny buffer left the run ~0.35 above AIAK. Raise it (set in _make_ov2_energon_dataset).
    shuffle_buffer_size: int = 100

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
            shuffle_buffer_size=self.shuffle_buffer_size,
            pg_collection=context.pg_collection,
            dataloader_load=self.dataloader_load,
            # AIAK parity (dataloader_provider.py:24-34 sets image_decode="pil"): force Energon to decode
            # images via PIL so pixel inputs are byte-level identical to AIAK. Without it the Energon
            # default decoder may differ. Flows through **kwargs -> get_train_dataset(image_decode=...).
            image_decode="pil",
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
    num_workers: int = 2, shuffle_buffer_size: int = 100,   # original/safe defaults (small samples + buffer)
    spatial_merge_size: Optional[int] = None,               # per-backbone merge override (None => read from processor)
) -> OV2EnergonProvider:
    from megatron.bridge.recipes.ov2.data.energon.task_encoder import OV2TaskEncoder

    te = OV2TaskEncoder(
        hf_processor_path=hf_processor_path, seq_length=seq_length,
        spatial_merge_size=spatial_merge_size,
    )
    return OV2EnergonProvider(
        path="",                                  # set via CLI: dataset.path=/vlm/data/blip_laion_cc_sbu_558k_wds
        tokenizer=None,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        num_workers=num_workers,                  # per-stage: stage-1/blip uses 4 (1 shard/worker across 64
        shuffle_buffer_size=shuffle_buffer_size,  # shards) + buffer 2000; stage-2/llava_next keeps 2/100
                                                  # (samples ~10x larger -> a big buffer OOMs the host)
        dataloader_type="external",
        task_encoder=te,
        pack_sequences_in_batch=False,
        dataloader_save=dataloader_save,
    )


# =============================================================================
# Shared recipe builder
# =============================================================================
def _ckpt_is_torch_dist(path) -> bool:
    """True if `path` is a Bridge torch_dist checkpoint (iter_<N>/.metadata, no AIAK release/mp_rank).

    The OV2 stitch loader (load_ov2_mcore_checkpoint) reads ONLY the AIAK .pt layout
    (release/mp_rank_00[_NNN]/model_optim_rng.pt). A torch_dist ckpt (e.g. built by A800/convert
    from_base, saved via save_megatron_model) must instead load as checkpoint.pretrained_checkpoint.
    Detect the format so _ov2_common routes it correctly (else: IsADirectoryError/FileNotFoundError).
    """
    if not path or not os.path.isdir(path):
        return False
    if os.path.isdir(os.path.join(path, "release")):
        return False  # AIAK stitch base (release/mp_rank_*/model_optim_rng.pt)
    if os.path.exists(os.path.join(path, ".metadata")):
        return True   # direct dist-checkpoint dir
    try:
        for d in os.listdir(path):
            if d.startswith("iter_") and os.path.exists(os.path.join(path, d, ".metadata")):
                return True
    except OSError:
        return False
    return False


def _ov2_common(
    backbone: str,
    stage: str,
    *,
    seq_len: int = SEQ_LEN,
    n_samples: Optional[int] = None,
    global_batch_size: Optional[int] = None,
    micro_batch_size: int = 1,
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    context_parallel_size: int = 1,
    sequence_parallel: bool = False,
    max_lr: float = 2e-5,
    min_lr: Optional[float] = None,
) -> ConfigContainer:
    """Assemble the OV2 ConfigContainer for one (backbone, stage).

    stage="stage1": freeze LLM + vision, train m33 adapter; AdamW + cosine (lr 2e-5 -> 1e-6,
        warmup-frac 0.002); gbs 256; adapter loaded fresh (load_ov2_adapter=False); base weights
        come from the backbone's assembled mcore ckpt via the provider stitch hook.
    stage="stage2": freeze LLM, TRAIN vision + adapter; distributed Muon, constant lr 2e-5
        (cosine flat, warmup 0, matched-adamw-rms 0.2, momentum 0.95, ns-steps 5); gbs 128; inits
        from the TRAINED stage-1 ckpt (load_ov2_adapter=True) when available, else the mcore base.
    stage="midtrain": TRAIN the FULL model (unfreeze LLM + vision + adapter), per AIAK date0528
        --trainable-modules "language_model adapter vision_model"; Muon (matched-adamw-rms 0.15),
        constant lr 2e-5, gbs 128, activation recompute ON; inits from a TRAINED prior-stage ckpt
        (CLI checkpoint.pretrained_checkpoint) else the mcore base. On a MoE backbone the experts are
        now trainable, so Muon would hit the EP backward all-to-all deadlock -> midtrain auto-uses
        AdamW(distopt=True) for MoE (dense backbones keep Muon).

    The OV2 model/vision/adapter/step are LLM-agnostic; only the provider's llm_hf_path + the
    stitch-ckpt path + (for MoE) EP/SP differ across backbones.
    """
    assert stage in ("stage1", "stage2", "midtrain"), f"stage must be 'stage1'|'stage2'|'midtrain', got {stage!r}"
    paths = _ov2_backbone_paths(backbone)
    is_moe = paths["is_moe"]

    if global_batch_size is None:
        global_batch_size = {"stage1": STAGE1_GBS, "stage2": STAGE2_GBS, "midtrain": MIDTRAIN_GBS}[stage]
    if n_samples is None:
        n_samples = {"stage1": N_SAMPLES, "stage2": STAGE2_N_SAMPLES, "midtrain": MIDTRAIN_N_SAMPLES}[stage]
    if min_lr is None:
        min_lr = 1e-6 if stage == "stage1" else max_lr  # stage-2 + midtrain: constant LR (min==max), per AIAK date0528

    cfg = _sft_common_vlm()
    train_iters = math.ceil(n_samples / global_batch_size)

    # ---- Model provider (built from the backbone's AutoBridge provider) ----
    # stage-1: load vision from the mcore base, train the adapter FRESH (load_ov2_adapter=False).
    # stage-2: chain from the TRAINED stage-1 ckpt (load_ov2_adapter=True) when one exists; else
    #          fall back to the mcore base (smoke) and train the adapter fresh.
    if stage == "stage1":
        ckpt_path = paths["mcore_ckpt"]
        load_adapter = False
        freeze_vision_model = True
        freeze_language_model = True
    else:
        # stage2: freeze LLM, train vision+adapter. midtrain: train the FULL model (unfreeze the LLM),
        # per AIAK date0528 --trainable-modules "language_model adapter vision_model".
        ckpt_path = paths["stage1_ckpt"] or paths["mcore_ckpt"]
        load_adapter = bool(paths["stage1_ckpt"])  # only load a merge-3 adapter from a real prior stage
        freeze_vision_model = __import__("os").environ.get("OV2_FREEZE_VISION") == "1"   # TRAIN vision (OV2_FREEZE_VISION=1 freezes it for ablation)
        freeze_language_model = (stage != "midtrain") or __import__("os").environ.get("OV2_FREEZE_LLM") == "1"  # midtrain unfreezes the LLM; stage2 keeps it frozen

    # OV2_SKIP_BASE_STITCH=1 (set by the GB200 launcher): mid-train resumes straight from a FULL
    # stage-2 ckpt, so skip the stage_0 base stitch entirely (no stage_0 needed on the box). SAFE-
    # GUARDED: a real OV2_INIT_CKPT must exist, else building on random weights -> refuse (the exact
    # NaN trap the always-stitch default guarded against). ckpt_path=None makes register_ov2_ckpt_hook
    # a no-op; checkpoint.pretrained_checkpoint then loads ALL weights from the stage-2 ckpt.
    if os.environ.get("OV2_SKIP_BASE_STITCH") == "1":
        _init = os.environ.get("OV2_INIT_CKPT", "")
        if not (_init and os.path.exists(_init)):
            raise FileNotFoundError(
                f"OV2_SKIP_BASE_STITCH=1 but OV2_INIT_CKPT not found: {_init!r}. Refusing to skip the "
                f"stage_0 stitch without a loadable resume (would train on RANDOM weights -> NaN)."
            )
        logging.getLogger(__name__).warning(
            "OV2_SKIP_BASE_STITCH=1: skipping stage_0 stitch; ALL weights come from OV2_INIT_CKPT=%s "
            "(verify iter-1 loss is finite, not NaN).", _init,
        )
        ckpt_path = None

    # Format/role auto-route (fixes torch_dist-vs-AIAK mismatch): a torch_dist mcore_ckpt/stage1_ckpt
    # (iter_*/.metadata; e.g. A800/convert from_base output) is a FULL Bridge ckpt and MUST load via
    # checkpoint.pretrained_checkpoint, NOT the AIAK release/mp_rank stitch (load_ov2_mcore_checkpoint
    # reads only .pt). Detect + route so stage1/2/mid work whether the base is AIAK (stitch) or
    # torch_dist (pretrained_checkpoint). AIAK backbones (4B/8B release/ ckpts) are unaffected.
    _pretrained_ckpt = None
    if ckpt_path is not None and _ckpt_is_torch_dist(ckpt_path):
        logging.getLogger(__name__).warning(
            "[ov2 recipe] mcore_ckpt %s is torch_dist -> loading via checkpoint.pretrained_checkpoint "
            "(skipping the AIAK release/mp_rank stitch).", ckpt_path)
        _pretrained_ckpt = ckpt_path
        ckpt_path = None

    cfg.model = LlavaOnevision2Provider.from_llm(
        paths["llm_hf"],
        is_moe=is_moe,
        ov2_mcore_ckpt_path=ckpt_path,
        load_ov2_adapter=load_adapter,
        load_ov2_vision=True,
        load_llm_weights=False,
        freeze_language_model=freeze_language_model,
        freeze_vision_model=freeze_vision_model,
        freeze_adapter=(__import__("os").environ.get("OV2_FREEZE_ADAPTER") == "1"),   # train adapter (OV2_FREEZE_ADAPTER=1 freezes it for ablation)
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        expert_model_parallel_size=expert_model_parallel_size,
        context_parallel_size=context_parallel_size,
        sequence_parallel=sequence_parallel,
        # Per-backbone vision-tower geometry (so the tower matches the backbone's ckpt, NOT a
        # hardcoded patch16). 4B = patch16/merge3/1024/24 (verified, unchanged).
        vision_patch_size=paths["vision_patch_size"],
        vision_spatial_merge_size=paths["vision_spatial_merge_size"],
        vision_hidden_size=paths["vision_hidden_size"],
        vision_num_layers=paths["vision_num_layers"],
        vision_model_name=paths["vision_model_name"],
    )
    cfg.model.seq_length = seq_len
    cfg.model.pipeline_dtype = None
    # midtrain trains the full model (LLM activations dominate at seq 32000) -> enable activation
    # recompute (full/uniform/1), matching AIAK date0528. provide() threads recompute_activations
    # into build_llava_ov2 (LLM-only; the vision tower keeps recompute off).
    if stage == "midtrain":
        cfg.model.recompute_activations = True
    # Fail fast with a clear message if the backbone's stitch base ckpt is missing on this host. The
    # 8B / 30B-A3B paths are server-specific; override _OV2_BACKBONES or the provider's
    # ov2_mcore_ckpt_path via CLI if your layout differs.
    if ckpt_path and not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"[ov2 recipe] backbone={backbone!r}: OV2 stitch ckpt not found: {ckpt_path}. "
            f"Verify the server layout or override ov2_mcore_ckpt_path via CLI."
        )
    # ALWAYS stitch the base — do NOT pass skip_if_resumable_load. A genuine torch_dist resume runs
    # AFTER this pre_wrap_hook (mcore load_checkpoint) and OVERWRITES the stitched weights, so the
    # skip was only an optimization. Worse, capturing cfg.checkpoint.load HERE grabs the BUILD-TIME
    # default (nemo_experiments/default/checkpoints) because CLI overrides apply later — so a stale
    # ckpt left at that default path would wrongly skip the base load and the model trains on RANDOM
    # weights (observed: OV2-30B-A3B NaN at iter 2 on all ranks). Always stitching is robust; the
    # extra ~30s load on a real resume is harmless.
    cfg.model.register_ov2_ckpt_hook()
    # torch_dist base (auto-routed above) loads as the pretrained_checkpoint default (CLI overrides win).
    if _pretrained_ckpt is not None:
        cfg.checkpoint.pretrained_checkpoint = _pretrained_ckpt
    # OV2_SKIP_BASE_STITCH=1 leaves ckpt_path=None (no stitch hook), so ALL weights must come from
    # the resume ckpt. Wire it from OV2_INIT_CKPT HERE (defense-in-depth) so the recipe is correct
    # even if the launcher forgets the checkpoint.pretrained_checkpoint CLI override -- otherwise the
    # model trains on RANDOM weights -> iter~2 NaN. A real CLI override still wins (applied later).
    elif os.environ.get("OV2_SKIP_BASE_STITCH") == "1":
        _skip_init = os.environ.get("OV2_INIT_CKPT", "")
        if _skip_init and os.path.exists(_skip_init):
            cfg.checkpoint.pretrained_checkpoint = _skip_init

    # ---- Train / validation ----
    cfg.train.train_iters = train_iters
    cfg.train.global_batch_size = global_batch_size
    cfg.train.micro_batch_size = micro_batch_size
    # OV2 fuses image features via masked_scatter and consumes a SINGLE cu_seqlens THD path -> it is
    # hardwired to micro_batch_size==1 (the model also asserts batch==1 at the masked_scatter). With
    # packed data + mbs>1 the loader silently DROPS per-sample cu_seqlens and runs cross-sample
    # FULL-CAUSAL attention (a quiet correctness bug). Fail LOUD at CONFIG time -- before any
    # collective exists, so this raise can never desync ranks (a per-rank raise in batch() could).
    if micro_batch_size != 1:
        raise ValueError(
            f"OV2 requires micro_batch_size==1 (packed THD + masked_scatter image fuse); got "
            f"{micro_batch_size}. Raise the GLOBAL batch size instead (grad-accum), or wire cu_seqlens "
            f"offset-concatenation in task_encoder.batch() + ov2_step before using mbs>1."
        )
    cfg.validation.eval_iters = 0                   # 558k / 780k have no usable val split

    # ---- Optimizer + schedule ----
    # OV2_STAGE2_ADAMW=1 routes stage-2 to the AdamW(distopt=True) path instead of distributed Muon.
    # WHY: distributed Muon forces use_distributed_optimizer=False, which on the MoE backbone (EP8)
    # deadlocks the expert-parallel backward all-to-all (NCCL ALLTOALL_BASE timeout in iter-1 backward).
    # Stage-1 AdamW + distopt=True runs EP8 cleanly, and the dense 4B stage-2 runs Muon+distopt-off
    # cleanly — only Muon + MoE/EP together hang. AdamW here keeps EP8 happy (optimizer differs from the
    # AIAK Muon recipe; acceptable until distributed-Muon+EP is fixed).
    # stage-2 FREEZES the LLM (experts frozen) -> distributed Muon touches ONLY the dense vision+adapter
    # 2-D matrices, never an expert-parallel matrix, so Muon+EP8 runs cleanly here. VERIFIED: MoE p16m33
    # stage-2 trains under dist_muon (OV2_STAGE2_ADAMW=0) with the EP8 all-to-all firing every step and
    # NO hang. MoE stage-2 therefore KEEPS Muon by default (matches the AIAK date0523 Muon recipe);
    # set OV2_STAGE2_ADAMW=1 to force AdamW instead. (NOT auto-routed on is_moe -- that would override an
    # intentional Muon stage-2 and discard Muon optimizer state on resume.)
    _stage2_adamw = stage == "stage2" and os.environ.get("OV2_STAGE2_ADAMW", "0") == "1"
    # midtrain trains the FULL model -> on a MoE backbone the experts UNFREEZE and become trainable, so
    # distributed Muon (use_distributed_optimizer=False) would orthogonalize expert-parallel matrices and
    # risks the EP backward all-to-all deadlock (UNVALIDATED), and AIAK date0528 uses AdamW for MoE
    # midtrain anyway. AUTO-route MoE midtrain to AdamW(distopt=True) (is_moe); dense midtrain keeps Muon.
    # OV2_MIDTRAIN_ADAMW=1 also forces AdamW on a dense backbone.
    _midtrain_adamw = stage == "midtrain" and os.environ.get("OV2_MIDTRAIN_MUON", "0") != "1" and (is_moe or os.environ.get("OV2_MIDTRAIN_ADAMW", "0") == "1")
    if stage == "stage1" or _stage2_adamw or _midtrain_adamw:
        # AIAK stage-1: AdamW(0.9,0.99,eps1e-5,wd0), lr 2e-5 -> cosine -> 1e-6, warmup-frac 0.002,
        # clip 1.0, bf16, 1 epoch over 558k.  (stage-2-AdamW: constant lr via min_lr==max_lr.)
        opt_cfg, sched_cfg = distributed_fused_adam_with_cosine_annealing(
            lr_warmup_iters=(max(1, int(0.002 * train_iters)) if stage == "stage1" else 0),  # AIAK: stage2/midtrain = constant LR, no warmup ramp
            lr_decay_iters=train_iters,
            max_lr=max_lr,
            min_lr=min_lr,
        )
        cfg.optimizer = opt_cfg
        cfg.scheduler = sched_cfg
        cfg.optimizer.adam_beta1 = 0.9
        cfg.optimizer.adam_beta2 = 0.99
        cfg.optimizer.adam_eps = 1e-5
        cfg.optimizer.weight_decay = 0.0
        cfg.optimizer.clip_grad = 1.0
        cfg.optimizer.use_precision_aware_optimizer = (__import__("os").environ.get("OV2_PRECISION_AWARE", "0") == "1")  # offload REQUIRES precision-aware (mcore HybridDeviceOptimizer reuses this path); dtypes stay fp32 so precision is unchanged
        cfg.optimizer.main_grads_dtype = torch.float32
        cfg.optimizer.main_params_dtype = torch.float32
        cfg.optimizer.exp_avg_dtype = torch.float32
        cfg.optimizer.exp_avg_sq_dtype = torch.float32
        use_dist_opt = True
    else:
        # AIAK date0523 stage-2: distributed Muon(momentum 0.95, ns-steps 5, matched-adamw-rms 0.2)
        # + AdamW(0.9,0.99,eps1e-5) for scalar/1-D params; lr 2e-5 CONSTANT; clip 1.0; wd 0.
        opt_cfg, sched_cfg = distributed_muon_with_cosine_annealing(
            max_lr=max_lr,
            min_lr=min_lr,                          # == max_lr => constant LR (cosine stays flat)
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
        # AIAK scales the orthogonalized Muon update by sqrt(max(d_out,d_in)) * matched_adamw_rms
        # (rms 0.2 stage-2 / 0.15 midtrain). In mcore's emerging Muon the sqrt(max) term comes from
        # muon_scale_mode='spectral' (the default) and the rms multiplier is **muon_extra_scale_factor**.
        # NOTE: there is NO `muon_matched_adamw_rms` field on mcore OptimizerConfig — setting it is
        # silently ignored, leaving extra_scale_factor at its default 1.0 => updates ~5-6.7x too large.
        _muon_rms = 0.15 if stage == "midtrain" else 0.2
        cfg.optimizer.muon_extra_scale_factor = _muon_rms   # the field mcore actually reads
        cfg.optimizer.muon_scale_mode = "spectral"          # supplies the sqrt(max(d_out,d_in)) term
        # OV2: the vision tower's fused QKV has a DIFFERENT head/group layout than the LLM, so Muon's
        # split_qkv (mcore default True) reshapes it with the LLM's qkv_split_shapes -> RuntimeError on
        # a backbone whose split-sum does not divide the vision QKV numel (e.g. 30B-A3B if it ever ran
        # Muon), or SILENT vision-QKV gradient corruption where it happens to divide (4B/8B). The LLM
        # QKV in stage-2 is FROZEN (and in dense midtrain a single-matrix orthogonalization is fine),
        # so orthogonalize the fused QKV as one matrix. Correct-by-construction here (previously this
        # was only set via the launch-script CLI optimizer.muon_split_qkv=false, which direct/dense
        # Muon runs did not get).
        cfg.optimizer.muon_split_qkv = False
        cfg.optimizer.adam_beta1 = 0.9              # Muon's AdamW for scalar/1-D params
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
        # still shards matrix state via the LayerWise legacy path); cheap here since only
        # vision+adapter train. Must set BOTH cfg.optimizer and cfg.ddp (a config hook force-enables
        # both if either True).
        cfg.optimizer.use_distributed_optimizer = False
        use_dist_opt = False

    # ---- Weight decay: AIAK uses wd=0. The optimizer_utils helpers hardcode the SCHEDULER's
    # start/end_weight_decay=0.033 (with override_opt_param_scheduler=True), and OptimizerParamScheduler.step()
    # writes param_group['weight_decay']=get_wd() EVERY iter -> it CLOBBERS cfg.optimizer.weight_decay=0.0
    # back to 0.033. Force the scheduler's wd to 0 too so the runtime wd is actually 0 (matches AIAK).
    cfg.scheduler.start_weight_decay = 0.0
    cfg.scheduler.end_weight_decay = 0.0

    # ---- Dataset (Energon; path via CLI) + native dataloader-state save ----
    if stage == "stage1":
        ds_workers, ds_buffer = 4, 2000             # blip 16-rank diversity fix (small samples -> mem-safe)
    else:
        ds_workers, ds_buffer = 2, 100              # llava_next samples ~10x larger -> small buffer
    cfg.dataset = _make_ov2_energon_dataset(
        hf_processor_path=paths["hf_proc"],
        seq_length=seq_len,
        micro_batch_size=cfg.train.micro_batch_size,
        global_batch_size=cfg.train.global_batch_size,
        dataloader_save=cfg.checkpoint.save,        # activates maybe_save_dataloader_state
        num_workers=ds_workers, shuffle_buffer_size=ds_buffer,
        # Make the <|image_pad|> expansion merge match the vision tower's merge. Each backbone uses
        # its OWN matched processor (4B=merge3, 8B/30B=merge2), so we pass the per-backbone merge
        # explicitly to keep the token-count expansion correct regardless of processor defaults.
        # OV2_SPATIAL_MERGE is the config override: set it to A/B a merge value (image AND video pad-count
        # + block layout both read this single value) without editing the per-card path profile; unset ->
        # the backbone's vision_spatial_merge_size. MUST equal the trained vision tower's merge, else the
        # collective n_feat==n_tok check in llava_ov2.forward aborts the job.
        spatial_merge_size=int(os.environ.get("OV2_SPATIAL_MERGE") or paths["vision_spatial_merge_size"]),
    )

    # ---- Token-weighted loss (AIAK): loss = sum(token losses)/sum(tokens) over DP+grad-accum.
    # calculate_per_token_loss=True makes mcore normalize by the GLOBAL token count (what the
    # standalone trainer did manually with /global_tok); average_in_collective=False sums grads
    # across DP so that normalization is correct. The provider.provide() also force-sets these on
    # model.config (the runtime config) — see the CRITICAL fix there. ----
    cfg.model.calculate_per_token_loss = True

    # ---- DDP / precision ----
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.use_distributed_optimizer = use_dist_opt   # must match cfg.optimizer (Muon note above)
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = False
    cfg.mixed_precision = "bf16_mixed"
    cfg.comm_overlap = None
    # CUDA graphs OFF (match AIAK, which trains without them). With the OV2 MIMO model + CUDA
    # graphs, the captured step bypasses the standard Python grad-finalize and mis-scales the
    # adapter gradient across NODES (2-node grad norm ~10x single-node / AIAK ~1.3). Disabling
    # graphs restores the eager grad-finalize path. (Single-node was unaffected; 2-node is.)
    cfg.model.cuda_graph_impl = "none"
    return cfg


# =============================================================================
# Per-size recipe functions
# =============================================================================
def ov2_4b_stage1() -> ConfigContainer:
    """OV2-4B (Qwen3-4B, p16m33) stage-1 adapter-only alignment (Bridge-native, VERIFIED base)."""
    return _ov2_common("qwen3-4b", "stage1")


def ov2_4b_stage2() -> ConfigContainer:
    """OV2-4B (Qwen3-4B, p16m33) stage-2 vit+adapter SFT (distributed Muon, VERIFIED base)."""
    return _ov2_common("qwen3-4b", "stage2")


def ov2_8b_stage1() -> ConfigContainer:
    """OV2-8B (Qwen3-8B, p16m33) stage-1 adapter-only alignment.

    Dense backbone, same code path as 4B (only the LLM width 4096 + paths differ; the adapter
    auto-sizes). PATHS for 8B (mcore ckpt subpath / processor) are UNVERIFIED — confirm on server
    or override via CLI (dataset.hf_processor_path / the provider ckpt path).
    """
    return _ov2_common("qwen3-8b", "stage1")


def ov2_8b_stage2() -> ConfigContainer:
    """OV2-8B (Qwen3-8B, p16m33) stage-2 vit+adapter SFT (distributed Muon).

    Needs a trained stage-1 ckpt to chain from (none wired by default for 8B) — set it via CLI:
    checkpoint.pretrained_checkpoint=<stage1> or edit _OV2_BACKBONES['qwen3-8b']['stage1_ckpt'].
    """
    return _ov2_common("qwen3-8b", "stage2")


def ov2_35b_a3b_stage1() -> ConfigContainer:
    """OV2 A3B-MoE stage-1 adapter-only alignment (Bridge-native).

    Backbone key "qwen3-30b-a3b": the existing OV2-MoE ckpt is built on Qwen3-30B-A3B (qwen3_moe,
    128 experts, ffn 768, 48L) — see the _OV2_BACKBONES note (the user's "qwen3.5-35b-a3b" name is a
    DIFFERENT 256-expert base with no OV2 ckpt). Uses LlavaOnevision2MoEProvider with EP=8 (single
    8xA100-80GB node, TP1/PP1). The EP8-sharded per-expert ckpt loads via the EP-aware stitch branch
    (the built model stays TE-grouped, moe_grouped_gemm=True; load_ov2_mcore_checkpoint remaps the
    per-expert ckpt keys -> grouped at load). The kept name ov2_35b_a3b_* preserves the
    user-facing recipe label.
    """
    return _ov2_common(
        "qwen3-30b-a3b",
        "stage1",
        expert_model_parallel_size=8,
        sequence_parallel=False,                    # TP=1 -> SP must be off (provider.finalize guards this)
    )


def ov2_35b_a3b_stage2() -> ConfigContainer:
    """OV2 A3B-MoE stage-2 vit+adapter SFT (distributed Muon, EP=8).

    Backbone "qwen3-30b-a3b" (see stage1 / the _OV2_BACKBONES note on the naming). Needs a trained
    stage-1 ckpt to chain from (none wired by default) — set via CLI. NOTE: the Muon path forces
    use_distributed_optimizer=False; EP+Muon interaction on the MoE backbone is UNVALIDATED and must
    be checked on GPU (see report).
    """
    return _ov2_common(
        "qwen3-30b-a3b",
        "stage2",
        expert_model_parallel_size=8,
        sequence_parallel=False,
    )


def ov2_30b_a3b_p16m33_stage1() -> ConfigContainer:
    """OV2 A3B-MoE p16m33 stage-1 adapter-only alignment. Backbone qwen3-30b-a3b-p16m33:
    Qwen3-30B-A3B MoE LLM + OV2.1 patch16/merge3 vision tower (HF onevision_encoder_patch16 baked into
    the combined mcore stitch) + FRESH merge3 adapter (1024*3^2=9216 -> 2048). EP=8. Build the combined
    ckpt first via A800/convert from_base --vision_hf <encoder>."""
    return _ov2_common(
        "qwen3-30b-a3b-p16m33",
        "stage1",
        expert_model_parallel_size=8,
        sequence_parallel=False,
    )


def ov2_30b_a3b_p16m33_stage2() -> ConfigContainer:
    """OV2 A3B-MoE p16m33 stage-2 vit+adapter SFT (distributed Muon, EP=8). Backbone
    qwen3-30b-a3b-p16m33. Needs a trained p16m33 stage-1 ckpt to chain from
    (set via CLI checkpoint.pretrained_checkpoint=<stage1>)."""
    return _ov2_common(
        "qwen3-30b-a3b-p16m33",
        "stage2",
        expert_model_parallel_size=8,
        sequence_parallel=False,
    )


# =============================================================================
# Mid-train (stage 1.5) recipe functions — train the FULL model (LLM+vision+adapter)
# =============================================================================
def ov2_4b_midtrain() -> ConfigContainer:
    """OV2-4B (Qwen3-4B, p16m33) mid-train: FULL-model SFT (LLM + vision + adapter), distributed Muon
    (matched-adamw-rms 0.15), lr 2e-5 constant, gbs 128, activation recompute ON. Mirrors AIAK
    date0528 (--trainable-modules language_model adapter vision_model). Chain from a trained prior
    stage via CLI checkpoint.pretrained_checkpoint=<ckpt> (else falls back to the mcore base for a smoke)."""
    return _ov2_common("qwen3-4b", "midtrain")


def ov2_8b_midtrain() -> ConfigContainer:
    """OV2-8B (Qwen3-8B, p16m33) mid-train: FULL-model SFT (Muon). Set the init ckpt via CLI
    checkpoint.pretrained_checkpoint=<trained stage>."""
    return _ov2_common("qwen3-8b", "midtrain")


def ov2_35b_a3b_midtrain() -> ConfigContainer:
    """OV2 A3B-MoE (qwen3-30b-a3b) mid-train: FULL-model SFT, EP=8. The LLM (incl. experts) is now
    trainable, so distributed Muon would hit the EP backward all-to-all deadlock (same as stage-2
    Muon) -> midtrain auto-uses AdamW(distopt=True) on this MoE backbone. NOTE: full-model 30B at
    seq 32000 is memory-heavy; recompute is ON but you likely need >2 nodes and/or TP>1 vs the
    vit+adapter stage-2 (validate on GPU). Chain via CLI checkpoint.pretrained_checkpoint."""
    return _ov2_common(
        "qwen3-30b-a3b",
        "midtrain",
        expert_model_parallel_size=8,
        sequence_parallel=False,
    )


def ov2_30b_a3b_p16m33_midtrain() -> ConfigContainer:
    """OV2 A3B-MoE p16m33 mid-train: FULL-model SFT (LLM+vision+adapter), EP=8, AdamW (MoE).
    Backbone qwen3-30b-a3b-p16m33. Chain from a trained p16m33 stage via CLI
    checkpoint.pretrained_checkpoint=<ckpt>."""
    return _ov2_common(
        "qwen3-30b-a3b-p16m33",
        "midtrain",
        expert_model_parallel_size=8,
        sequence_parallel=False,
    )


# =============================================================================
# Back-compat aliases (running scripts import these original 4B names)
# =============================================================================
def ov2_1_stage1_adapter_only_config() -> ConfigContainer:
    """Back-compat alias for ov2_4b_stage1 (OV2.1-4B stage-1 adapter-only)."""
    return ov2_4b_stage1()


def ov2_1_stage2_vit_adapter_muon_config() -> ConfigContainer:
    """Back-compat alias for ov2_4b_stage2 (OV2.1-4B stage-2 vit+adapter Muon)."""
    return ov2_4b_stage2()


# Legacy module-level path constants kept for any importer (now sourced from the 4B factory entry).
_OV2_4B = _OV2_BACKBONES["qwen3-4b"]
OV2_LLM_HF = _OV2_4B["llm_hf"]
OV2_MCORE_CKPT = _OV2_4B["mcore_ckpt"]
OV2_HF_PROC = _OV2_4B["hf_proc"]
AIAK_STAGE1_CKPT = _OV2_4B["stage1_ckpt"]


__all__ = [
    # per-size recipe functions (run via run_recipe.py --recipe <name>)
    "ov2_4b_stage1", "ov2_4b_stage2",
    "ov2_8b_stage1", "ov2_8b_stage2",
    "ov2_35b_a3b_stage1", "ov2_35b_a3b_stage2",
    # mid-train (stage 1.5) — full-model SFT
    "ov2_4b_midtrain", "ov2_8b_midtrain", "ov2_35b_a3b_midtrain",
    # back-compat aliases (original 4B recipe names)
    "ov2_1_stage1_adapter_only_config", "ov2_1_stage2_vit_adapter_muon_config",
]
