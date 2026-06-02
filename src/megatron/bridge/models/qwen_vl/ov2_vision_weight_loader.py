"""Loader to splice OV2.1's trained vision-tower weights into Bridge's OV2VisionTower.

Usage (called from training launch script):
    from megatron.bridge.models.qwen_vl.ov2_vision_weight_loader import load_ov2_vision_into_tower
    load_ov2_vision_into_tower(model.vision_model, "/ov2/feilong/reshard_30b_a3b/align_iter500_mcore_tp1_pp1_ep8")

For 30B MoE stage_1 alignment we additionally load the adapter from OV2 stage_0:
    from megatron.bridge.models.qwen_vl.ov2_vision_weight_loader import load_ov2_adapter_into_tower
    load_ov2_adapter_into_tower(model.vision_model, "/ov2/pretrain_models/llava_onevision2/llava_onevision2_30b_a3b/stage_0_tp1_pp1_ep8")

For the 8B/35B SFT runs we intentionally leave the adapter random (these
recipes only call load_ov2_vision_into_tower).
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn


def _mp_rank_path(ckpt_dir: str, rank: int = 0):
    """Return the mp_rank_00[_NNN]/model_optim_rng.pt path under ckpt_dir/release.

    Accepts both <ckpt_dir>/release/mp_rank_00 (TP=1 PP=1 EP=1) and
    <ckpt_dir>/release/mp_rank_00_<rank:03d> (EP-sharded) layouts.
    """
    rel = Path(ckpt_dir) / "release"
    p = rel / f"mp_rank_00_{rank:03d}" / "model_optim_rng.pt"
    if p.exists():
        return p
    if rank == 0:
        q = rel / "mp_rank_00" / "model_optim_rng.pt"
        if q.exists():
            return q
    return p  # surface the missing _000 path for the FileNotFoundError below


def _load_ov2_shard(ckpt_dir: str, rank: int = 0):
    """Load a single OV2 legacy mcore shard from rank ``rank`` and return its ``model`` dict.

    Accepts both legacy EP-sharded (mp_rank_00_NNN) and single-rank (mp_rank_00) layouts.
    """
    p = _mp_rank_path(ckpt_dir, rank=rank)
    if not p.exists():
        raise FileNotFoundError(f"OV2 vision shard not found at {p}")
    blob = torch.load(p.as_posix(), map_location="cpu", weights_only=False)
    if "model" not in blob:
        raise KeyError(
            f"expected top-level 'model' key in {p}; got {list(blob.keys())}"
        )
    return blob["model"]


def _extract_vision_subdict(ov2_sd: Mapping[str, torch.Tensor]) -> dict:
    """Strip the ``vision_model.`` prefix and re-prefix as ``vit.`` for OV2VisionTower.

    The result feeds directly into ``tower.load_state_dict(...)`` where ``tower``
    is the ``OV2VisionTower`` instance (which IS ``model.vision_model``).
    """
    out: dict = {}
    for k, v in ov2_sd.items():
        if not k.startswith("vision_model."):
            continue
        new_k = "vit." + k[len("vision_model.") :]
        out[new_k] = v
    return out


def _extract_adapter_subdict(ov2_sd: Mapping[str, torch.Tensor]) -> dict:
    """Strip the ``adapter.`` prefix and re-prefix as ``adapter.`` for OV2VisionTower.

    The OV2 ckpt's adapter keys already live at ``adapter.<...>``; we re-emit
    them under the same prefix because that's exactly where
    :class:`OV2VisionTower` exposes the adapter submodule. The pass-through
    rename keeps the API symmetric with :func:`_extract_vision_subdict`.
    """
    out: dict = {}
    for k, v in ov2_sd.items():
        if not k.startswith("adapter."):
            continue
        out[k] = v
    return out


def load_ov2_vision_into_tower(
    tower: nn.Module,
    ckpt_dir: str,
    *,
    strict: bool = False,
) -> dict:
    """Load ``vision_model.*`` tensors from an OV2 legacy mcore ckpt into Bridge's OV2VisionTower.

    Args:
        tower: the ``nn.Module`` instance (typically ``model.vision_model``,
            an :class:`OV2VisionTower`).
        ckpt_dir: path to OV2 ckpt root (the dir containing ``release/``).
        strict: passed through to ``load_state_dict``. ``False`` allows
            extra/missing keys.

    Returns:
        A dict with the following keys:

        * ``missing`` -- list of keys in ``tower`` but not in the source
          (excluding ``adapter.*`` -- adapter is random by design, and excluding
          ``._extra_state`` -- TE rebuilds those at first forward).
        * ``unexpected`` -- list of keys in source not in ``tower``.
        * ``loaded_count`` -- number of tensors fed into ``load_state_dict``.
        * ``skipped_extra_states`` -- number of ``._extra_state`` BytesIO blobs
          dropped from the source side (TE-FP8 state objects, version-fragile).
        * ``ov2_vision_keys_total`` -- raw count of ``vision_model.*`` keys in
          the source (sanity check, expect 387 for the OV2.1 24-layer encoder).
    """
    ov2_sd = _load_ov2_shard(ckpt_dir, rank=0)
    vision_sd = _extract_vision_subdict(ov2_sd)

    # Drop TE _extra_state entries -- they are TE-FP8 state objects that don't
    # transfer cleanly across versions. TE will rebuild them at first forward.
    cleaned = {k: v for k, v in vision_sd.items() if not k.endswith("._extra_state")}
    skipped_extras = [k for k in vision_sd if k.endswith("._extra_state")]

    # Identity-fill any final_layernorm the tower has but the source ckpt
    # doesn't. The OV2.1 trained encoder uses pre_layernorm + per-layer
    # layernorms only; MCore's TransformerBlock unconditionally instantiates a
    # ``decoder.final_layernorm`` whose weights aren't in the ckpt. Setting it
    # to identity (weight=1, bias=0) makes the loaded tower numerically
    # equivalent to the source. Done in-tensor here so it shows up in
    # ``loaded_count`` and so ``load_state_dict`` doesn't flag it as missing.
    identity_filled: list = []
    tower_keys = set(tower.state_dict().keys())
    for ln_key_root in ("vit.decoder.final_layernorm",):
        w_key = f"{ln_key_root}.weight"
        b_key = f"{ln_key_root}.bias"
        if w_key in tower_keys and w_key not in cleaned:
            ref = tower.state_dict()[w_key]
            cleaned[w_key] = torch.ones_like(ref)
            identity_filled.append(w_key)
        if b_key in tower_keys and b_key not in cleaned:
            ref = tower.state_dict()[b_key]
            cleaned[b_key] = torch.zeros_like(ref)
            identity_filled.append(b_key)

    # patch_embed Linear -> Conv2d reshape: OV2 4B p16m33 stores patch_embed.proj
    # as nn.Linear([1024, 3*K*K]); Bridge's OV2VisionTower uses Conv2d
    # ([1024, 3, K, K]). Reshape on load — the op is identical when input is
    # flattened in C-H-W order.
    pe_key = "vit.patch_embed.proj.weight"
    if pe_key in cleaned and pe_key in tower.state_dict():
        src = cleaned[pe_key]
        dst_shape = tower.state_dict()[pe_key].shape
        if src.shape != dst_shape and src.numel() == int(torch.tensor(dst_shape).prod().item()):
            print(f"[ov2_vision_loader] reshaping {pe_key} from {tuple(src.shape)} to {tuple(dst_shape)}")
            cleaned[pe_key] = src.view(dst_shape)

    missing, unexpected = tower.load_state_dict(cleaned, strict=strict)

    # Filter "expected" missing:
    #   - keys under ``adapter.*`` (random init by design, or loaded separately
    #     by :func:`load_ov2_adapter_into_tower`)
    #   - any ``._extra_state`` keys the tower exposes (TE rebuilds at first fwd)
    real_missing = [
        k for k in missing
        if not k.startswith("adapter.") and not k.endswith("._extra_state")
    ]

    return {
        "missing": real_missing,
        "unexpected": list(unexpected),
        "loaded_count": len(cleaned),
        "skipped_extra_states": len(skipped_extras),
        "ov2_vision_keys_total": len(vision_sd),
        "identity_filled": identity_filled,
    }


def load_ov2_adapter_into_tower(
    tower: nn.Module,
    ckpt_dir: str,
    *,
    strict: bool = False,
) -> dict:
    """Load ``adapter.*`` tensors from an OV2 legacy mcore ckpt into Bridge's OV2VisionTower.

    The adapter is EP-replicated in OV2 (verified by ``/tmp/inspect_ov2_stage0.py``),
    so reading rank-0 of an ``ep<n>`` ckpt is sufficient. The OV2 30B-A3B
    stage_0 adapter is a 6-tensor block:

        adapter.layernorm.{weight,bias}    shape=(1024,)
        adapter.linear_fc1.{weight,bias}   shape=(4096, 4096) / (4096,)
        adapter.linear_fc2.{weight,bias}   shape=(2048, 4096) / (2048,)

    plus 2 ``._extra_state`` blobs (TE-FP8 state, skipped).

    Args:
        tower: the ``nn.Module`` instance (typically ``model.vision_model``,
            an :class:`OV2VisionTower`).
        ckpt_dir: path to OV2 ckpt root (the dir containing ``release/``).
        strict: passed through to ``load_state_dict``. ``False`` allows
            extra/missing keys (and is the default — the tower also contains
            the ``vit.*`` tree which this loader doesn't touch).

    Returns:
        A dict with:

        * ``missing`` -- tower keys NOT in the source. Filtered to exclude
          ``vit.*`` (loaded separately) and ``._extra_state``.
        * ``unexpected`` -- source keys not in the tower.
        * ``loaded_count`` -- number of tensors fed into ``load_state_dict``
          (expect 6 for the 30B-A3B stage_0 adapter).
        * ``skipped_extra_states`` -- number of ``._extra_state`` BytesIO
          blobs dropped from the source side.
        * ``ov2_adapter_keys_total`` -- raw count of ``adapter.*`` keys in
          the source.
    """
    ov2_sd = _load_ov2_shard(ckpt_dir, rank=0)
    adapter_sd = _extract_adapter_subdict(ov2_sd)

    # Drop TE _extra_state entries.
    cleaned = {k: v for k, v in adapter_sd.items() if not k.endswith("._extra_state")}
    skipped_extras = [k for k in adapter_sd if k.endswith("._extra_state")]

    # patch_embed Linear -> Conv2d reshape: OV2 4B p16m33 stores patch_embed.proj
    # as nn.Linear([1024, 3*K*K]); Bridge's OV2VisionTower uses Conv2d
    # ([1024, 3, K, K]). Reshape on load — the op is identical when input is
    # flattened in C-H-W order.
    pe_key = "vit.patch_embed.proj.weight"
    if pe_key in cleaned and pe_key in tower.state_dict():
        src = cleaned[pe_key]
        dst_shape = tower.state_dict()[pe_key].shape
        if src.shape != dst_shape and src.numel() == int(torch.tensor(dst_shape).prod().item()):
            print(f"[ov2_vision_loader] reshaping {pe_key} from {tuple(src.shape)} to {tuple(dst_shape)}")
            cleaned[pe_key] = src.view(dst_shape)

    missing, unexpected = tower.load_state_dict(cleaned, strict=strict)

    # Filter "expected" missing:
    #   - keys under ``vit.*`` (loaded separately, or random if vision ckpt not set)
    #   - any ``._extra_state`` keys (TE rebuilds at first fwd)
    real_missing = [
        k for k in missing
        if not k.startswith("vit.") and not k.endswith("._extra_state")
    ]

    return {
        "missing": real_missing,
        "unexpected": list(unexpected),
        "loaded_count": len(cleaned),
        "skipped_extra_states": len(skipped_extras),
        "ov2_adapter_keys_total": len(adapter_sd),
    }
