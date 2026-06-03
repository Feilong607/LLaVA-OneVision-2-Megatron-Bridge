# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Energon task encoder for OV2.1 stage-1 (MultiMixQASample -> the OV2 step's batch dict).

Reuses the verified standalone-trainer logic (manual Qwen chat prompt + HF AutoProcessor +
prompt-prefix masking), then applies the next-token label shift (roll -1, mask the wrapped
last position) so that `labels[t]` supervises `input_ids[t+1]` — required because mcore's
`compute_language_model_loss` does NOT shift internally. This matches QwenVLTaskEncoder.
The OV2 step passes `images=pixel_values` + `image_grid_thw` straight to LlavaOnevision2.forward.
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from megatron.energon import Batch, DefaultTaskEncoder, SkipSample

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


def _to_pil(im):
    from PIL import Image

    if isinstance(im, Image.Image):
        return im.convert("RGB")
    if torch.is_tensor(im):
        a = im.detach().cpu()
        if a.dtype.is_floating_point:
            a = (a * 255).clamp(0, 255).to(torch.uint8)
        a = a.numpy()
        if a.ndim == 3 and a.shape[0] in (1, 3):
            a = np.transpose(a, (1, 2, 0))
        return Image.fromarray(a).convert("RGB")
    return Image.fromarray(np.asarray(im)).convert("RGB")


@dataclass
class OV2TaskSample:
    __key__: str
    __subflavors__: Optional[Dict]
    text: torch.Tensor                       # input_ids
    target: torch.Tensor                     # next-token labels (rolled -1, prompt masked)
    pixel_values: Optional[torch.Tensor] = None
    image_grid_thw: Optional[torch.Tensor] = None


@dataclass
class OV2TaskBatch(Batch):
    __keys__: List[str]
    tokens: torch.Tensor
    labels: torch.Tensor
    loss_mask: torch.Tensor
    position_ids: torch.Tensor
    attention_mask: Optional[torch.Tensor]
    pixel_values: Optional[torch.Tensor]
    image_grid_thw: Optional[torch.Tensor]


class OV2TaskEncoder(DefaultTaskEncoder):
    """OV2.1 stage-1 SFT encoder: MultiMixQASample -> OV2 step batch (token-aligned labels rolled -1)."""

    VISION = "<|vision_start|><|image_pad|><|vision_end|>"

    def __init__(self, hf_processor_path: str, seq_length: int = 32000,
                 default_system: str = "You are a helpful assistant.", **_unused):
        super().__init__()
        from transformers import AutoProcessor

        # trust_remote_code=False: the dir's custom auto_map (configuration_llava_onevision2) is the
        # MODEL config, not needed for the processor, and compiling it concurrently across DP ranks
        # races on the shared HF dynamic-module cache. The processor is a standard Qwen2_5_VLProcessor
        # (verified: merge_size=3, <|image_pad|> expands to grid.prod//merge^2 either way).
        self.proc = AutoProcessor.from_pretrained(hf_processor_path, trust_remote_code=False)
        tok = getattr(self.proc, "tokenizer", self.proc)
        self.pad_id = int(getattr(tok, "pad_token_id", None) or getattr(tok, "eos_token_id", 0) or 0)
        self.seq_length = self.seq_len = int(seq_length)
        self.default_system = default_system

    def encode_sample(self, s) -> OV2TaskSample:
        try:
            msgs = s.messages or []
            user = next(m["content"] for m in msgs if m.get("role") == "user")
            ans = next(m["content"] for m in msgs if m.get("role") == "assistant")
            imgs = [_to_pil(x) for x in (s.image or [])]
            user = user.replace("<image>", self.VISION) if "<image>" in user else (
                self.VISION + "\n" + user if imgs else user
            )
            _sys = getattr(s, "system", None) or self.default_system
            prompt = (
                f"<|im_start|>system\n{_sys}<|im_end|>\n"
                f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
            )
            ft = self.proc(text=[prompt + ans + "<|im_end|>"], images=imgs or None, return_tensors="pt")
            pt = self.proc(text=[prompt], images=imgs or None, return_tensors="pt")
            ids = ft["input_ids"][0]
            if ids.shape[0] > self.seq_length:
                raise SkipSample()
            labels = ids.clone()
            labels[: pt["input_ids"].shape[1]] = IGNORE_INDEX           # mask system+user+image prompt
            labels = torch.roll(labels, shifts=-1, dims=0)              # next-token shift (mcore does NOT shift)
            labels[-1] = IGNORE_INDEX
            return OV2TaskSample(
                __key__=getattr(s, "__key__", "?"),
                __subflavors__=getattr(s, "__subflavors__", None),
                text=ids,
                target=labels,
                pixel_values=ft.get("pixel_values"),
                image_grid_thw=ft.get("image_grid_thw"),
            )
        except SkipSample:
            raise
        except Exception as e:
            logger.warning("OV2 encode_sample skip key=%s: %r", getattr(s, "__key__", "?"), e)
            raise SkipSample()

    def batch(self, samples: List[OV2TaskSample]) -> OV2TaskBatch:
        n = len(samples)
        max_len = min(self.seq_length, max(int(s.text.shape[0]) for s in samples))
        toks = torch.full((n, max_len), self.pad_id, dtype=torch.long)
        labs = torch.full((n, max_len), IGNORE_INDEX, dtype=torch.long)
        for i, s in enumerate(samples):
            tl = min(max_len, int(s.text.shape[0]))
            ll = min(max_len, int(s.target.shape[0]))
            toks[i, :tl] = s.text[:tl].long()
            labs[i, :ll] = s.target[:ll].long()
        loss_mask = (labs != IGNORE_INDEX).float()
        pos = torch.arange(max_len, dtype=torch.long).unsqueeze(0).expand(n, -1).contiguous()
        pvs = [s.pixel_values for s in samples if s.pixel_values is not None]
        grids = [s.image_grid_thw for s in samples if s.image_grid_thw is not None]
        return OV2TaskBatch(
            __keys__=[s.__key__ for s in samples],
            tokens=toks,
            labels=labs,
            loss_mask=loss_mask,
            position_ids=pos,
            attention_mask=None,                       # mcore builds causal mask
            pixel_values=torch.cat(pvs, dim=0) if pvs else None,
            image_grid_thw=torch.cat(grids, dim=0) if grids else None,
        )

    def encode_batch(self, batch: OV2TaskBatch) -> dict:
        return {
            "tokens": batch.tokens,
            "labels": batch.labels,
            "loss_mask": batch.loss_mask,
            "position_ids": batch.position_ids,
            "attention_mask": batch.attention_mask,
            "pixel_values": batch.pixel_values,
            "image_grid_thw": batch.image_grid_thw,
        }
