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
                 default_system: str = "You are a helpful assistant.",
                 spatial_merge_size: Optional[int] = None, **_unused):
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
        # Explicit per-backbone merge override. The <|image_pad|> expansion count MUST match the
        # vision tower's spatial_merge_size (img_counts = grid.prod // merge^2). All three backbones
        # currently share the 4B p16m33 processor (merge_size=3), which is WRONG for 8B/35B (merge=2):
        # leaving merge=3 would expand too few image tokens vs the merge=2 tower's features and trip
        # LlavaOnevision2.forward's "image features != image tokens" assert. When set, this overrides
        # the processor's merge_size. None => read merge_size from the processor (4B path, unchanged).
        self.spatial_merge_size = spatial_merge_size

    # Qwen2-VL / Qwen3 special-token ids (verified on the p16m33 processor).
    IMG_PAD_ID = 151655
    VIS_START_ID = 151652
    VIS_END_ID = 151653

    def encode_sample(self, s) -> OV2TaskSample:
        try:
            msgs = s.messages or []
            # Split optional system; collect user/assistant turns in order.
            system = None
            turns = []
            for m in msgs:
                role = m.get("role")
                content = m.get("content", "")
                if role == "system":
                    system = content
                elif role in ("user", "assistant"):
                    turns.append((role, content))
            if not system:
                system = getattr(s, "system", None) or self.default_system

            # MULTI-TURN: llava_next is ~65% multi-turn; AIAK (encode_multiturn) supervises EVERY
            # assistant response, so we must too (loss-口径 alignment). Pair consecutive user->assistant.
            pairs = []
            i = 0
            while i < len(turns) - 1:
                if turns[i][0] == "user" and turns[i + 1][0] == "assistant":
                    pairs.append((turns[i][1], turns[i + 1][1]))
                    i += 2
                else:
                    i += 1
            if not pairs:
                raise SkipSample()

            imgs = [_to_pil(x) for x in (s.image or [])]
            if not imgs:
                # Text-only sample (no image): OV2 stage-1/stage-2 FREEZE the LLM, so the only trainable
                # params (vision tower + adapter) are absent from this sample's autograd graph -> the loss
                # has requires_grad=False -> backward() is a no-op on this rank -> it issues NONE of the
                # backward MoE expert-parallel all-to-alls and runs ahead to the next microbatch's forward,
                # while ranks holding image samples execute the backward -> cross-rank EP collective desync
                # -> NCCL ALLTOALL_BASE deadlock (mbs=1: one text-only sample on one rank hangs the job).
                # A text-only sample also trains nothing in these stages. Skip it.
                raise SkipSample()
            # Image features + per-image expanded token counts (grid//merge^2, merge=3) — replicates
            # the full processor's <|image_pad|> expansion so masked_scatter gets the right #tokens.
            pixel_values = None
            image_grid_thw = None
            img_counts = []
            if imgs:
                ip = getattr(self.proc, "image_processor", None)
                img_out = ip(images=imgs, return_tensors="pt")
                pixel_values = img_out["pixel_values"]
                image_grid_thw = img_out["image_grid_thw"]
                # Per-backbone merge override wins; else read from the processor (4B path unchanged).
                merge = int(self.spatial_merge_size if self.spatial_merge_size is not None
                            else getattr(ip, "merge_size", 2))
                img_counts = [
                    int(image_grid_thw[k].prod().item()) // (merge * merge)
                    for k in range(image_grid_thw.shape[0])
                ]

            tok = getattr(self.proc, "tokenizer", self.proc)

            def _enc(text):
                return tok(text, add_special_tokens=False)["input_ids"]

            # Reconcile <image> markers with len(imgs): more markers than images is malformed; fewer
            # (image with no marker) -> prepend the missing to turn 0. Guarantees #<|image_pad|>==#imgs.
            n_img = len(imgs)
            total_markers = sum(str(uc).count("<image>") for uc, _ac in pairs)
            if total_markers > n_img:
                raise SkipSample()
            missing = n_img - total_markers

            # Build SEGMENT-WISE (matches AIAK encode_multiturn): mask system + every user/assistant
            # header; supervise each assistant response + closing <|im_end|>.
            input_ids, labels = [], []
            seg = _enc(f"<|im_start|>system\n{system}<|im_end|>\n")
            input_ids += seg
            labels += [IGNORE_INDEX] * len(seg)
            for ti, (user_c, asst_c) in enumerate(pairs):
                user_c = str(user_c)
                if ti == 0 and missing > 0:
                    user_c = self.VISION * missing + "\n" + user_c
                if "<image>" in user_c:
                    user_c = user_c.replace("<image>", self.VISION)
                sep = "" if ti == 0 else "\n"
                src = _enc(f"{sep}<|im_start|>user\n{user_c}<|im_end|>\n<|im_start|>assistant\n")
                input_ids += src
                labels += [IGNORE_INDEX] * len(src)
                tgt = _enc(f"{asst_c}<|im_end|>")
                input_ids += tgt
                labels += tgt

            # Expand each single <|image_pad|> placeholder to its real token count.
            if img_counts:
                e_ids, e_lab, k = [], [], 0
                for tid, lab in zip(input_ids, labels):
                    if tid == self.IMG_PAD_ID:
                        cnt = img_counts[k] if k < len(img_counts) else 1
                        k += 1
                        e_ids += [self.IMG_PAD_ID] * cnt
                        e_lab += [IGNORE_INDEX] * cnt
                    else:
                        e_ids.append(tid)
                        e_lab.append(lab)
                input_ids, labels = e_ids, e_lab
                if k != len(img_counts):
                    raise SkipSample()

            ids = torch.tensor(input_ids, dtype=torch.long)
            labels = torch.tensor(labels, dtype=torch.long)
            for vid in (self.VIS_START_ID, self.IMG_PAD_ID, self.VIS_END_ID):
                labels[ids == vid] = IGNORE_INDEX
            if ids.shape[0] > self.seq_length:
                raise SkipSample()
            labels = torch.roll(labels, shifts=-1, dims=0)              # next-token shift (mcore does NOT shift)
            labels[-1] = IGNORE_INDEX
            if int((labels != IGNORE_INDEX).sum()) == 0:
                # No supervised tokens left (image pads + the next-token shift can leave a sample with
                # zero learnable positions) -> the per-token loss becomes sum/0 = 0/0 = NaN, which
                # crashes the whole step. This is a real cause of the OV2-30B-A3B iter-2 NaN: its
                # patch14/merge2 processor expands each image to MORE tokens than 4B's patch16/merge3,
                # so a short-caption sample can end up all-image/no-text. Skip it.
                raise SkipSample()
            return OV2TaskSample(
                __key__=getattr(s, "__key__", "?"),
                __subflavors__=getattr(s, "__subflavors__", None),
                text=ids,
                target=labels,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
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
