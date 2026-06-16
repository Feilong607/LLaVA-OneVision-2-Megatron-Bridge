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
import re
from dataclasses import dataclass, fields
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


def _to_block_layout(positions: torch.Tensor, t: int, h: int, w: int, sms: int) -> torch.Tensor:
    """Reorder row-major (t,h,w) patch positions into the merge x merge spatial-block order the OV2
    vision tower's pixel patches are arranged in. Verbatim port of AIAK convert_positions_to_block_layout
    (qwen2vl_task_encoder.py:62-106) EXCEPT spatial_merge_size is a parameter (AIAK hardcodes 2; OV2
    p16m33 = 3). t stays outermost so flat[::sms*sms, 0] gives one frame-index per merged token."""
    if sms == 1:
        return positions
    idx = torch.arange(t * h * w).view(t, h, w)
    idx = idx.view(t, h // sms, sms, w // sms, sms)
    idx = idx.permute(0, 1, 3, 2, 4).contiguous().view(t * h * w)
    return positions[idx]


@dataclass
class OV2TaskSample:
    __key__: str
    __subflavors__: Optional[Dict]
    text: torch.Tensor                       # input_ids
    target: torch.Tensor                     # next-token labels (rolled -1, prompt masked)
    pixel_values: Optional[torch.Tensor] = None
    image_grid_thw: Optional[torch.Tensor] = None
    cu_seqlens: Optional[torch.Tensor] = None        # packed: per-sub-sample boundaries; None=non-packed
    patch_positions: Optional[torch.Tensor] = None   # video: [total_patches,3]=(t,h,w) block-layout; None=image (encoder self-derives)


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
    cu_seqlens: Optional[torch.Tensor] = None
    patch_positions: Optional[torch.Tensor] = None


_OV2_BATCH_DUNDERS = {f.name for f in fields(OV2TaskBatch)} & {"__key__", "__restore_key__"}


def _ov2_batch_dunders(samples):
    """energon >=6/7 makes Batch a kw-only dataclass that REQUIRES __key__/__restore_key__; energon
    5.x (the A100 mbridge image) has neither and rejects them. Pass them only when THIS energon's
    Batch actually declares them, so one task_encoder runs unchanged on both image generations."""
    extra = {}
    if "__key__" in _OV2_BATCH_DUNDERS:
        extra["__key__"] = samples[0].__key__ if samples else "ov2batch"
    if "__restore_key__" in _OV2_BATCH_DUNDERS:
        extra["__restore_key__"] = ()  # placeholder; the savable loader reassigns the real key
    return extra


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
        self.proc = AutoProcessor.from_pretrained(hf_processor_path, trust_remote_code=False, local_files_only=True)
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

    def encode_sample(self, s):
        # Offline-packed captioning (e.g. seed85m PackedCaptioningSample): one energon sample is N
        # sub-samples already packed together. Encode each sub-sample with the verified MultiMix path,
        # concatenate into ONE sequence, and emit per-sub-sample cu_seqlens so ov2_step builds THD
        # block-diagonal attention. Non-packed MultiMixQASample is unchanged.
        if type(s).__name__ == "PackedCaptioningSample":
            return self._encode_packed(s)
        return self._encode_multimix(s)

    def _encode_packed(self, s) -> OV2TaskSample:
        from types import SimpleNamespace
        n = len(getattr(s, "images", []) or [])
        if n == 0:
            raise SkipSample()
        prompts = getattr(s, "prompts", None)
        captions = getattr(s, "captions", None)
        subs = []
        for i in range(n):
            ca = captions[i] if (captions is not None and i < len(captions)) else None
            if ca is None:
                continue
            pr = prompts[i] if (prompts is not None and i < len(prompts)) else ""
            pr = [pr] if isinstance(pr, str) else list(pr)
            ca = [ca] if isinstance(ca, str) else list(ca)
            msgs = []
            for p, c in zip(pr, ca):
                msgs.append({"role": "user", "content": p})
                msgs.append({"role": "assistant", "content": c})
            imgs_i = s.images[i]
            if imgs_i is None:
                imgs_i = []
            elif not isinstance(imgs_i, list):
                imgs_i = [imgs_i]
            # Per-sub-sample video metadata (the seed85m cooker bakes patch_positions as .npy [P,3]=(t,h,w)
            # and fps for VIDEO sub-samples; image/caption sub-samples carry None/'' here -> image path).
            _pp = getattr(s, "patch_positions", None)
            pp_i = _pp[i] if (_pp is not None and i < len(_pp)) else None
            _fps = getattr(s, "fps", None)
            fps_i = _fps[i] if (_fps is not None and i < len(_fps)) else None
            syn = SimpleNamespace(messages=msgs, system=None, image=imgs_i,
                                  patch_positions=pp_i, fps=fps_i,
                                  __key__=f"{getattr(s, '__key__', '?')}.sub{i:03d}",
                                  __subflavors__=getattr(s, "__subflavors__", None))
            try:
                subs.append(self._encode_multimix(syn))
            except SkipSample:
                continue
        if not subs:
            raise SkipSample()
        # Keep whole sub-samples until the seq_length budget is exhausted (preserve boundaries).
        texts, targs, kept_pvs, kept_grids, kept_pps, cu = [], [], [], [], [], [0]
        total = 0
        # The encoder slices ONE [total_patches,3] tensor per grid row by patch offset, so patch_positions
        # MUST cover every grid row or be None. Both video AND image sub-samples now emit positions (image
        # = default t=0; see _encode_multimix), so a normal pack is full-coverage, pp_consistent stays True,
        # the cat is emitted, and a mixed video+image pack KEEPS its real video temporal RoPE. pp_consistent
        # is the safety net: if any kept sub-sample lacks positions (e.g. a non-merge-divisible image fell
        # back to None) it nulls the whole pack so the tower self-derives RoPE for all grids, instead of
        # feeding a too-short tensor (which would crash late inside apply_rotary_pos_emb_vision).
        pp_consistent = True
        for t in subs:
            L = int(t.text.shape[0])
            if total + L > self.seq_length:
                # An offline pack whose sub-samples sum > seq_length was built for a LARGER seq_length
                # than this run uses. Two wrong ways to handle it: (1) the old silent `break` kept a
                # prefix -> trains on a TRUNCATED pack and skews the data with no signal; (2) a hard
                # `raise ValueError` is NOT caught (encode_sample dispatches _encode_packed with no
                # try/except) -> aborts the WHOLE multi-node job on one bad pack. So: drop the whole
                # pack via SkipSample (energon retries the next sample) and log LOUDLY -- logger.error,
                # because warnings are swallowed in this module. Root cause: raise OV2_SEQ_LEN to >= the
                # packing length (you cannot lower seq for OOM with pre-packed data -- use TP/recompute/
                # offload), or repack the corpus to this seq_length.
                logger.error(
                    "OV2 packed sample %s: sub-samples exceed seq_length=%d at sub#%d (running %d+%d). "
                    "The offline pack was built for a larger seq_length; raise OV2_SEQ_LEN or repack. "
                    "Skipping this pack.", getattr(s, "__key__", "?"), self.seq_length, len(texts), total, L,
                )
                raise SkipSample()
            texts.append(t.text)
            targs.append(t.target)
            if t.pixel_values is not None:
                kept_pvs.append(t.pixel_values)
            if t.image_grid_thw is not None:
                kept_grids.append(t.image_grid_thw)
                if t.patch_positions is not None:
                    kept_pps.append(t.patch_positions)
                else:
                    pp_consistent = False        # image sub-sample (no positions) among the kept grids
            total += L
            cu.append(total)
        if len(texts) == 0:
            raise SkipSample()
        return OV2TaskSample(
            __key__=getattr(s, "__key__", "?"),
            __subflavors__=getattr(s, "__subflavors__", None),
            text=torch.cat(texts, dim=0),
            target=torch.cat(targs, dim=0),
            pixel_values=torch.cat(kept_pvs, dim=0) if kept_pvs else None,
            image_grid_thw=torch.cat(kept_grids, dim=0) if kept_grids else None,
            cu_seqlens=torch.tensor(cu, dtype=torch.int32),
            patch_positions=torch.cat(kept_pps, dim=0) if (kept_pps and pp_consistent) else None,
        )

    def _encode_multimix(self, s) -> OV2TaskSample:
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
                system = system or getattr(s, "system", None) or self.default_system

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

            imgs_all = [_to_pil(x) for x in (s.image or [])]
            if not imgs_all:
                # Text-only sample (no image): OV2 stage-1/stage-2 FREEZE the LLM, so the only trainable
                # params (vision tower + adapter) are absent from this sample's autograd graph -> the loss
                # has requires_grad=False -> backward() is a no-op on this rank -> it issues NONE of the
                # backward MoE expert-parallel all-to-alls and runs ahead to the next microbatch's forward,
                # while ranks holding image samples execute the backward -> cross-rank EP collective desync
                # -> NCCL ALLTOALL_BASE deadlock (mbs=1: one text-only sample on one rank hangs the job).
                # A text-only sample also trains nothing in these stages. Skip it.
                raise SkipSample()

            ip = getattr(self.proc, "image_processor", None)
            # Per-backbone merge override wins; else read from the processor (4B path unchanged).
            merge = int(self.spatial_merge_size if self.spatial_merge_size is not None
                        else getattr(ip, "merge_size", 2))
            tok = getattr(self.proc, "tokenizer", self.proc)

            def _enc(text):
                return tok(text, add_special_tokens=False)["input_ids"]

            # VIDEO branch (AIAK process_sft_qa video path + _rewrap_vision_by_frame): the seed85m cooker
            # bakes per-patch (t,h,w) positions (t = REAL frame index) as .npy for video sub-samples and
            # emits one <image> per frame. Detect via non-empty patch_positions; route to the temporal
            # build (collapse per-frame grids into ONE [[F,h,w]] + block-layout positions + <N seconds>
            # timestamps) instead of treating the frames as F independent images.
            _raw_pp = getattr(s, "patch_positions", None)
            _pp_arrays = []
            if _raw_pp is not None:
                for _p in (_raw_pp if isinstance(_raw_pp, (list, tuple)) else [_raw_pp]):
                    if _p is None or (isinstance(_p, str) and _p == ""):
                        continue
                    _pp_arrays.append(torch.as_tensor(np.asarray(_p), dtype=torch.int64))
            if _pp_arrays:
                _fps = getattr(s, "fps", None)
                if isinstance(_fps, (list, tuple)):
                    _fps = _fps[0] if _fps else None
                return self._encode_video(s, pairs, imgs_all, ip, merge, _pp_arrays, _fps, system)

            # Reconcile <image> markers with len(imgs). AIAK-PARITY (mm_plugin.py:212-231): on ANY
            # count mismatch (n_img>0 and markers!=n_img), AIAK clears ALL <image> and prepends the real
            # image count to the first user turn. We match that by stripping every <image> and zeroing
            # total_markers -> the truncation loop below then sees missing = n_img-0 = n_img and prepends
            # all n_img images to turn 0 (identical layout). (Old Bridge behavior: skip when markers>imgs,
            # partial-fill when markers<imgs -> dropped more samples / different text layout than AIAK.)
            n_img = len(imgs_all)
            total_markers = sum(str(uc).count("<image>") for uc, _ac in pairs)
            if n_img > 0 and total_markers != n_img:
                pairs = [(str(uc).replace("<image>", ""), ac) for uc, ac in pairs]
                total_markers = 0

            # AIAK-FAITHFUL TRUNCATION (qwen2vl_task_encoder.encode / _remove_last_qa_round): if the
            # built sequence exceeds seq_length, drop the LAST user->assistant round and retry, looping
            # until it fits. Each drop also trims the trailing images the dropped round's <image> markers
            # consumed (images are consumed in order, so keep the first n_used = missing + markers-left).
            # Only Skip when a single round still overflows (image tokens alone > seq_length) -- AIAK
            # asserts "no QA rounds left" there; we Skip per Bridge convention. Re-encodes (incl. the HF
            # image processor) per iteration, exactly like AIAK; only fires on over-length samples.
            cur_pairs = list(pairs)
            while True:
                markers = sum(str(uc).count("<image>") for uc, _ac in cur_pairs)
                n_used = (n_img - total_markers) + markers      # = missing (no-marker imgs) + markers-left
                missing = n_used - markers                      # == n_img - total_markers (constant)
                imgs = imgs_all[:n_used]
                if not imgs:
                    raise SkipSample()                          # truncated down to text-only -> skip (see above)

                # Image features + per-image expanded token counts (grid//merge^2) -- replicates the
                # full processor's <|image_pad|> expansion so masked_scatter gets the right #tokens.
                img_out = ip(images=imgs, return_tensors="pt")
                pixel_values = img_out["pixel_values"]
                image_grid_thw = img_out["image_grid_thw"]
                img_counts = [
                    int(image_grid_thw[k].prod().item()) // (merge * merge)
                    for k in range(image_grid_thw.shape[0])
                ]

                # Build SEGMENT-WISE (matches AIAK encode_multiturn): mask system + every user/assistant
                # header; supervise each assistant response + closing <|im_end|>.
                input_ids, labels = [], []
                seg = _enc(f"<|im_start|>system\n{system}<|im_end|>\n")
                input_ids += seg
                labels += [IGNORE_INDEX] * len(seg)
                for ti, (user_c, asst_c) in enumerate(cur_pairs):
                    user_c = str(user_c)
                    if ti == 0 and missing > 0:
                        user_c = "\n".join([self.VISION] * missing) + "\n" + user_c.lstrip("\n")
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

                if ids.shape[0] <= self.seq_length:
                    break
                if len(cur_pairs) <= 1:
                    raise SkipSample()                          # one round still overflows -> discard (AIAK)
                cur_pairs = cur_pairs[:-1]                       # drop last QA round (_remove_last_qa_round)

            labels = torch.roll(labels, shifts=-1, dims=0)              # next-token shift (mcore does NOT shift)
            labels[-1] = IGNORE_INDEX
            if int((labels != IGNORE_INDEX).sum()) == 0:
                # No supervised tokens left (image pads + the next-token shift can leave a sample with
                # zero learnable positions) -> the per-token loss becomes sum/0 = 0/0 = NaN, which
                # crashes the whole step. This is a real cause of the OV2-30B-A3B iter-2 NaN: its
                # patch14/merge2 processor expands each image to MORE tokens than 4B's patch16/merge3,
                # so a short-caption sample can end up all-image/no-text. Skip it.
                raise SkipSample()
            # AIAK parity (qwen2vl_task_encoder.py:356-372): give IMAGE rows EXPLICIT positions too
            # (t=0, spatial h/w, block layout) so patch_positions ALWAYS covers EVERY grid row. A video
            # sub-sample packed alongside images then keeps its REAL-frame-index temporal RoPE -- the
            # all-or-none net would otherwise null the whole pack and the video would fall back to the
            # tower grid-based path (sequential 0..F-1 frame index, not the cookers real indices). For
            # images the RoPE is byte-identical to the grid-based path (same t=0 + the same block
            # permutation as convert_rope_to_block_layout), so pure-image stage-1/2 is unchanged.
            img_pps = []
            for k in range(image_grid_thw.shape[0]):
                tk = int(image_grid_thw[k][0]); hk = int(image_grid_thw[k][1]); wk = int(image_grid_thw[k][2])
                if hk % merge or wk % merge:
                    img_pps = None; break              # non-merge-divisible -> let the all-or-none net handle it
                hc = torch.arange(hk, dtype=torch.int64).repeat_interleave(wk).repeat(tk)
                wc = torch.arange(wk, dtype=torch.int64).repeat(hk).repeat(tk)
                tc = torch.zeros(tk * hk * wk, dtype=torch.int64)
                img_pps.append(_to_block_layout(torch.stack([tc, hc, wc], dim=1), tk, hk, wk, merge))
            img_patch_positions = torch.cat(img_pps, dim=0) if img_pps else None
            return OV2TaskSample(
                __key__=getattr(s, "__key__", "?"),
                __subflavors__=getattr(s, "__subflavors__", None),
                text=ids,
                target=labels,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                patch_positions=img_patch_positions,
            )
        except SkipSample:
            raise
        except Exception as e:
            logger.warning("OV2 encode_sample skip key=%s: %r", getattr(s, "__key__", "?"), e)
            raise SkipSample()

    def _encode_video(self, s, pairs, frames, ip, merge, pp_arrays, fps, system=None) -> OV2TaskSample:
        """Temporal video encode, matching AIAK qwen2vl_task_encoder (process_sft_qa video branch +
        _rewrap_vision_by_frame + convert_positions_to_block_layout), PARAMETERIZED by `merge` (AIAK
        hardcodes 2; OV2 p16m33 = 3). Multi-turn IS supported (every assistant response supervised;
        the video block goes in turn 0 only), though seed85m video sub-samples are in fact single-turn."""
        try:
            if not pairs:
                raise SkipSample()
            tok = getattr(self.proc, "tokenizer", self.proc)

            def _enc(text):
                return tok(text, add_special_tokens=False)["input_ids"]

            # Process ALL frames; collapse the per-frame grids [[1,h,w]]*F into ONE temporal grid
            # [[F,h,w]] (AIAK qwen2vl_task_encoder.py:377) so the OV2 tower runs its frame-windowed
            # temporal path (VideoRotaryEmbeddingSplit466) instead of F independent images.
            img_out = ip(images=frames, return_tensors="pt")
            pixel_values = img_out["pixel_values"]
            grid = img_out["image_grid_thw"]
            F = int(grid.shape[0]); H = int(grid[0][1].item()); W = int(grid[0][2].item())
            # Non-uniform per-frame h/w would make the [[F,H,W]] collapse + F*H*W math wrong (we use
            # grid[0] for every frame); and H/W not divisible by merge would crash _to_block_layout's
            # reshape. Guard both with a clear reason instead of a cryptic view() error swallowed as Skip.
            if any(int(grid[k][1].item()) != H or int(grid[k][2].item()) != W for k in range(F)):
                logger.warning("OV2 _encode_video skip key=%s: non-uniform frame grid (need same h,w)",
                               getattr(s, "__key__", "?"))
                raise SkipSample()
            if H % merge != 0 or W % merge != 0:
                logger.warning("OV2 _encode_video skip key=%s: grid (h=%d,w=%d) not divisible by merge=%d",
                               getattr(s, "__key__", "?"), H, W, merge)
                raise SkipSample()
            image_grid_thw = torch.tensor([[F, H, W]], dtype=grid.dtype)

            # Patch positions: the cooker's .npy is row-major (frame,h,w) with col0 = REAL frame index.
            # Reorder into the tower's spatial-merge block layout (same merge the pad-count uses).
            flat = torch.cat(pp_arrays, dim=0)
            if flat.shape[0] != F * H * W:
                raise SkipSample()                               # grid/positions disagree -> drop
            flat = _to_block_layout(flat, F, H, W, merge)

            # Per-frame merged-token counts (block layout keeps t outermost -> same-t merged tokens are
            # contiguous; sample one t per merge*merge patches). sum(counts) == nmt == #<|image_pad|>.
            # Record each run's REAL frame index (col0=t) so the timestamp is computed from THAT frame --
            # robust to non-monotonic / duplicate frame indices (no sorted-unique vs run-order skew).
            msu = merge * merge
            nmt = flat.shape[0] // msu
            tvals = flat[::msu, 0].tolist()
            counts, frame_t, i = [], [], 0
            while i < nmt:
                j = i + 1
                while j < nmt and tvals[j] == tvals[i]:
                    j += 1
                counts.append(j - i); frame_t.append(int(tvals[i])); i = j

            # Per-frame timestamped vision string (AIAK _rewrap_vision_by_frame): <ts> VS PAD*K VE \n ...
            # The FULL expanded <|image_pad|> run is built here at string level (no later expander).
            use_ts = bool(fps) and float(fps) > 0
            parts = []
            for c, ft in zip(counts, frame_t):
                if use_ts:
                    parts.append(f"<{round(ft / float(fps), 1):.1f} seconds>")
                parts.append("<|vision_start|>" + "<|image_pad|>" * c + "<|vision_end|>\n")
            video_vision_str = "".join(parts)

            system = system or getattr(s, "system", None) or self.default_system
            # MULTI-TURN (AIAK encode_multiturn): supervise EVERY assistant response. The timestamped video
            # block replaces the <image> run in the FIRST user turn only; later turns are plain-text QA about
            # the same video. Single-turn (the seed85m video case) is byte-identical to before.
            # AIAK-FAITHFUL TRUNCATION (_remove_last_qa_round): if the built sequence overflows seq_length,
            # drop the LAST QA round and retry. The video block lives in turn 0 (never dropped) so nmt /
            # patch_positions / pixel_values are invariant across drops; only trailing text turns are shed.
            # Skip only when a single round still overflows (the video tokens alone exceed seq_length).
            cur_pairs = list(pairs)
            while True:
                input_ids, labels = [], []
                seg = _enc(f"<|im_start|>system\n{system}<|im_end|>\n")
                input_ids += seg; labels += [IGNORE_INDEX] * len(seg)
                for ti, (user_c, asst_c) in enumerate(cur_pairs):
                    user_c = str(user_c)
                    if ti == 0:
                        # Cooker emits one <image> per frame (contiguous run); collapse into the timestamped
                        # video block (keep trailing question text). lambda repl avoids re backslash/group
                        # interpretation; strip leftover stray <image> (one video/sub-sample, safe to drop).
                        if "<image>" in user_c:
                            user_c = re.sub(r"(?:<image>\s*)+", lambda _m: video_vision_str, user_c, count=1)
                            user_c = user_c.replace("<image>", "")
                        else:
                            user_c = video_vision_str + user_c
                    else:
                        user_c = user_c.replace("<image>", "")   # later turns carry no vision markers
                    sep = "" if ti == 0 else "\n"
                    src = _enc(f"{sep}<|im_start|>user\n{user_c}<|im_end|>\n<|im_start|>assistant\n")
                    input_ids += src; labels += [IGNORE_INDEX] * len(src)
                    tgt = _enc(f"{asst_c}<|im_end|>")
                    input_ids += tgt; labels += tgt
                if len(input_ids) <= self.seq_length:
                    break
                if len(cur_pairs) <= 1:
                    raise SkipSample()                           # one round still overflows -> discard (AIAK)
                cur_pairs = cur_pairs[:-1]                        # drop last QA round (_remove_last_qa_round)

            ids = torch.tensor(input_ids, dtype=torch.long)
            labels = torch.tensor(labels, dtype=torch.long)
            for vid in (self.VIS_START_ID, self.IMG_PAD_ID, self.VIS_END_ID):
                labels[ids == vid] = IGNORE_INDEX
            if int((ids == self.IMG_PAD_ID).sum()) != nmt:
                raise SkipSample()                               # pad count MUST == vision feature count
            labels = torch.roll(labels, shifts=-1, dims=0); labels[-1] = IGNORE_INDEX
            if int((labels != IGNORE_INDEX).sum()) == 0:
                raise SkipSample()
            return OV2TaskSample(
                __key__=getattr(s, "__key__", "?"),
                __subflavors__=getattr(s, "__subflavors__", None),
                text=ids, target=labels,
                pixel_values=pixel_values, image_grid_thw=image_grid_thw, patch_positions=flat,
            )
        except SkipSample:
            raise
        except Exception as e:
            logger.warning("OV2 _encode_video skip key=%s: %r", getattr(s, "__key__", "?"), e)
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
        # patch_positions must cover EVERY grid row or none: the tower slices one [P,3] across all grid
        # rows by offset, so a microbatch mixing a video sample (has positions) with an image sample
        # (none) would feed a too-short tensor -> truncated RoPE -> shape-mismatch crash. All-or-none.
        _grid_samps = [s for s in samples if s.image_grid_thw is not None]
        _pps_ok = bool(_grid_samps) and all(s.patch_positions is not None for s in _grid_samps)
        pps = [s.patch_positions for s in _grid_samps] if _pps_ok else []
        # cu_seqlens (offline-pack THD segment boundaries) is per-sample; ov2_step only consumes a SINGLE
        # cu tensor (n==1 path below). With micro_batch_size>1 we would need offset-concatenation across
        # samples, which is not wired yet -> the n>1 branch DROPS cu_seqlens and the pack silently runs
        # FULL-CAUSAL (later sub-samples attend to earlier ones). mbs==1 today so this never fires; make
        # it LOUD (logger.error, not a raise -- a per-rank raise in the loader could desync collectives)
        # if anyone raises mbs with packed data.
        if n > 1 and any(s.cu_seqlens is not None for s in samples):
            logger.error(
                "OV2 batch(): micro_batch_size=%d (>1) with offline-packed sample(s) -- per-sample "
                "cu_seqlens cannot be merged into ov2_step's single-cu THD path, so pack segment "
                "boundaries are DROPPED and the pack runs FULL-CAUSAL. Use mbs=1 with packed data, or add "
                "offset-concatenation of cu_seqlens here + in ov2_step.", n,
            )
        return OV2TaskBatch(
            __keys__=[s.__key__ for s in samples],
            tokens=toks,
            labels=labs,
            loss_mask=loss_mask,
            position_ids=pos,
            attention_mask=None,                       # mcore builds causal mask
            pixel_values=torch.cat(pvs, dim=0) if pvs else None,
            image_grid_thw=torch.cat(grids, dim=0) if grids else None,
            cu_seqlens=(samples[0].cu_seqlens if (n == 1 and samples[0].cu_seqlens is not None) else None),
            patch_positions=torch.cat(pps, dim=0).to(torch.int64) if pps else None,   # #8: RoPE temporal/spatial indices MUST be int64 (block-layout/temporal paths already are; this enforces it for offline-pack passthrough too)
            **_ov2_batch_dunders(samples),
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
            "cu_seqlens": batch.cu_seqlens,
            "patch_positions": batch.patch_positions,
        }
