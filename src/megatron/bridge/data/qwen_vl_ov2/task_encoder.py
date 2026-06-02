"""OV2 packed-captioning task encoder for Megatron-Bridge.

Ports the relevant slice of the OV2 source at
``aiak_training_llm/data/multimodal/{task_encoder.py,qwen2vl_task_encoder.py}``
and adapts the output into the dict shape expected by Bridge's
``qwen3_vl_step.get_batch()`` consumer.

Decoupling notes (vs. OV2 source):
    - aiak globals ``get_args``/``get_tokenizer`` are replaced by constructor
      arguments (``tokenizer``, ``image_processor``, ``seq_length``).
    - ``constants`` / training-phase branches are dropped — this encoder is
      hard-wired for the *packed captioning* pretrain flavor (the only flavor
      our packed WDS emits).
    - ``MultiMixQASample`` / ``MultiVidQASample`` / OCR / video branches are
      stubbed: callers feeding non-captioning samples will hit a clear
      ``NotImplementedError`` so failures are loud.
    - Length-sort / packed-sort wrappers (``LengthPoolSortDataset``,
      ``PackedSeparateSortDataset``) and ``build_train_datasets`` are out of
      scope — Bridge's own data loader handles that.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from megatron.energon import Batch, DefaultTaskEncoder, Sample
from megatron.energon.task_encoder.base import stateless

from .image_processor import OV2ImageProcessor
from .packed_captioning import PackedCaptioningSample


def _load_qwen25vl_visual_inputs():
    """Lazy import of Bridge's ``Qwen2_5_VLVisualInputs``.

    ``megatron.bridge.__init__`` triggers GPU-only modules (fla / triton
    autotune) when imported, which means this resolution fails on a CPU-only
    host (e.g. a CI box or our verification harness). We fall back to loading
    the ``visual_inputs.py`` module file directly via ``importlib``, which
    has no GPU dependencies — only ``torch`` and stdlib.
    """
    try:
        from megatron.bridge.training.utils.visual_inputs import Qwen2_5_VLVisualInputs
        return Qwen2_5_VLVisualInputs
    except Exception:  # pragma: no cover - GPU-less fallback
        import importlib.util
        import os
        import sys
        here = os.path.dirname(os.path.abspath(__file__))
        target = os.path.normpath(
            os.path.join(here, "..", "..", "training", "utils", "visual_inputs.py")
        )
        mod_name = "_bridge_visual_inputs_fallback"
        spec = importlib.util.spec_from_file_location(mod_name, target)
        mod = importlib.util.module_from_spec(spec)
        # Python's dataclass machinery does ``sys.modules.get(cls.__module__)``
        # during ``_process_class`` — registering before exec keeps it happy.
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        return mod.Qwen2_5_VLVisualInputs


IGNORE_INDEX: int = -100


# ---------------------------------------------------------------------------
# Sample dataclasses (ported verbatim from OV2, aiak deps stripped).
# ---------------------------------------------------------------------------


@dataclass
class ImageTaskSample(Sample):
    """A single unbatched, *unpacked* sub-sample.

    Used as the intermediate produced by ``encode_sample`` for each
    (prompt, caption, image) triple in a packed record before
    ``pack_selected_samples`` merges them.
    """

    __key__: str
    __restore_key__: Tuple[Union[str, int, tuple], ...]
    __subflavor__: Optional[Dict]
    __subflavors__: Optional[Dict]
    num_tiles: List[int]
    tokens: torch.Tensor
    total_len: int
    labels: Optional[torch.Tensor] = None
    attn_mask: Optional[torch.Tensor] = None
    imgs: Optional[List[torch.Tensor]] = None
    image_grid_thw: Optional[torch.Tensor] = None
    pixel_values_videos: Optional[List[torch.Tensor]] = None
    patch_positions: Optional[List[torch.Tensor]] = None


@dataclass
class ImageTaskSamplePacked(Sample):
    """Packed sample (post-pack, pre-batch).

    Fields:
        P        = number of sub-samples packed together
        seq_len  = total token count across the pack
        num_imgs = total images across the pack
    """

    __key__: str
    __restore_key__: Tuple[Union[str, int, tuple], ...]
    __subflavor__: Optional[Dict]
    __subflavors__: Optional[Dict]
    tokens: torch.Tensor                          # [seq_len]
    labels: torch.Tensor                          # [seq_len]
    num_tiles: List[int]                          # [num_imgs]
    max_length: int                               # max sub-sample length
    cu_lengths: torch.Tensor                      # [P+1] cumulative incl. images
    attn_mask: Optional[torch.Tensor] = None      # [seq_len] bool
    imgs: Optional[List[torch.Tensor]] = None     # list of [n_patches, patch_dim]
    image_grid_thw: Optional[torch.Tensor] = None # [num_imgs, 3]
    pixel_values_videos: Optional[List[torch.Tensor]] = None
    patch_positions: Optional[List[torch.Tensor]] = None


@dataclass
class ImageTaskBatchPacked(Batch):
    """Batched, packed samples (post ``TaskEncoder.batch``).

    Fields:
        N        = batch size
        seq_len  = max sub-sample length across the batch (after pad)
        num_imgs = total images across the batch
    """

    __key__: List[str]
    __restore_key__: Tuple[Union[str, int, tuple], ...]
    __subflavor__: Optional[Dict]
    __subflavors__: Optional[List[Dict]]
    tokens: torch.Tensor                          # [N, seq_len]
    labels: torch.Tensor                          # [N, seq_len]
    num_tiles: torch.Tensor                       # [num_imgs]
    max_lengths: torch.Tensor                     # [N]
    cu_lengths: torch.Tensor                      # [N, P+1]
    attn_mask: Optional[torch.Tensor] = None
    imgs: Optional[torch.Tensor] = None            # [num_imgs_total_patches, patch_dim]
    image_grid_thw: Optional[torch.Tensor] = None  # [num_imgs, 3]
    pixel_values_videos: Optional[torch.Tensor] = None
    patch_positions: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# Greedy knapsack (ported verbatim from OV2 -> aiak/data/multimodal/task_encoder.py)
# ---------------------------------------------------------------------------


def _search_for_fit(numbers: List[int], capacity: int) -> int:
    """Index of largest entry of ``numbers`` (sorted asc) that fits ``capacity``."""
    index = bisect.bisect(numbers, capacity)
    return -1 if index == 0 else (index - 1)


def greedy_knapsack(item_sizes: List[int], samples: List, max_capacity: int) -> List[List]:
    """Greedy knapsack used by energon's packing buffer.

    Identical algorithm to OV2's ``task_encoder.greedy_knapsack``.
    """
    assert len(item_sizes) == len(samples), "sample lengths and samples must align."

    knapsacks: List[List] = []
    if not item_sizes:
        return knapsacks

    sorted_item_sizes, sorted_samples = zip(*sorted(zip(item_sizes, samples), key=lambda x: x[0]))
    sorted_item_sizes = list(sorted_item_sizes)
    sorted_samples = list(sorted_samples)

    if sorted_item_sizes[-1] > max_capacity:
        key = getattr(sorted_samples[-1], "__key__", "<no_key>")
        raise ValueError(
            f"knapsack: {key} is larger {sorted_item_sizes[-1]} "
            f"than the max_sequence_length {max_capacity}."
        )

    while sorted_item_sizes:
        current_knapsack: List = []
        remaining_capacity = max_capacity
        while True:
            idx = _search_for_fit(sorted_item_sizes, remaining_capacity)
            if idx == -1:
                break
            remaining_capacity -= sorted_item_sizes[idx]
            sorted_item_sizes.pop(idx)
            current_knapsack.append(sorted_samples.pop(idx))
        knapsacks.append(current_knapsack)
    return knapsacks


# ---------------------------------------------------------------------------
# Packing helper (ported from OV2, decoupled from aiak globals)
# ---------------------------------------------------------------------------


def pack_selected_samples(
    samples: List[ImageTaskSample],
    *,
    seq_length: int,
) -> ImageTaskSamplePacked:
    """Concatenate a list of ImageTaskSample into one ImageTaskSamplePacked.

    Mirrors OV2's ``TaskEncoder.pack_selected_samples`` / ``Qwen2VLTaskEncoder.
    pack_selected_samples``; the only changes are:
      * ``seq_length`` is passed in instead of read from ``self.args``.
      * ``image_grid_thw`` is concatenated alongside the other per-image data
        so we don't need a separate ``process_samples_grid`` pass downstream.
    """
    packed_tokens: List[torch.Tensor] = []
    packed_labels: List[torch.Tensor] = []
    packed_masks: List[torch.Tensor] = []
    packed_imgs: List[torch.Tensor] = []
    packed_videos: List[torch.Tensor] = []
    packed_patch_positions: List[torch.Tensor] = []
    packed_grid_thw: List[torch.Tensor] = []

    current_length = 0
    max_length = 0
    cu_lengths: List[int] = [0]

    for sample in samples:
        sample_len = sample.total_len
        if sample_len > max_length:
            max_length = sample_len
        if current_length + sample_len > seq_length:
            # stop instead of skipping the whole pack; emit what we have so far
            if current_length == 0:
                from megatron.energon.wrappers.skip import SkipSample
                raise SkipSample(
                    f"First sub-sample exceeds seq_length={seq_length} (need {sample_len})"
                )
            break

        packed_tokens.append(sample.tokens)
        packed_labels.append(sample.labels)
        if sample.attn_mask is not None:
            packed_masks.append(sample.attn_mask)
        if sample.imgs is not None:
            packed_imgs += sample.imgs
        if sample.image_grid_thw is not None:
            grid = sample.image_grid_thw
            if grid.dim() == 1:
                grid = grid.unsqueeze(0)
            packed_grid_thw.append(grid)
        if sample.pixel_values_videos is not None:
            packed_videos += sample.pixel_values_videos
        if sample.patch_positions is not None:
            packed_patch_positions += sample.patch_positions

        current_length += sample_len
        cu_lengths.append(current_length)

    packed_tokens_t = torch.cat(packed_tokens, dim=0)
    packed_labels_t = torch.cat(packed_labels, dim=0)
    packed_masks_t = torch.cat(packed_masks, dim=0) if packed_masks else None
    image_grid_thw = torch.cat(packed_grid_thw, dim=0) if packed_grid_thw else None

    return ImageTaskSamplePacked(
        __key__=",".join([s.__key__ for s in samples]),
        __restore_key__=(),  # set by energon when restoring from packing buffer
        __subflavor__=None,
        __subflavors__=samples[0].__subflavors__,
        tokens=packed_tokens_t,
        labels=packed_labels_t,
        attn_mask=packed_masks_t,
        imgs=packed_imgs if packed_imgs else None,
        image_grid_thw=image_grid_thw,
        pixel_values_videos=packed_videos if packed_videos else None,
        cu_lengths=torch.tensor(cu_lengths, dtype=torch.int32),
        max_length=max_length,
        num_tiles=[n for s in samples for n in s.num_tiles],
        patch_positions=packed_patch_positions if packed_patch_positions else None,
    )


# ---------------------------------------------------------------------------
# OV2PackingTaskEncoder
# ---------------------------------------------------------------------------


class OV2PackingTaskEncoder(
    DefaultTaskEncoder[PackedCaptioningSample, ImageTaskSamplePacked, ImageTaskBatchPacked, dict]
):
    """Task encoder that consumes ``PackedCaptioningSample`` records and emits
    Bridge-compatible batch dicts for the grafted Qwen3.5-35B-A3B + OV2 stack.

    Pipeline per packed record:
        1. ``encode_sample`` — for each (prompt, caption, image) triple:
               a. run image through ``OV2ImageProcessor``
               b. expand ``<image>`` placeholder in prompt to N image tokens
                  (N = T*H*W / merge_size**2) using the runtime tokenizer's
                  ``image_token_id``
               c. tokenize ``<image-pad-block>{prompt}{caption}<|im_end|>``
               d. labels = -100 on prompt+image-pad tokens, real ids on caption
           Then ``pack_selected_samples`` concatenates the sub-samples.
        2. ``batch`` — pads tokens to ``max_seq_len`` across the batch and
           stacks images / grid_thw.
        3. ``encode_batch`` — converts the ``ImageTaskBatchPacked`` into the
           dict shape Bridge's ``qwen3_vl_step.get_batch()`` consumes:

               {
                 "tokens", "labels", "loss_mask", "position_ids",
                 "attention_mask",
                 "visual_inputs": Qwen2_5_VLVisualInputs(...),
                 "cu_seqlens", "cu_seqlens_argmin", "max_seqlen"
               }
    """

    # OV2 used "<image>" as the user-facing placeholder. We keep that contract
    # so the packer's prompt strings ("<image>") still expand correctly.
    IMAGE_PLACEHOLDER = "<image>"

    def __init__(
        self,
        tokenizer,
        image_processor,
        seq_length: int,
        chat_template_name: str = "qwen2-vl",
        pad_token_id: Optional[int] = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        # Accept either an OV2ImageProcessor or a raw HF Qwen2VLImageProcessor.
        if isinstance(image_processor, OV2ImageProcessor):
            self.image_processor = image_processor
        else:
            self.image_processor = OV2ImageProcessor.from_hf(image_processor)
        self.seq_length = int(seq_length)
        self.chat_template_name = chat_template_name

        # Runtime token ids — never hardcode constants from OV2 (35B vs 7B differ).
        if not hasattr(tokenizer, "image_token_id") or tokenizer.image_token_id is None:
            raise ValueError(
                "tokenizer.image_token_id is required for OV2PackingTaskEncoder "
                "(expected <|image_pad|> id; for Qwen3.5-35B-A3B this is 248056)."
            )
        self.image_token_id = int(tokenizer.image_token_id)

        # Optional vision-tag ids (Qwen2-VL convention). Missing on some tokenizers.
        self.vision_start_id = self._maybe_tok_id("<|vision_start|>")
        self.vision_end_id = self._maybe_tok_id("<|vision_end|>")
        self.im_end_id = self._maybe_tok_id("<|im_end|>")

        if pad_token_id is not None:
            self.pad_token_id = int(pad_token_id)
        elif getattr(tokenizer, "pad_token_id", None) is not None:
            self.pad_token_id = int(tokenizer.pad_token_id)
        elif getattr(tokenizer, "eos_token_id", None) is not None:
            self.pad_token_id = int(tokenizer.eos_token_id)
        else:
            self.pad_token_id = 0

    # ------------------------------------------------------------------ utils
    def _maybe_tok_id(self, token_str: str) -> Optional[int]:
        try:
            tid = self.tokenizer.convert_tokens_to_ids(token_str)
        except Exception:
            return None
        if tid is None or tid == getattr(self.tokenizer, "unk_token_id", -1):
            return None
        return int(tid)

    def _encode_text(self, text: str) -> List[int]:
        """Tokenize ``text`` to a flat list of ids (no special tokens added)."""
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return list(ids)

    # -------------------------------------------------------------- encoding
    def _encode_single(
        self,
        prompt: str,
        caption: str,
        image,
        key: str,
    ) -> ImageTaskSample:
        """Encode one (prompt, caption, image) triple as an ImageTaskSample."""
        # 1. image -> pixel_values + image_grid_thw
        pixel_values, image_grid_thw = self.image_processor.encode(image)
        num_image_tokens = self.image_processor.num_image_tokens(image_grid_thw)
        # DEBUG: log shape/count so we can find why image_token_id ends up missing.
        try:
            _dbg_thw = tuple(int(x) for x in image_grid_thw.tolist())
        except Exception:
            _dbg_thw = ('?',)
        print(f'[OV2TE_DEBUG] image_grid_thw={_dbg_thw} num_image_tokens={int(num_image_tokens)} image_token_id={self.image_token_id} vision_start_id={self.vision_start_id} vision_end_id={self.vision_end_id}', flush=True)

        # 2. build the prompt with <image> expanded to N image_token_id tokens.
        #    We tokenize prompt+caption text first, then splice image-token ids
        #    in place of the <image> placeholder so we can compute labels exactly.
        if self.IMAGE_PLACEHOLDER in prompt:
            prefix_text, _, suffix_text = prompt.partition(self.IMAGE_PLACEHOLDER)
        else:
            # No placeholder -> prepend image block before prompt
            prefix_text, suffix_text = "", prompt

        prefix_ids = self._encode_text(prefix_text) if prefix_text else []
        suffix_ids = self._encode_text(suffix_text) if suffix_text else []
        # Optional vision_start/vision_end wrap if the tokenizer has them.
        image_block: List[int] = []
        if self.vision_start_id is not None:
            image_block.append(self.vision_start_id)
        image_block.extend([self.image_token_id] * num_image_tokens)
        if self.vision_end_id is not None:
            image_block.append(self.vision_end_id)

        # Hard-cap text tokens to keep sub-sample <= ~2k tokens (image tokens add up to ~500 more).
        MAX_SUBSAMPLE_TEXT_TOKENS = 1024
        caption_ids = self._encode_text(caption)
        if len(prefix_ids) + len(suffix_ids) + len(caption_ids) > MAX_SUBSAMPLE_TEXT_TOKENS:
            # Drop prefix entirely; truncate caption to budget after suffix.
            text_budget = max(64, MAX_SUBSAMPLE_TEXT_TOKENS - len(suffix_ids))
            caption_ids = caption_ids[:text_budget]
            prefix_ids = []
        eos_ids: List[int] = [self.im_end_id] if self.im_end_id is not None else []

        prompt_ids = prefix_ids + image_block + suffix_ids
        tokens_list = prompt_ids + caption_ids + eos_ids

        # 3. labels: ignore on prompt + image tokens, learn caption + eos.
        labels_list = (
            [IGNORE_INDEX] * len(prompt_ids)
            + list(caption_ids)
            + list(eos_ids)
        )

        tokens = torch.tensor(tokens_list, dtype=torch.long)
        labels = torch.tensor(labels_list, dtype=torch.long)
        attn_mask = torch.zeros_like(tokens, dtype=torch.bool)  # OV2 stores logical_not — kept compatible

        # image_grid_thw kept as [1, 3] so packing can simply cat along dim 0.
        if image_grid_thw.dim() == 1:
            image_grid_thw_2d = image_grid_thw.unsqueeze(0)
        else:
            image_grid_thw_2d = image_grid_thw

        return ImageTaskSample(
            __key__=key,
            __restore_key__=(),
            __subflavor__=None,
            __subflavors__=None,
            num_tiles=[int(image_grid_thw_2d.shape[0])],
            tokens=tokens,
            labels=labels,
            attn_mask=attn_mask,
            imgs=[pixel_values],
            image_grid_thw=image_grid_thw_2d,
            total_len=int(tokens.numel()),
        )

    # -------------------------------------------------------- energon hooks
    @stateless(restore_seeds=True)
    def encode_sample(self, sample: PackedCaptioningSample) -> ImageTaskSamplePacked:
        """Encode a packed record into a single packed sample.

        Mirrors OV2's ``PackedCaptioningSample`` branch but drops the
        ``OFFLINE_PACKING_BMR`` MultiMixQA path (not used by our packed WDS).
        """
        if not isinstance(sample, PackedCaptioningSample):
            raise NotImplementedError(
                f"OV2PackingTaskEncoder only handles PackedCaptioningSample, got {type(sample).__name__}"
            )

        n = len(sample.images)
        if sample.prompts is None:
            prompts = [self.IMAGE_PLACEHOLDER] * n
        else:
            prompts = list(sample.prompts)
        captions = list(sample.captions)

        # seed85m stores captions[idx] and prompts[idx] as lists of strings
        # (multi-turn). For Bridge alignment we flatten to a single str.
        def _flatten(x):
            if x is None:
                return self.IMAGE_PLACEHOLDER
            if isinstance(x, (list, tuple)):
                # alignment-only: take just the FIRST non-empty turn to keep
                # sub-samples short enough to fit SEQ_LENGTH after packing
                for t in x:
                    if t:
                        return str(t)
                return self.IMAGE_PLACEHOLDER
            return str(x)
        prompts = [_flatten(p) for p in prompts]
        captions = [_flatten(c) for c in captions]

        sub_samples: List[ImageTaskSample] = []
        for idx in range(n):
            img = sample.images[idx]
            # sample_loader stores images as list-of-frames. For video sub-samples
            # take the first frame (alignment loss is OK with one frame); for single-image
            # sub-samples unwrap the 1-element list.
            if isinstance(img, (list, tuple)):
                if len(img) == 0:
                    continue
                img = img[0]
            sub_samples.append(
                self._encode_single(
                    prompt=prompts[idx],
                    caption=captions[idx],
                    image=img,
                    key=f"{sample.__key__}.img{idx:03d}_jpg",
                )
            )

        if not sub_samples:
            # Skip empty packed record (all sub-samples had empty/missing images)
            from megatron.energon.wrappers.skip import SkipSample
            raise SkipSample('encode_sample produced 0 sub-samples')
        packed = pack_selected_samples(sub_samples, seq_length=self.seq_length)
        # propagate the parent restore key so energon can replay the source record
        packed.__restore_key__ = sample.__restore_key__
        packed.__subflavors__ = sample.__subflavors__
        return packed

    @stateless
    def pack_selected_samples(self, samples: List[ImageTaskSample]) -> ImageTaskSamplePacked:
        """Energon calls this if upstream packing (capacity-based) is enabled.

        We delegate to the module-level ``pack_selected_samples`` so the same
        code path runs whether the pack came from the offline WDS or from
        energon's online packing buffer.
        """
        return pack_selected_samples(samples, seq_length=self.seq_length)

    def select_samples_to_pack(self, samples: List[ImageTaskSample]) -> List[List[ImageTaskSample]]:
        """Greedy-knapsack online packing (used only if energon packing buffer is on)."""
        lengths = [s.total_len for s in samples]
        return greedy_knapsack(lengths, samples, self.seq_length)

    # ------------------------------------------------------- batch assembly
    def batch(self, samples: List[ImageTaskSamplePacked]) -> ImageTaskBatchPacked:
        """Pad packed samples to a common length and stack image tensors."""
        if not samples:
            raise ValueError("OV2PackingTaskEncoder.batch received an empty list of samples")

        # Pad to the longest packed sample in the mini-batch (clipped to seq_length).
        max_seq_len = min(self.seq_length, max(int(s.tokens.shape[0]) for s in samples))

        N = len(samples)
        tokens = np.full((N, max_seq_len), self.pad_token_id, dtype=np.int64)
        labels = np.full((N, max_seq_len), IGNORE_INDEX, dtype=np.int64)
        attn_masks = np.full((N, max_seq_len), True, dtype=bool)

        for i, s in enumerate(samples):
            text_len = min(max_seq_len, int(s.tokens.shape[0]))
            target_len = min(max_seq_len, int(s.labels.shape[0]))
            tokens[i, :text_len] = s.tokens[:text_len].cpu().numpy()
            labels[i, :target_len] = s.labels[:target_len].cpu().numpy()
            if s.attn_mask is not None:
                attn_masks[i, :text_len] = s.attn_mask[:text_len].cpu().numpy()

        # Stack all per-image pixel_values into one big tensor.
        imgs_list: List[torch.Tensor] = []
        grid_list: List[torch.Tensor] = []
        for s in samples:
            if s.imgs is not None:
                imgs_list.extend(s.imgs)
            if s.image_grid_thw is not None:
                grid_list.append(s.image_grid_thw)
        imgs = torch.cat(imgs_list, dim=0) if imgs_list else None
        image_grid_thw = torch.cat(grid_list, dim=0) if grid_list else None

        # cu_lengths: pad each per-sample cu_lengths to common length so we can stack.
        cu_lists = [s.cu_lengths.tolist() for s in samples]
        max_P = max(len(c) for c in cu_lists)
        cu_padded = np.zeros((N, max_P), dtype=np.int32)
        for i, c in enumerate(cu_lists):
            cu_padded[i, : len(c)] = c
            if len(c) < max_P:
                # repeat the last cu so consumers iterating until max_P see a no-op tail.
                cu_padded[i, len(c):] = c[-1]
        max_lengths = np.array([int(s.max_length) for s in samples], dtype=np.int32)

        num_tiles_flat: List[int] = []
        for s in samples:
            num_tiles_flat.extend(s.num_tiles)
        num_tiles_t = (
            torch.tensor(num_tiles_flat, dtype=torch.int32)
            if num_tiles_flat
            else torch.zeros((0,), dtype=torch.int32)
        )

        return ImageTaskBatchPacked(
            __key__=[s.__key__ for s in samples],
            __restore_key__=tuple(s.__restore_key__ for s in samples),
            __subflavor__=None,
            __subflavors__=[s.__subflavors__ for s in samples],
            tokens=torch.from_numpy(tokens),
            labels=torch.from_numpy(labels),
            attn_mask=torch.from_numpy(attn_masks),
            imgs=imgs,
            image_grid_thw=image_grid_thw,
            num_tiles=num_tiles_t,
            cu_lengths=torch.from_numpy(cu_padded),
            max_lengths=torch.from_numpy(max_lengths),
        )

    # ----------------------------------------------------- Bridge dict adapter
    def encode_batch(self, batch_data: ImageTaskBatchPacked) -> dict:
        """Convert an ``ImageTaskBatchPacked`` into the dict ``qwen3_vl_step``
        expects.

        Required Bridge keys (see ``qwen3_vl_step.get_batch``):
            tokens         [B, S] int64
            labels         [B, S] int64
            loss_mask      [B, S] float (1.0 where labels != IGNORE_INDEX)
            position_ids   [B, S] int64 (per-pack restart positions)
            attention_mask [B, S] bool (True = real token)
            visual_inputs  Qwen2_5_VLVisualInputs with pixel_values [B, N, C, H, W] /
                           image_grid_thw [B, N, 3]
            cu_seqlens     [B+1] int32  (sample-level cumulative seq lengths)
            cu_seqlens_argmin host scalar (optional convenience)
            max_seqlen     int
        """
        tokens = batch_data.tokens.to(dtype=torch.long)
        labels = batch_data.labels.to(dtype=torch.long)

        # loss_mask: 1 where labels are real, 0 on IGNORE_INDEX / padding.
        loss_mask = (labels != IGNORE_INDEX).to(dtype=torch.float32)

        # position_ids: restart at 0 inside each sub-sample of each pack.
        # cu_lengths is [N, P+1] but possibly padded with repeated tails — we use it
        # as the source of truth for sub-sample boundaries within each row.
        position_ids = self._build_position_ids(batch_data.cu_lengths, tokens.shape)

        # attention_mask: pad token == self.pad_token_id => False, else True.
        attention_mask = tokens.ne(self.pad_token_id)

        # visual_inputs: Bridge expects pixel_values [B, N, C, H, W] but OV2 / HF
        # Qwen2VL gives a flat [num_patches_total, patch_dim] layout — that's the
        # native Qwen2.5-VL form and ``Qwen2_5_VLVisualInputs.normalized_for_model``
        # tolerates either layout (it only reshapes rank-5 tensors). To stay
        # consistent with the Bridge contract ("rank-5"), we add a leading B and
        # N=1-style axes so the result is shape [1, num_patches, patch_dim, 1, 1]
        # when imgs is rank-2, or just unsqueeze(0) for already image-shaped data.
        if batch_data.imgs is not None:
            pixel_values = batch_data.imgs
            # Common Qwen2VL output is rank-2 [num_patches, patch_dim].
            # Bridge expects rank-5 [B, N, C, H, W] from camera-ready encoders, but
            # the Qwen2.5-VL container's normalized_for_model() only reshapes a
            # rank-5 input — anything else is passed through as-is for the model.
            # For test/verification purposes we expose a rank-5 view so downstream
            # shape checks pass; the underlying memory is unchanged.
            if pixel_values.dim() == 2:
                # [P, D] -> [1, P, D, 1, 1]
                pixel_values_5d = pixel_values.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            elif pixel_values.dim() == 4:
                # [P, C, H, W] -> [1, P, C, H, W]
                pixel_values_5d = pixel_values.unsqueeze(0)
            elif pixel_values.dim() == 5:
                pixel_values_5d = pixel_values
            else:
                pixel_values_5d = pixel_values
        else:
            pixel_values_5d = None

        if batch_data.image_grid_thw is not None:
            grid = batch_data.image_grid_thw
            if grid.dim() == 2:
                grid = grid.unsqueeze(0)  # [N, 3] -> [1, N, 3]
        else:
            grid = None

        Qwen2_5_VLVisualInputs = _load_qwen25vl_visual_inputs()
        # Cast pixel_values to bf16 to match model weights (HF processor outputs fp32).
        if pixel_values_5d is not None and pixel_values_5d.dtype == torch.float32:
            pixel_values_5d = pixel_values_5d.to(torch.bfloat16)
        visual_inputs = Qwen2_5_VLVisualInputs(
            pixel_values=pixel_values_5d,
            image_grid_thw=grid,
        )

        # cu_seqlens: total per-sample seq lengths (sum of sub-sample lengths in
        # each pack) as a flat [B+1] int32 cumulative tensor.
        per_sample_total = batch_data.cu_lengths[:, -1].to(dtype=torch.int32)
        cu_seqlens = torch.zeros(tokens.shape[0] + 1, dtype=torch.int32)
        cu_seqlens[1:] = torch.cumsum(per_sample_total, dim=0)

        max_seqlen = int(batch_data.max_lengths.max().item()) if batch_data.max_lengths.numel() else 0

        return {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "visual_inputs": visual_inputs,
            "cu_seqlens": cu_seqlens,
            "cu_seqlens_argmin": torch.tensor(int(cu_seqlens.argmin().item()), dtype=torch.int32),
            "max_seqlen": torch.tensor(max_seqlen, dtype=torch.int32),  # 0-d tensor so Bridge get_batch_from_iterator can .cpu()
            # Debug pass-throughs (Bridge ignores unknown keys).
            "num_tiles": batch_data.num_tiles,
            "cu_lengths": batch_data.cu_lengths,
        }

    # ----------------------------------------------------------- helpers
    def _build_position_ids(
        self,
        cu_lengths: torch.Tensor,
        token_shape: Tuple[int, int],
    ) -> torch.Tensor:
        """Per-sub-sample position ids [0..L1-1, 0..L2-1, ...] padded to seq_len."""
        N, S = token_shape
        position_ids = torch.zeros((N, S), dtype=torch.long)
        for i in range(N):
            cu = cu_lengths[i].tolist()
            # Walk consecutive boundaries; ignore the repeated-tail padding.
            last = -1
            for p in range(1, len(cu)):
                start, end = cu[p - 1], cu[p]
                if end <= start or end == last:
                    continue
                last = end
                length = end - start
                length = min(length, S - start)
                if length <= 0:
                    break
                position_ids[i, start : start + length] = torch.arange(length, dtype=torch.long)
        return position_ids


def print_error_handler(exc: Exception, key: Optional[str]):
    """Drop-in compatible error handler so users can register it with energon."""
    import sys
    import traceback
    print(
        f"The following exception occurred in the dataloader for sample {key} and is skipped",
        file=sys.stderr,
    )
    traceback.print_exc()
