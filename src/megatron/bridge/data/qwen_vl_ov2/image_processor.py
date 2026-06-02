"""Qwen2-VL image preprocessing helpers used by the OV2 packed captioning pipeline.

This module wraps HuggingFace's ``Qwen2VLImageProcessor`` so the rest of the
package can stay framework-light. The preprocessor config (mean/std, patch
size, merge size, min/max pixels) lives at
``/ov2/pretrain_models/preprocessor/preprocessor_llava_onevision1_5/``.

The OV2 source mostly delegates to HF's processor; we keep the same contract
(``smart_resize`` + ``Qwen2VLImageProcessor.preprocess``) and expose a thin
``OV2ImageProcessor`` wrapper that returns ``(pixel_values, image_grid_thw)``
per image.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from PIL import Image


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280,
) -> Tuple[int, int]:
    """Rescale (height, width) so each axis is a multiple of ``factor`` and the
    total pixel count lies within ``[min_pixels, max_pixels]``.

    Ported from ``qwen_vl_utils.vision_process.smart_resize`` (Qwen2-VL recipe).
    The OV2 task encoder calls this prior to invoking ``Qwen2VLImageProcessor``
    so the resulting (H, W) divides cleanly into (patch * merge) tiles.
    """
    if height < factor or width < factor:
        # Bump up to at least one tile per axis so the image processor doesn't fail.
        scale = factor / min(height, width)
        height = int(math.ceil(height * scale))
        width = int(math.ceil(width * scale))

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    # Guarantee at least one tile per axis after rounding.
    h_bar = max(h_bar, factor)
    w_bar = max(w_bar, factor)

    return h_bar, w_bar


class OV2ImageProcessor:
    """Thin wrapper around HF's ``Qwen2VLImageProcessor``.

    Construction:
        proc = OV2ImageProcessor.from_pretrained(
            "/ov2/pretrain_models/preprocessor/preprocessor_llava_onevision1_5"
        )

    Per-image call:
        pixel_values, image_grid_thw = proc.encode(pil_image)

    ``pixel_values`` is shaped ``[num_patches, patch_dim]`` (the flattened
    per-patch token features the processor returns) and ``image_grid_thw``
    is a 1-D tensor of ``(T, H_patches, W_patches)`` for that image. The
    number of LLM image tokens after the adapter's 2×2 spatial merge is
    ``T * H_patches * W_patches / (merge_size ** 2)``.
    """

    def __init__(self, hf_processor):
        self._hf = hf_processor
        # Qwen2VL processor exposes these on the instance.
        self.patch_size: int = int(getattr(hf_processor, "patch_size", 14))
        self.merge_size: int = int(getattr(hf_processor, "merge_size", 2))
        self.temporal_patch_size: int = int(getattr(hf_processor, "temporal_patch_size", 1))
        # smart_resize factor = patch_size * merge_size
        self._resize_factor = self.patch_size * self.merge_size
        self.min_pixels: int = int(getattr(hf_processor, "min_pixels", 56 * 56))
        self.max_pixels: int = int(getattr(hf_processor, "max_pixels", 14 * 14 * 4 * 1280))

    @classmethod
    def from_pretrained(cls, path: str) -> "OV2ImageProcessor":
        """Load the HF Qwen2VLImageProcessor from disk and wrap it."""
        # Local import keeps top-level light.
        from transformers import AutoImageProcessor

        hf_processor = AutoImageProcessor.from_pretrained(path)
        return cls(hf_processor)

    @classmethod
    def from_hf(cls, hf_processor) -> "OV2ImageProcessor":
        """Wrap an already-loaded HF image processor."""
        return cls(hf_processor)

    def _resize(self, image: Image.Image) -> Image.Image:
        """Apply Qwen2-VL smart_resize so H and W are multiples of patch*merge."""
        new_h, new_w = smart_resize(
            image.height,
            image.width,
            factor=self._resize_factor,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        if (new_h, new_w) != (image.height, image.width):
            image = image.resize((new_w, new_h))
        return image

    def encode(
        self, image: Image.Image
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a single PIL image into ``(pixel_values, image_grid_thw)``.

        ``pixel_values`` shape: ``[num_patches, patch_dim]`` (HF Qwen2VL layout)
        ``image_grid_thw`` shape: ``[3]`` -> (T, H_patches, W_patches)
        """
        # Energon webdataset default decoder turns .jpg bytes into a torch.Tensor;
        # OV2's original loader expected a PIL.Image. Coerce here so both work.
        if not hasattr(image, "convert"):
            import numpy as _np
            import torch as _torch
            from PIL import Image as _PILImage
            if isinstance(image, _torch.Tensor):
                arr = image.detach().cpu().numpy()
            else:
                arr = _np.asarray(image)
            # Squeeze leading singleton dims, e.g. (T=1, C=1, H, W) -> (H, W)
            while arr.ndim > 3 and arr.shape[0] == 1:
                arr = arr[0]
            if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
                arr = arr.transpose(1, 2, 0)
            if arr.dtype != _np.uint8:
                arr = (arr * 255 if arr.max() <= 1.0 else arr).astype(_np.uint8)
            if arr.ndim == 3 and arr.shape[2] == 1:
                arr = arr[..., 0]
            image = _PILImage.fromarray(arr)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = self._resize(image)

        out = self._hf(images=image, return_tensors="pt")
        pixel_values: torch.Tensor = out["pixel_values"]
        image_grid_thw: torch.Tensor = out["image_grid_thw"]
        # HF returns image_grid_thw shape [num_images, 3]; we collapse for one image.
        if image_grid_thw.dim() == 2 and image_grid_thw.shape[0] == 1:
            image_grid_thw = image_grid_thw[0]
        return pixel_values, image_grid_thw

    def num_image_tokens(self, image_grid_thw: torch.Tensor) -> int:
        """Return the number of LLM image tokens *after* the 2x2 adapter merge."""
        t, h, w = (int(x) for x in image_grid_thw.tolist())
        merged = (self.merge_size ** 2)
        return (t * h * w) // merged
