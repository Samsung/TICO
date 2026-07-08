# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

import torch


@dataclass(frozen=True)
class StaticGemma4Layout:
    """Static layout information required by the Gemma4 E2B runtime.

    Attributes:
        max_seq: Static text sequence length used by prefill graphs.
        visual_start_idx: First token slot occupied by visual soft tokens.
        num_visual_tokens: Number of visual soft tokens inserted into the prompt.
        batch_size: Static batch size. The initial runtime targets batch size 1.
    """

    max_seq: int
    visual_start_idx: int
    num_visual_tokens: int
    batch_size: int = 1

    def validate(self) -> None:
        """Validate that the static multimodal layout is internally consistent."""
        if self.batch_size != 1:
            raise ValueError(
                "Gemma4 E2B static runtime currently supports batch_size=1 only."
            )
        if self.max_seq <= 0:
            raise ValueError(f"max_seq must be positive, got {self.max_seq}.")
        if self.visual_start_idx < 0:
            raise ValueError(
                f"visual_start_idx must be non-negative, got {self.visual_start_idx}."
            )
        if self.num_visual_tokens < 0:
            raise ValueError(
                f"num_visual_tokens must be non-negative, got {self.num_visual_tokens}."
            )
        end = self.visual_start_idx + self.num_visual_tokens
        if end > self.max_seq:
            raise ValueError(
                "Visual token range exceeds max_seq: "
                f"visual_start_idx={self.visual_start_idx}, "
                f"num_visual_tokens={self.num_visual_tokens}, max_seq={self.max_seq}."
            )


def assert_gemma4_e2b_no_moe(model_or_config: Any) -> None:
    """Raise if the supplied Gemma4 model or config enables MoE blocks.

    The initial Gemma4 E2B runtime supports only dense decoder layers. This
    guard should be called both in recipe loading and wrapper construction.
    """

    config = getattr(model_or_config, "config", model_or_config)
    if hasattr(config, "get_text_config"):
        text_config = config.get_text_config()
    else:
        text_config = getattr(config, "text_config", config)

    if bool(getattr(text_config, "enable_moe_block", False)):
        raise ValueError(
            "Gemma4 E2B static runtime supports dense decoder layers only, "
            "but text_config.enable_moe_block=True."
        )

    if hasattr(model_or_config, "named_modules"):
        for name, module in model_or_config.named_modules():
            cls_name = type(module).__name__
            if cls_name in {"Gemma4TextRouter", "Gemma4TextExperts"}:
                raise ValueError(
                    f"Unexpected MoE module in Gemma4 E2B model: {name} ({cls_name})."
                )


def _ensure_batched_visual_embeds(visual_embeds: torch.Tensor) -> torch.Tensor:
    """Return visual embeddings with an explicit batch dimension."""
    if visual_embeds.dim() == 2:
        return visual_embeds.unsqueeze(0)
    return visual_embeds


def _normalize_image_mask(
    image_mask: torch.Tensor,
    *,
    batch: int,
    seq_len: int,
    device: torch.device,
) -> torch.BoolTensor:
    """Return an image placeholder mask with shape ``(B, S)``."""
    if image_mask.dim() == 3:
        image_mask = image_mask[..., 0]
    if image_mask.dim() != 2:
        raise ValueError(
            "image_mask must have shape `(B, S)` or `(B, S, D)`, "
            f"got shape={tuple(image_mask.shape)}."
        )
    if tuple(image_mask.shape) != (batch, seq_len):
        raise ValueError(
            "image_mask shape is incompatible with text_embeds: "
            f"mask={tuple(image_mask.shape)}, expected={(batch, seq_len)}."
        )
    return image_mask.to(device=device, dtype=torch.bool)


def dynamic_placeholder_fuse(
    text_embeds: torch.Tensor,
    visual_embeds: torch.Tensor,
    image_mask: torch.Tensor,
) -> torch.Tensor:
    """Fuse visual embeddings into the actual image placeholder positions.

    This eager/PTQ helper follows the original HF-style multimodal semantics:
    visual embeddings are written to the token positions selected by
    ``image_mask`` instead of assuming a static contiguous visual-token slot.
    The helper intentionally uses boolean indexing and is not meant for static
    export graphs.

    Args:
        text_embeds: Text embedding tensor with shape ``(B, S, D)``.
        visual_embeds: Visual embedding tensor with shape ``(B, V, D)`` or
            ``(V, D)``. A missing batch dimension is inserted.
        image_mask: Boolean image placeholder mask with shape ``(B, S)`` or
            ``(B, S, D)``.

    Returns:
        Fused embedding tensor with the same shape as ``text_embeds``.

    Raises:
        ValueError: If tensor ranks, batch dimensions, hidden dimensions, or
            placeholder counts are incompatible.
    """
    visual_embeds = _ensure_batched_visual_embeds(visual_embeds)

    if text_embeds.dim() != 3:
        raise ValueError(
            f"text_embeds must be rank 3, got shape={tuple(text_embeds.shape)}."
        )
    if visual_embeds.dim() != 3:
        raise ValueError(
            f"visual_embeds must be rank 3, got shape={tuple(visual_embeds.shape)}."
        )

    batch, seq_len, hidden = text_embeds.shape
    if visual_embeds.shape[0] != batch or visual_embeds.shape[2] != hidden:
        raise ValueError(
            "visual_embeds shape is incompatible with text_embeds: "
            f"text={tuple(text_embeds.shape)}, visual={tuple(visual_embeds.shape)}."
        )

    image_mask = _normalize_image_mask(
        image_mask,
        batch=batch,
        seq_len=seq_len,
        device=text_embeds.device,
    )
    visual_len = int(visual_embeds.shape[1])
    token_counts = image_mask.sum(dim=1)
    if not torch.all(token_counts == visual_len):
        raise ValueError(
            "Image placeholder count must match visual embedding length for each "
            "batch row: "
            f"expected={visual_len}, actual={token_counts.detach().cpu().tolist()}."
        )

    fused = text_embeds.clone()
    visual_embeds = visual_embeds.to(device=fused.device, dtype=fused.dtype)
    for batch_idx in range(batch):
        fused[batch_idx, image_mask[batch_idx], :] = visual_embeds[batch_idx]
    return fused


def validate_static_visual_layout(
    image_mask: torch.Tensor,
    *,
    visual_start_idx: int,
    num_visual_tokens: int,
    seq_len: int | None = None,
) -> None:
    """Validate that image placeholders match the static visual-token span.

    This validator is intended for static-runtime calibration and pre-export
    checks. It verifies that each batch row has image placeholders exactly in
    the range ``[visual_start_idx, visual_start_idx + num_visual_tokens)`` and
    nowhere else.

    Args:
        image_mask: Boolean image placeholder mask with shape ``(B, S)`` or
            ``(B, S, D)``.
        visual_start_idx: Start index of the static visual-token span.
        num_visual_tokens: Expected length of the static visual-token span.
        seq_len: Optional expected sequence length. When provided, it must match
            the mask sequence dimension.

    Raises:
        ValueError: If the static span is invalid or the placeholders do not
            exactly match the configured span.
    """
    if image_mask.dim() == 3:
        image_mask = image_mask[..., 0]
    if image_mask.dim() != 2:
        raise ValueError(
            "image_mask must have shape `(B, S)` or `(B, S, D)`, "
            f"got shape={tuple(image_mask.shape)}."
        )

    mask = image_mask.to(dtype=torch.bool)
    _, actual_seq_len = mask.shape
    if seq_len is not None and int(seq_len) != int(actual_seq_len):
        raise ValueError(
            "image_mask sequence length does not match the expected sequence "
            f"length: mask_seq_len={actual_seq_len}, expected={int(seq_len)}."
        )

    start = int(visual_start_idx)
    count = int(num_visual_tokens)
    if start < 0:
        raise ValueError(f"visual_start_idx must be non-negative, got {start}.")
    if count < 0:
        raise ValueError(f"num_visual_tokens must be non-negative, got {count}.")
    end = start + count
    if end > actual_seq_len:
        raise ValueError(
            "Static visual-token span exceeds the sequence length: "
            f"visual_start_idx={start}, num_visual_tokens={count}, "
            f"seq_len={actual_seq_len}."
        )

    expected = torch.zeros_like(mask, dtype=torch.bool)
    if count:
        expected[:, start:end] = True
    if not torch.equal(mask, expected):
        actual_counts = mask.sum(dim=1).detach().cpu().tolist()
        raise ValueError(
            "Image placeholder layout does not match the configured static "
            "visual-token span: "
            f"visual_start_idx={start}, num_visual_tokens={count}, "
            f"seq_len={actual_seq_len}, actual_counts={actual_counts}."
        )


def fixed_slot_fuse(
    text_embeds: torch.Tensor,
    visual_embeds: torch.Tensor,
    *,
    visual_start_idx: int,
    num_visual_tokens: Optional[int] = None,
) -> torch.Tensor:
    """Insert visual embeddings into fixed token slots.

    This function intentionally avoids data-dependent masking and scatter. It is
    suitable for static-shape export when the processor guarantees that visual
    tokens occupy a fixed contiguous slot range.

    Args:
        text_embeds: Text embedding tensor with shape ``(B, S, D)``.
        visual_embeds: Visual embedding tensor with shape ``(B, V, D)`` or
            ``(V, D)``. A missing batch dimension is inserted.
        visual_start_idx: Start index of the static visual token slot.
        num_visual_tokens: Expected number of visual tokens. When ``None``, the
            length is inferred from ``visual_embeds``.

    Returns:
        Fused embedding tensor with the same shape as ``text_embeds``.
    """

    visual_embeds = _ensure_batched_visual_embeds(visual_embeds)

    if text_embeds.dim() != 3:
        raise ValueError(
            f"text_embeds must be rank 3, got shape={tuple(text_embeds.shape)}."
        )
    if visual_embeds.dim() != 3:
        raise ValueError(
            f"visual_embeds must be rank 3, got shape={tuple(visual_embeds.shape)}."
        )

    batch, seq_len, hidden = text_embeds.shape
    if visual_embeds.shape[0] != batch or visual_embeds.shape[2] != hidden:
        raise ValueError(
            "visual_embeds shape is incompatible with text_embeds: "
            f"text={tuple(text_embeds.shape)}, visual={tuple(visual_embeds.shape)}."
        )

    visual_len = int(visual_embeds.shape[1])
    # Use actual visual token count - config num_visual_tokens is a hint for
    # static export, but calibration images may produce different token counts.
    expected_len = visual_len if num_visual_tokens is None else int(num_visual_tokens)
    if visual_len != expected_len:
        raise ValueError(
            "Visual token count mismatch for fixed-slot fusion: "
            f"expected {expected_len}, got {visual_len}."
        )

    end = int(visual_start_idx) + expected_len
    if visual_start_idx < 0 or end > seq_len:
        raise ValueError(
            f"Invalid visual slot range [{visual_start_idx}, {end}) for seq_len={seq_len}."
        )

    return torch.cat(
        [
            text_embeds[:, :visual_start_idx, :],
            visual_embeds,
            text_embeds[:, end:, :],
        ],
        dim=1,
    )


def build_decode_attention_mask(
    *,
    batch_size: int,
    past_len: int,
    max_seq: int,
    device: torch.device,
    dtype: torch.dtype,
    mask_value: float,
) -> torch.Tensor:
    """Build a static decode attention mask for a single-token decode step."""

    if past_len < 0 or past_len >= max_seq:
        raise ValueError(
            f"past_len must be in [0, max_seq), got past_len={past_len}, max_seq={max_seq}."
        )

    mask = torch.full(
        (batch_size, 1, max_seq), float(mask_value), device=device, dtype=dtype
    )
    if past_len > 0:
        mask[:, :, :past_len] = 0.0
    mask[:, :, past_len] = 0.0
    return mask


def extract_text_config(config: Any) -> Any:
    """Return the Gemma4 text config from a model or config object."""

    config = getattr(config, "config", config)
    if hasattr(config, "get_text_config"):
        return config.get_text_config()
    return getattr(config, "text_config", config)


def ensure_static_shape(
    name: str, tensor: torch.Tensor, expected: Iterable[int]
) -> None:
    """Validate that a tensor has the expected static shape."""

    expected_tuple = tuple(int(v) for v in expected)
    actual_tuple = tuple(int(v) for v in tensor.shape)
    if actual_tuple != expected_tuple:
        raise ValueError(f"{name} expected shape {expected_tuple}, got {actual_tuple}.")
