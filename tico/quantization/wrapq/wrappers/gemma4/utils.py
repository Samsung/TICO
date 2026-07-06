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

    if visual_embeds.dim() == 2:
        visual_embeds = visual_embeds.unsqueeze(0)

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
        # Warn instead of raising error - use actual token count for flexibility
        import warnings

        warnings.warn(
            f"Visual token count mismatch: expected {expected_len}, got {visual_len}. "
            "Using actual count. If this is unexpected, check image resolution settings."
        )
        expected_len = visual_len

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
