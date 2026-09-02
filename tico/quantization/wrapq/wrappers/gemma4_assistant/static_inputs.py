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

"""Static-shape canonicalization for the Gemma4 assistant draft-one core.

Hugging Face assisted generation produces shared KV states whose length
varies per prompt. The NPU core graph consumes fixed-capacity inputs, so this
module pads/crops the dynamic tensors and rebuilds the additive masks so that
padded slots can never influence the output.

Mask semantics mirror ``Gemma4AssistantForCausalLM.create_attention_masks``:

- ``full_attention`` is bidirectional over every valid shared-KV position.
- ``sliding_attention`` keeps only the last ``sliding_window + 1`` valid
  positions (the HF bidirectional sliding overlay is distance-inclusive, and
  the assistant flips the future-looking window into a past-looking one).
"""

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

import torch

from tico.quantization.wrapq.wrappers.gemma4_assistant.utils import (
    assistant_layer_type_head_dim,
    assistant_shared_kv_num_heads,
    extract_assistant_text_config,
)


LayerKV = Tuple[torch.Tensor, torch.Tensor]

GEMMA4_ASSISTANT_CORE_INPUT_NAMES: tuple[str, ...] = (
    "assistant_input",
    "full_key",
    "full_value",
    "sliding_key",
    "sliding_value",
    "full_attention_mask",
    "sliding_attention_mask",
    "full_cos",
    "full_sin",
    "sliding_cos",
    "sliding_sin",
)

GEMMA4_ASSISTANT_CORE_OUTPUT_NAMES: tuple[str, ...] = (
    "projected_state",
    "assistant_hidden",
    "centroid_logits",
)


@dataclass(frozen=True)
class Gemma4AssistantStaticShapeConfig:
    """Fixed input-shape contract of the assistant draft-one core graph."""

    full_kv_length: int
    sliding_kv_length: int
    batch_size: int = 1
    query_length: int = 1

    def validate(self, text_config: Any) -> None:
        """Validate the static contract against the assistant text config."""
        if self.batch_size != 1:
            raise ValueError(
                "Gemma4 assistant static core supports batch_size=1 only, "
                f"got {self.batch_size}."
            )
        if self.query_length != 1:
            raise ValueError(
                "Gemma4 assistant static core supports query_length=1 only, "
                f"got {self.query_length}."
            )
        if self.full_kv_length < 1:
            raise ValueError(
                f"full_kv_length must be positive, got {self.full_kv_length}."
            )
        if self.sliding_kv_length < 1:
            raise ValueError(
                f"sliding_kv_length must be positive, got {self.sliding_kv_length}."
            )
        max_positions = int(text_config.max_position_embeddings)
        if self.full_kv_length > max_positions:
            raise ValueError(
                "full_kv_length exceeds max_position_embeddings: "
                f"full_kv_length={self.full_kv_length}, "
                f"max_position_embeddings={max_positions}."
            )
        window = int(getattr(text_config, "sliding_window", 0) or 0)
        if "sliding_attention" in tuple(text_config.layer_types):
            # The bidirectional sliding overlay is inclusive, so the visible
            # span is (sliding_window + 1) positions. A smaller capacity would
            # silently drop visible KV entries during canonicalization.
            required = min(window + 1, self.full_kv_length)
            if self.sliding_kv_length < required:
                raise ValueError(
                    "sliding_kv_length is too small for the configured "
                    "sliding window: the assistant can attend to the last "
                    f"sliding_window + 1 = {window + 1} valid positions, so "
                    f"sliding_kv_length must be at least "
                    f"min(sliding_window + 1, full_kv_length) = {required}, "
                    f"got {self.sliding_kv_length}."
                )

    @classmethod
    def from_model_args(
        cls, model_args: Optional[Mapping[str, Any]]
    ) -> "Gemma4AssistantStaticShapeConfig":
        """Build the static contract from ``model_args["assistant"]``."""
        assistant_args = (model_args or {}).get("assistant")
        if not isinstance(assistant_args, Mapping):
            raise ValueError(
                "Gemma4 assistant static export requires "
                "model_args.assistant with full_kv_length and "
                "sliding_kv_length."
            )
        missing = [
            key
            for key in ("full_kv_length", "sliding_kv_length")
            if assistant_args.get(key) is None
        ]
        if missing:
            raise ValueError(
                f"model_args.assistant is missing required keys: {missing}."
            )
        return cls(
            full_kv_length=int(assistant_args["full_kv_length"]),
            sliding_kv_length=int(assistant_args["sliding_kv_length"]),
            batch_size=int(assistant_args.get("batch_size", 1)),
            query_length=int(assistant_args.get("query_length", 1)),
        )


@dataclass(frozen=True)
class Gemma4AssistantStaticInputs:
    """Canonicalized fixed-shape inputs of the assistant core graph."""

    assistant_input: torch.Tensor
    full_key: torch.Tensor
    full_value: torch.Tensor
    sliding_key: torch.Tensor
    sliding_value: torch.Tensor
    full_attention_mask: torch.Tensor
    sliding_attention_mask: torch.Tensor
    full_cos: torch.Tensor
    full_sin: torch.Tensor
    sliding_cos: torch.Tensor
    sliding_sin: torch.Tensor

    def as_tuple(self) -> tuple[torch.Tensor, ...]:
        """Return the flattened export-input tuple in ABI order."""
        return tuple(getattr(self, name) for name in GEMMA4_ASSISTANT_CORE_INPUT_NAMES)

    def attention_mask_mapping(self) -> dict[str, torch.Tensor]:
        """Return per-layer-type masks for eager wrapper execution."""
        return {
            "full_attention": self.full_attention_mask,
            "sliding_attention": self.sliding_attention_mask,
        }

    def shared_kv_mapping(self) -> dict[str, LayerKV]:
        """Return per-layer-type shared KV tuples for eager execution."""
        return {
            "full_attention": (self.full_key, self.full_value),
            "sliding_attention": (self.sliding_key, self.sliding_value),
        }

    def position_embeddings_mapping(
        self,
    ) -> dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Return per-layer-type RoPE tables for eager execution."""
        return {
            "full_attention": (self.full_cos, self.full_sin),
            "sliding_attention": (self.sliding_cos, self.sliding_sin),
        }


def _normalize_position_ids(
    position_ids: Any, *, batch_size: int, query_length: int
) -> torch.Tensor:
    """Normalize position ids to a ``(B, Q)`` long tensor."""
    if position_ids is None:
        raise ValueError(
            "Gemma4 assistant canonicalization requires explicit position_ids."
        )
    if not isinstance(position_ids, torch.Tensor):
        position_ids = torch.tensor(position_ids, dtype=torch.long)
    position_ids = position_ids.long().reshape(-1)
    if position_ids.numel() == 1:
        position_ids = position_ids.expand(batch_size * query_length)
    if position_ids.numel() != batch_size * query_length:
        raise ValueError(
            "position_ids must contain one position per query token: "
            f"got {position_ids.numel()} values for batch={batch_size}, "
            f"query_length={query_length}."
        )
    if int(position_ids.min()) < 0:
        raise ValueError("position_ids must be non-negative.")
    return position_ids.view(batch_size, query_length)


def _validate_kv_pair(
    layer_type: str,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    num_kv_heads: int,
    head_dim: int,
) -> int:
    """Validate one shared KV pair and return its dynamic length."""
    for name, tensor in (("key", key), ("value", value)):
        if tensor.dim() != 4:
            raise ValueError(
                f"{layer_type} shared {name} must be rank 4 "
                f"(B, kv_heads, S, head_dim), got {tuple(tensor.shape)}."
            )
        if tensor.shape[0] != 1:
            raise ValueError(
                f"{layer_type} shared {name} must have batch size 1, "
                f"got {tensor.shape[0]}."
            )
        if tensor.shape[1] != num_kv_heads:
            raise ValueError(
                f"{layer_type} shared {name} has {tensor.shape[1]} KV heads, "
                f"expected {num_kv_heads}."
            )
        if tensor.shape[3] != head_dim:
            raise ValueError(
                f"{layer_type} shared {name} head dim is {tensor.shape[3]}, "
                f"expected {head_dim}."
            )
    if key.shape[2] != value.shape[2]:
        raise ValueError(
            f"{layer_type} shared key/value lengths differ: "
            f"key={key.shape[2]}, value={value.shape[2]}."
        )
    return int(key.shape[2])


def _pad_kv_to_capacity(tensor: torch.Tensor, capacity: int) -> torch.Tensor:
    """Right-pad a KV tensor with zeros along the sequence axis."""
    valid = int(tensor.shape[2])
    if valid == capacity:
        return tensor
    padded = tensor.new_zeros(
        (tensor.shape[0], tensor.shape[1], capacity, tensor.shape[3])
    )
    padded[:, :, :valid, :] = tensor
    return padded


def canonicalize_gemma4_assistant_static_inputs(
    *,
    inputs_embeds: torch.Tensor,
    position_ids: Any,
    shared_kv_states: Mapping[str, LayerKV],
    shape: Gemma4AssistantStaticShapeConfig,
    model_or_config: Any,
    rotary_emb: Any,
    attention_mask: Optional[torch.Tensor] = None,
    mask_fill_value: float = -120.0,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> Gemma4AssistantStaticInputs:
    """Canonicalize dynamic assistant inputs to the fixed NPU core ABI.

    Args:
        inputs_embeds: Concatenated target embedding/hidden input with shape
            ``(1, 1, 2 * backbone_hidden_size)``.
        position_ids: Constant MTP position (scalar-like or ``(1, 1)``).
        shared_kv_states: Target shared KV mapping with ``full_attention``
            and ``sliding_attention`` entries.
        shape: Fixed capacity contract of the core graph.
        model_or_config: Assistant model or config used for validation.
        rotary_emb: Assistant rotary-embedding module used to build the
            per-layer-type cos/sin tables on the host.
        attention_mask: Optional HF 2D padding mask over the target sequence.
        mask_fill_value: Additive value for masked positions (follow
            ``PTQConfig.attention_mask_fill_value``).
        dtype: Floating dtype of the canonicalized tensors.
        device: Device of the canonicalized tensors.

    Raises:
        ValueError: On batch/query violations, over-capacity full KV, or a
            sliding crop that would drop positions still visible through the
            sliding window.
    """
    # Normalize device early to ensure all tensor operations use consistent device.
    device = torch.device(device)

    text_config = extract_assistant_text_config(model_or_config)
    shape.validate(text_config)

    if inputs_embeds.dim() != 3:
        raise ValueError(
            "inputs_embeds must be rank 3 (B, Q, 2 * backbone_hidden), "
            f"got {tuple(inputs_embeds.shape)}."
        )
    batch_size, q_len = int(inputs_embeds.shape[0]), int(inputs_embeds.shape[1])
    if batch_size != shape.batch_size:
        raise ValueError(
            f"inputs_embeds batch size {batch_size} does not match the static "
            f"contract batch_size={shape.batch_size}."
        )
    if q_len != shape.query_length:
        raise ValueError(
            f"inputs_embeds query length {q_len} does not match the static "
            f"contract query_length={shape.query_length}."
        )

    for layer_type in ("full_attention", "sliding_attention"):
        if shared_kv_states.get(layer_type) is None:
            raise ValueError(f"shared_kv_states is missing the {layer_type!r} entry.")

    num_kv_heads = assistant_shared_kv_num_heads(text_config)
    full_key, full_value = shared_kv_states["full_attention"]
    sliding_key, sliding_value = shared_kv_states["sliding_attention"]

    full_valid = _validate_kv_pair(
        "full_attention",
        full_key,
        full_value,
        num_kv_heads=num_kv_heads,
        head_dim=assistant_layer_type_head_dim(text_config, "full_attention"),
    )
    sliding_valid = _validate_kv_pair(
        "sliding_attention",
        sliding_key,
        sliding_value,
        num_kv_heads=num_kv_heads,
        head_dim=assistant_layer_type_head_dim(text_config, "sliding_attention"),
    )

    if full_valid > shape.full_kv_length:
        raise ValueError(
            "Full shared KV exceeds the static capacity and full attention "
            "must see every valid position: "
            f"valid={full_valid}, full_kv_length={shape.full_kv_length}."
        )
    if sliding_valid > full_valid:
        raise ValueError(
            "Sliding shared KV cannot be longer than the full shared KV: "
            f"sliding={sliding_valid}, full={full_valid}."
        )

    # Base padding mask over the valid full-KV span (HF cuts the 2D mask to
    # the full shared-KV length).
    if attention_mask is None:
        base_valid = torch.ones(batch_size, full_valid, dtype=torch.bool)
    else:
        if attention_mask.dim() != 2:
            raise ValueError(
                "attention_mask must be a 2D padding mask, "
                f"got shape={tuple(attention_mask.shape)}."
            )
        if attention_mask.shape[1] < full_valid:
            raise ValueError(
                "attention_mask is shorter than the full shared KV span: "
                f"mask_len={attention_mask.shape[1]}, kv_len={full_valid}."
            )
        base_valid = attention_mask[:, :full_valid].to(torch.bool).cpu()

    # ``sliding_attention`` sees the last ``window + 1`` valid positions of
    # the sliding span, aligned to the last ``sliding_valid`` entries of the
    # full padding mask (see module docstring).
    window = int(getattr(text_config, "sliding_window", 0) or 0)
    sliding_base_valid = base_valid[:, full_valid - sliding_valid :]
    first_visible = max(0, sliding_valid - (window + 1))

    if sliding_valid > shape.sliding_kv_length:
        raise ValueError(
            "Sliding shared KV exceeds the configured static capacity: "
            f"valid={sliding_valid}, sliding_kv_length={shape.sliding_kv_length}. "
            "Inputs that exceed capacity must not be silently cropped; "
            "increase sliding_kv_length or reduce the input sequence length."
        )

    fill = float(mask_fill_value)

    full_mask = torch.full(
        (batch_size, 1, q_len, shape.full_kv_length), fill, dtype=dtype
    )
    full_mask[:, 0, :, :full_valid] = torch.where(
        base_valid[:, None, :],
        torch.zeros((), dtype=dtype),
        torch.full((), fill, dtype=dtype),
    )

    sliding_mask = torch.full(
        (batch_size, 1, q_len, shape.sliding_kv_length), fill, dtype=dtype
    )
    sliding_keep = sliding_base_valid.clone()
    if first_visible > 0:
        sliding_keep[:, :first_visible] = False
    sliding_mask[:, 0, :, :sliding_valid] = torch.where(
        sliding_keep[:, None, :],
        torch.zeros((), dtype=dtype),
        torch.full((), fill, dtype=dtype),
    )

    position_ids = _normalize_position_ids(
        position_ids, batch_size=batch_size, query_length=q_len
    )
    rope_ref = inputs_embeds.to(dtype=dtype)
    with torch.no_grad():
        full_cos, full_sin = rotary_emb(rope_ref, position_ids, "full_attention")
        sliding_cos, sliding_sin = rotary_emb(
            rope_ref, position_ids, "sliding_attention"
        )

    def _finalize(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().to(device=device, dtype=dtype).contiguous()

    return Gemma4AssistantStaticInputs(
        assistant_input=_finalize(inputs_embeds),
        full_key=_finalize(_pad_kv_to_capacity(full_key, shape.full_kv_length)),
        full_value=_finalize(_pad_kv_to_capacity(full_value, shape.full_kv_length)),
        sliding_key=_finalize(
            _pad_kv_to_capacity(sliding_key, shape.sliding_kv_length)
        ),
        sliding_value=_finalize(
            _pad_kv_to_capacity(sliding_value, shape.sliding_kv_length)
        ),
        full_attention_mask=_finalize(full_mask),
        sliding_attention_mask=_finalize(sliding_mask),
        full_cos=_finalize(full_cos),
        full_sin=_finalize(full_sin),
        sliding_cos=_finalize(sliding_cos),
        sliding_sin=_finalize(sliding_sin),
    )
