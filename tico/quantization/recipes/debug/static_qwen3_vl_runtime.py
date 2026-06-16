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

import argparse
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

from tico.quantization import prepare
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.evaluation.metric import compute_peir
from tico.quantization.wrapq.wrappers.qwen_vl.export_adapters import (
    Qwen3VLVisionPrefillExportAdapter,
    Qwen3VLVisualEmbeddingFusionAdapter,
)
from tico.quantization.wrapq.wrappers.qwen_vl.quant_text_decoder_layer import (
    QuantQwen3VLTextDecoderLayer,
)


DEFAULT_IMAGE_MAX_PIXELS = 1280 * 28 * 28
DEFAULT_IMAGE_MIN_PIXELS = None


@dataclass
class LayerCache:
    """Store one decoder layer's static KV cache tensors."""

    past_k: torch.Tensor
    past_v: torch.Tensor


@dataclass
class StaticQwen3VLRuntimeConfig:
    """Configuration for the static Qwen3-VL runtime smoke test."""

    model: str = "Qwen/Qwen3-VL-2B-Instruct"
    max_seq: int = 2048
    padding_side: str = "right"
    device: str = "cpu"
    prompt: str = "Describe this image."
    image: Optional[str] = None
    image_max_pixels: Optional[int] = DEFAULT_IMAGE_MAX_PIXELS
    image_min_pixels: Optional[int] = DEFAULT_IMAGE_MIN_PIXELS
    use_chat_template: bool = True
    visual_start_idx: Optional[int] = None
    enable_deepstack: bool = True
    verify_steps: int = 6
    gen_steps: int = 16
    trust_remote_code: bool = True


def _clone_quant_layer(layer: nn.Module) -> nn.Module:
    """Build a quantized decoder layer wrapper for runtime simulation."""
    return prepare(layer, PTQConfig())


def _make_qwen_model(
    model_name: str, device: str, trust_remote_code: bool
) -> nn.Module:
    """Load a Qwen3-VL model with a transformers-version-compatible dtype argument."""
    kwargs = {"trust_remote_code": trust_remote_code}
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            dtype=torch.float32,
            **kwargs,
        )
    except TypeError:
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            **kwargs,
        )
    return model.eval().to(device)


def _set_text_max_position_embeddings(model: nn.Module, max_seq: int) -> None:
    """Set Qwen3-VL text-position capacity before quant wrappers are created."""
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        text_config.max_position_embeddings = max_seq

    qwen_model = getattr(model, "model", None)
    language_model = getattr(qwen_model, "language_model", None)
    if language_model is None:
        raise RuntimeError("Expected model.model.language_model for Qwen3-VL.")

    language_model.config.max_position_embeddings = max_seq
    for layer in language_model.layers:
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "config"):
            layer.self_attn.config.max_position_embeddings = max_seq


def _normalize_valid_token_mask(
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor],
    *,
    pad_token_id: Optional[int],
    device: torch.device,
) -> torch.Tensor:
    """Build a boolean mask where True marks real prompt tokens."""
    if attention_mask is None:
        if pad_token_id is None:
            valid = torch.ones(input_ids.shape, device=device, dtype=torch.bool)
        else:
            valid = input_ids.to(device).ne(int(pad_token_id))
    else:
        if tuple(attention_mask.shape) != tuple(input_ids.shape):
            raise ValueError(
                "attention_mask must have the same shape as input_ids. "
                f"Got attention_mask={tuple(attention_mask.shape)}, "
                f"input_ids={tuple(input_ids.shape)}."
            )
        valid = attention_mask.to(device=device).bool()

    if torch.any(valid.sum(dim=1) == 0):
        raise ValueError("Each batch row must contain at least one real token.")

    return valid


def _validate_padding_layout(valid_token_mask: torch.Tensor, padding_side: str) -> None:
    """Validate that each batch row uses contiguous left or right padding."""
    batch_size, seq_len = valid_token_mask.shape
    valid_lengths = valid_token_mask.sum(dim=1)
    positions = torch.arange(seq_len, device=valid_token_mask.device).unsqueeze(0)

    if padding_side == "right":
        expected = positions < valid_lengths.unsqueeze(1)
    elif padding_side == "left":
        expected = positions >= (seq_len - valid_lengths).unsqueeze(1)
    else:
        raise ValueError(
            f"padding_side must be 'left' or 'right'. got {padding_side!r}"
        )

    if not torch.equal(valid_token_mask, expected):
        raise ValueError(
            "Input padding layout does not match padding_side. "
            f"Expected contiguous {padding_side} padding for shape "
            f"(B={batch_size}, S={seq_len})."
        )


def _build_text_position_ids_from_valid_token_mask(
    valid_token_mask: torch.Tensor,
) -> torch.LongTensor:
    """Build compact text-only position IDs for padded prefill inputs."""
    position_ids = valid_token_mask.to(torch.long).cumsum(dim=1) - 1
    position_ids = torch.clamp(position_ids, min=0)
    return position_ids.masked_fill(~valid_token_mask, 0)


def _last_valid_token_indices(valid_token_mask: torch.Tensor) -> torch.LongTensor:
    """Return the input index of the last real token for each batch row."""
    batch_size, seq_len = valid_token_mask.shape
    positions = torch.arange(seq_len, device=valid_token_mask.device)
    positions = positions.unsqueeze(0).expand(batch_size, -1)
    return positions.masked_fill(~valid_token_mask, 0).max(dim=1).values


def _gather_last_token_logits(
    logits: torch.Tensor,
    valid_token_mask: torch.Tensor,
) -> torch.Tensor:
    """Gather logits at the last real token index of each padded prefill row."""
    last_indices = _last_valid_token_indices(valid_token_mask).to(logits.device)
    batch_indices = torch.arange(logits.size(0), device=logits.device)
    return logits[batch_indices, last_indices, :]


def _build_prefill_attention_mask(
    valid_token_mask: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    mask_value: float = -120.0,
) -> torch.Tensor:
    """Build a static additive causal + padding mask for prefill."""
    batch_size, seq_len = valid_token_mask.shape
    causal = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
    causal = torch.tril(causal).unsqueeze(0).expand(batch_size, -1, -1)
    key_valid = valid_token_mask.to(device).unsqueeze(1).expand(-1, seq_len, -1)
    query_valid = valid_token_mask.to(device).unsqueeze(2).expand(-1, -1, seq_len)
    valid = causal & key_valid & query_valid
    mask = torch.zeros(batch_size, 1, seq_len, seq_len, device=device, dtype=dtype)
    return mask.masked_fill(~valid.unsqueeze(1), mask_value)


def _build_decode_attention_mask(
    batch_size: int,
    past_len: int,
    max_seq: int,
    device: torch.device,
    dtype: torch.dtype,
    mask_value: float = -120.0,
) -> torch.Tensor:
    """
    Build a static additive decode mask for a `(max_seq - 1) + 1` KV layout.

    The CPU cache stores valid past tokens at the beginning of the fixed cache.
    The decode adapter appends the current token after the fixed past tensor, so
    the current key is located at index `max_seq - 1` inside the attention matmul.
    """
    if past_len < 0 or past_len >= max_seq:
        raise ValueError(f"past_len must be within [0, {max_seq}), got {past_len}.")

    mask = torch.full(
        (batch_size, 1, 1, max_seq),
        mask_value,
        device=device,
        dtype=dtype,
    )
    if past_len > 0:
        mask[:, :, :, :past_len] = 0.0
    mask[:, :, :, max_seq - 1] = 0.0
    return mask


def _pad_token_batch(
    batch: dict[str, Any],
    *,
    max_seq: int,
    pad_token_id: int,
    padding_side: str,
) -> dict[str, Any]:
    """Pad or validate processor/tokenizer outputs to the static sequence length."""
    input_ids = batch["input_ids"]
    if input_ids.dim() != 2:
        raise ValueError(f"Expected input_ids rank 2, got {tuple(input_ids.shape)}.")

    batch_size, seq_len = input_ids.shape
    if seq_len > max_seq:
        raise ValueError(
            f"Input sequence length {seq_len} exceeds static max_seq {max_seq}."
        )
    if seq_len == max_seq:
        return batch

    pad_len = max_seq - seq_len
    pad_ids = torch.full(
        (batch_size, pad_len),
        int(pad_token_id),
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    attention_mask = batch.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    pad_mask = torch.zeros(
        (batch_size, pad_len),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )

    out = dict(batch)
    if padding_side == "right":
        out["input_ids"] = torch.cat([input_ids, pad_ids], dim=1)
        out["attention_mask"] = torch.cat([attention_mask, pad_mask], dim=1)
    elif padding_side == "left":
        out["input_ids"] = torch.cat([pad_ids, input_ids], dim=1)
        out["attention_mask"] = torch.cat([pad_mask, attention_mask], dim=1)
    else:
        raise ValueError(f"Unsupported padding_side: {padding_side!r}")

    for key in ("position_ids", "mm_token_type_ids"):
        if key in out and isinstance(out[key], torch.Tensor) and out[key].dim() == 2:
            value = out[key]
            pad_value = torch.zeros(
                (batch_size, pad_len), dtype=value.dtype, device=value.device
            )
            if padding_side == "right":
                out[key] = torch.cat([value, pad_value], dim=1)
            else:
                out[key] = torch.cat([pad_value, value], dim=1)

    return out


def _move_batch_to_device(
    batch: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    """Move tensor values in a processor/tokenizer batch to the target device."""
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def _processor_accepts_kwarg(callable_obj, name: str) -> bool:
    """Return whether a callable appears to accept a keyword argument."""
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return True
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return name in signature.parameters


def _set_processor_image_budget(
    processor,
    *,
    image_min_pixels: Optional[int],
    image_max_pixels: Optional[int],
) -> None:
    """Best-effort update of processor image pixel-budget attributes."""
    targets = [processor, getattr(processor, "image_processor", None)]
    for target in targets:
        if target is None:
            continue
        if image_min_pixels is not None and hasattr(target, "min_pixels"):
            setattr(target, "min_pixels", int(image_min_pixels))
        if image_max_pixels is not None and hasattr(target, "max_pixels"):
            setattr(target, "max_pixels", int(image_max_pixels))


def _resize_image_to_pixel_budget(image, image_max_pixels: Optional[int]):
    """Downscale a PIL image so its area does not exceed a pixel budget."""
    if image_max_pixels is None:
        return image
    image_max_pixels = int(image_max_pixels)
    if image_max_pixels <= 0:
        raise ValueError(f"image_max_pixels must be positive, got {image_max_pixels}.")
    width, height = image.size
    if width * height <= image_max_pixels:
        return image

    scale = (float(image_max_pixels) / float(width * height)) ** 0.5
    target_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    try:
        from PIL import Image

        resample = Image.Resampling.LANCZOS
    except Exception:  # pragma: no cover - Pillow compatibility fallback.
        resample = 1

    resized = image.copy()
    resized.thumbnail(target_size, resample=resample)
    return resized


def _processor_call_kwargs_for_image_budget(
    processor,
    *,
    image_min_pixels: Optional[int],
    image_max_pixels: Optional[int],
) -> dict[str, int]:
    """Return processor call kwargs accepted by the installed processor version."""
    kwargs: dict[str, int] = {}
    if image_min_pixels is not None and _processor_accepts_kwarg(
        processor.__call__, "min_pixels"
    ):
        kwargs["min_pixels"] = int(image_min_pixels)
    if image_max_pixels is not None and _processor_accepts_kwarg(
        processor.__call__, "max_pixels"
    ):
        kwargs["max_pixels"] = int(image_max_pixels)
    return kwargs


def _image_budget_candidates(
    *,
    image_min_pixels: Optional[int],
    image_max_pixels: Optional[int],
) -> list[Optional[int]]:
    """Build descending image pixel-budget candidates for static sequence fitting."""
    if image_max_pixels is None:
        return [None]

    floor = int(image_min_pixels) if image_min_pixels is not None else 4 * 28 * 28
    floor = max(1, floor)
    current = int(image_max_pixels)
    if current < floor:
        return [current]

    candidates: list[Optional[int]] = []
    while current >= floor:
        candidates.append(current)
        next_current = current // 2
        if next_current == current:
            break
        current = next_current
    if candidates[-1] != floor:
        candidates.append(floor)
    return candidates


def _format_optional_tensor(value: Any) -> str:
    """Format a tensor shape/value summary for diagnostics."""
    if isinstance(value, torch.Tensor):
        if value.numel() <= 16:
            return f"shape={tuple(value.shape)}, value={value.detach().cpu().tolist()}"
        return f"shape={tuple(value.shape)}"
    return str(value)


def _validate_unpadded_static_length(
    batch: dict[str, Any],
    *,
    max_seq: int,
    image_budget: Optional[int],
) -> None:
    """Raise a detailed error if the unpadded processor output exceeds max_seq."""
    seq_len = int(batch["input_ids"].shape[1])
    if seq_len <= max_seq:
        return

    grid = _format_optional_tensor(batch.get("image_grid_thw"))
    pixel_values = _format_optional_tensor(batch.get("pixel_values"))
    raise ValueError(
        f"Input sequence length {seq_len} exceeds static max_seq {max_seq}. "
        f"image_grid_thw={grid}; pixel_values={pixel_values}; "
        f"image_budget={image_budget}. Lower --image-max-pixels or increase --max-seq."
    )


def _compact_token_batch(batch: dict[str, Any], valid: torch.Tensor) -> dict[str, Any]:
    """Remove static padding from token-like tensors for reference execution."""
    if valid.size(0) != 1:
        raise ValueError("Reference compaction currently supports batch_size=1 only.")
    compact: dict[str, Any] = {}
    token_keys = {"input_ids", "attention_mask", "mm_token_type_ids"}
    for key, value in batch.items():
        if key in token_keys and isinstance(value, torch.Tensor) and value.dim() == 2:
            compact[key] = value[:, valid[0]]
        else:
            compact[key] = value
    if "attention_mask" in compact:
        compact["attention_mask"] = torch.ones_like(compact["input_ids"])
    return compact


def _append_token_type_zero(
    mm_token_type_ids: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Append a text-token type value for one generated token."""
    if mm_token_type_ids is None:
        return None
    zeros = torch.zeros(
        (mm_token_type_ids.size(0), 1),
        dtype=mm_token_type_ids.dtype,
        device=mm_token_type_ids.device,
    )
    return torch.cat([mm_token_type_ids, zeros], dim=1)


def _image_grid_tuple(image_grid_thw: torch.Tensor) -> tuple[int, int, int]:
    """Convert a single-image grid tensor to a Python tuple."""
    if image_grid_thw is None:
        raise ValueError("image_grid_thw is required for image prefill.")
    if (
        image_grid_thw.dim() != 2
        or image_grid_thw.size(0) != 1
        or image_grid_thw.size(1) != 3
    ):
        raise ValueError(
            "Only one fixed image is supported. Expected image_grid_thw shape `(1, 3)`, "
            f"got {tuple(image_grid_thw.shape)}."
        )
    values = [int(x) for x in image_grid_thw[0].tolist()]
    return (values[0], values[1], values[2])


def _processor_input_keys(batch: dict[str, Any]) -> dict[str, Any]:
    """Select model-forward keys from a tokenizer or processor batch."""
    keys = {
        "input_ids",
        "attention_mask",
        "pixel_values",
        "image_grid_thw",
        "mm_token_type_ids",
    }
    return {key: value for key, value in batch.items() if key in keys}


class StaticQwen3VLTextLayerRuntime:
    """
    Static-shape Qwen3-VL runtime over separately wrapped decoder layers.

    CPU responsibilities:
      - tokenizer/processor padding,
      - token embedding,
      - image embedding fusion,
      - mRoPE lookup,
      - attention-mask construction,
      - KV-cache allocation and updates,
      - final norm and LM head.

    NPU-simulated responsibilities:
      - fixed-grid vision prefill adapter,
      - dense decoder-layer prefill and decode adapters.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        max_seq: int,
        device: str = "cpu",
        padding_side: str = "right",
        layers: Optional[Sequence[nn.Module]] = None,
        visual_start_idx: Optional[int] = None,
        enable_deepstack: bool = True,
    ):
        self.model = model.eval().to(device)
        self.tokenizer = tokenizer
        self.max_seq = max_seq
        self.device = torch.device(device)
        self.padding_side = padding_side
        self.visual_start_idx = visual_start_idx
        self.enable_deepstack = enable_deepstack

        self.qwen_model = self.model.model
        self.text_model = self.qwen_model.language_model
        self.embed_tokens = self.text_model.embed_tokens
        self.final_norm = self.text_model.norm
        self.lm_head = self.model.lm_head
        self.rotate_lm_head = getattr(self.model, "rotate_lm_head", None)
        self.layers_ref = self.text_model.layers

        if layers is None:
            self.layers = nn.ModuleList(
                [_clone_quant_layer(layer) for layer in self.layers_ref]
            ).to(self.device)
        else:
            self.layers = nn.ModuleList(layers).to(self.device)

        for layer in self.layers:
            assert hasattr(layer, "wrapped")
            assert isinstance(layer.wrapped, QuantQwen3VLTextDecoderLayer)

        self.prefill_layers = nn.ModuleList(
            [
                layer.wrapped.as_export_module("prefill", return_kv=True)
                for layer in self.layers
            ]
        ).to(self.device)
        self.decode_layers = nn.ModuleList(
            [
                layer.wrapped.as_export_module("decode", return_kv=True)
                for layer in self.layers
            ]
        ).to(self.device)

        self.config = self.text_model.config
        self.hidden_size = self.config.hidden_size
        self.num_hidden_layers = self.config.num_hidden_layers
        self.num_kv_heads = self.config.num_key_value_heads
        self.head_dim = getattr(self.config, "head_dim", None) or (
            self.hidden_size // self.config.num_attention_heads
        )

        self.layer_caches: list[LayerCache] = []
        self.past_len = 0
        self.rope_delta = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        self.vision_prefill_adapter: Optional[Qwen3VLVisionPrefillExportAdapter] = None
        self.vision_grid_thw: Optional[tuple[int, int, int]] = None
        self.visual_fusion_adapter: Optional[Qwen3VLVisualEmbeddingFusionAdapter] = None

    def reset_cache(self) -> None:
        """Clear CPU-managed KV caches."""
        self.layer_caches = []
        self.past_len = 0
        self.rope_delta = torch.zeros(1, 1, dtype=torch.long, device=self.device)

    def _allocate_empty_cache(
        self,
        batch_size: int,
        dtype: torch.dtype,
    ) -> list[LayerCache]:
        """Allocate static `(max_seq - 1)` past-cache tensors for every layer."""
        caches = []
        for _ in range(self.num_hidden_layers):
            past_k = torch.zeros(
                batch_size,
                self.num_kv_heads,
                self.max_seq - 1,
                self.head_dim,
                device=self.device,
                dtype=dtype,
            )
            caches.append(LayerCache(past_k=past_k, past_v=torch.zeros_like(past_k)))
        return caches

    def _position_embeddings(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Qwen3-VL mRoPE embeddings on CPU/runtime side."""
        cos, sin = self.text_model.rotary_emb(hidden_states, position_ids)
        return cos.to(dtype=hidden_states.dtype), sin.to(dtype=hidden_states.dtype)

    def _text_position_ids(self, valid: torch.Tensor) -> torch.LongTensor:
        """Build text-only 3D-compatible position IDs."""
        ids_2d = _build_text_position_ids_from_valid_token_mask(valid)
        return ids_2d.unsqueeze(0).expand(3, -1, -1).to(self.device)

    def _compute_rope_delta(
        self,
        position_ids: torch.LongTensor,
        valid: torch.Tensor,
    ) -> torch.LongTensor:
        """Compute the decode rope delta from compact cache length and mRoPE max index."""
        deltas = []
        for b in range(valid.size(0)):
            row_valid = valid[b]
            max_pos = position_ids[:, b, row_valid].max()
            valid_len = row_valid.sum()
            deltas.append(max_pos + 1 - valid_len)
        return torch.stack(deltas).view(-1, 1).to(device=self.device, dtype=torch.long)

    def _compute_multimodal_position_ids(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        image_grid_thw: torch.Tensor,
        mm_token_type_ids: Optional[torch.Tensor],
    ) -> torch.LongTensor:
        """Compute Qwen3-VL 3D position IDs for a padded single-image prompt."""
        get_rope_index = getattr(self.qwen_model, "get_rope_index", None)
        if callable(get_rope_index):
            kwargs: dict[str, Any] = {
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": None,
                "attention_mask": attention_mask,
            }
            if (
                mm_token_type_ids is not None
                and "mm_token_type_ids" in inspect.signature(get_rope_index).parameters
            ):
                kwargs["mm_token_type_ids"] = mm_token_type_ids
            position_ids, _ = get_rope_index(input_ids, **kwargs)
            return position_ids.to(device=self.device, dtype=torch.long)

        return self._compute_single_image_position_ids_fallback(
            input_ids,
            attention_mask,
            image_grid_thw,
        )

    def _compute_single_image_position_ids_fallback(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.LongTensor:
        """Fallback single-image mRoPE position ID builder."""
        if input_ids.size(0) != 1:
            raise ValueError("Fallback mRoPE builder supports batch_size=1 only.")
        grid_t, grid_h, grid_w = _image_grid_tuple(image_grid_thw)
        spatial_merge_size = int(self.qwen_model.visual.spatial_merge_size)
        llm_grid_t = grid_t
        llm_grid_h = grid_h // spatial_merge_size
        llm_grid_w = grid_w // spatial_merge_size
        image_token_id = int(self.model.config.image_token_id)
        valid_ids = input_ids[0][attention_mask[0].bool()]
        image_positions = torch.nonzero(valid_ids == image_token_id, as_tuple=True)[0]
        if image_positions.numel() == 0:
            return self._text_position_ids(attention_mask.bool())
        visual_start = int(image_positions[0].item())
        num_visual_tokens = llm_grid_t * llm_grid_h * llm_grid_w
        visual_end = visual_start + num_visual_tokens

        segments: list[torch.Tensor] = []
        if visual_start > 0:
            segments.append(
                torch.arange(visual_start, device=self.device).view(1, -1).expand(3, -1)
            )

        t_index = (
            torch.arange(llm_grid_t, device=self.device)
            .view(-1, 1)
            .expand(-1, llm_grid_h * llm_grid_w)
            .flatten()
        )
        h_index = (
            torch.arange(llm_grid_h, device=self.device)
            .view(1, -1, 1)
            .expand(llm_grid_t, -1, llm_grid_w)
            .flatten()
        )
        w_index = (
            torch.arange(llm_grid_w, device=self.device)
            .view(1, 1, -1)
            .expand(llm_grid_t, llm_grid_h, -1)
            .flatten()
        )
        st_idx = segments[-1].max() + 1 if segments else 0
        segments.append(torch.stack([t_index, h_index, w_index]) + st_idx)

        valid_len = int(attention_mask[0].sum().item())
        if visual_end < valid_len:
            st_idx = segments[-1].max() + 1
            text_len = valid_len - visual_end
            segments.append(
                torch.arange(text_len, device=self.device).view(1, -1).expand(3, -1)
                + st_idx
            )

        valid_positions = torch.cat(segments, dim=1)
        position_ids = torch.zeros(
            3,
            1,
            input_ids.size(1),
            dtype=torch.long,
            device=self.device,
        )
        position_ids[:, 0, attention_mask[0].bool()] = valid_positions
        return position_ids

    def _create_vision_prefill_adapter(
        self,
        image_grid_thw: torch.Tensor,
        visual_start_idx: int,
    ) -> None:
        """Create or refresh the fixed-grid vision prefill adapter."""
        grid_tuple = _image_grid_tuple(image_grid_thw)
        if (
            self.vision_prefill_adapter is not None
            and self.vision_grid_thw == grid_tuple
        ):
            return

        spatial_merge_size = int(self.qwen_model.visual.spatial_merge_size)
        qcfg = PTQConfig(
            model_args={
                "vision": {
                    "grid_thw": grid_tuple,
                    "visual_start_idx": visual_start_idx,
                    "spatial_merge_size": spatial_merge_size,
                }
            }
        )
        wrapped_visual = prepare(self.qwen_model.visual, qcfg).to(self.device)
        self.vision_prefill_adapter = Qwen3VLVisionPrefillExportAdapter(wrapped_visual)
        self.vision_grid_thw = grid_tuple

    def _find_visual_span(
        self,
        input_ids: torch.LongTensor,
        valid: torch.Tensor,
        image_embeds: torch.Tensor,
    ) -> tuple[int, int]:
        """Find and validate the contiguous image placeholder span."""
        if input_ids.size(0) != 1:
            raise ValueError("Milestone 2 supports batch_size=1 for image prefill.")
        image_token_id = int(self.model.config.image_token_id)
        image_mask = input_ids[0].eq(image_token_id) & valid[0]
        image_positions = torch.nonzero(image_mask, as_tuple=True)[0]
        if image_positions.numel() == 0:
            raise ValueError("No image placeholder tokens were found in input_ids.")

        start = int(image_positions[0].item())
        visual_len = int(image_embeds.size(0))
        expected = torch.arange(start, start + visual_len, device=input_ids.device)
        if image_positions.numel() != visual_len or not torch.equal(
            image_positions, expected
        ):
            raise ValueError(
                "Image placeholder tokens must be one contiguous span matching image_embeds. "
                f"placeholder_count={image_positions.numel()}, image_embeds={visual_len}, "
                f"start={start}."
            )

        if self.visual_start_idx is not None and self.visual_start_idx != start:
            raise ValueError(
                "Configured visual_start_idx does not match processor output. "
                f"configured={self.visual_start_idx}, actual={start}."
            )
        self.visual_start_idx = start
        return start, visual_len

    def _make_deepstack_deltas(
        self,
        deepstack_features: Sequence[torch.Tensor],
        *,
        batch_size: int,
        seq_len: int,
        hidden_size: int,
        visual_start_idx: int,
        dtype: torch.dtype,
    ) -> Optional[list[torch.Tensor]]:
        """Convert sparse DeepStack visual features to dense static deltas."""
        if not self.enable_deepstack or not deepstack_features:
            return None

        deltas = []
        for feature in deepstack_features:
            if feature.dim() != 2:
                raise ValueError(
                    "DeepStack feature must have shape `(V, H)`, "
                    f"got {tuple(feature.shape)}."
                )
            visual_len = feature.size(0)
            delta = torch.zeros(
                batch_size,
                seq_len,
                hidden_size,
                device=self.device,
                dtype=dtype,
            )
            delta[:, visual_start_idx : visual_start_idx + visual_len, :] = feature.to(
                device=self.device,
                dtype=dtype,
            ).unsqueeze(0)
            deltas.append(delta)
        return deltas

    def _run_vision_prefill(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        visual_start_idx: int,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Run the fixed-grid vision adapter and return image/DeepStack features."""
        self._create_vision_prefill_adapter(image_grid_thw, visual_start_idx)
        assert self.vision_prefill_adapter is not None
        image_embeds, deepstack_features = self.vision_prefill_adapter(
            pixel_values.to(self.device),
            image_grid_thw.to(self.device),
        )
        return image_embeds.to(self.device), tuple(
            feature.to(self.device) for feature in deepstack_features
        )

    def _prepare_inputs_embeds_and_positions(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.Tensor],
        mm_token_type_ids: Optional[torch.Tensor],
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.LongTensor, Optional[list[torch.Tensor]],]:
        """Prepare fused embeddings, position IDs, and optional DeepStack deltas."""
        input_ids = input_ids.to(self.device)
        inputs_embeds = self.embed_tokens(input_ids)
        deepstack_deltas = None

        if pixel_values is None:
            position_ids = self._text_position_ids(valid)
            self.rope_delta = self._compute_rope_delta(position_ids, valid)
            return inputs_embeds, position_ids, deepstack_deltas

        if image_grid_thw is None:
            raise ValueError(
                "image_grid_thw is required when pixel_values is provided."
            )

        # Bootstrap visual_start_idx from input IDs if the caller did not fix it.
        image_token_id = int(self.model.config.image_token_id)
        bootstrap_mask = input_ids[0].eq(image_token_id) & valid[0]
        if bootstrap_mask.sum().item() == 0:
            raise ValueError(
                "pixel_values were provided but no image tokens were found."
            )
        bootstrap_start = int(torch.nonzero(bootstrap_mask, as_tuple=True)[0][0].item())
        configured_start = (
            self.visual_start_idx
            if self.visual_start_idx is not None
            else bootstrap_start
        )

        image_embeds, deepstack_features = self._run_vision_prefill(
            pixel_values,
            image_grid_thw,
            configured_start,
        )
        visual_start_idx, visual_len = self._find_visual_span(
            input_ids,
            valid,
            image_embeds,
        )
        del visual_len

        if (
            self.visual_fusion_adapter is None
            or self.visual_fusion_adapter.visual_start_idx != visual_start_idx
        ):
            self.visual_fusion_adapter = Qwen3VLVisualEmbeddingFusionAdapter(
                visual_start_idx
            )
        inputs_embeds = self.visual_fusion_adapter(inputs_embeds, image_embeds)

        position_ids = self._compute_multimodal_position_ids(
            input_ids,
            attention_mask,
            image_grid_thw.to(self.device),
            mm_token_type_ids,
        )
        self.rope_delta = self._compute_rope_delta(position_ids, valid)

        deepstack_deltas = self._make_deepstack_deltas(
            deepstack_features,
            batch_size=input_ids.size(0),
            seq_len=input_ids.size(1),
            hidden_size=inputs_embeds.size(-1),
            visual_start_idx=visual_start_idx,
            dtype=inputs_embeds.dtype,
        )
        return inputs_embeds, position_ids, deepstack_deltas

    @torch.no_grad()
    def prefill(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run static prefill and return last-real-token logits."""
        assert (
            input_ids.dim() == 2
        ), f"Expected input_ids as (B, S), got {tuple(input_ids.shape)}"
        batch_size, seq_len = input_ids.shape
        assert seq_len == self.max_seq, (
            f"Static prefill expects padded length == max_seq. "
            f"Got seq_len={seq_len}, max_seq={self.max_seq}."
        )

        input_ids = input_ids.to(self.device)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        else:
            attention_mask = attention_mask.to(self.device)
        if mm_token_type_ids is not None:
            mm_token_type_ids = mm_token_type_ids.to(self.device)
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(self.device)
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.device)

        valid = _normalize_valid_token_mask(
            input_ids,
            attention_mask,
            pad_token_id=self.tokenizer.pad_token_id,
            device=self.device,
        )
        _validate_padding_layout(valid, self.padding_side)

        (
            hidden_states,
            position_ids,
            deepstack_deltas,
        ) = self._prepare_inputs_embeds_and_positions(
            input_ids,
            attention_mask,
            pixel_values,
            image_grid_thw,
            mm_token_type_ids,
            valid,
        )
        runtime_dtype = hidden_states.dtype

        self.layer_caches = self._allocate_empty_cache(batch_size, runtime_dtype)
        attn_mask = _build_prefill_attention_mask(valid, self.device, runtime_dtype)
        position_embeddings = self._position_embeddings(hidden_states, position_ids)

        for layer_idx, layer in enumerate(self.prefill_layers):
            out = layer(
                hidden_states=hidden_states,
                attention_mask=attn_mask,
                position_embeddings=position_embeddings,
            )
            if not isinstance(out, tuple) or len(out) != 3:
                raise RuntimeError(
                    "Expected prefill adapter output as (hidden_states, new_k, new_v)."
                )
            hidden_states, new_k, new_v = out

            valid_lengths = valid.sum(dim=1)
            for b in range(batch_size):
                length = int(valid_lengths[b].item())
                src_start = 0 if self.padding_side == "right" else self.max_seq - length
                src_end = src_start + length
                self.layer_caches[layer_idx].past_k[b, :, :length, :] = new_k[
                    b, :, src_start:src_end, :
                ]
                self.layer_caches[layer_idx].past_v[b, :, :length, :] = new_v[
                    b, :, src_start:src_end, :
                ]

            if deepstack_deltas is not None and layer_idx < len(deepstack_deltas):
                hidden_states = hidden_states + deepstack_deltas[layer_idx]

        if torch.unique(valid.sum(dim=1)).numel() != 1:
            raise ValueError(
                "Static decode currently requires equal valid prompt length for all batch rows."
            )
        self.past_len = int(valid.sum(dim=1)[0].item())

        hidden_states = self.final_norm(hidden_states)
        if self.rotate_lm_head is not None:
            hidden_states = self.rotate_lm_head(hidden_states)
        logits = self.lm_head(hidden_states)
        return _gather_last_token_logits(logits, valid)

    def _decode_position_ids(self, batch_size: int) -> torch.LongTensor:
        """Build mRoPE decode position IDs for one generated token."""
        delta = self.rope_delta.to(self.device)
        if delta.size(0) != batch_size:
            delta = delta.expand(batch_size, -1)
        pos = (
            torch.full(
                (batch_size, 1),
                int(self.past_len),
                dtype=torch.long,
                device=self.device,
            )
            + delta
        )
        return pos.unsqueeze(0).expand(3, -1, -1)

    @torch.no_grad()
    def decode_one(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Run one static decode step and return logits for the current token."""
        assert input_ids.dim() == 2 and input_ids.size(1) == 1
        assert len(self.layer_caches) == self.num_hidden_layers, "Call prefill() first."
        assert self.past_len < self.max_seq

        batch_size = input_ids.size(0)
        hidden_states = self.embed_tokens(input_ids.to(self.device))

        attention_mask = _build_decode_attention_mask(
            batch_size,
            self.past_len,
            self.max_seq,
            self.device,
            hidden_states.dtype,
        )
        position_ids = self._decode_position_ids(batch_size)
        position_embeddings = self._position_embeddings(hidden_states, position_ids)

        for layer_idx, layer in enumerate(self.decode_layers):
            cache = self.layer_caches[layer_idx]
            out = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                past_key_values=(cache.past_k, cache.past_v),
            )
            if not isinstance(out, tuple) or len(out) != 3:
                raise RuntimeError(
                    "Expected decode adapter output as (hidden_states, new_k, new_v)."
                )
            hidden_states, new_k, new_v = out
            cache.past_k[:, :, self.past_len : self.past_len + 1, :] = new_k
            cache.past_v[:, :, self.past_len : self.past_len + 1, :] = new_v

        self.past_len += 1
        hidden_states = self.final_norm(hidden_states)
        if self.rotate_lm_head is not None:
            hidden_states = self.rotate_lm_head(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits[:, -1, :]

    @torch.no_grad()
    def prefill_batch(self, batch: dict[str, Any]) -> torch.Tensor:
        """Run prefill from a padded tokenizer/processor batch."""
        return self.prefill(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            pixel_values=batch.get("pixel_values"),
            image_grid_thw=batch.get("image_grid_thw"),
            mm_token_type_ids=batch.get("mm_token_type_ids"),
        )

    @torch.no_grad()
    def generate_greedy_from_batch(
        self,
        batch: dict[str, Any],
        max_new_tokens: int,
        eos_token_id: Optional[int] = None,
    ) -> torch.LongTensor:
        """Generate tokens greedily with a preprocessed static batch."""
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id

        self.reset_cache()
        logits = self.prefill_batch(batch)
        valid = _normalize_valid_token_mask(
            input_ids,
            attention_mask,
            pad_token_id=self.tokenizer.pad_token_id,
            device=self.device,
        )
        prompt_len = int(valid.sum(dim=1)[0].item())
        generated = input_ids[valid].reshape(1, prompt_len).clone()

        for _ in range(max_new_tokens):
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break
            logits = self.decode_one(next_token)
        return generated

    @torch.no_grad()
    def verify_batch_against_reference(
        self,
        batch: dict[str, Any],
        steps: int = 8,
        verbose: bool = True,
    ) -> None:
        """Compare static runtime logits with the HF reference model."""
        batch = _move_batch_to_device(batch, self.device)
        valid = _normalize_valid_token_mask(
            batch["input_ids"],
            batch.get("attention_mask"),
            pad_token_id=self.tokenizer.pad_token_id,
            device=self.device,
        )
        compact_batch = _compact_token_batch(batch, valid)

        self.reset_cache()
        logits_rt = self.prefill_batch(batch)
        ref_out = self.model(**_processor_input_keys(compact_batch))
        logits_ref = ref_out.logits[:, -1, :]
        self._print_diff(
            "Step 0: prefill last-token logits",
            logits_rt,
            logits_ref,
            verbose,
        )

        generated_input_ids = compact_batch["input_ids"].clone()
        generated_mm_types = compact_batch.get("mm_token_type_ids")
        next_token = torch.argmax(logits_rt, dim=-1, keepdim=True)
        generated_input_ids = torch.cat([generated_input_ids, next_token], dim=1)
        generated_mm_types = _append_token_type_zero(generated_mm_types)

        static_image_payload = {
            key: compact_batch[key]
            for key in ("pixel_values", "image_grid_thw")
            if key in compact_batch
        }

        for step in range(1, steps + 1):
            logits_rt = self.decode_one(next_token)
            ref_batch = {
                "input_ids": generated_input_ids,
                "attention_mask": torch.ones_like(generated_input_ids),
                **static_image_payload,
            }
            if generated_mm_types is not None:
                ref_batch["mm_token_type_ids"] = generated_mm_types
            ref_out = self.model(**ref_batch)
            logits_ref = ref_out.logits[:, -1, :]
            self._print_diff(
                f"Step {step}: decode logits",
                logits_rt,
                logits_ref,
                verbose,
            )

            next_token = torch.argmax(logits_rt, dim=-1, keepdim=True)
            generated_input_ids = torch.cat([generated_input_ids, next_token], dim=1)
            generated_mm_types = _append_token_type_zero(generated_mm_types)
            if generated_input_ids.size(1) >= self.max_seq:
                if verbose:
                    print("Stopped because the static decode window is full.")
                break

    @staticmethod
    def _print_diff(
        title: str,
        actual: torch.Tensor,
        expected: torch.Tensor,
        verbose: bool,
    ) -> None:
        """Print absolute-difference and PEIR metrics for two logits tensors."""
        if not verbose:
            return
        diff = (actual - expected).abs()
        print("=" * 100)
        print(title)
        print(f"mean|diff| = {diff.mean().item():.8f}")
        print(f" max|diff| = {diff.max().item():.8f}")
        print(f"PEIR       = {compute_peir(actual, expected) * 100:.6f} %")


def _build_text_only_batch(
    tokenizer,
    prompt: str,
    *,
    max_seq: int,
    padding_side: str,
) -> dict[str, Any]:
    """Build a padded text-only batch for static runtime verification."""
    tokenizer.padding_side = padding_side
    batch = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
        padding=False,
        truncation=False,
    )
    return _pad_token_batch(
        dict(batch),
        max_seq=max_seq,
        pad_token_id=int(tokenizer.pad_token_id),
        padding_side=padding_side,
    )


def _build_image_batch(
    processor,
    prompt: str,
    image_path: str,
    *,
    max_seq: int,
    padding_side: str,
    use_chat_template: bool,
    image_min_pixels: Optional[int],
    image_max_pixels: Optional[int],
) -> dict[str, Any]:
    """Build a padded single-image Qwen3-VL processor batch within max_seq."""
    from PIL import Image

    original_image = Image.open(Path(image_path)).convert("RGB")

    text = prompt
    if use_chat_template and hasattr(processor, "apply_chat_template"):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("Qwen3-VL processor is expected to expose `.tokenizer`.")

    last_batch: Optional[dict[str, Any]] = None
    last_budget: Optional[int] = None
    for candidate_max_pixels in _image_budget_candidates(
        image_min_pixels=image_min_pixels,
        image_max_pixels=image_max_pixels,
    ):
        _set_processor_image_budget(
            processor,
            image_min_pixels=image_min_pixels,
            image_max_pixels=candidate_max_pixels,
        )
        image = _resize_image_to_pixel_budget(original_image, candidate_max_pixels)
        processor_kwargs = _processor_call_kwargs_for_image_budget(
            processor,
            image_min_pixels=image_min_pixels,
            image_max_pixels=candidate_max_pixels,
        )
        batch = dict(
            processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding=False,
                **processor_kwargs,
            )
        )
        last_batch = batch
        last_budget = candidate_max_pixels
        if int(batch["input_ids"].shape[1]) <= max_seq:
            break
    else:
        assert last_batch is not None
        _validate_unpadded_static_length(
            last_batch,
            max_seq=max_seq,
            image_budget=last_budget,
        )

    assert last_batch is not None
    _validate_unpadded_static_length(
        last_batch,
        max_seq=max_seq,
        image_budget=last_budget,
    )

    seq_len = int(last_batch["input_ids"].shape[1])
    grid = _format_optional_tensor(last_batch.get("image_grid_thw"))
    print(
        "Static image batch: "
        f"unpadded_seq_len={seq_len}, image_grid_thw={grid}, "
        f"image_max_pixels={last_budget}"
    )

    return _pad_token_batch(
        last_batch,
        max_seq=max_seq,
        pad_token_id=int(tokenizer.pad_token_id),
        padding_side=padding_side,
    )


def run_static_qwen3_vl_runtime(cfg: StaticQwen3VLRuntimeConfig) -> None:
    """Load a Qwen3-VL model and run static runtime smoke tests."""
    torch.set_grad_enabled(False)

    model = _make_qwen_model(cfg.model, cfg.device, cfg.trust_remote_code)
    _set_text_max_position_embeddings(model, cfg.max_seq)

    if cfg.image:
        processor = AutoProcessor.from_pretrained(
            cfg.model,
            trust_remote_code=cfg.trust_remote_code,
        )
        tokenizer = processor.tokenizer
    else:
        processor = None
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model,
            trust_remote_code=cfg.trust_remote_code,
        )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = cfg.padding_side

    runtime = StaticQwen3VLTextLayerRuntime(
        model=model,
        tokenizer=tokenizer,
        max_seq=cfg.max_seq,
        device=cfg.device,
        padding_side=cfg.padding_side,
        visual_start_idx=cfg.visual_start_idx,
        enable_deepstack=cfg.enable_deepstack,
    )

    if cfg.image:
        assert processor is not None
        batch = _build_image_batch(
            processor,
            cfg.prompt,
            cfg.image,
            max_seq=cfg.max_seq,
            padding_side=cfg.padding_side,
            use_chat_template=cfg.use_chat_template,
            image_min_pixels=cfg.image_min_pixels,
            image_max_pixels=cfg.image_max_pixels,
        )
    else:
        batch = _build_text_only_batch(
            tokenizer,
            cfg.prompt,
            max_seq=cfg.max_seq,
            padding_side=cfg.padding_side,
        )

    batch = _move_batch_to_device(batch, torch.device(cfg.device))
    runtime.verify_batch_against_reference(
        batch=batch,
        steps=cfg.verify_steps,
        verbose=True,
    )

    out_ids = runtime.generate_greedy_from_batch(
        batch=batch,
        max_new_tokens=cfg.gen_steps,
        eos_token_id=tokenizer.eos_token_id,
    )

    print("=" * 100)
    print("Generated text:")
    print(tokenizer.decode(out_ids[0], skip_special_tokens=True))


def _parse_args() -> StaticQwen3VLRuntimeConfig:
    """Parse command-line arguments for the static runtime smoke test."""
    parser = argparse.ArgumentParser(
        description="Run the static Qwen3-VL text/image-prefill runtime."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=StaticQwen3VLRuntimeConfig.model,
        help="HF model name or local model path.",
    )
    parser.add_argument(
        "--max-seq",
        type=int,
        default=StaticQwen3VLRuntimeConfig.max_seq,
        help="Static sequence length used by prefill and decode.",
    )
    parser.add_argument(
        "--padding-side",
        type=str,
        choices=("left", "right"),
        default=StaticQwen3VLRuntimeConfig.padding_side,
        help="Padding direction for static prefill inputs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=StaticQwen3VLRuntimeConfig.device,
        help="Execution device, such as cpu or cuda.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=StaticQwen3VLRuntimeConfig.prompt,
        help="Prompt used for verification and greedy generation.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Optional image path. When provided, single-image prefill is used.",
    )
    parser.add_argument(
        "--visual-start-idx",
        type=int,
        default=None,
        help="Optional fixed visual token start index. If omitted, the processor output is used.",
    )
    parser.add_argument(
        "--image-max-pixels",
        type=int,
        default=StaticQwen3VLRuntimeConfig.image_max_pixels,
        help=(
            "Maximum image pixel budget before processor tokenization. "
            "The default keeps typical single-image prompts under max_seq=2048."
        ),
    )
    parser.add_argument(
        "--image-min-pixels",
        type=int,
        default=StaticQwen3VLRuntimeConfig.image_min_pixels,
        help="Optional minimum image pixel budget passed to the processor.",
    )
    parser.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Do not wrap the image prompt with the processor chat template.",
    )
    parser.add_argument(
        "--no-deepstack",
        action="store_true",
        help="Disable dense DeepStack delta injection during image prefill.",
    )
    parser.add_argument(
        "--verify-steps",
        type=int,
        default=StaticQwen3VLRuntimeConfig.verify_steps,
        help="Number of decode steps for reference verification.",
    )
    parser.add_argument(
        "--gen-steps",
        type=int,
        default=StaticQwen3VLRuntimeConfig.gen_steps,
        help="Maximum number of generated tokens.",
    )
    parser.add_argument(
        "--no-trust-remote-code",
        action="store_true",
        help="Disable trust_remote_code when loading the model and tokenizer/processor.",
    )
    args = parser.parse_args()

    return StaticQwen3VLRuntimeConfig(
        model=args.model,
        max_seq=args.max_seq,
        padding_side=args.padding_side,
        device=args.device,
        prompt=args.prompt,
        image=args.image,
        image_max_pixels=args.image_max_pixels,
        image_min_pixels=args.image_min_pixels,
        use_chat_template=not args.no_chat_template,
        visual_start_idx=args.visual_start_idx,
        enable_deepstack=not args.no_deepstack,
        verify_steps=args.verify_steps,
        gen_steps=args.gen_steps,
        trust_remote_code=not args.no_trust_remote_code,
    )


if __name__ == "__main__":
    run_static_qwen3_vl_runtime(_parse_args())
