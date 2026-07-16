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

from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import torch


_DEFAULT_MAX_CHUNK_ELEMENTS = 4 * 1024 * 1024
_ALL_SCOPE = "All quantized weights"
_UNCLASSIFIED_SCOPE = "Unclassified quantized weights"


@dataclass(frozen=True)
class WeightTensorSparsity:
    """Store semantic-zero statistics for one quantized weight use."""

    module_name: str
    weight_name: str
    qdtype: str
    zero_count: int
    numel: int
    dedup_key: str

    @property
    def sparsity_percent(self) -> float:
        """Return the semantic-zero percentage for this tensor."""

        if self.numel == 0:
            return 0.0
        return 100.0 * self.zero_count / self.numel


@dataclass(frozen=True)
class SparsityRow:
    """Store one aggregated summary-table row."""

    scope: str
    qdtype: str
    zero_count: int
    numel: int

    @property
    def sparsity_percent(self) -> float | None:
        """Return the weighted semantic-zero percentage for this scope."""

        if self.numel == 0:
            return None
        return 100.0 * self.zero_count / self.numel


@dataclass(frozen=True)
class LayerSparsityRow:
    """Store one layer-and-scope sparsity-table row."""

    layer: str
    scope: str
    qdtype: str
    zero_count: int
    numel: int

    @property
    def sparsity_percent(self) -> float | None:
        """Return the weighted semantic-zero percentage for this layer scope."""

        if self.numel == 0:
            return None
        return 100.0 * self.zero_count / self.numel


@dataclass(frozen=True)
class WeightSparsityReport:
    """Store summary and layer-level rows produced from one tensor scan."""

    summary_rows: tuple[SparsityRow, ...]
    layer_rows: tuple[LayerSparsityRow, ...]


class WeightSparsityError(RuntimeError):
    """Report an invalid or unsupported post-convert sparsity state."""


def _is_identity_observer(observer: Any) -> bool:
    """Return whether an observer intentionally leaves a weight in floating point."""

    return observer.__class__.__name__ == "IdentityObserver"


def _extract_direct_weight(module: torch.nn.Module) -> torch.Tensor | None:
    """Return the weight owned by a leaf quantization wrapper."""

    wrapped_module = getattr(module, "module", None)
    weight = getattr(wrapped_module, "weight", None)
    if isinstance(weight, torch.Tensor):
        return weight

    weight = getattr(module, "weight", None)
    if isinstance(weight, torch.Tensor):
        return weight
    return None


def _get_direct_weight_observer(module: torch.nn.Module) -> Any | None:
    """Return the direct weight observer without traversing child wrappers."""

    get_observer = getattr(module, "get_observer", None)
    if not callable(get_observer):
        return None

    try:
        return get_observer("weight", recurse=False)
    except TypeError as exc:
        raise WeightSparsityError(
            f"{type(module).__name__}.get_observer() must accept recurse=False."
        ) from exc


def _require_post_convert_mode(module: torch.nn.Module, weight_name: str) -> None:
    """Verify that a quantization wrapper is in its final QUANT mode."""

    mode = getattr(module, "_mode", None)
    mode_name = getattr(mode, "name", None)
    if mode is not None and mode_name != "QUANT":
        raise WeightSparsityError(
            f"Weight {weight_name!r} is not in post-convert QUANT mode: {mode!r}."
        )


def _require_affine_qparams(
    observer: Any, weight_name: str
) -> tuple[torch.Tensor, torch.Tensor, int, int, int | None, str]:
    """Read finalized affine quantization parameters from a TICO observer."""

    scale = getattr(observer, "_cached_scale", None)
    zero_point = getattr(observer, "_cached_zp", None)
    dtype = getattr(observer, "dtype", None)
    channel_axis = getattr(observer, "channel_axis", None)

    if not isinstance(scale, torch.Tensor) or scale.numel() == 0:
        raise WeightSparsityError(
            f"Weight {weight_name!r} does not have a finalized affine scale."
        )
    if not isinstance(zero_point, torch.Tensor) or zero_point.numel() == 0:
        raise WeightSparsityError(
            f"Weight {weight_name!r} does not have a finalized affine zero-point."
        )
    if dtype is None or not hasattr(dtype, "qmin") or not hasattr(dtype, "qmax"):
        raise WeightSparsityError(
            f"Weight {weight_name!r} uses an unsupported non-affine observer "
            f"{type(observer).__name__}."
        )

    qmin = int(dtype.qmin)
    qmax = int(dtype.qmax)
    qdtype = str(dtype)
    if channel_axis is not None:
        channel_axis = int(channel_axis)

    return scale, zero_point, qmin, qmax, channel_axis, qdtype


def _count_block_zeros(
    block: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    qmin: int,
    qmax: int,
) -> int:
    """Count integer codes equal to the broadcast zero-point in one block."""

    values = block.detach().to(dtype=torch.float32)
    scale_f = scale.to(device=values.device, dtype=torch.float32)
    zero_f = zero_point.to(device=values.device, dtype=torch.float32)

    quantized = torch.round(values / scale_f) + zero_f
    quantized.clamp_(qmin, qmax)
    return int(torch.count_nonzero(quantized == zero_f).item())


def _count_per_tensor_zeros(
    weight: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    qmin: int,
    qmax: int,
    max_chunk_elements: int,
) -> int:
    """Count semantic zeros for a per-tensor affine weight."""

    if scale.numel() != 1 or zero_point.numel() != 1:
        raise WeightSparsityError(
            "Per-tensor quantization requires scalar scale and zero-point values."
        )

    zero_count = 0
    flattened = weight.detach().reshape(-1)
    for start in range(0, flattened.numel(), max_chunk_elements):
        block = flattened[start : start + max_chunk_elements]
        zero_count += _count_block_zeros(
            block,
            scale.reshape(()),
            zero_point.reshape(()),
            qmin,
            qmax,
        )
    return zero_count


def _count_per_channel_zeros(
    weight: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    qmin: int,
    qmax: int,
    channel_axis: int,
    max_chunk_elements: int,
) -> int:
    """Count semantic zeros for a per-channel affine weight."""

    axis = channel_axis if channel_axis >= 0 else weight.ndim + channel_axis
    if axis < 0 or axis >= weight.ndim:
        raise WeightSparsityError(
            f"Invalid channel axis {channel_axis} for weight shape "
            f"{tuple(weight.shape)}."
        )

    channels = int(weight.shape[axis])
    scale_flat = scale.detach().reshape(-1)
    zero_flat = zero_point.detach().reshape(-1)
    if scale_flat.numel() != channels or zero_flat.numel() != channels:
        raise WeightSparsityError(
            "Per-channel scale and zero-point lengths must match the quantized axis: "
            f"channels={channels}, scale={scale_flat.numel()}, "
            f"zero_point={zero_flat.numel()}."
        )

    channel_major = weight.detach().movedim(axis, 0).reshape(channels, -1)
    inner_size = int(channel_major.shape[1])
    columns_per_chunk = min(inner_size, max_chunk_elements)
    rows_per_chunk = max(1, max_chunk_elements // columns_per_chunk)

    zero_count = 0
    for row_start in range(0, channels, rows_per_chunk):
        row_end = min(channels, row_start + rows_per_chunk)
        row_scale = scale_flat[row_start:row_end].reshape(-1, 1)
        row_zero = zero_flat[row_start:row_end].reshape(-1, 1)

        for column_start in range(0, inner_size, columns_per_chunk):
            column_end = min(inner_size, column_start + columns_per_chunk)
            block = channel_major[row_start:row_end, column_start:column_end]
            zero_count += _count_block_zeros(
                block,
                row_scale,
                row_zero,
                qmin,
                qmax,
            )
    return zero_count


def _count_semantic_zeros(
    weight: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    qmin: int,
    qmax: int,
    channel_axis: int | None,
    max_chunk_elements: int,
) -> int:
    """Count weight values whose integer code equals their affine zero-point."""

    if weight.device.type == "meta":
        raise WeightSparsityError("Meta-device weights cannot be measured.")
    if weight.numel() == 0:
        return 0

    with torch.no_grad():
        if channel_axis is None:
            return _count_per_tensor_zeros(
                weight,
                scale,
                zero_point,
                qmin,
                qmax,
                max_chunk_elements,
            )
        return _count_per_channel_zeros(
            weight,
            scale,
            zero_point,
            qmin,
            qmax,
            channel_axis,
            max_chunk_elements,
        )


def _tensor_digest(tensor: torch.Tensor) -> str:
    """Return a stable digest for a small quantization-parameter tensor."""

    data = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha1(data).hexdigest()


def _weight_storage_key(weight: torch.Tensor) -> str:
    """Return a storage-and-view identity for a weight tensor."""

    try:
        storage_pointer = weight.untyped_storage().data_ptr()
    except (AttributeError, RuntimeError):
        storage_pointer = id(weight)

    device_index = -1 if weight.device.index is None else int(weight.device.index)
    return ":".join(
        [
            weight.device.type,
            str(device_index),
            str(storage_pointer),
            str(weight.storage_offset()),
            repr(tuple(weight.shape)),
            repr(tuple(weight.stride())),
        ]
    )


def _make_dedup_key(
    weight: torch.Tensor,
    qdtype: str,
    channel_axis: int | None,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    module_name: str,
    deduplicate_shared_weights: bool,
) -> str:
    """Build a key that deduplicates shared weights with identical qparams."""

    if not deduplicate_shared_weights:
        return f"module:{module_name}"

    return ":".join(
        [
            _weight_storage_key(weight),
            qdtype,
            str(channel_axis),
            _tensor_digest(scale),
            _tensor_digest(zero_point),
        ]
    )


def collect_weight_tensor_sparsity(
    model: torch.nn.Module,
    *,
    max_chunk_elements: int = _DEFAULT_MAX_CHUNK_ELEMENTS,
    deduplicate_shared_weights: bool = True,
) -> list[WeightTensorSparsity]:
    """Collect post-convert semantic-zero statistics from TICO weight observers.

    The function visits only direct ``weight`` observers on leaf quantization
    wrappers. Identity observers are skipped because they intentionally keep the
    associated weight in floating point. Affine integer codes are reconstructed
    in bounded chunks, and a semantic zero is counted when ``qcode == zero_point``.

    Args:
        model: A TICO model immediately after PTQ ``convert()``.
        max_chunk_elements: Maximum number of weight elements processed at once.
        deduplicate_shared_weights: Deduplicate shared storage only when the
            quantization dtype and finalized qparams are also identical.

    Returns:
        One statistics record per quantized weight use.

    Raises:
        ValueError: If ``max_chunk_elements`` is not positive.
        WeightSparsityError: If the model is not in a measurable post-convert
            affine quantization state.
    """

    if max_chunk_elements <= 0:
        raise ValueError("max_chunk_elements must be positive.")

    records: list[WeightTensorSparsity] = []
    for module_name, module in model.named_modules():
        observer = _get_direct_weight_observer(module)
        if observer is None or _is_identity_observer(observer):
            continue

        weight_name = str(getattr(module, "fp_name", None) or module_name)
        _require_post_convert_mode(module, weight_name)

        weight = _extract_direct_weight(module)
        if weight is None:
            raise WeightSparsityError(
                f"Quantized weight observer for {weight_name!r} has no direct "
                "weight tensor."
            )

        scale, zero_point, qmin, qmax, channel_axis, qdtype = _require_affine_qparams(
            observer, weight_name
        )
        zero_count = _count_semantic_zeros(
            weight,
            scale,
            zero_point,
            qmin,
            qmax,
            channel_axis,
            max_chunk_elements,
        )
        dedup_key = _make_dedup_key(
            weight,
            qdtype,
            channel_axis,
            scale,
            zero_point,
            module_name,
            deduplicate_shared_weights,
        )
        records.append(
            WeightTensorSparsity(
                module_name=module_name,
                weight_name=weight_name,
                qdtype=qdtype,
                zero_count=zero_count,
                numel=weight.numel(),
                dedup_key=dedup_key,
            )
        )

    if not records:
        raise WeightSparsityError(
            "No finalized affine weight observers were found. Run the analysis "
            "immediately after the PTQ convert stage."
        )
    return records


def _path_has_suffix(path: str, suffix: str) -> bool:
    """Return whether a module path matches a complete dotted suffix."""

    return path == suffix or path.endswith(f".{suffix}")


def _contains_segment(path: str, *segments: str) -> bool:
    """Return whether any complete path segment is present."""

    path_segments = set(path.split("."))
    return any(segment in path_segments for segment in segments)


def _contains_path(path: str, fragment: str) -> bool:
    """Return whether a complete dotted subpath is present."""

    return (
        path == fragment
        or path.startswith(f"{fragment}.")
        or path.endswith(f".{fragment}")
        or f".{fragment}." in path
    )


def _classify_llama(weight_name: str) -> str:
    """Return the projection-level Llama scope for a weight path."""

    if _path_has_suffix(weight_name, "model.embed_tokens") or _path_has_suffix(
        weight_name, "embed_tokens"
    ):
        return "Token embedding"
    if _path_has_suffix(weight_name, "lm_head"):
        return "LM head"
    if _contains_segment(weight_name, "rotate_embedding", "rotate_lm_head"):
        return "SpinQuant rotation weights"

    projection_scopes = {
        "self_attn.q_proj": "Attention / q_proj",
        "self_attn.k_proj": "Attention / k_proj",
        "self_attn.v_proj": "Attention / v_proj",
        "self_attn.o_proj": "Attention / o_proj",
        "mlp.gate_proj": "MLP / gate_proj",
        "mlp.up_proj": "MLP / up_proj",
        "mlp.down_proj": "MLP / down_proj",
    }
    for suffix, scope in projection_scopes.items():
        if _path_has_suffix(weight_name, suffix):
            return scope

    if _contains_segment(
        weight_name,
        "input_layernorm",
        "post_attention_layernorm",
        "norm",
    ):
        return "Norm weights"
    return _UNCLASSIFIED_SCOPE


def _classify_qwen3_vl(weight_name: str) -> str:
    """Return the projection-level Qwen3-VL scope for a weight path."""

    if _contains_path(weight_name, "visual.patch_embed") and _path_has_suffix(
        weight_name, "patch_embed.proj"
    ):
        return "Vision / patch_embed.proj"

    vision_projection_scopes = {
        "attn.qkv": "Vision attention / qkv",
        "attn.proj": "Vision attention / proj",
        "mlp.linear_fc1": "Vision MLP / linear_fc1",
        "mlp.linear_fc2": "Vision MLP / linear_fc2",
    }
    if _contains_path(weight_name, "visual.blocks"):
        for suffix, scope in vision_projection_scopes.items():
            if _path_has_suffix(weight_name, suffix):
                return scope

    merger_scopes = {
        "linear_fc1": "Vision merger / linear_fc1",
        "linear_fc2": "Vision merger / linear_fc2",
    }
    if _contains_path(weight_name, "visual.merger"):
        for suffix, scope in merger_scopes.items():
            if _path_has_suffix(weight_name, suffix):
                return scope
    if _contains_path(weight_name, "visual.deepstack_merger_list"):
        for suffix, scope in merger_scopes.items():
            if _path_has_suffix(weight_name, suffix):
                return scope.replace("Vision merger", "Deepstack merger")

    if _contains_path(weight_name, "language_model"):
        if _path_has_suffix(weight_name, "embed_tokens"):
            return "Text / token embedding"

        text_projection_scopes = {
            "self_attn.q_proj": "Text attention / q_proj",
            "self_attn.k_proj": "Text attention / k_proj",
            "self_attn.v_proj": "Text attention / v_proj",
            "self_attn.o_proj": "Text attention / o_proj",
            "mlp.gate_proj": "Text MLP / gate_proj",
            "mlp.up_proj": "Text MLP / up_proj",
            "mlp.down_proj": "Text MLP / down_proj",
        }
        for suffix, scope in text_projection_scopes.items():
            if _path_has_suffix(weight_name, suffix):
                return scope

    if _path_has_suffix(weight_name, "lm_head"):
        return "LM head"
    if _contains_segment(weight_name, "rotate_embedding", "rotate_lm_head"):
        return "SpinQuant rotation weights"
    if _contains_segment(
        weight_name,
        "norm",
        "norm1",
        "norm2",
        "input_layernorm",
        "post_attention_layernorm",
        "q_norm",
        "k_norm",
    ):
        return "Norm weights"
    return _UNCLASSIFIED_SCOPE


_LLAMA_SCOPE_ORDER = (
    "Token embedding",
    "Attention / q_proj",
    "Attention / k_proj",
    "Attention / v_proj",
    "Attention / o_proj",
    "MLP / gate_proj",
    "MLP / up_proj",
    "MLP / down_proj",
    "LM head",
    "Norm weights",
    "SpinQuant rotation weights",
)

_QWEN3_VL_SCOPE_ORDER = (
    "Vision / patch_embed.proj",
    "Vision attention / qkv",
    "Vision attention / proj",
    "Vision MLP / linear_fc1",
    "Vision MLP / linear_fc2",
    "Vision merger / linear_fc1",
    "Vision merger / linear_fc2",
    "Deepstack merger / linear_fc1",
    "Deepstack merger / linear_fc2",
    "Text / token embedding",
    "Text attention / q_proj",
    "Text attention / k_proj",
    "Text attention / v_proj",
    "Text attention / o_proj",
    "Text MLP / gate_proj",
    "Text MLP / up_proj",
    "Text MLP / down_proj",
    "LM head",
    "Norm weights",
    "SpinQuant rotation weights",
)

_LLAMA_DECODER_LAYER_RE = re.compile(r"(?:^|\.)(?:model\.)?layers\.(\d+)(?:\.|$)")
_QWEN_VISION_BLOCK_RE = re.compile(r"(?:^|\.)(?:model\.)?visual\.blocks\.(\d+)(?:\.|$)")
_QWEN_DEEPSTACK_MERGER_RE = re.compile(
    r"(?:^|\.)(?:model\.)?visual\.deepstack_merger_list\.(\d+)(?:\.|$)"
)
_QWEN_TEXT_LAYER_RE = re.compile(
    r"(?:^|\.)(?:model\.)?language_model\.layers\.(\d+)(?:\.|$)"
)


def _matched_index(pattern: re.Pattern[str], path: str) -> int | None:
    """Return an integer index captured from a module path."""

    match = pattern.search(path)
    if match is None:
        return None
    return int(match.group(1))


def _classify_llama_layer(weight_name: str) -> str:
    """Return a normalized Llama layer name for a weight path."""

    layer_idx = _matched_index(_LLAMA_DECODER_LAYER_RE, weight_name)
    if layer_idx is not None:
        return f"model.layers.{layer_idx}"
    if _path_has_suffix(weight_name, "model.embed_tokens") or _path_has_suffix(
        weight_name, "embed_tokens"
    ):
        return "model.embed_tokens"
    if _contains_segment(weight_name, "rotate_embedding"):
        return "model.rotate_embedding"
    if _path_has_suffix(weight_name, "model.norm") or weight_name == "norm":
        return "model.norm"
    if _contains_segment(weight_name, "rotate_lm_head"):
        return "rotate_lm_head"
    if _path_has_suffix(weight_name, "lm_head"):
        return "lm_head"
    return weight_name or "<unnamed>"


def _classify_qwen3_vl_layer(weight_name: str) -> str:
    """Return a normalized Qwen3-VL layer name for a weight path."""

    vision_idx = _matched_index(_QWEN_VISION_BLOCK_RE, weight_name)
    if vision_idx is not None:
        return f"model.visual.blocks.{vision_idx}"

    deepstack_idx = _matched_index(_QWEN_DEEPSTACK_MERGER_RE, weight_name)
    if deepstack_idx is not None:
        return f"model.visual.deepstack_merger_list.{deepstack_idx}"

    text_idx = _matched_index(_QWEN_TEXT_LAYER_RE, weight_name)
    if text_idx is not None:
        return f"model.language_model.layers.{text_idx}"

    if _contains_path(weight_name, "visual.patch_embed"):
        return "model.visual.patch_embed"
    if _contains_path(weight_name, "visual.merger"):
        return "model.visual.merger"
    if _contains_path(weight_name, "language_model.embed_tokens"):
        return "model.language_model.embed_tokens"
    if _contains_path(weight_name, "language_model.rotate_embedding"):
        return "model.language_model.rotate_embedding"
    if _path_has_suffix(weight_name, "language_model.norm"):
        return "model.language_model.norm"
    if _contains_segment(weight_name, "rotate_lm_head"):
        return "rotate_lm_head"
    if _path_has_suffix(weight_name, "lm_head"):
        return "lm_head"
    return weight_name or "<unnamed>"


def _llama_layer_sort_key(layer: str) -> tuple[int, int, str]:
    """Return a stable architecture-aware sort key for Llama layer names."""

    if layer == "model.embed_tokens":
        return (0, 0, layer)
    if layer == "model.rotate_embedding":
        return (1, 0, layer)
    match = re.fullmatch(r"model\.layers\.(\d+)", layer)
    if match is not None:
        return (2, int(match.group(1)), layer)
    if layer == "model.norm":
        return (3, 0, layer)
    if layer == "rotate_lm_head":
        return (4, 0, layer)
    if layer == "lm_head":
        return (5, 0, layer)
    return (99, 0, layer)


def _qwen3_vl_layer_sort_key(layer: str) -> tuple[int, int, str]:
    """Return a stable architecture-aware sort key for Qwen3-VL layer names."""

    if layer == "model.visual.patch_embed":
        return (0, 0, layer)
    match = re.fullmatch(r"model\.visual\.blocks\.(\d+)", layer)
    if match is not None:
        return (1, int(match.group(1)), layer)
    if layer == "model.visual.merger":
        return (2, 0, layer)
    match = re.fullmatch(r"model\.visual\.deepstack_merger_list\.(\d+)", layer)
    if match is not None:
        return (3, int(match.group(1)), layer)
    if layer == "model.language_model.embed_tokens":
        return (4, 0, layer)
    if layer == "model.language_model.rotate_embedding":
        return (5, 0, layer)
    match = re.fullmatch(r"model\.language_model\.layers\.(\d+)", layer)
    if match is not None:
        return (6, int(match.group(1)), layer)
    if layer == "model.language_model.norm":
        return (7, 0, layer)
    if layer == "rotate_lm_head":
        return (8, 0, layer)
    if layer == "lm_head":
        return (9, 0, layer)
    return (99, 0, layer)


def _family_definition(
    family: str,
) -> tuple[
    Callable[[str], str],
    tuple[str, ...],
    Callable[[str], str],
    Callable[[str], tuple[int, int, str]],
]:
    """Return scope and layer classifiers for a supported model family."""

    normalized = family.strip().lower().replace("-", "_")
    if normalized == "llama":
        return (
            _classify_llama,
            _LLAMA_SCOPE_ORDER,
            _classify_llama_layer,
            _llama_layer_sort_key,
        )
    if normalized in {"qwen3_vl", "qwen3vl"}:
        return (
            _classify_qwen3_vl,
            _QWEN3_VL_SCOPE_ORDER,
            _classify_qwen3_vl_layer,
            _qwen3_vl_layer_sort_key,
        )
    raise ValueError(
        f"Unsupported model family {family!r}. Supported families: llama, qwen3_vl."
    )


def _aggregate_records(
    scope: str, records: Iterable[WeightTensorSparsity]
) -> SparsityRow:
    """Aggregate records by element count while deduplicating equivalent weights."""

    seen: set[str] = set()
    qdtypes: set[str] = set()
    zero_count = 0
    numel = 0

    for record in records:
        if record.dedup_key in seen:
            continue
        seen.add(record.dedup_key)
        qdtypes.add(record.qdtype)
        zero_count += record.zero_count
        numel += record.numel

    if not qdtypes:
        qdtype = "-"
    elif len(qdtypes) == 1:
        qdtype = next(iter(qdtypes))
    else:
        qdtype = f"mixed ({', '.join(sorted(qdtypes))})"

    return SparsityRow(
        scope=scope,
        qdtype=qdtype,
        zero_count=zero_count,
        numel=numel,
    )


def aggregate_weight_sparsity(
    records: Sequence[WeightTensorSparsity],
    family: str,
    *,
    include_empty_scopes: bool = False,
) -> list[SparsityRow]:
    """Aggregate collected tensor statistics into the three-column summary report."""

    classifier, scope_order, _, _ = _family_definition(family)
    grouped: dict[str, list[WeightTensorSparsity]] = {
        scope: [] for scope in scope_order
    }
    grouped[_UNCLASSIFIED_SCOPE] = []
    for record in records:
        grouped.setdefault(classifier(record.weight_name), []).append(record)

    rows = [_aggregate_records(_ALL_SCOPE, records)]
    for scope in scope_order:
        row = _aggregate_records(scope, grouped.get(scope, ()))
        if include_empty_scopes or row.numel > 0:
            rows.append(row)

    unclassified = _aggregate_records(_UNCLASSIFIED_SCOPE, grouped[_UNCLASSIFIED_SCOPE])
    if unclassified.numel > 0:
        rows.append(unclassified)
    return rows


def _to_layer_row(layer: str, row: SparsityRow) -> LayerSparsityRow:
    """Attach a normalized layer name to an aggregated scope row."""

    return LayerSparsityRow(
        layer=layer,
        scope=row.scope,
        qdtype=row.qdtype,
        zero_count=row.zero_count,
        numel=row.numel,
    )


def aggregate_layer_weight_sparsity(
    records: Sequence[WeightTensorSparsity],
    family: str,
    *,
    include_layer_totals: bool = True,
) -> list[LayerSparsityRow]:
    """Aggregate tensor statistics by normalized layer and projection scope."""

    classifier, scope_order, layer_classifier, layer_sort_key = _family_definition(
        family
    )
    grouped: dict[str, dict[str, list[WeightTensorSparsity]]] = {}
    layer_records: dict[str, list[WeightTensorSparsity]] = {}

    for record in records:
        layer = layer_classifier(record.weight_name)
        scope = classifier(record.weight_name)
        layer_records.setdefault(layer, []).append(record)
        grouped.setdefault(layer, {}).setdefault(scope, []).append(record)

    rows: list[LayerSparsityRow] = []
    for layer in sorted(layer_records, key=layer_sort_key):
        if include_layer_totals:
            rows.append(
                _to_layer_row(
                    layer,
                    _aggregate_records(_ALL_SCOPE, layer_records[layer]),
                )
            )

        by_scope = grouped[layer]
        for scope in scope_order:
            row = _aggregate_records(scope, by_scope.get(scope, ()))
            if row.numel > 0:
                rows.append(_to_layer_row(layer, row))

        unclassified = _aggregate_records(
            _UNCLASSIFIED_SCOPE,
            by_scope.get(_UNCLASSIFIED_SCOPE, ()),
        )
        if unclassified.numel > 0:
            rows.append(_to_layer_row(layer, unclassified))
    return rows


def measure_weight_sparsity(
    model: torch.nn.Module,
    family: str,
    *,
    max_chunk_elements: int = _DEFAULT_MAX_CHUNK_ELEMENTS,
    deduplicate_shared_weights: bool = True,
    include_empty_scopes: bool = False,
) -> list[SparsityRow]:
    """Measure post-convert weight sparsity and build a three-column summary report.

    Scope-level sparsity is computed from the total semantic-zero count divided by
    the total number of elements in that scope. It is not an unweighted average of
    per-tensor sparsity values.

    Args:
        model: A TICO model immediately after PTQ ``convert()``.
        family: ``llama`` or ``qwen3_vl``.
        max_chunk_elements: Maximum number of elements processed in one block.
        deduplicate_shared_weights: Deduplicate shared weights with identical
            finalized affine qparams inside each report row.
        include_empty_scopes: Include predefined scopes that have no matching weight.

    Returns:
        Rows ordered as ``Scope``, ``Qdtype``, and ``Sparsity (%)``.
    """

    records = collect_weight_tensor_sparsity(
        model,
        max_chunk_elements=max_chunk_elements,
        deduplicate_shared_weights=deduplicate_shared_weights,
    )
    return aggregate_weight_sparsity(
        records,
        family,
        include_empty_scopes=include_empty_scopes,
    )


def measure_layer_weight_sparsity(
    model: torch.nn.Module,
    family: str,
    *,
    max_chunk_elements: int = _DEFAULT_MAX_CHUNK_ELEMENTS,
    deduplicate_shared_weights: bool = True,
    include_layer_totals: bool = True,
) -> list[LayerSparsityRow]:
    """Measure post-convert sparsity by layer and projection scope."""

    records = collect_weight_tensor_sparsity(
        model,
        max_chunk_elements=max_chunk_elements,
        deduplicate_shared_weights=deduplicate_shared_weights,
    )
    return aggregate_layer_weight_sparsity(
        records,
        family,
        include_layer_totals=include_layer_totals,
    )


def measure_weight_sparsity_report(
    model: torch.nn.Module,
    family: str,
    *,
    max_chunk_elements: int = _DEFAULT_MAX_CHUNK_ELEMENTS,
    deduplicate_shared_weights: bool = True,
    include_empty_scopes: bool = False,
    include_layer_totals: bool = True,
) -> WeightSparsityReport:
    """Measure summary and layer reports from a single post-convert tensor scan."""

    records = collect_weight_tensor_sparsity(
        model,
        max_chunk_elements=max_chunk_elements,
        deduplicate_shared_weights=deduplicate_shared_weights,
    )
    return WeightSparsityReport(
        summary_rows=tuple(
            aggregate_weight_sparsity(
                records,
                family,
                include_empty_scopes=include_empty_scopes,
            )
        ),
        layer_rows=tuple(
            aggregate_layer_weight_sparsity(
                records,
                family,
                include_layer_totals=include_layer_totals,
            )
        ),
    )


def _format_sparsity_percent(value: float | None, precision: int) -> str:
    """Format an optional sparsity percentage for text output."""

    if value is None:
        return "-"
    return f"{value:.{precision}f}"


def format_weight_sparsity_table(
    rows: Sequence[SparsityRow], *, precision: int = 6
) -> str:
    """Format the scope-level Markdown sparsity table."""

    if precision < 0:
        raise ValueError("precision must be non-negative.")

    lines = [
        "| Scope | Qdtype | Sparsity (%) |",
        "|---|---|---:|",
    ]
    for row in rows:
        sparsity = _format_sparsity_percent(row.sparsity_percent, precision)
        scope = row.scope.replace("|", "\\|")
        qdtype = row.qdtype.replace("|", "\\|")
        lines.append(f"| {scope} | {qdtype} | {sparsity} |")
    return "\n".join(lines)


def format_layer_weight_sparsity_table(
    rows: Sequence[LayerSparsityRow], *, precision: int = 6
) -> str:
    """Format the layer-and-scope Markdown sparsity table."""

    if precision < 0:
        raise ValueError("precision must be non-negative.")

    lines = [
        "| Layer | Scope | Qdtype | Sparsity (%) |",
        "|---|---|---|---:|",
    ]
    for row in rows:
        sparsity = _format_sparsity_percent(row.sparsity_percent, precision)
        layer = row.layer.replace("|", "\\|")
        scope = row.scope.replace("|", "\\|")
        qdtype = row.qdtype.replace("|", "\\|")
        lines.append(f"| {layer} | {scope} | {qdtype} | {sparsity} |")
    return "\n".join(lines)


def write_weight_sparsity_csv(
    rows: Sequence[SparsityRow],
    path: str | Path,
    *,
    precision: int = 6,
) -> Path:
    """Write the three-column summary report as CSV."""

    if precision < 0:
        raise ValueError("precision must be non-negative.")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["scope", "qdtype", "sparsity_percent"])
        for row in rows:
            value = (
                ""
                if row.sparsity_percent is None
                else f"{row.sparsity_percent:.{precision}f}"
            )
            writer.writerow([row.scope, row.qdtype, value])
    return output_path


def write_layer_weight_sparsity_csv(
    rows: Sequence[LayerSparsityRow],
    path: str | Path,
    *,
    precision: int = 6,
) -> Path:
    """Write the layer-and-scope sparsity report as CSV."""

    if precision < 0:
        raise ValueError("precision must be non-negative.")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["layer", "scope", "qdtype", "sparsity_percent"])
        for row in rows:
            value = (
                ""
                if row.sparsity_percent is None
                else f"{row.sparsity_percent:.{precision}f}"
            )
            writer.writerow([row.layer, row.scope, row.qdtype, value])
    return output_path


def write_weight_sparsity_markdown(
    rows: Sequence[SparsityRow],
    path: str | Path,
    *,
    precision: int = 6,
) -> Path:
    """Write the three-column summary report as Markdown."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        format_weight_sparsity_table(rows, precision=precision) + "\n",
        encoding="utf-8",
    )
    return output_path


def write_layer_weight_sparsity_markdown(
    rows: Sequence[LayerSparsityRow],
    path: str | Path,
    *,
    precision: int = 6,
) -> Path:
    """Write the layer-and-scope sparsity report as Markdown."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        format_layer_weight_sparsity_table(rows, precision=precision) + "\n",
        encoding="utf-8",
    )
    return output_path
