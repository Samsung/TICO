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
from typing import Any, Callable, Iterable, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.observers.affine_base import AffineObserverBase
from tico.quantization.wrapq.observers.base import ObserverBase
from tico.quantization.wrapq.observers.identity import IdentityObserver

# ``(tensor, observer) -> tensor`` hook applied at every PLE observer boundary.
# ``QuantModuleBase._fq`` provides calibration/quant/no-quant semantics for the
# text model; export adapters pass a frozen fake-quant or identity function.
PLEObserverFn = Callable[[torch.Tensor, Any], torch.Tensor]


@dataclass(frozen=True)
class StaticGemma4Layout:
    """Static layout information required by the dense Gemma4 runtime.

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
                "Dense Gemma4 static runtime currently supports batch_size=1 only."
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


def assert_gemma4_dense_no_moe(model_or_config: Any) -> None:
    """Raise if the supplied Gemma4 model or config enables MoE blocks.

    The dense Gemma4 E2B/E4B runtime supports only dense decoder layers. This
    guard should be called both in recipe loading and wrapper construction.
    """

    config = getattr(model_or_config, "config", model_or_config)
    if hasattr(config, "get_text_config"):
        text_config = config.get_text_config()
    else:
        text_config = getattr(config, "text_config", config)

    if bool(getattr(text_config, "enable_moe_block", False)):
        raise ValueError(
            "Dense Gemma4 static runtime supports dense decoder layers only, "
            "but text_config.enable_moe_block=True."
        )

    if hasattr(model_or_config, "named_modules"):
        for name, module in model_or_config.named_modules():
            cls_name = type(module).__name__
            if cls_name in {"Gemma4TextRouter", "Gemma4TextExperts"}:
                raise ValueError(
                    f"Unexpected MoE module in dense Gemma4 model: {name} ({cls_name})."
                )


# Retained for external users and the existing Gemma4 assistant path.
assert_gemma4_e2b_no_moe = assert_gemma4_dense_no_moe


def resolve_gemma4_shared_kv_suffix(
    layers: Iterable[nn.Module],
) -> tuple[int, dict[int, int]]:
    """Resolve the trailing read-only KV consumers and their storage producers.

    Inspect the instantiated layers, rather than assuming a config-specific
    boundary or one producer for every attention type. The mapping uses decoder
    list indices and includes consumers before the trailing suffix as well.
    A model without a trailing shared consumer returns ``len(layers)``.

    Explicit producer bindings must agree with the same-type
    ``store_full_length_kv`` producer used by the eager text model. Malformed
    sharing is an error, not a reason to silently use another execution path.
    """
    decoders = [getattr(layer, "wrapped", layer) for layer in layers]
    sources: dict[str, tuple[int, Any]] = {}
    producers: dict[int, int] = {}
    boundary = 0
    for index, decoder in enumerate(decoders):
        if bool(getattr(decoder, "enable_moe_block", False)):
            raise ValueError("Gemma4 shared-KV suffix requires dense decoder layers.")
        attention_wrapper = getattr(decoder, "self_attn", None)
        attention = getattr(attention_wrapper, "wrapped", attention_wrapper)
        if attention is None:
            raise ValueError(f"Gemma4 decoder layer {index} has no self attention.")
        layer_type = getattr(attention, "layer_type", None)
        if layer_type not in {"full_attention", "sliding_attention"}:
            raise ValueError(
                f"Unsupported Gemma4 shared-KV layer type at layer {index}: "
                f"{layer_type!r}."
            )
        if not bool(getattr(attention, "is_kv_shared_layer", False)):
            boundary = index + 1
            if bool(getattr(attention, "store_full_length_kv", False)):
                sources[layer_type] = (index, attention)
            continue
        if bool(getattr(attention, "store_full_length_kv", False)) or any(
            getattr(attention, name, None) is not None for name in ("k_proj", "v_proj")
        ):
            raise ValueError(
                f"Gemma4 shared-KV consumer layer {index} must not produce K/V."
            )
        source_entry = sources.get(layer_type)
        if source_entry is None:
            raise ValueError(
                "Gemma4 shared-KV layer has no preceding storage producer: "
                f"layer_idx={index}, layer_type={layer_type!r}."
            )
        source_index, source = source_entry
        if (
            hasattr(attention, "_cache_source")
            and attention._cache_source is not source
        ):
            raise ValueError(
                f"Gemma4 shared-KV layer {index} cache binding does not match "
                f"storage producer {source_index}."
            )
        if getattr(attention, "head_dim", None) != getattr(source, "head_dim", None):
            raise ValueError(
                f"Gemma4 shared-KV layer {index} and producer {source_index} "
                "have incompatible head dimensions."
            )
        head_counts = []
        for owner in (attention, source):
            config = getattr(owner, "config", None)
            per_layer = getattr(config, "per_layer_config", None)
            if per_layer is not None:
                config = per_layer[int(owner.layer_idx)]
            head_counts.append(getattr(config, "num_key_value_heads", None))
        if (
            all(count is not None for count in head_counts)
            and head_counts[0] != head_counts[1]
        ):
            raise ValueError(
                f"Gemma4 shared-KV layer {index} and producer {source_index} "
                "have incompatible KV head counts."
            )
        producers[index] = source_index
    return boundary, producers


def validate_gemma4_single_token_quantization(
    layers: Iterable[nn.Module],
    *,
    per_layer_input_observer: Optional[ObserverBase] = None,
) -> None:
    """Require token-independent frozen activation grids for suffix slicing.

    The initial implementation supports the canonical static backend's
    per-tensor affine or identity activations. Arbitrary per-channel activation
    axes and dynamic MX exponent groups can depend on the removed query axis;
    they need a separate shape/domain proof. Weight and bias policies are
    unaffected, including per-channel and MX weights. This check never collects
    statistics, switches modes, or replaces frozen grids.
    """
    issue = _gemma4_single_token_quantization_issue(
        layers, per_layer_input_observer=per_layer_input_observer
    )
    if issue is not None:
        raise ValueError(
            "Gemma4 single-token finalize requires per-tensor affine or identity "
            f"activation observers; unsupported policy at {issue!r}. "
            "Disable last_token_only_prefill for this checkpoint."
        )


def gemma4_single_token_quantization_supported(
    layers: Iterable[nn.Module],
    *,
    per_layer_input_observer: Optional[ObserverBase] = None,
) -> bool:
    """Report policy support for manifests without disabling ordinary export."""
    return (
        _gemma4_single_token_quantization_issue(
            layers, per_layer_input_observer=per_layer_input_observer
        )
        is None
    )


def _gemma4_single_token_quantization_issue(
    layers: Iterable[nn.Module],
    *,
    per_layer_input_observer: Optional[ObserverBase],
) -> Optional[str]:
    quantized = False
    for layer in layers:
        decoder = getattr(layer, "wrapped", layer)
        attention_wrapper = getattr(decoder, "self_attn", None)
        attention = getattr(attention_wrapper, "wrapped", attention_wrapper)
        if (
            attention is not None
            and attention.training
            and float(getattr(attention, "attention_dropout", 0.0)) > 0.0
        ):
            raise ValueError(
                "Gemma4 single-token finalize requires attention dropout to be "
                "disabled; use eval() before generation."
            )
        mode = getattr(decoder, "_mode", Mode.NO_QUANT)
        if mode is Mode.CALIB:
            raise ValueError(
                "Gemma4 single-token finalize cannot run during calibration."
            )
        if mode is not Mode.QUANT:
            continue
        quantized = True
        for path, observer in decoder.named_modules():
            if not isinstance(observer, ObserverBase):
                continue
            if path.rsplit(".", 1)[-1] in {"obs_weight", "obs_bias"}:
                continue
            if not _gemma4_single_token_observer_supported(observer):
                return path
    if quantized and per_layer_input_observer is not None:
        if not _gemma4_single_token_observer_supported(per_layer_input_observer):
            return "per_layer_inputs"
    return None


def _gemma4_single_token_observer_supported(observer: ObserverBase) -> bool:
    return isinstance(observer, IdentityObserver) or (
        isinstance(observer, AffineObserverBase)
        and observer.channel_axis is None
        and not observer.qscheme.is_per_channel()
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
) -> torch.Tensor:
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

    start = int(visual_start_idx)
    end = start + expected_len
    if start < 0 or end > seq_len:
        raise ValueError(
            f"Invalid visual slot range [{start}, {end}) for seq_len={seq_len}."
        )

    # Circle does not support zero-length Slice outputs. Build the concatenation
    # from non-empty static ranges so boundary placements such as start=0 do not
    # leave ``slice(0, 0)`` in the exported graph.
    segments: list[torch.Tensor] = []
    if start > 0:
        segments.append(text_embeds[:, :start, :])
    segments.append(visual_embeds)
    if end < seq_len:
        segments.append(text_embeds[:, end:, :])

    if len(segments) == 1:
        return segments[0]
    return torch.cat(segments, dim=1)


def build_decode_attention_mask(
    *,
    batch_size: int,
    past_len: int,
    max_seq: int,
    device: torch.device,
    dtype: torch.dtype,
    mask_value: float,
) -> torch.Tensor:
    """Build a mask for fixed past-cache storage plus one current-token slot.

    The first ``max_seq - 1`` positions correspond to persistent past-cache
    storage. The decode graph appends the current token at position
    ``max_seq - 1``, independent of the logical ``past_len``.
    """

    if past_len < 0 or past_len >= max_seq:
        raise ValueError(
            "past_len must be in [0, max_seq), got "
            f"past_len={past_len}, max_seq={max_seq}."
        )

    mask = torch.full(
        (batch_size, 1, max_seq), float(mask_value), device=device, dtype=dtype
    )
    if past_len > 0:
        mask[:, :, :past_len] = 0.0
    mask[:, :, max_seq - 1] = 0.0
    return mask


def lookup_gemma4_per_layer_token_inputs(
    input_ids: torch.Tensor,
    *,
    embed_tokens_per_layer: nn.Module,
    num_hidden_layers: int,
    hidden_size_per_layer_input: int,
    fq: PLEObserverFn,
    token_inputs_observer: Any,
) -> torch.Tensor:
    """Return the token-identity Per-Layer Embedding (PLE) component.

    This is the shared implementation behind
    ``QuantGemma4TextModel.get_per_layer_inputs`` and the CPU
    ``ple_embedding`` export adapter. The packed lookup
    ``[B, S, L * P]`` is reshaped to ``[B, S, L, P]`` and observed with the
    text-model ``per_layer_token_inputs`` observer.
    """
    result = embed_tokens_per_layer(input_ids).reshape(
        *input_ids.shape,
        int(num_hidden_layers),
        int(hidden_size_per_layer_input),
    )
    return fq(result, token_inputs_observer)


def validate_gemma4_ple_projection_state(model: nn.Module) -> None:
    """Reject wrappers missing the PLE scale buffers or arithmetic observers."""
    required = (
        "per_layer_model_projection_scale_tensor",
        "per_layer_input_scale_tensor",
        "obs_per_layer_model_projection_scale",
        "obs_per_layer_input_scale",
        "obs_per_layer_sum",
    )
    missing = [name for name in required if getattr(model, name, None) is None]
    if missing:
        raise RuntimeError(
            "Gemma4 PLE projection quantization state is missing: "
            f"{', '.join(missing)}. Re-prepare the model and rerun PTQ calibration "
            "before quantization/export."
        )


def validate_gemma4_ple_projection_slicing(model: nn.Module) -> None:
    """Validate the independent layer/P-axis domains of a bounded producer.

    RMSNorm reduces only the final P axis. Frozen scalar activation grids are
    pointwise, so removing tokens or complete layers preserves every boundary.
    Projection weights may use the original per-output-row affine grid.
    Norm weights keep their complete P axis and existing policy, including MX;
    neither their values nor their groups change when layers are removed.
    """
    validate_gemma4_ple_projection_state(model)
    mode = getattr(model, "_mode", Mode.NO_QUANT)
    if mode is Mode.CALIB:
        raise ValueError("Gemma4 PLE slicing cannot run during calibration.")
    projection: Any = getattr(
        model.per_layer_model_projection, "wrapped", model.per_layer_model_projection
    )
    norm: Any = getattr(
        model.per_layer_projection_norm, "wrapped", model.per_layer_projection_norm
    )
    from tico.quantization.wrapq.wrappers.nn.quant_linear import QuantLinear

    if not isinstance(projection, (QuantLinear, nn.Linear)):
        raise ValueError("Gemma4 PLE slicing requires a Linear projection.")
    _validate_gemma4_ple_projection_norm(
        norm, int(getattr(model, "hidden_size_per_layer_input"))
    )
    if mode is not Mode.QUANT:
        return
    observers = {
        name: getattr(model, name)
        for name in (
            "obs_per_layer_token_inputs",
            "obs_per_layer_projection",
            "obs_per_layer_sum",
            "obs_per_layer_inputs",
            "obs_per_layer_model_projection_scale",
            "obs_per_layer_input_scale",
        )
    }
    for name, module in (("projection", projection), ("norm", norm)):
        for path, observer in module.named_modules():
            if (
                isinstance(observer, ObserverBase)
                and path.rsplit(".", 1)[-1] != "obs_weight"
            ):
                observers[f"{name}.{path}"] = observer
    for path, observer in observers.items():
        _require_gemma4_ple_pointwise_activation(observer, path)
    norm_weight_observer = getattr(norm, "obs_weight", None)
    if isinstance(norm_weight_observer, ObserverBase):
        norm_weight_observer.require_frozen()
    if isinstance(projection, QuantLinear):
        observer = projection.obs_weight
        if not isinstance(observer, (IdentityObserver, AffineObserverBase)) or (
            isinstance(observer, AffineObserverBase)
            and observer.channel_axis is not None
            and (not -2 <= observer.channel_axis < 2 or observer.channel_axis % 2 != 0)
        ):
            raise ValueError(
                "Gemma4 PLE slicing supports identity, per-tensor affine or "
                "per-output-row affine projection weights; unsupported weight policy."
            )
        observer.require_frozen()


def validate_gemma4_ple_slicing_state(model: nn.Module) -> None:
    """Validate upstream lookup and projection before selecting a split phase."""
    from tico.quantization.wrapq.wrappers.gemma4.ple_embedding_host import (
        validate_gemma4_ple_embedding_slicing,
    )

    validate_gemma4_ple_projection_slicing(model)
    validate_gemma4_ple_embedding_slicing(
        getattr(model, "embed_tokens_per_layer"),
        getattr(model, "obs_per_layer_token_inputs"),
        quantized=getattr(model, "_mode", Mode.NO_QUANT) is Mode.QUANT,
    )


def _validate_gemma4_ple_projection_norm(norm: nn.Module, width: int) -> None:
    from tico.quantization.wrapq.wrappers.gemma4.quant_rmsnorm import QuantGemma4RMSNorm

    if (
        not isinstance(norm, (QuantGemma4RMSNorm, nn.RMSNorm))
        and type(norm).__name__ != "Gemma4RMSNorm"
    ) or (isinstance(norm, nn.RMSNorm) and tuple(norm.normalized_shape) != (width,)):
        raise ValueError("Gemma4 PLE slicing requires RMSNorm over the final P axis.")
    if getattr(norm, "_mode", Mode.NO_QUANT) is Mode.CALIB:
        raise ValueError("Gemma4 PLE slicing cannot run during calibration.")


def _require_gemma4_ple_pointwise_activation(observer: ObserverBase, path: str) -> None:
    if not _gemma4_single_token_observer_supported(observer):
        raise ValueError(
            "Gemma4 PLE slicing requires frozen per-tensor affine or identity "
            f"activation observers; unsupported policy at {path!r}. "
            "Disable ple_prefill_split for this checkpoint."
        )
    observer.require_frozen()


def _project_gemma4_layer_range(
    inputs_embeds: torch.Tensor,
    projection: nn.Module,
    start: int,
    end: int,
) -> torch.Tensor:
    """Select output rows before Linear, retaining all QuantLinear boundaries."""
    from tico.quantization.wrapq.wrappers.nn.quant_linear import QuantLinear

    linear = getattr(projection, "wrapped", projection)
    if isinstance(linear, nn.Linear):
        bias = linear.bias[start:end] if linear.bias is not None else None
        return F.linear(inputs_embeds, linear.weight[start:end], bias)
    if not isinstance(linear, QuantLinear):
        raise ValueError(
            "Gemma4 PLE layer-range projection requires QuantLinear or Linear."
        )
    linear.require_entry_available()
    if linear._mode is Mode.CALIB:
        raise ValueError("Gemma4 PLE slicing cannot run during calibration.")
    if linear._mode is Mode.QUANT:
        for name in ("obs_act_in", "obs_act_out"):
            _require_gemma4_ple_pointwise_activation(
                getattr(linear, name), f"projection.{name}"
            )
    x = linear._fq(inputs_embeds, linear.obs_act_in)
    weight = linear.module.weight[start:end]
    if linear._mode is Mode.QUANT:
        observer = linear.obs_weight
        if isinstance(observer, IdentityObserver) or (
            isinstance(observer, AffineObserverBase) and observer.channel_axis is None
        ):
            weight = observer.fake_quant(weight)
        elif (
            isinstance(observer, AffineObserverBase)
            and observer.channel_axis is not None
            and -2 <= observer.channel_axis < 2
            and observer.channel_axis % 2 == 0
        ):
            state = observer.require_frozen()
            observer.validate_execution(weight.device)
            if state.fake_quant_enabled:
                assert state.scale is not None and state.zero_point is not None
                weight = torch.fake_quantize_per_channel_affine(
                    weight,
                    state.scale[start:end].to(weight.device),
                    state.zero_point[start:end].to(weight.device, dtype=torch.int),
                    axis=0,
                    quant_min=observer.dtype.qmin,
                    quant_max=observer.dtype.qmax,
                )
        else:
            raise ValueError(
                "Unsupported Gemma4 PLE projection weight policy for layer slicing."
            )
    bias = linear.module.bias[start:end] if linear.module.bias is not None else None
    return linear._fq(F.linear(x, weight, bias), linear.obs_act_out)


def project_gemma4_per_layer_inputs(
    inputs_embeds: torch.Tensor,
    per_layer_token_inputs: Optional[torch.Tensor],
    *,
    projection: nn.Module,
    projection_norm: nn.Module,
    projection_scale: float | torch.Tensor,
    input_scale: float | torch.Tensor,
    num_hidden_layers: int,
    hidden_size_per_layer_input: int,
    fq: PLEObserverFn,
    projection_observer: Any,
    sum_observer: Any,
    inputs_observer: Any,
    layer_start: int = 0,
    layer_end: Optional[int] = None,
) -> torch.Tensor:
    """Project decoder inputs and combine them with token-identity PLE.

    This is the shared implementation behind
    ``QuantGemma4TextModel.project_per_layer_inputs`` and the NPU
    ``ple_projection`` export adapter. The numerical contract is:

    ``projected = norm(reshape(projection(inputs_embeds) * projection_scale))``
    ``projected = fq(projected, projection_observer)``
    ``summed = fq(projected + per_layer_token_inputs, sum_observer)``
    ``combined = fq(summed * input_scale, inputs_observer)``

    Callers supply observed scale tensors in QUANT mode and retain the original
    Python scalars in NO_QUANT/CALIB mode. The ADD output has its own range;
    the final observer remains the producer/decoder boundary after scaling.

    Only ``inputs_embeds`` passes through the projection; the token-identity
    tensor bypasses it and is added after normalization and observation. When
    ``per_layer_token_inputs`` is ``None`` (Hugging Face ``inputs_embeds``-only
    entry), the observed projection is returned without the final scale.

    ``[layer_start, layer_end)`` selects projection output rows before Linear;
    the token tensor must already have that layer extent. The default keeps the
    full eager/calibration computation and its existing observer lifecycle.
    """
    end = int(num_hidden_layers) if layer_end is None else int(layer_end)
    start = int(layer_start)
    if not 0 <= start < end <= int(num_hidden_layers):
        raise ValueError(f"Invalid Gemma4 PLE layer range [{start}, {end}).")
    if per_layer_token_inputs is not None and (
        start != 0 or end != int(num_hidden_layers)
    ):
        expected_shape = (
            *inputs_embeds.shape[:-1],
            end - start,
            int(hidden_size_per_layer_input),
        )
        if tuple(per_layer_token_inputs.shape) != expected_shape:
            raise ValueError(
                f"Gemma4 PLE token inputs expected shape {expected_shape}, got {tuple(per_layer_token_inputs.shape)}."
            )
    if start == 0 and end == int(num_hidden_layers):
        projected = projection(inputs_embeds)
    else:
        width = int(hidden_size_per_layer_input)
        linear = getattr(projection, "wrapped", projection)
        norm = getattr(projection_norm, "wrapped", projection_norm)
        _validate_gemma4_ple_projection_norm(norm, width)
        if getattr(linear, "_mode", Mode.NO_QUANT) is Mode.QUANT:
            for path, observer in (
                ("per_layer_projection", projection_observer),
                ("per_layer_inputs", inputs_observer),
            ):
                _require_gemma4_ple_pointwise_activation(observer, path)
            if per_layer_token_inputs is not None:
                _require_gemma4_ple_pointwise_activation(sum_observer, "per_layer_sum")
        if getattr(norm, "_mode", Mode.NO_QUANT) is Mode.QUANT:
            for path, observer in norm.named_modules():
                if (
                    isinstance(observer, ObserverBase)
                    and path.rsplit(".", 1)[-1] != "obs_weight"
                ):
                    _require_gemma4_ple_pointwise_activation(observer, f"norm.{path}")
        projected = _project_gemma4_layer_range(
            inputs_embeds, projection, start * width, end * width
        )
    if isinstance(projection_scale, torch.Tensor):
        projection_scale = projection_scale.to(
            dtype=projected.dtype, device=projected.device
        )
    projected = projected * projection_scale
    projected = projected.reshape(
        *inputs_embeds.shape[:-1],
        end - start,
        int(hidden_size_per_layer_input),
    )
    projected = projection_norm(projected)
    projected = fq(projected, projection_observer)
    if per_layer_token_inputs is None:
        return fq(projected, inputs_observer)
    summed = fq(projected + per_layer_token_inputs, sum_observer)
    if isinstance(input_scale, torch.Tensor):
        input_scale = input_scale.to(dtype=summed.dtype, device=summed.device)
    combined = summed * input_scale
    return fq(combined, inputs_observer)


def extract_text_config(config: Any) -> Any:
    """Return the Gemma4 text config from a model or config object."""

    config = getattr(config, "config", config)
    if hasattr(config, "get_text_config"):
        return config.get_text_config()
    return getattr(config, "text_config", config)


def gemma4_layer_type_head_dim(text_config: Any, layer_type: str) -> int:
    """Return the attention head dim used by one Gemma4 text layer type.

    Mirrors ``Gemma4TextAttention.__init__``. transformers 5.15+ consumes the
    ``global_head_dim`` constructor argument into a full-attention override in
    ``per_layer_config`` and no longer exposes it as a plain attribute, so the
    per-layer view is authoritative whenever the config owns a layer of the
    requested type. Older configs expose ``global_head_dim`` directly; sliding
    layers always use ``head_dim``.
    """
    per_layer_config = getattr(text_config, "per_layer_config", None)
    layer_types = tuple(getattr(text_config, "layer_types", ()) or ())
    if per_layer_config is not None and layer_type in layer_types:
        return int(per_layer_config[layer_type].head_dim)
    global_head_dim = getattr(text_config, "global_head_dim", None)
    if layer_type == "full_attention" and global_head_dim:
        return int(global_head_dim)
    return int(text_config.head_dim)


def ensure_static_shape(
    name: str, tensor: torch.Tensor, expected: Iterable[int]
) -> None:
    """Validate that a tensor has the expected static shape."""

    expected_tuple = tuple(int(v) for v in expected)
    actual_tuple = tuple(int(v) for v in tensor.shape)
    if actual_tuple != expected_tuple:
        raise ValueError(f"{name} expected shape {expected_tuple}, got {actual_tuple}.")
