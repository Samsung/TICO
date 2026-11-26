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

"""Static per-stage Circle export for dense Gemma4 E2B/E4B."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, cast, Mapping

import torch

from tico.config import CompileConfigBase

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.recipes.export.circle import (
    convert_and_save_circle as _convert_and_save,
)
from tico.quantization.wrapq.observers.base import ObserverBase
from tico.quantization.wrapq.utils.static_export import ExportCapabilityGuard
from tico.quantization.wrapq.wrap_helper import PTQWrapHelper
from tico.quantization.wrapq.wrappers.gemma4.export_adapters import (
    build_gemma4_vision_prefill_export_module,
    Gemma4LMHeadExportAdapter,
    Gemma4MMFusionExportAdapter,
    Gemma4PLEEmbeddingExportAdapter,
    Gemma4PLEProjectionExportAdapter,
    Gemma4TokenEmbeddingExportAdapter,
)
from tico.quantization.wrapq.wrappers.gemma4.ple_embedding_host import (
    CIRCLE_FLATBUFFER_LIMIT_BYTES,
    estimate_gemma4_ple_embedding_circle_bytes,
    save_gemma4_ple_embedding_artifact,
)
from tico.quantization.wrapq.wrappers.gemma4.quant_for_conditional_generation import (
    QuantGemma4ForConditionalGeneration,
)
from tico.quantization.wrapq.wrappers.gemma4.quant_model import QuantGemma4Model
from tico.quantization.wrapq.wrappers.gemma4.quant_text_attention import (
    QuantGemma4TextAttention,
)
from tico.quantization.wrapq.wrappers.gemma4.quant_text_decoder_layer import (
    QuantGemma4TextDecoderLayer,
)
from tico.quantization.wrapq.wrappers.gemma4.quant_text_model import (
    QuantGemma4TextModel,
)
from tico.quantization.wrapq.wrappers.gemma4.quant_vision_model import (
    QuantGemma4VisionModel,
)
from tico.quantization.wrapq.wrappers.gemma4.static_vision_profile import (
    build_gemma4_static_vision_profile,
    canonicalize_gemma4_static_vision_model_args,
    Gemma4StaticVisionProfile,
)
from tico.quantization.wrapq.wrappers.gemma4.utils import (
    assert_gemma4_dense_no_moe,
    gemma4_decode_capacity,
    gemma4_single_token_quantization_supported,
    resolve_gemma4_shared_kv_suffix,
    validate_gemma4_compact_decode_quantization,
)
from tico.quantization.wrapq.wrappers.gemma4.vision_split_export import (
    build_gemma4_vision_split_export_bundle,
    Gemma4VisionSplitExportBundle,
)
from tico.quantization.wrapq.wrappers.llama.export_adapters import (
    make_token_embedding_dynamic_shapes,
    register_fake_quant_meta_kernels_for_dynamic_export,
)
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper

_VISION_EXPORT_GRANULARITIES = frozenset({"monolithic", "encoder", "layer", "both"})
_PLE_EMBEDDING_FORMATS = frozenset({"auto", "circle", "pt"})


def _is_wrapped_export_model(model: torch.nn.Module) -> bool:
    """Return whether a model already exposes the PTQ wrapper export layout."""
    wrapped = getattr(model, "wrapped", None)
    return (
        wrapped is not None
        and hasattr(wrapped, "model")
        and hasattr(wrapped, "lm_head")
    )


_FLOAT_ARTIFACT_TAG = "f32"
_QUANTIZED_ARTIFACT_TAG = "q"


def _float_artifact_tag(model: torch.nn.Module) -> str:
    """Validate a floating-point export model and return its precision tag."""
    try:
        dtype = next(model.parameters()).dtype
    except StopIteration:
        dtype = torch.float32

    if dtype is not torch.float32:
        raise TypeError(
            "Floating-point Gemma4 export currently supports float32 only. "
            f"Got parameter dtype {dtype}."
        )
    return _FLOAT_ARTIFACT_TAG


def _normalize_model_args(
    model_args: Mapping[str, Any] | None,
    *,
    max_seq_len: int,
) -> dict[str, Any]:
    """Normalize the fixed Gemma4 runtime contract used by export wrappers."""
    normalized = canonicalize_gemma4_static_vision_model_args(model_args)

    vision = normalized.setdefault("vision", {})
    if not isinstance(vision, dict):
        raise TypeError("model_args.vision must be a mapping.")

    if "visual_start_idx" not in vision:
        raise ValueError(
            "Gemma4 Circle export requires model_args.vision.visual_start_idx."
        )
    vision["visual_start_idx"] = int(vision["visual_start_idx"])
    if vision["visual_start_idx"] < 0:
        raise ValueError("model_args.vision.visual_start_idx must be non-negative.")

    if "num_visual_tokens" not in vision:
        raise ValueError(
            "Gemma4 Circle export requires model_args.vision.num_visual_tokens."
        )
    vision["num_visual_tokens"] = int(vision["num_visual_tokens"])
    if vision["num_visual_tokens"] <= 0:
        raise ValueError("model_args.vision.num_visual_tokens must be positive.")

    if "max_soft_tokens" in vision:
        vision["max_soft_tokens"] = int(vision["max_soft_tokens"])
        if vision["max_soft_tokens"] <= 0:
            raise ValueError("model_args.vision.max_soft_tokens must be positive.")

    for key in ("patch_grid_height", "patch_grid_width"):
        if key not in vision:
            raise ValueError(f"Gemma4 Circle export requires model_args.vision.{key}.")
        vision[key] = int(vision[key])
        if vision[key] <= 0:
            raise ValueError(f"model_args.vision.{key} must be positive.")

    text = normalized.setdefault("text", {})
    if not isinstance(text, dict):
        raise TypeError("model_args.text must be a mapping.")
    configured_max_seq = text.get("max_seq")
    if configured_max_seq is not None and int(configured_max_seq) != max_seq_len:
        raise ValueError(
            "model_args.text.max_seq must match export.max_seq_len: "
            f"text.max_seq={int(configured_max_seq)}, "
            f"export.max_seq_len={max_seq_len}."
        )
    text["max_seq"] = int(max_seq_len)

    return normalized


def _prepare_gemma4_export_model(
    model: torch.nn.Module,
    model_args: Mapping[str, Any],
) -> tuple[torch.nn.Module, str]:
    """Normalize a checkpoint or FP model for staged Gemma4 export.

    Floating-point models are structurally wrapped in ``NO_QUANT`` mode. No
    calibration or fake quantization is introduced. Converted checkpoints keep
    their existing quantization state.
    """
    model = model.eval().cpu()
    if _is_wrapped_export_model(model):
        return model, _QUANTIZED_ARTIFACT_TAG

    artifact_tag = _float_artifact_tag(model)
    wrapper_config = PTQConfig(
        model_args=deepcopy(dict(model_args)),
        strict_wrap=True,
    )
    export_model = PTQWrapHelper(strict_wrap=True).wrap_supported(
        model,
        wrapper_config,
    )
    if not _is_wrapped_export_model(export_model):
        raise TypeError(
            "Gemma4 staged export requires a top-level PTQ wrapper exposing "
            "the wrapped model and LM head."
        )
    return export_model, artifact_tag


def _circle_name(stem: str, artifact_tag: str) -> str:
    """Build a Circle artifact name with an explicit precision tag."""
    return f"{stem}.{artifact_tag}.circle"


def _unwrap_gemma4_components(
    export_model: torch.nn.Module,
) -> tuple[
    QuantGemma4ForConditionalGeneration,
    QuantGemma4Model,
    QuantGemma4VisionModel,
    QuantGemma4TextModel,
]:
    """Return top-level, multimodal, vision, and text Gemma4 wrappers."""
    qmodel = cast(
        QuantGemma4ForConditionalGeneration, cast(PTQWrapper, export_model).wrapped
    )
    gemma_model = cast(QuantGemma4Model, qmodel.model.wrapped)
    if getattr(gemma_model, "vision_tower", None) is None:
        raise ValueError("Gemma4 Circle export requires an image vision tower.")
    assert gemma_model.vision_tower is not None
    qvision = cast(QuantGemma4VisionModel, gemma_model.vision_tower.wrapped)
    qtext = cast(QuantGemma4TextModel, gemma_model.language_model.wrapped)
    return qmodel, gemma_model, qvision, qtext


def _resolve_vision_contract(
    *,
    gemma_model: torch.nn.Module,
    qvision: torch.nn.Module,
    model_args: Mapping[str, Any],
    max_seq_len: int,
) -> Gemma4StaticVisionProfile:
    """Build and validate the canonical static vision profile."""
    profile = build_gemma4_static_vision_profile(
        model_args,
        vision_config=qvision.config,
        max_seq_len=max_seq_len,
    )
    profile.validate_wrapped_model(gemma_model)
    return profile


def _make_vision_inputs(
    *,
    profile: Gemma4StaticVisionProfile,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create padded patch values and canonical 2-D position IDs."""
    pixel_position_ids = profile.build_image_position_ids()
    pixel_values = torch.rand(
        1,
        profile.num_patches,
        profile.patch_vector_size,
        device="cpu",
    )
    if profile.num_padding_patches:
        pixel_values[:, profile.num_valid_patches :, :] = 0.0
    return pixel_values, pixel_position_ids


def _normalize_vision_export_granularity(granularity: str) -> str:
    """Validate and normalize the Gemma4 vision export granularity."""
    normalized = str(granularity).strip().lower()
    if normalized not in _VISION_EXPORT_GRANULARITIES:
        choices = ", ".join(sorted(_VISION_EXPORT_GRANULARITIES))
        raise ValueError(
            "Unsupported Gemma4 vision export granularity "
            f"{granularity!r}. Expected one of: {choices}."
        )
    return normalized


def _vision_encoder_stage_stem(layer_index: int) -> str:
    """Return the artifact stem for one vision encoder layer."""
    if layer_index < 0:
        raise ValueError(
            f"Gemma4 vision layer index must be non-negative, got {layer_index}."
        )
    return f"vision_encoder_layer_{layer_index:02d}"


def _tensor_manifest(name: str, tensor: torch.Tensor) -> dict[str, Any]:
    """Return a JSON-safe tensor contract for a split vision manifest."""
    return {
        "name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
    }


def _small_tensor_values(tensor: torch.Tensor) -> dict[str, Any]:
    """Return compact JSON metadata for an observer parameter tensor."""
    tensor = tensor.detach().cpu()
    metadata: dict[str, Any] = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "sha256": hashlib.sha256(
            tensor.reshape(-1).contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest(),
    }
    if tensor.numel() <= 16:
        metadata["values"] = tensor.reshape(-1).tolist()
    return metadata


def _observer_manifest(observer: ObserverBase) -> dict[str, Any]:
    """Return the quantization contract owned by one boundary observer."""
    contract: dict[str, Any] = {
        "name": str(getattr(observer, "name", type(observer).__name__)),
        "type": type(observer).__name__,
    }
    dtype = getattr(observer, "dtype", None)
    if dtype is not None:
        contract["dtype"] = str(dtype)
    qscheme = getattr(observer, "qscheme", None)
    if qscheme is not None:
        contract["qscheme"] = str(qscheme)
    channel_axis = getattr(observer, "channel_axis", None)
    if channel_axis is not None:
        contract["channel_axis"] = int(channel_axis)

    snapshot = observer.frozen_state
    contract.update(
        {
            "enabled": bool(observer.enabled),
            "fake_quant_enabled": bool(observer.fake_quant_enabled),
            "frozen": snapshot is not None,
        }
    )
    if snapshot is not None and snapshot.scale is not None:
        assert snapshot.zero_point is not None
        contract["scale"] = _small_tensor_values(snapshot.scale)
        contract["zero_point"] = _small_tensor_values(snapshot.zero_point)
    return contract


def _save_vision_context(
    *,
    path: Path,
    bundle: Gemma4VisionSplitExportBundle,
) -> None:
    """Save the shared raw mask and RoPE tensors used by every encoder stage."""
    torch.save(
        {
            "attention_mask": bundle.attention_mask,
            "position_embeddings_cos": bundle.position_embeddings_cos,
            "position_embeddings_sin": bundle.position_embeddings_sin,
        },
        path,
    )


def _save_vision_pipeline_manifest(
    *,
    path: Path,
    profile: Gemma4StaticVisionProfile,
    artifact_tag: str,
    granularity: str,
    bundle: Gemma4VisionSplitExportBundle,
    stage_artifacts: list[dict[str, Any]],
    monolithic_artifact: str | None,
    context_artifact: str,
) -> None:
    """Write the execution and shared-input contract for split vision graphs."""
    manifest = {
        "schema_version": 1,
        "profile": profile.name,
        "static_profile": profile.to_manifest(),
        "artifact_tag": artifact_tag,
        "granularity": granularity,
        "context_externalized": True,
        "direct_chainable_boundaries": True,
        "monolithic_artifact": monolithic_artifact,
        "shared_encoder_inputs_artifact": context_artifact,
        "shared_encoder_inputs": [
            _tensor_manifest("attention_mask", bundle.attention_mask),
            _tensor_manifest(
                "position_embeddings_cos",
                bundle.position_embeddings_cos,
            ),
            _tensor_manifest(
                "position_embeddings_sin",
                bundle.position_embeddings_sin,
            ),
        ],
        "shared_encoder_input_observers": {
            name: _observer_manifest(observer)
            for name, observer in (
                ("attention_mask", bundle.encoder.attention_mask_observer),
                ("position_embeddings_cos", bundle.encoder.position_cos_observer),
                ("position_embeddings_sin", bundle.encoder.position_sin_observer),
            )
        },
        "quantized_export": bundle.encoder.quantized,
        "boundaries": [
            {
                "name": boundary.name,
                "producer": boundary.producer,
                "consumer": boundary.consumer,
                "observer": _observer_manifest(boundary.observer),
            }
            for boundary in (
                bundle.encoder_boundary_contracts
                if granularity == "encoder"
                else bundle.boundary_contracts
            )
        ],
        "stages": stage_artifacts,
    }
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _export_gemma4_vision_stages(
    *,
    gemma_model: torch.nn.Module,
    qvision: torch.nn.Module,
    profile: Gemma4StaticVisionProfile,
    pixel_values: torch.Tensor,
    pixel_position_ids: torch.Tensor,
    text_hidden_size: int,
    artifact_tag: str,
    output_dir: Path,
    granularity: str,
    spatial_pooling: bool,
    strict: bool,
    compile_config: CompileConfigBase | None = None,
) -> None:
    """Export the selected Gemma4 static vision pipeline composition."""
    granularity = _normalize_vision_export_granularity(granularity)

    if granularity == "monolithic":
        vision_prefill = build_gemma4_vision_prefill_export_module(
            gemma_model,
            pixel_position_ids=pixel_position_ids,
            spatial_pooling=spatial_pooling,
        )
        _convert_and_save(
            vision_prefill,
            (pixel_values,),
            output_dir / _circle_name("vision_prefill", artifact_tag),
            strict=strict,
            config=compile_config,
        )
        return

    bundle = build_gemma4_vision_split_export_bundle(
        gemma_model,
        pixel_position_ids=pixel_position_ids,
        output_dtype=pixel_values.dtype,
        spatial_pooling=spatial_pooling,
    )
    if bundle.post_projection.num_valid_pool_outputs != profile.num_visual_tokens:
        raise RuntimeError(
            "Gemma4 split vision valid-output count does not match the static "
            "profile: "
            f"module={bundle.post_projection.num_valid_pool_outputs}, "
            f"profile={profile.num_visual_tokens}."
        )

    context_artifact = "vision_context.pt"
    _save_vision_context(
        path=output_dir / context_artifact,
        bundle=bundle,
    )

    monolithic_artifact: str | None = None
    if granularity == "both":
        monolithic_artifact = _circle_name("vision_prefill", artifact_tag)
        _convert_and_save(
            bundle.monolithic,
            (pixel_values,),
            output_dir / monolithic_artifact,
            strict=strict,
            config=compile_config,
        )

    qvision = cast(QuantGemma4VisionModel, qvision)
    vision_hidden_size = int(qvision.config.hidden_size)
    hidden_states = torch.randn(
        1,
        profile.num_patches,
        vision_hidden_size,
        device="cpu",
    )
    pooled_hidden_states = torch.randn(
        1,
        profile.max_soft_tokens,
        vision_hidden_size,
        device="cpu",
    )

    stage_artifacts: list[dict[str, Any]] = []

    patch_artifact = _circle_name("vision_patch_embedder", artifact_tag)
    _convert_and_save(
        bundle.patch_embedder,
        (pixel_values,),
        output_dir / patch_artifact,
        strict=strict,
        config=compile_config,
    )
    stage_artifacts.append(
        {
            "id": "patch_embedder",
            "kind": "patch_embedder",
            "artifact": patch_artifact,
            "inputs": [_tensor_manifest("pixel_values", pixel_values)],
            "outputs": [
                {
                    "name": "hidden_states",
                    "shape": [1, profile.num_patches, vision_hidden_size],
                    "compute_dtype": str(hidden_states.dtype).removeprefix("torch."),
                }
            ],
        }
    )

    shared_context = (
        bundle.attention_mask,
        bundle.position_embeddings_cos,
        bundle.position_embeddings_sin,
    )
    encoder_stages = (
        [
            (
                "encoder",
                "vision_encoder",
                bundle.encoder,
                {"num_layers": len(bundle.encoder_layers)},
            )
        ]
        if granularity == "encoder"
        else [
            (
                f"encoder_layer_{layer_index}",
                _vision_encoder_stage_stem(layer_index),
                layer,
                {"layer_index": layer_index},
            )
            for layer_index, layer in enumerate(bundle.encoder_layers)
        ]
    )
    for stage_id, stem, encoder_stage, metadata in encoder_stages:
        artifact = _circle_name(stem, artifact_tag)
        _convert_and_save(
            encoder_stage,
            (hidden_states, *shared_context),
            output_dir / artifact,
            strict=strict,
            config=compile_config,
        )
        stage_artifacts.append(
            {
                "id": stage_id,
                "kind": "encoder" if granularity == "encoder" else "encoder_layer",
                "artifact": artifact,
                **metadata,
                "inputs": [
                    {
                        "name": "hidden_states",
                        "shape": list(hidden_states.shape),
                        "compute_dtype": str(hidden_states.dtype).removeprefix(
                            "torch."
                        ),
                    },
                    *[
                        _tensor_manifest(name, tensor)
                        for name, tensor in zip(
                            (
                                "attention_mask",
                                "position_embeddings_cos",
                                "position_embeddings_sin",
                            ),
                            shared_context,
                        )
                    ],
                ],
                "outputs": [
                    {
                        "name": "hidden_states",
                        "shape": list(hidden_states.shape),
                        "compute_dtype": str(hidden_states.dtype).removeprefix(
                            "torch."
                        ),
                    }
                ],
            }
        )

    pooler_artifact = _circle_name("vision_pooler", artifact_tag)
    _convert_and_save(
        bundle.pooler,
        (hidden_states,),
        output_dir / pooler_artifact,
        strict=strict,
        config=compile_config,
    )
    stage_artifacts.append(
        {
            "id": "pooler",
            "kind": "pooler",
            "artifact": pooler_artifact,
            "inputs": [
                {
                    "name": "hidden_states",
                    "shape": list(hidden_states.shape),
                    "compute_dtype": str(hidden_states.dtype).removeprefix("torch."),
                }
            ],
            "outputs": [
                {
                    "name": "pooled_hidden_states",
                    "shape": list(pooled_hidden_states.shape),
                    "compute_dtype": str(pooled_hidden_states.dtype).removeprefix(
                        "torch."
                    ),
                }
            ],
        }
    )

    post_projection_artifact = _circle_name("vision_post_projection", artifact_tag)
    _convert_and_save(
        bundle.post_projection,
        (pooled_hidden_states,),
        output_dir / post_projection_artifact,
        strict=strict,
        config=compile_config,
    )
    stage_artifacts.append(
        {
            "id": "post_projection",
            "kind": "post_projection",
            "artifact": post_projection_artifact,
            "inputs": [
                {
                    "name": "pooled_hidden_states",
                    "shape": list(pooled_hidden_states.shape),
                    "compute_dtype": str(pooled_hidden_states.dtype).removeprefix(
                        "torch."
                    ),
                }
            ],
            "outputs": [
                {
                    "name": "visual_embeds",
                    "shape": [profile.num_visual_tokens, text_hidden_size],
                    "compute_dtype": str(pixel_values.dtype).removeprefix("torch."),
                }
            ],
        }
    )

    _save_vision_pipeline_manifest(
        path=output_dir / "vision_pipeline.json",
        profile=profile,
        artifact_tag=artifact_tag,
        granularity=granularity,
        bundle=bundle,
        stage_artifacts=stage_artifacts,
        monolithic_artifact=monolithic_artifact,
        context_artifact=context_artifact,
    )


def _make_prefill_attention_mask(
    *,
    max_seq_len: int,
    sliding_window: int | None,
) -> torch.Tensor:
    """Create a full or sliding additive causal mask for tracing."""
    query_positions = torch.arange(max_seq_len).unsqueeze(1)
    key_positions = torch.arange(max_seq_len).unsqueeze(0)
    allowed = key_positions <= query_positions
    if sliding_window is not None:
        allowed = allowed & (key_positions > query_positions - sliding_window)

    mask = torch.full((max_seq_len, max_seq_len), -120.0)
    mask.masked_fill_(allowed, 0.0)
    return mask.unsqueeze(0).unsqueeze(0)


def _make_decode_attention_mask(
    *,
    max_seq_len: int,
    sliding_window: int | None,
) -> torch.Tensor:
    """Create an additive single-token decode mask at maximum cache length."""
    mask = torch.full((1, 1, max_seq_len), -120.0)
    start = 0 if sliding_window is None else max(0, max_seq_len - sliding_window)
    mask[..., start:max_seq_len] = 0.0
    return mask


def _make_position_embeddings(
    *,
    seq_len: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create example Gemma4 RoPE cosine and sine tensors."""
    return (
        torch.randn(1, seq_len, head_dim, device="cpu"),
        torch.randn(1, seq_len, head_dim, device="cpu"),
    )


def _attention_contract(
    layer: torch.nn.Module,
    *,
    max_seq_len: int,
) -> tuple[torch.nn.Module, int, int, int | None, bool]:
    """Return attention dimensions and sharing mode for one text layer."""
    decoder = cast(QuantGemma4TextDecoderLayer, cast(PTQWrapper, layer).wrapped)
    attention = cast(QuantGemma4TextAttention, decoder.self_attn.wrapped)
    attention_capacity = int(getattr(attention, "max_seq", max_seq_len))
    if max_seq_len > attention_capacity:
        raise ValueError(
            "max_seq_len exceeds the wrapped Gemma4 attention capacity: "
            f"layer={int(getattr(attention, 'layer_idx', -1))}, "
            f"max_seq_len={max_seq_len}, capacity={attention_capacity}."
        )

    num_heads = int(attention.config.num_attention_heads)
    num_kv_groups = int(attention.num_key_value_groups)
    if num_kv_groups <= 0 or num_heads % num_kv_groups:
        raise ValueError(
            "Invalid Gemma4 grouped-query attention contract: "
            f"num_heads={num_heads}, num_key_value_groups={num_kv_groups}."
        )
    num_kv_heads = num_heads // num_kv_groups
    head_dim = int(attention.head_dim)
    sliding_window = (
        int(cast(int, attention.sliding_window))
        if bool(getattr(attention, "is_sliding", False))
        else None
    )
    is_shared = bool(getattr(attention, "is_kv_shared_layer", False))
    return attention, num_kv_heads, head_dim, sliding_window, is_shared


def _make_ple_projection_inputs(
    *,
    seq_len: int,
    hidden_size: int,
    num_hidden_layers: int,
    ple_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create ``(inputs_embeds, per_layer_token_inputs)`` for one PLE graph."""
    inputs_embeds = torch.randn(1, seq_len, hidden_size, device="cpu")
    per_layer_token_inputs = torch.randn(
        1,
        seq_len,
        num_hidden_layers,
        ple_dim,
        device="cpu",
    )
    return inputs_embeds, per_layer_token_inputs


def _normalize_ple_embedding_format(embedding_format: str) -> str:
    """Validate and normalize the ``ple_embedding`` artifact format."""
    normalized = str(embedding_format).strip().lower()
    if normalized not in _PLE_EMBEDDING_FORMATS:
        choices = ", ".join(sorted(_PLE_EMBEDDING_FORMATS))
        raise ValueError(
            "Unsupported Gemma4 ple_embedding format "
            f"{embedding_format!r}. Expected one of: {choices}."
        )
    return normalized


def _resolve_ple_embedding_format(
    embedding_format: str,
    *,
    estimated_circle_bytes: int,
) -> str:
    """Return ``circle`` or ``pt`` for the shared PLE lookup artifact.

    ``auto`` writes a Circle graph when the packed table fits the flatbuffer
    limit and falls back to the host ``.pt`` artifact otherwise. Both float and
    quantized exports follow the same rule.
    """
    normalized = _normalize_ple_embedding_format(embedding_format)
    if normalized != "auto":
        return normalized
    if estimated_circle_bytes >= CIRCLE_FLATBUFFER_LIMIT_BYTES:
        return "pt"
    return "circle"


def _export_gemma4_ple_embedding_stage(
    *,
    qtext: torch.nn.Module,
    max_seq_len: int,
    vocab_size_per_layer_input: int,
    artifact_tag: str,
    output_dir: Path,
    embedding_format: str,
    strict: bool,
    compile_config: CompileConfigBase | None = None,
) -> dict[str, Any]:
    """Export the shared PLE lookup as a dynamic Circle graph or ``.pt`` table.

    Returns the manifest entry describing the chosen format and the
    ``(1, S)`` input contract shared with ``token_embedding``.
    """
    ple_embedding = Gemma4PLEEmbeddingExportAdapter(qtext)
    estimated_bytes = estimate_gemma4_ple_embedding_circle_bytes(ple_embedding)
    resolved = _resolve_ple_embedding_format(
        embedding_format,
        estimated_circle_bytes=estimated_bytes,
    )

    manifest: dict[str, Any] = {
        "id": "ple_embedding",
        "kind": "ple_embedding",
        "format": resolved,
        "requested_format": _normalize_ple_embedding_format(embedding_format),
        "estimated_circle_bytes": int(estimated_bytes),
        "layer_range_lookup": resolved == "pt",
        "shared_by": ["prefill", "decode"],
        "circle_flatbuffer_limit_bytes": int(CIRCLE_FLATBUFFER_LIMIT_BYTES),
        "inputs": [
            {
                "name": "input_ids",
                "shape": [1, "S"],
                "dtype": "int64",
                "sequence_range": [1, int(max_seq_len)],
            }
        ],
        "outputs": [
            {
                "name": "per_layer_token_inputs",
                "shape": [
                    1,
                    "S",
                    int(ple_embedding.num_hidden_layers),
                    int(ple_embedding.hidden_size_per_layer_input),
                ],
            }
        ],
    }

    if resolved == "pt":
        if manifest["requested_format"] == "auto":
            print(
                "ple_embedding table is estimated at "
                f"{estimated_bytes / 2**30:.2f} GiB, above the Circle flatbuffer "
                "limit; saving the host .pt artifact instead."
            )
        artifact = f"ple_embedding.{artifact_tag}.pt"
        save_gemma4_ple_embedding_artifact(ple_embedding, output_dir / artifact)
        manifest["artifact"] = artifact
        return manifest

    artifact = _circle_name("ple_embedding", artifact_tag)
    ple_input_ids = torch.randint(
        low=0,
        high=vocab_size_per_layer_input,
        size=(1, max_seq_len),
        dtype=torch.long,
        device="cpu",
    )
    _convert_and_save(
        ple_embedding,
        (ple_input_ids,),
        output_dir / artifact,
        dynamic_shapes=make_token_embedding_dynamic_shapes(max_seq_len),
        strict=strict,
        config=compile_config,
    )
    manifest["artifact"] = artifact
    return manifest


def _export_gemma4_ple_projection_stages(
    *,
    qtext: torch.nn.Module,
    max_seq_len: int,
    hidden_size: int,
    ple_dim: int,
    artifact_tag: str,
    output_dir: Path,
    prefill_decode: bool,
    strict: bool,
    compile_config: CompileConfigBase | None = None,
    suffix_start: int | None = None,
) -> list[dict[str, Any]]:
    """Export the fixed-shape PLE projection graphs for prefill and decode.

    A single adapter instance is traced once per static sequence length. The
    projection graphs never contain ``embed_tokens_per_layer``; the token
    lookup is the separate shared ``ple_embedding`` artifact. Returns one
    manifest entry per exported graph.
    """
    num_hidden_layers = int(getattr(qtext, "config").num_hidden_layers)
    ple_projection = Gemma4PLEProjectionExportAdapter(qtext)
    stages: list[dict[str, Any]] = []
    projection = getattr(
        qtext.per_layer_model_projection, "wrapped", qtext.per_layer_model_projection
    )

    def tensor_contract(
        name: str, tensor: torch.Tensor, observer: ObserverBase | None, owner: str
    ) -> dict[str, Any]:
        contract = _tensor_manifest(name, tensor)
        contract["example_dtype"] = contract["dtype"]
        contract["dtype_semantics"] = "floating fake-quant example"
        contract["qparam_owner"] = owner
        snapshot = None if observer is None else observer.frozen_state
        contract["abi_dtype"] = (
            str(observer.dtype)
            if ple_projection.quantized
            and observer is not None
            and observer.fake_quant_enabled
            and snapshot is not None
            and snapshot.scale is not None
            else contract["dtype"]
        )
        if observer is not None:
            contract["observer"] = _observer_manifest(observer)
        return contract

    def _export(
        stem: str,
        seq_len: int,
        stage_id: str,
        layer_start: int = 0,
        layer_end: int = num_hidden_layers,
    ) -> None:
        adapter = (
            ple_projection
            if layer_start == 0 and layer_end == num_hidden_layers
            else Gemma4PLEProjectionExportAdapter(
                qtext, layer_start=layer_start, layer_end=layer_end
            )
        )
        inputs = _make_ple_projection_inputs(
            seq_len=seq_len,
            hidden_size=hidden_size,
            num_hidden_layers=layer_end - layer_start,
            ple_dim=ple_dim,
        )
        artifact = _circle_name(stem, artifact_tag)
        _convert_and_save(
            adapter,
            inputs,
            output_dir / artifact,
            strict=strict,
            config=compile_config,
        )
        stages.append(
            {
                "id": stage_id,
                "kind": "ple_projection",
                "format": "circle",
                "artifact": artifact,
                "phase": (
                    "prefill"
                    if stage_id == "ple_projection"
                    else stage_id.removeprefix("ple_projection_")
                ),
                "layer_start": layer_start,
                "layer_end": layer_end,
                "original_layer_indices": list(range(layer_start, layer_end)),
                "projection_output_channel_range": [
                    layer_start * ple_dim,
                    layer_end * ple_dim,
                ],
                "ple_output_elements": seq_len * (layer_end - layer_start) * ple_dim,
                "projection_macs": seq_len
                * (layer_end - layer_start)
                * ple_dim
                * hidden_size,
                "shared_lookup": "ple_embedding",
                "inputs": [
                    tensor_contract(
                        "inputs_embeds",
                        inputs[0],
                        getattr(projection, "obs_act_in", None),
                        "per_layer_model_projection.obs_act_in",
                    ),
                    tensor_contract(
                        "per_layer_token_inputs",
                        inputs[1],
                        cast(ObserverBase, qtext.obs_per_layer_token_inputs),
                        "obs_per_layer_token_inputs",
                    ),
                ],
                "outputs": [
                    tensor_contract(
                        "per_layer_inputs",
                        inputs[1],
                        cast(ObserverBase, qtext.obs_per_layer_inputs),
                        "obs_per_layer_inputs",
                    )
                ],
            }
        )

    prefill_stem = "ple_projection_prefill" if prefill_decode else "ple_projection"
    if suffix_start is None:
        _export(prefill_stem, max_seq_len, prefill_stem)
    else:
        _export(
            "ple_projection_prefill_prefix",
            max_seq_len,
            "ple_projection_prefill_prefix",
            0,
            suffix_start,
        )
        _export(
            "ple_projection_prefill_suffix",
            1,
            "ple_projection_prefill_suffix",
            suffix_start,
            num_hidden_layers,
        )
    if prefill_decode:
        _export("ple_projection_decode", 1, "ple_projection_decode")
    return stages


def _save_ple_pipeline_manifest(
    *,
    path: Path,
    qtext: torch.nn.Module,
    artifact_tag: str,
    embedding: dict[str, Any],
    projections: list[dict[str, Any]],
    suffix_start: int | None = None,
) -> None:
    """Write the PLE stage contract, including format and observer boundaries."""
    boundaries = []
    for name, attr, producer, consumer in (
        (
            "per_layer_token_inputs",
            "obs_per_layer_token_inputs",
            "ple_embedding",
            "ple_projection",
        ),
        (
            "per_layer_projection",
            "obs_per_layer_projection",
            "ple_projection",
            "ple_projection",
        ),
        (
            "per_layer_inputs",
            "obs_per_layer_inputs",
            "ple_projection",
            "decoder_layer",
        ),
    ):
        observer = getattr(qtext, attr, None)
        if observer is None:
            continue
        boundaries.append(
            {
                "name": name,
                "producer": producer,
                "consumer": consumer,
                "observer": _observer_manifest(observer),
            }
        )

    producer_observers = {}
    for component in (
        "embed_tokens_per_layer",
        "per_layer_model_projection",
        "per_layer_projection_norm",
    ):
        module = getattr(qtext, component, None)
        if module is not None:
            for name, observer in module.named_modules():
                if isinstance(observer, ObserverBase):
                    producer_observers[f"{component}.{name}"] = _observer_manifest(
                        observer
                    )
    for name in (
        "obs_per_layer_model_projection_scale",
        "obs_per_layer_input_scale",
        "obs_per_layer_sum",
    ):
        observer = getattr(qtext, name, None)
        if observer is not None:
            producer_observers[name] = _observer_manifest(observer)

    manifest = {
        "schema_version": 2 if suffix_start is not None else 1,
        "artifact_tag": artifact_tag,
        "ple_prefill_split": suffix_start is not None,
        "suffix_start": suffix_start,
        "prefill_contract": (
            {
                "inputs_embeds": "H0: original decoder layer-0 input; multimodal fusion output",
                "suffix_token_index": "valid_length - 1",
                "padding": "batch_size_1_right_padding",
                "prefix_tokens": "all static max_seq slots",
                "suffix_tokens": "one final valid prompt token",
                "suffix_uses_decoder_prefix_output": False,
                "decoder_last_token_only_required": suffix_start is not None,
                "layer_mapping": "local layer j maps to original layer_start + j",
                "lookup": "one shared table; select token rows and layer columns before dequantization",
            }
            if suffix_start is not None
            else {"inputs_embeds": "H0", "tokens": "all static max_seq slots"}
        ),
        "embedding": embedding,
        "projections": projections,
        "boundaries": boundaries,
        "producer_observers": producer_observers,
        "quantization_contract": {
            "activation_slicing": (
                "frozen per-tensor affine or identity"
                if suffix_start is not None
                else "not required for full producer"
            ),
            "weight_slicing": "original frozen qparams; no recalibration",
            "boundary_owner": "source checkpoint observers",
            "decoder_input_observer": "obs_per_layer_inputs",
        },
    }
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _gemma4_text_pipeline_manifest(
    qtext: QuantGemma4TextModel,
    *,
    max_seq_len: int,
    artifact_tag: str,
    decode_kv_policy: str = "compact_sliding_v1",
) -> dict[str, Any]:
    """Describe the existing Q=1 consumer graph as a prefill-finalize alias.

    Both adapters execute the same wrapped consumer and frozen observers. They
    read external storage K/V of layer capacity C, return no K/V and perform
    no writes.
    The host supplies the original prefill query row/RoPE/PLE, rather than the
    normal decode mask or a new logical position. No new graph or qparam owner
    is needed for this alias.
    """
    if not isinstance(decode_kv_policy, str) or decode_kv_policy not in {
        "compact_sliding_v1",
        "full_capacity_v1",
    }:
        raise ValueError("Unsupported Gemma4 export.decode_kv_policy.")
    compact = decode_kv_policy == "compact_sliding_v1"
    boundary, producers = resolve_gemma4_shared_kv_suffix(qtext.layers)
    finalize_supported = gemma4_single_token_quantization_supported(
        qtext.layers[boundary:],
        per_layer_input_observer=getattr(qtext, "obs_per_layer_inputs", None),
    )
    layers = []
    for index, layer in enumerate(qtext.layers):
        attention, heads, head_dim, window, shared = _attention_contract(
            layer,
            max_seq_len=max_seq_len,
        )
        layers.append(
            {
                "layer_index": index,
                "layer_type": attention.layer_type,
                "is_shared": shared,
                "store_full_length_kv": bool(attention.store_full_length_kv),
                "producer_index": producers.get(index),
                "num_kv_heads": heads,
                "head_dim": head_dim,
                "sliding_window": window,
                "decode_capacity": gemma4_decode_capacity(
                    max_seq_len, window, compact=compact
                ),
                "prefill_artifact": _circle_name(
                    f"decoder_layer_prefill_{index}", artifact_tag
                ),
                "decode_artifact": _circle_name(
                    f"decoder_layer_decode_{index}", artifact_tag
                ),
                "finalize_artifact": (
                    _circle_name(f"decoder_layer_decode_{index}", artifact_tag)
                    if index >= boundary
                    else None
                ),
            }
        )
    return {
        "schema_version": 2,
        "model_family": "gemma4",
        "decode_kv_policy": decode_kv_policy,
        "finalize_contract": "shared_consumer_decode_alias_v2",
        "finalize_supported": finalize_supported,
        "finalize_unsupported_reason": (
            None
            if finalize_supported
            else "Shape-dependent activation quantization is not supported."
        ),
        "rope_convention": "pre_negated_sin",
        "execution_mode": "fake_quantize" if artifact_tag == "q" else "native",
        "max_seq_len": max_seq_len,
        "hidden_size": int(qtext.config.hidden_size),
        "vocab_size": int(qtext.config.vocab_size),
        "ple_dim": int(getattr(qtext.config, "hidden_size_per_layer_input", 0) or 0),
        "suffix_start": boundary,
        "lm_head_artifact": _circle_name("lm_head", artifact_tag),
        "layers": layers,
    }


def export_gemma4_per_layer(
    *,
    q_model: torch.nn.Module,
    max_seq_len: int,
    output_dir: str | Path,
    model_args: Mapping[str, Any],
    prefill_decode: bool = True,
    strict: bool = False,
    vision_granularity: str = "monolithic",
    spatial_pooling: bool = True,
    ple_embedding_format: str = "auto",
    ple_prefill_split: bool = False,
    decode_kv_policy: str = "compact_sliding_v1",
    compile_config: CompileConfigBase | None = None,
) -> None:
    """Export a floating-point or PTQ-wrapped dense Gemma4 E2B/E4B by runtime stage.

    The generated Circle set contains the configured image vision pipeline,
    dynamic token embedding, fixed-slot multimodal fusion, every text decoder
    layer, and final norm/LM head. ``vision_granularity`` selects the existing
    monolithic vision graph, a split patch/encoder-stack/pool/post-projection
    pipeline (``encoder``), per-layer encoder artifacts (``layer``), or monolithic
    plus per-layer artifacts (``both``). Split encoders share mask and RoPE inputs instead
    of embedding duplicate profile tensors.

    When Per-Layer Embeddings are enabled (``hidden_size_per_layer_input > 0``)
    the exporter additionally emits:

    - ``ple_embedding``: one shared CPU lookup owning ``embed_tokens_per_layer``
      with the same ``(1, S)`` dynamic contract as ``token_embedding``.
      ``ple_embedding_format`` selects ``circle`` (dynamic Circle graph),
      ``pt`` (host ``.pt`` table loaded by
      ``Gemma4PLEEmbeddingHostTable``), or ``auto`` (Circle when the packed
      table fits the 2 GiB flatbuffer limit, otherwise ``.pt``);
    - ``ple_projection_prefill`` (``S = max_seq_len``) and
      ``ple_projection_decode`` (``S = 1``): fixed-shape NPU graphs running the
      complete projection/norm/combine stage. With ``prefill_decode=False`` only
      the unsuffixed ``ple_projection`` prefill graph is written.
    - With ``ple_prefill_split=True``, the full PLE prefill graph is replaced by
      ``ple_projection_prefill_prefix`` and ``ple_projection_prefill_suffix``.
      This requires paired prefill/decode export and the shared ``.pt`` lookup.
      The suffix uses the original fusion input at ``valid_length - 1``.

    With ``prefill_decode=True``, each text layer is emitted once for full
    prefill and once for single-token decode. The CPU runtime orchestrates the
    PLE lookup and slices the packed projection output; each decoder Circle
    receives only its ``per_layer_input`` slice, matching
    ``StaticGemma4Runtime``. ``decode_kv_policy="compact_sliding_v1"`` uses
    C=min(S,W) keys for causal sliding decode; ``"full_capacity_v1"`` retains
    the full key capacity for comparison and shape-dependent observer policies.
    """
    if max_seq_len < 1:
        raise ValueError(f"max_seq_len must be positive, got {max_seq_len}.")

    if not isinstance(decode_kv_policy, str) or decode_kv_policy not in {
        "compact_sliding_v1",
        "full_capacity_v1",
    }:
        raise ValueError("Unsupported Gemma4 export.decode_kv_policy.")
    compact_decode = decode_kv_policy == "compact_sliding_v1"
    if type(ple_prefill_split) is not bool:
        raise TypeError("export.ple_prefill_split must be a boolean.")
    if ple_prefill_split and not prefill_decode:
        raise ValueError(
            "export.ple_prefill_split requires export.prefill_decode=true."
        )
    if not isinstance(spatial_pooling, bool):
        raise TypeError("export.vision.spatial_pooling must be a boolean.")
    vision_granularity = _normalize_vision_export_granularity(vision_granularity)
    ple_embedding_format = _normalize_ple_embedding_format(ple_embedding_format)
    normalized_model_args = _normalize_model_args(
        model_args,
        max_seq_len=max_seq_len,
    )
    output_dir = Path(output_dir)
    export_model, artifact_tag = _prepare_gemma4_export_model(
        q_model,
        normalized_model_args,
    )
    qmodel, gemma_model, qvision, qtext = _unwrap_gemma4_components(export_model)
    # The arithmetic-only fusion stage has no quantized child entry to guard it.
    ExportCapabilityGuard(gemma_model, "multimodal_fusion")
    assert_gemma4_dense_no_moe(qtext.config)
    text_manifest = (
        _gemma4_text_pipeline_manifest(
            qtext,
            max_seq_len=max_seq_len,
            artifact_tag=artifact_tag,
            decode_kv_policy=decode_kv_policy,
        )
        if prefill_decode
        else None
    )

    if prefill_decode and compact_decode:
        assert text_manifest is not None
        shrinking_layers = [
            layer
            for layer, contract in zip(qtext.layers, text_manifest["layers"])
            if contract["decode_capacity"] < max_seq_len
        ]
        validate_gemma4_compact_decode_quantization(shrinking_layers)

    text_capacity = int(qtext.config.max_position_embeddings)
    if max_seq_len > text_capacity:
        raise ValueError(
            "max_seq_len exceeds the wrapped Gemma4 text capacity: "
            f"max_seq_len={max_seq_len}, capacity={text_capacity}."
        )

    vision_profile = _resolve_vision_contract(
        gemma_model=gemma_model,
        qvision=qvision,
        model_args=normalized_model_args,
        max_seq_len=max_seq_len,
    )
    visual_start_idx = vision_profile.visual_start_idx
    num_visual_tokens = vision_profile.num_visual_tokens

    config = qtext.config
    hidden_size = int(config.hidden_size)
    vocab_size = int(config.vocab_size)
    ple_dim = int(getattr(config, "hidden_size_per_layer_input", 0) or 0)
    per_layer_input_observer = (
        getattr(qtext, "obs_per_layer_inputs", None) if ple_dim else None
    )
    if ple_dim and per_layer_input_observer is None:
        raise RuntimeError(
            "Gemma4 PLE split export requires the text-model "
            "obs_per_layer_inputs observer."
        )

    ple_suffix_start: int | None = None
    if ple_prefill_split and ple_dim:
        boundary, _ = resolve_gemma4_shared_kv_suffix(qtext.layers)
        if boundary < len(qtext.layers):
            if text_manifest is not None and not text_manifest["finalize_supported"]:
                raise ValueError(
                    "export.ple_prefill_split requires supported single-token decoder "
                    "quantization; use ple_prefill_split=false for full PLE export."
                )
            lookup = Gemma4PLEEmbeddingExportAdapter(qtext)
            resolved_format = _resolve_ple_embedding_format(
                ple_embedding_format,
                estimated_circle_bytes=estimate_gemma4_ple_embedding_circle_bytes(
                    lookup
                ),
            )
            if resolved_format != "pt":
                raise ValueError(
                    "export.ple_prefill_split requires the shared .pt host lookup; "
                    "set export.ple_embedding_format=pt or ple_prefill_split=false. "
                    "Circle lookup retains its all-layer output ABI."
                )
            for start, end in ((0, boundary), (boundary, len(qtext.layers))):
                Gemma4PLEEmbeddingExportAdapter(qtext, layer_start=start, layer_end=end)
                Gemma4PLEProjectionExportAdapter(
                    qtext, layer_start=start, layer_end=end
                )
            ple_suffix_start = boundary

    output_dir.mkdir(parents=True, exist_ok=True)
    # A partially rewritten directory must never advertise the previous bundle.
    (output_dir / "text_pipeline.json").unlink(missing_ok=True)
    (output_dir / "ple_pipeline.json").unlink(missing_ok=True)
    pixel_values, pixel_position_ids = _make_vision_inputs(
        profile=vision_profile,
    )
    vision_profile.save_manifest(output_dir / "vision_profile.json")
    _export_gemma4_vision_stages(
        gemma_model=gemma_model,
        qvision=qvision,
        profile=vision_profile,
        pixel_values=pixel_values,
        pixel_position_ids=pixel_position_ids,
        text_hidden_size=hidden_size,
        artifact_tag=artifact_tag,
        output_dir=output_dir,
        granularity=vision_granularity,
        spatial_pooling=spatial_pooling,
        strict=strict,
        compile_config=compile_config,
    )

    # Dynamic-sequence QUANT export traces frozen fake-quant ops with
    # FakeTensors; reuse the shared meta-kernel registration.
    register_fake_quant_meta_kernels_for_dynamic_export()
    token_input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(1, max_seq_len),
        dtype=torch.long,
        device="cpu",
    )
    _convert_and_save(
        Gemma4TokenEmbeddingExportAdapter(qtext),
        (token_input_ids,),
        output_dir / _circle_name("token_embedding", artifact_tag),
        dynamic_shapes=make_token_embedding_dynamic_shapes(max_seq_len),
        strict=strict,
        config=compile_config,
    )
    ple_embedding_manifest: dict[str, Any] | None = None
    if ple_dim:
        # One shared lookup for both phases; the large
        # ``embed_tokens_per_layer`` table must exist only here.
        ple_embedding_manifest = _export_gemma4_ple_embedding_stage(
            qtext=qtext,
            max_seq_len=max_seq_len,
            vocab_size_per_layer_input=int(
                getattr(config, "vocab_size_per_layer_input", vocab_size)
            ),
            artifact_tag=artifact_tag,
            output_dir=output_dir,
            embedding_format=ple_embedding_format,
            strict=strict,
            compile_config=compile_config,
        )

    text_embeds = torch.randn(1, max_seq_len, hidden_size, device="cpu")
    visual_embeds = torch.randn(num_visual_tokens, hidden_size, device="cpu")
    fusion_name = "multimodal_fusion_prefill" if prefill_decode else "multimodal_fusion"
    _convert_and_save(
        Gemma4MMFusionExportAdapter(
            visual_start_idx=visual_start_idx,
            num_visual_tokens=num_visual_tokens,
        ),
        (text_embeds, visual_embeds),
        output_dir / _circle_name(fusion_name, artifact_tag),
        strict=strict,
        config=compile_config,
    )

    prefill_hidden = torch.randn(1, max_seq_len, hidden_size, device="cpu")
    decode_hidden = torch.randn(1, 1, hidden_size, device="cpu")

    if ple_dim:
        projection_manifest = _export_gemma4_ple_projection_stages(
            qtext=qtext,
            max_seq_len=max_seq_len,
            hidden_size=hidden_size,
            ple_dim=ple_dim,
            artifact_tag=artifact_tag,
            output_dir=output_dir,
            prefill_decode=prefill_decode,
            strict=strict,
            compile_config=compile_config,
            suffix_start=ple_suffix_start,
        )
        assert ple_embedding_manifest is not None
        if ple_suffix_start is not None:
            ple_embedding_manifest["shared_by"] = [
                "prefill_prefix",
                "prefill_suffix",
                "decode",
            ]
            ple_embedding_manifest["selected_lookup_contract"] = {
                "layer_start": "inclusive original decoder index",
                "layer_end": "exclusive original decoder index",
                "output_shape": [1, "Q", "layer_end - layer_start", ple_dim],
                "prefix": {
                    "query_length": max_seq_len,
                    "layer_start": 0,
                    "layer_end": ple_suffix_start,
                },
                "suffix": {
                    "query_length": 1,
                    "layer_start": ple_suffix_start,
                    "layer_end": len(qtext.layers),
                },
                "decode": {
                    "query_length": 1,
                    "layer_start": 0,
                    "layer_end": len(qtext.layers),
                },
            }
        _save_ple_pipeline_manifest(
            path=output_dir / "ple_pipeline.json",
            qtext=qtext,
            artifact_tag=artifact_tag,
            embedding=ple_embedding_manifest,
            projections=projection_manifest,
            suffix_start=ple_suffix_start,
        )

    for layer_idx, layer in enumerate(qtext.layers):
        (
            _attention,
            num_kv_heads,
            head_dim,
            sliding_window,
            is_shared,
        ) = _attention_contract(layer, max_seq_len=max_seq_len)

        prefill_kwargs: dict[str, Any] = {
            "attention_mask": _make_prefill_attention_mask(
                max_seq_len=max_seq_len,
                sliding_window=sliding_window,
            ),
            "position_embeddings": _make_position_embeddings(
                seq_len=max_seq_len,
                head_dim=head_dim,
            ),
        }
        if ple_dim:
            prefill_kwargs["per_layer_input"] = torch.randn(
                1,
                max_seq_len,
                ple_dim,
                device="cpu",
            )
        if is_shared:
            shared_key = torch.randn(
                1,
                num_kv_heads,
                max_seq_len,
                head_dim,
                device="cpu",
            )
            prefill_kwargs["shared_key_value"] = (
                shared_key,
                torch.randn_like(shared_key),
            )

        prefill_stem = (
            f"decoder_layer_prefill_{layer_idx}"
            if prefill_decode
            else f"decoder_layer_{layer_idx}"
        )
        prefill_export_kwargs: dict[str, Any] = {
            "return_kv": prefill_decode,
        }
        if per_layer_input_observer is not None:
            prefill_export_kwargs["per_layer_input_observer"] = per_layer_input_observer
        _convert_and_save(
            cast(QuantGemma4TextDecoderLayer, layer.wrapped).as_export_module(
                "prefill",
                **prefill_export_kwargs,
            ),
            (prefill_hidden,),
            output_dir / _circle_name(prefill_stem, artifact_tag),
            kwargs=prefill_kwargs,
            strict=strict,
            config=compile_config,
        )

        if not prefill_decode:
            continue

        decode_capacity = gemma4_decode_capacity(
            max_seq_len, sliding_window, compact=compact_decode
        )
        decode_kwargs: dict[str, Any] = {
            "attention_mask": _make_decode_attention_mask(
                max_seq_len=decode_capacity,
                sliding_window=sliding_window,
            ),
            "position_embeddings": _make_position_embeddings(
                seq_len=1,
                head_dim=head_dim,
            ),
        }
        if ple_dim:
            decode_kwargs["per_layer_input"] = torch.randn(
                1,
                1,
                ple_dim,
                device="cpu",
            )
        if is_shared:
            shared_key = torch.randn(
                1,
                num_kv_heads,
                decode_capacity,
                head_dim,
                device="cpu",
            )
            decode_kwargs["shared_key_value"] = (
                shared_key,
                torch.randn_like(shared_key),
            )
        else:
            past_key = torch.randn(
                1,
                num_kv_heads,
                decode_capacity - 1,
                head_dim,
                device="cpu",
            )
            decode_kwargs["past_key_value"] = (
                past_key,
                torch.randn_like(past_key),
            )

        decode_export_kwargs: dict[str, Any] = {"return_kv": True}
        if per_layer_input_observer is not None:
            decode_export_kwargs["per_layer_input_observer"] = per_layer_input_observer
        _convert_and_save(
            cast(QuantGemma4TextDecoderLayer, layer.wrapped).as_export_module(
                "decode", **decode_export_kwargs
            ),
            (decode_hidden,),
            output_dir
            / _circle_name(f"decoder_layer_decode_{layer_idx}", artifact_tag),
            kwargs=decode_kwargs,
            strict=strict,
            config=compile_config,
        )

    lm_head_hidden = torch.randn(1, 1, hidden_size, device="cpu")
    _convert_and_save(
        Gemma4LMHeadExportAdapter(qmodel),
        (lm_head_hidden,),
        output_dir / _circle_name("lm_head", artifact_tag),
        strict=strict,
        config=compile_config,
    )
    if text_manifest is not None:
        (output_dir / "text_pipeline.json").write_text(
            json.dumps(text_manifest, indent=2) + "\n"
        )
