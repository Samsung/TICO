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

"""Static per-stage Circle export for Gemma4 E2B."""

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import torch

import tico
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.wrap_helper import PTQWrapHelper
from tico.quantization.wrapq.wrappers.gemma4.export_adapters import (
    build_gemma4_vision_prefill_export_module,
    Gemma4LMHeadExportAdapter,
    Gemma4MMFusionExportAdapter,
    Gemma4TokenEmbeddingExportAdapter,
)
from tico.quantization.wrapq.wrappers.gemma4.static_vision_profile import (
    build_gemma4_static_vision_profile,
    canonicalize_gemma4_static_vision_model_args,
    Gemma4StaticVisionProfile,
)
from tico.quantization.wrapq.wrappers.gemma4.utils import assert_gemma4_e2b_no_moe
from tico.quantization.wrapq.wrappers.llama.export_adapters import (
    make_token_embedding_dynamic_shapes,
)
from tico.utils.utils import SuppressWarning


def _convert_and_save(
    module: torch.nn.Module,
    example_inputs: tuple[Any, ...],
    save_path: Path,
    *,
    kwargs: dict[str, Any] | None = None,
    dynamic_shapes: Any | None = None,
    strict: bool = False,
) -> None:
    """Convert one Gemma4 export stage to Circle and save it."""
    print(f"Saving {save_path.name} to {save_path.resolve()}")
    with torch.no_grad(), SuppressWarning(UserWarning, ".*"):
        circle_model = tico.convert(
            module.eval(),
            example_inputs,
            kwargs=kwargs,
            dynamic_shapes=dynamic_shapes,
            strict=strict,
        )
    circle_model.save(save_path)


def _is_wrapped_export_model(model: torch.nn.Module) -> bool:
    """Return whether a model already exposes the PTQ wrapper export layout."""
    wrapped = getattr(model, "wrapped", None)
    return (
        wrapped is not None
        and hasattr(wrapped, "model")
        and hasattr(wrapped, "lm_head")
    )


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
    return "f32"


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
        return model, "q"

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
) -> tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module, torch.nn.Module]:
    """Return top-level, multimodal, vision, and text Gemma4 wrappers."""
    qmodel = export_model.wrapped
    gemma_model = qmodel.model.wrapped
    if getattr(gemma_model, "vision_tower", None) is None:
        raise ValueError("Gemma4 Circle export requires an image vision tower.")
    qvision = gemma_model.vision_tower.wrapped
    qtext = gemma_model.language_model.wrapped
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
    attention = layer.wrapped.self_attn.wrapped
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
        int(attention.sliding_window)
        if bool(getattr(attention, "is_sliding", False))
        else None
    )
    is_shared = bool(getattr(attention, "is_kv_shared_layer", False))
    return attention, num_kv_heads, head_dim, sliding_window, is_shared


def export_gemma4_per_layer(
    *,
    q_model: torch.nn.Module,
    max_seq_len: int,
    output_dir: str | Path,
    model_args: Mapping[str, Any],
    prefill_decode: bool = True,
    strict: bool = False,
) -> None:
    """Export a floating-point or PTQ-wrapped Gemma4 E2B by runtime stage.

    The generated Circle set contains the image vision prefill stage, dynamic
    token embedding, fixed-slot multimodal fusion, every text decoder layer,
    and final norm/LM head. With ``prefill_decode=True``, each text layer is
    emitted once for full prefill and once for single-token decode.

    PLE token lookup and the packed context projection intentionally remain CPU
    runtime responsibilities. Each decoder Circle receives only its sliced
    ``per_layer_input`` tensor, matching ``StaticGemma4Runtime``.
    """
    if max_seq_len < 1:
        raise ValueError(f"max_seq_len must be positive, got {max_seq_len}.")

    normalized_model_args = _normalize_model_args(
        model_args,
        max_seq_len=max_seq_len,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    export_model, artifact_tag = _prepare_gemma4_export_model(
        q_model,
        normalized_model_args,
    )
    qmodel, gemma_model, qvision, qtext = _unwrap_gemma4_components(export_model)
    assert_gemma4_e2b_no_moe(qtext.config)

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

    pixel_values, pixel_position_ids = _make_vision_inputs(
        profile=vision_profile,
    )
    vision_profile.save_manifest(output_dir / "vision_profile.json")
    vision_prefill = build_gemma4_vision_prefill_export_module(
        gemma_model,
        pixel_position_ids=pixel_position_ids,
    )
    _convert_and_save(
        vision_prefill,
        (pixel_values,),
        output_dir / _circle_name("vision_prefill", artifact_tag),
        strict=strict,
    )

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
    )

    prefill_hidden = torch.randn(1, max_seq_len, hidden_size, device="cpu")
    decode_hidden = torch.randn(1, 1, hidden_size, device="cpu")

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
        _convert_and_save(
            layer.wrapped.as_export_module(
                "prefill",
                return_kv=prefill_decode,
            ),
            (prefill_hidden,),
            output_dir / _circle_name(prefill_stem, artifact_tag),
            kwargs=prefill_kwargs,
            strict=strict,
        )

        if not prefill_decode:
            continue

        decode_kwargs: dict[str, Any] = {
            "attention_mask": _make_decode_attention_mask(
                max_seq_len=max_seq_len,
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
                max_seq_len,
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
                max_seq_len - 1,
                head_dim,
                device="cpu",
            )
            decode_kwargs["past_key_value"] = (
                past_key,
                torch.randn_like(past_key),
            )

        _convert_and_save(
            layer.wrapped.as_export_module("decode", return_kv=True),
            (decode_hidden,),
            output_dir
            / _circle_name(f"decoder_layer_decode_{layer_idx}", artifact_tag),
            kwargs=decode_kwargs,
            strict=strict,
        )

    lm_head_hidden = torch.randn(1, 1, hidden_size, device="cpu")
    _convert_and_save(
        Gemma4LMHeadExportAdapter(qmodel),
        (lm_head_hidden,),
        output_dir / _circle_name("lm_head", artifact_tag),
        strict=strict,
    )
