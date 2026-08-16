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

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import torch

import tico
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.config.qwen3_vl_attention import (
    is_npu_export_text_attention_options,
)
from tico.quantization.wrapq.wrap_helper import PTQWrapHelper
from tico.quantization.wrapq.wrappers.llama.export_adapters import (
    make_token_embedding_dynamic_shapes,
    register_fake_quant_meta_kernels_for_dynamic_export,
)
from tico.quantization.wrapq.wrappers.qwen_vl.export_adapters import (
    Qwen3VLDeepstackFusionExportAdapter,
    Qwen3VLLMHeadExportAdapter,
    Qwen3VLMultimodalEmbeddingExportAdapter,
    Qwen3VLTextEmbeddingExportAdapter,
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
    """Convert one Qwen3-VL export stage to Circle and save it."""
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
            "Floating-point Qwen3-VL export currently supports float32 only. "
            f"Got parameter dtype {dtype}."
        )
    return "f32"


def _normalize_model_args(model_args: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize model-specific export arguments before structural wrapping."""
    normalized = deepcopy(dict(model_args or {}))
    profile = normalized.setdefault("profile", "npu_export")
    if profile != "npu_export":
        raise ValueError(
            "Qwen3-VL Circle export requires model_args.profile='npu_export', "
            f"got {profile!r}."
        )

    vision = normalized.setdefault("vision", {})
    if not isinstance(vision, dict):
        raise TypeError("model_args.vision must be a mapping.")

    grid_thw = vision.get("grid_thw")
    if grid_thw is None:
        raise ValueError("Qwen3-VL Circle export requires model_args.vision.grid_thw.")
    if not isinstance(grid_thw, (list, tuple)) or len(grid_thw) != 3:
        raise ValueError(
            "model_args.vision.grid_thw must contain exactly three values "
            "(temporal, height, width)."
        )
    vision["grid_thw"] = tuple(int(value) for value in grid_thw)

    if "visual_start_idx" not in vision:
        raise ValueError(
            "Qwen3-VL Circle export requires model_args.vision.visual_start_idx."
        )
    vision["visual_start_idx"] = int(vision["visual_start_idx"])
    if vision["visual_start_idx"] < 0:
        raise ValueError("model_args.vision.visual_start_idx must be non-negative.")

    if "spatial_merge_size" not in vision:
        raise ValueError(
            "Qwen3-VL Circle export requires model_args.vision.spatial_merge_size."
        )
    vision["spatial_merge_size"] = int(vision["spatial_merge_size"])
    if vision["spatial_merge_size"] <= 0:
        raise ValueError("model_args.vision.spatial_merge_size must be positive.")

    return normalized


def _prepare_qwen3_vl_export_model(
    model: torch.nn.Module,
    model_args: Mapping[str, Any] | None,
) -> tuple[torch.nn.Module, str]:
    """Normalize a checkpoint or FP model for staged Qwen3-VL export.

    Floating-point models are structurally wrapped with the configured fixed
    vision layout. The wrappers stay in ``NO_QUANT`` mode, so no calibration or
    fake quantization is introduced. Converted checkpoints retain their
    existing quantization state.
    """
    model = model.eval().cpu()
    if _is_wrapped_export_model(model):
        return model, "q"

    artifact_tag = _float_artifact_tag(model)
    wrapper_config = PTQConfig(
        model_args=_normalize_model_args(model_args),
        strict_wrap=True,
    )
    export_model = PTQWrapHelper(strict_wrap=True).wrap_supported(
        model,
        wrapper_config,
    )
    if not _is_wrapped_export_model(export_model):
        raise TypeError(
            "Qwen3-VL staged export requires a top-level PTQ wrapper exposing "
            "the wrapped model and LM head."
        )
    return export_model, artifact_tag


def _circle_name(stem: str, artifact_tag: str) -> str:
    """Build a Circle artifact name with an explicit precision tag."""
    return f"{stem}.{artifact_tag}.circle"


def _unwrap_qwen3_vl_components(
    export_model: torch.nn.Module,
) -> tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module, torch.nn.Module]:
    """Return the top-level, multimodal, vision, and text quant wrappers."""
    qmodel = export_model.wrapped
    qwen_model = qmodel.model.wrapped
    qvision = qwen_model.visual.wrapped
    qtext = qwen_model.language_model.wrapped
    return qmodel, qwen_model, qvision, qtext


def _validate_npu_export_attention(qtext: torch.nn.Module) -> None:
    """Reject wrapped attention layouts that are not NPU-export compatible."""
    for layer_idx, layer in enumerate(qtext.layers):
        self_attn = getattr(layer.wrapped, "self_attn", None)
        attention = getattr(self_attn, "wrapped", None)
        options = getattr(attention, "attn_options", None)
        if options is None:
            continue
        if not is_npu_export_text_attention_options(options):
            raise ValueError(
                "Qwen3-VL Circle export requires the NPU export attention "
                f"layout, but layer {layer_idx} uses {options!r}."
            )


def _resolve_vision_contract(
    *,
    qwen_model: torch.nn.Module,
    qvision: torch.nn.Module,
    model_args: Mapping[str, Any],
    max_seq_len: int,
) -> tuple[tuple[int, int, int], int, int, int]:
    """Validate and resolve the fixed image/video export contract."""
    vision_args = model_args["vision"]
    requested_grid = tuple(int(value) for value in vision_args["grid_thw"])
    wrapped_grid_tensor = getattr(qvision, "vision_grid_thw", None)
    if not isinstance(wrapped_grid_tensor, torch.Tensor):
        raise TypeError("Wrapped Qwen3-VL vision model has no vision_grid_thw tensor.")
    wrapped_grid = tuple(int(value) for value in wrapped_grid_tensor[0].tolist())
    if requested_grid != wrapped_grid:
        raise ValueError(
            "Configured grid_thw does not match the wrapped checkpoint: "
            f"configured={requested_grid}, wrapped={wrapped_grid}."
        )

    visual_start_idx = int(vision_args["visual_start_idx"])
    wrapped_start_idx = int(getattr(qwen_model, "visual_start_idx"))
    if visual_start_idx != wrapped_start_idx:
        raise ValueError(
            "Configured visual_start_idx does not match the wrapped checkpoint: "
            f"configured={visual_start_idx}, wrapped={wrapped_start_idx}."
        )

    spatial_merge_size = int(vision_args["spatial_merge_size"])
    wrapped_merge_size = int(getattr(qvision, "spatial_merge_size"))
    if spatial_merge_size != wrapped_merge_size:
        raise ValueError(
            "Configured spatial_merge_size does not match the wrapped checkpoint: "
            f"configured={spatial_merge_size}, wrapped={wrapped_merge_size}."
        )

    grid_t, grid_h, grid_w = wrapped_grid
    if min(grid_t, grid_h, grid_w) <= 0:
        raise ValueError(f"grid_thw values must be positive, got {wrapped_grid}.")
    if grid_h % spatial_merge_size or grid_w % spatial_merge_size:
        raise ValueError(
            "Vision grid height and width must be divisible by spatial_merge_size: "
            f"grid_thw={wrapped_grid}, spatial_merge_size={spatial_merge_size}."
        )

    visual_tokens = (
        grid_t * (grid_h // spatial_merge_size) * (grid_w // spatial_merge_size)
    )
    if visual_start_idx + visual_tokens > max_seq_len:
        raise ValueError(
            "The fixed visual-token span exceeds max_seq_len: "
            f"start={visual_start_idx}, visual_tokens={visual_tokens}, "
            f"max_seq_len={max_seq_len}."
        )

    return (grid_t, grid_h, grid_w), visual_start_idx, spatial_merge_size, visual_tokens


def _make_vision_inputs(
    qvision: torch.nn.Module,
    grid_thw: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create flattened processor-style pixel values for fixed-grid export."""
    patch_embed = qvision.patch_embed.wrapped
    patch_vector_size = (
        int(patch_embed.in_channels)
        * int(patch_embed.temporal_patch_size)
        * int(patch_embed.patch_size)
        * int(patch_embed.patch_size)
    )
    patch_count = int(grid_thw[0] * grid_thw[1] * grid_thw[2])
    pixel_values = torch.randn(patch_count, patch_vector_size, device="cpu")
    image_grid_thw = torch.tensor([grid_thw], dtype=torch.long, device="cpu")
    return pixel_values, image_grid_thw


def _make_prefill_attention_mask(max_seq_len: int) -> torch.Tensor:
    """Create a static additive causal mask for full-length prefill export."""
    mask = torch.full((1, 1, max_seq_len, max_seq_len), -120.0)
    return mask.triu_(1)


def _make_decode_attention_mask(max_seq_len: int) -> torch.Tensor:
    """Create a static additive mask for one-token decode export."""
    mask = torch.full((1, 1, 1, max_seq_len), -120.0)
    effective_past_len = max(max_seq_len // 2, 1)
    mask[..., :effective_past_len] = 0.0
    mask[..., max_seq_len - 1] = 0.0
    return mask


def _make_position_embeddings(
    *,
    seq_len: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create example mRoPE cosine and sine tensors."""
    return (
        torch.randn(1, seq_len, head_dim, device="cpu"),
        torch.randn(1, seq_len, head_dim, device="cpu"),
    )


def export_qwen3_vl_per_layer(
    *,
    q_model: torch.nn.Module,
    max_seq_len: int,
    output_dir: str | Path,
    model_args: Mapping[str, Any],
    prefill_decode: bool = True,
    strict: bool = False,
) -> None:
    """Export a floating-point or PTQ-wrapped Qwen3-VL model by runtime stage.

    The exported contract mirrors the static Qwen3-VL runtime: the vision model,
    text/multimodal embedding stages, every text decoder layer, optional
    DeepStack additions, and the final norm/LM head are separate Circle models.
    KV-cache storage, attention-mask construction, and mRoPE lookup remain
    runtime responsibilities.

    Args:
        q_model: Floating-point or converted PTQ Qwen3-VL model.
        max_seq_len: Static prefill sequence and decode key length.
        output_dir: Directory where Circle artifacts are written.
        model_args: Fixed Qwen3-VL wrapper arguments, including vision grid and
            visual-token placement.
        prefill_decode: Export both full prefill and one-token decode graphs when
            True. When False, only prefill graphs are produced.
        strict: Forwarded to ``tico.convert``.
    """
    if max_seq_len < 1:
        raise ValueError(f"max_seq_len must be positive, got {max_seq_len}.")

    normalized_model_args = _normalize_model_args(model_args)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    export_model, artifact_tag = _prepare_qwen3_vl_export_model(
        q_model,
        normalized_model_args,
    )

    if artifact_tag == "q":
        register_fake_quant_meta_kernels_for_dynamic_export()
    qmodel, qwen_model, qvision, qtext = _unwrap_qwen3_vl_components(export_model)
    _validate_npu_export_attention(qtext)

    text_capacity = int(qtext.config.max_position_embeddings)
    if max_seq_len > text_capacity:
        raise ValueError(
            "max_seq_len exceeds the wrapped Qwen3-VL text capacity: "
            f"max_seq_len={max_seq_len}, capacity={text_capacity}."
        )

    (
        grid_thw,
        visual_start_idx,
        _spatial_merge_size,
        visual_tokens,
    ) = _resolve_vision_contract(
        qwen_model=qwen_model,
        qvision=qvision,
        model_args=normalized_model_args,
        max_seq_len=max_seq_len,
    )

    config = qtext.config
    hidden_size = int(config.hidden_size)
    head_dim = int(
        getattr(config, "head_dim", None)
        or hidden_size // int(config.num_attention_heads)
    )
    num_kv_heads = int(config.num_key_value_heads)
    vocab_size = int(config.vocab_size)

    pixel_values, image_grid_thw = _make_vision_inputs(qvision, grid_thw)
    vision_export = qvision.as_export_module(mode="prefill")
    _convert_and_save(
        vision_export,
        (pixel_values, image_grid_thw),
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
        Qwen3VLTextEmbeddingExportAdapter(qmodel),
        (token_input_ids,),
        output_dir / _circle_name("token_embedding", artifact_tag),
        dynamic_shapes=make_token_embedding_dynamic_shapes(max_seq_len),
        strict=strict,
    )

    prefill_input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(1, max_seq_len),
        dtype=torch.long,
        device="cpu",
    )
    image_embeds = torch.randn(visual_tokens, hidden_size, device="cpu")
    multimodal_embedding_name = (
        "multimodal_embedding_prefill" if prefill_decode else "multimodal_embedding"
    )
    _convert_and_save(
        Qwen3VLMultimodalEmbeddingExportAdapter(
            qmodel,
            visual_start_idx=visual_start_idx,
        ),
        (prefill_input_ids, image_embeds),
        output_dir / _circle_name(multimodal_embedding_name, artifact_tag),
        strict=strict,
    )

    prefill_hidden = torch.randn(1, max_seq_len, hidden_size, device="cpu")
    prefill_mask = _make_prefill_attention_mask(max_seq_len)
    prefill_position_embeddings = _make_position_embeddings(
        seq_len=max_seq_len,
        head_dim=head_dim,
    )

    deepstack_count = len(qvision.deepstack_merger_list)
    if deepstack_count > len(qtext.layers):
        raise ValueError(
            "The number of DeepStack outputs exceeds the number of text layers: "
            f"deepstack={deepstack_count}, text_layers={len(qtext.layers)}."
        )

    for layer_idx, layer in enumerate(qtext.layers):
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
            kwargs={
                "attention_mask": prefill_mask,
                "position_embeddings": prefill_position_embeddings,
            },
            strict=strict,
        )

        if layer_idx < deepstack_count:
            deepstack_features = torch.randn(
                visual_tokens,
                hidden_size,
                device="cpu",
            )
            _convert_and_save(
                Qwen3VLDeepstackFusionExportAdapter(
                    visual_start_idx=visual_start_idx,
                ),
                (prefill_hidden, deepstack_features),
                output_dir
                / _circle_name(f"deepstack_fusion_{layer_idx}", artifact_tag),
                strict=strict,
            )

        if prefill_decode:
            decode_hidden = torch.randn(1, 1, hidden_size, device="cpu")
            decode_position_embeddings = _make_position_embeddings(
                seq_len=1,
                head_dim=head_dim,
            )
            decode_mask = _make_decode_attention_mask(max_seq_len)
            past_key = torch.randn(
                1,
                num_kv_heads,
                max_seq_len - 1,
                head_dim,
                device="cpu",
            )
            past_value = torch.randn_like(past_key)
            _convert_and_save(
                layer.wrapped.as_export_module("decode", return_kv=True),
                (decode_hidden,),
                output_dir
                / _circle_name(f"decoder_layer_decode_{layer_idx}", artifact_tag),
                kwargs={
                    "attention_mask": decode_mask,
                    "position_embeddings": decode_position_embeddings,
                    "past_key_values": (past_key, past_value),
                },
                strict=strict,
            )

    lm_head_hidden = torch.randn(1, 1, hidden_size, device="cpu")
    _convert_and_save(
        Qwen3VLLMHeadExportAdapter(qmodel),
        (lm_head_hidden,),
        output_dir / _circle_name("lm_head", artifact_tag),
        strict=strict,
    )
