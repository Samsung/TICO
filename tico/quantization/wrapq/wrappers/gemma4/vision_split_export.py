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

"""Split-stage export adapters for the static Gemma4 vision pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn

from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.observers.base import ObserverBase
from tico.quantization.wrapq.wrappers.gemma4.export_adapters import (
    build_gemma4_vision_prefill_export_module,
)


def _unwrap(module: nn.Module) -> nn.Module:
    """Return the quantized wrapper stored by a PTQ-style container."""
    return getattr(module, "wrapped", module)


def _quantized_export(mode: Mode, *, stage_name: str) -> bool:
    """Validate an export mode and return whether fake quantization is active."""
    if mode is Mode.QUANT:
        return True
    if mode is Mode.NO_QUANT:
        return False
    raise RuntimeError(
        f"{stage_name} export requires NO_QUANT or QUANT mode, got {mode}."
    )


def _apply_export_observer(
    tensor: torch.Tensor,
    observer: ObserverBase,
    *,
    quantized: bool,
) -> torch.Tensor:
    """Apply a frozen observer only for a quantized export graph."""
    if quantized:
        return observer.fake_quant(tensor)
    return tensor


class Gemma4VisionPatchStageExportAdapter(nn.Module):
    """Export patch embedding with the encoder-input boundary quantization.

    The patch embedder already owns pixel normalization, projection, positional
    embedding, and its local output observer. The additional boundary observer
    matches the first operation of the split encoder pipeline so the generated
    Circle output can be passed directly to the first encoder artifact.
    """

    def __init__(
        self,
        patch_embedder: nn.Module,
        *,
        mode: Mode,
        output_observer: ObserverBase,
    ) -> None:
        super().__init__()
        self.patch_embedder = patch_embedder
        self.output_observer = output_observer
        self.quantized = _quantized_export(
            mode,
            stage_name="Gemma4 vision patch stage",
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return patch states in the first encoder input quantization domain."""
        hidden_states = self.patch_embedder(pixel_values)
        return _apply_export_observer(
            hidden_states,
            self.output_observer,
            quantized=self.quantized,
        )


class Gemma4VisionEncoderLayerExportAdapter(nn.Module):
    """Export one Gemma4 vision encoder layer.

    The large static attention mask and the two RoPE tensors remain runtime
    inputs. Every layer applies the encoder-owned context observers before using
    them, which preserves the monolithic quantization path without embedding the
    same tensors in every Circle artifact.

    The final output observers are selected by the builder. A non-final layer is
    requantized with the next layer's input observer. The final layer applies the
    encoder output observer followed by the pooler input observer. Therefore all
    neighboring split artifacts have directly compatible tensor boundaries.
    """

    def __init__(
        self,
        layer: nn.Module,
        *,
        mode: Mode,
        attention_mask_observer: ObserverBase,
        position_cos_observer: ObserverBase,
        position_sin_observer: ObserverBase,
        input_observer: ObserverBase | None = None,
        output_observers: Iterable[ObserverBase] = (),
    ) -> None:
        super().__init__()
        self.layer = layer
        self.attention_mask_observer = attention_mask_observer
        self.position_cos_observer = position_cos_observer
        self.position_sin_observer = position_sin_observer
        self.input_observer = input_observer
        self.output_observers = nn.ModuleList(tuple(output_observers))
        self.quantized = _quantized_export(
            mode,
            stage_name="Gemma4 vision encoder layer",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings_cos: torch.Tensor,
        position_embeddings_sin: torch.Tensor,
    ) -> torch.Tensor:
        """Run one encoder layer with shared external mask and RoPE inputs."""
        if self.input_observer is not None:
            hidden_states = _apply_export_observer(
                hidden_states,
                self.input_observer,
                quantized=self.quantized,
            )

        attention_mask = _apply_export_observer(
            attention_mask,
            self.attention_mask_observer,
            quantized=self.quantized,
        )
        position_embeddings_cos = _apply_export_observer(
            position_embeddings_cos,
            self.position_cos_observer,
            quantized=self.quantized,
        )
        position_embeddings_sin = _apply_export_observer(
            position_embeddings_sin,
            self.position_sin_observer,
            quantized=self.quantized,
        )
        hidden_states = self.layer(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=(
                position_embeddings_cos,
                position_embeddings_sin,
            ),
        )

        for observer in self.output_observers:
            hidden_states = _apply_export_observer(
                hidden_states,
                observer,
                quantized=self.quantized,
            )
        return hidden_states


class Gemma4VisionPostProjectionExportAdapter(nn.Module):
    """Export static pool-output finalization and text-width projection.

    The input observer duplicates the pooler's final output observer only at the
    artifact boundary. Applying the same frozen affine quantizer twice is
    idempotent, and it gives the consumer Circle input the same quantization
    parameters as the producer Circle output.
    """

    def __init__(
        self,
        *,
        mode: Mode,
        input_observer: ObserverBase,
        strip_padding_observer: ObserverBase,
        last_hidden_state_observer: ObserverBase,
        num_valid_pool_outputs: int,
        hidden_size: int,
        output_dtype: torch.dtype,
        vision_projection: nn.Module,
        standardize: bool,
        minus_bias_observer: ObserverBase | None = None,
        std_bias_observer: ObserverBase | None = None,
        std_scale_observer: ObserverBase | None = None,
        std_bias: torch.Tensor | None = None,
        std_scale: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if num_valid_pool_outputs <= 0:
            raise ValueError(
                "num_valid_pool_outputs must be positive for Gemma4 vision export."
            )
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive for Gemma4 vision export.")

        self.input_observer = input_observer
        self.strip_padding_observer = strip_padding_observer
        self.last_hidden_state_observer = last_hidden_state_observer
        self.num_valid_pool_outputs = int(num_valid_pool_outputs)
        self.hidden_size = int(hidden_size)
        self.output_dtype = output_dtype
        self.vision_projection = vision_projection
        self.standardize = bool(standardize)
        self.quantized = _quantized_export(
            mode,
            stage_name="Gemma4 vision post-projection stage",
        )

        self.minus_bias_observer = minus_bias_observer
        self.std_bias_observer = std_bias_observer
        self.std_scale_observer = std_scale_observer

        if self.standardize:
            if (
                minus_bias_observer is None
                or std_bias_observer is None
                or std_scale_observer is None
                or std_bias is None
                or std_scale is None
            ):
                raise ValueError(
                    "Standardized Gemma4 vision export requires standardization "
                    "buffers and observers."
                )
            self.register_buffer(
                "std_bias",
                std_bias.detach().clone(),
                persistent=False,
            )
            self.register_buffer(
                "std_scale",
                std_scale.detach().clone(),
                persistent=False,
            )
        else:
            self.register_buffer("std_bias", torch.empty(0), persistent=False)
            self.register_buffer("std_scale", torch.empty(0), persistent=False)

    def forward(self, pooled_hidden_states: torch.Tensor) -> torch.Tensor:
        """Finalize pooled vision features and project them to text width."""
        hidden_states = _apply_export_observer(
            pooled_hidden_states,
            self.input_observer,
            quantized=self.quantized,
        )

        hidden_states = hidden_states[:, : self.num_valid_pool_outputs, :]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        hidden_states = _apply_export_observer(
            hidden_states,
            self.strip_padding_observer,
            quantized=self.quantized,
        )

        if self.standardize:
            assert self.minus_bias_observer is not None
            assert self.std_bias_observer is not None
            assert self.std_scale_observer is not None
            std_bias = _apply_export_observer(
                self.std_bias,
                self.std_bias_observer,
                quantized=self.quantized,
            )
            std_scale = _apply_export_observer(
                self.std_scale,
                self.std_scale_observer,
                quantized=self.quantized,
            )
            hidden_states = hidden_states - std_bias.float()
            hidden_states = _apply_export_observer(
                hidden_states,
                self.minus_bias_observer,
                quantized=self.quantized,
            )
            hidden_states = hidden_states * std_scale.float()

        hidden_states = hidden_states.to(dtype=self.output_dtype)
        hidden_states = _apply_export_observer(
            hidden_states,
            self.last_hidden_state_observer,
            quantized=self.quantized,
        )
        return self.vision_projection(hidden_states)


@dataclass(frozen=True)
class Gemma4VisionBoundaryContract:
    """Describe one directly connected split-vision tensor boundary."""

    name: str
    producer: str
    consumer: str
    observer: ObserverBase


@dataclass(frozen=True)
class Gemma4VisionSplitExportBundle:
    """Hold monolithic and split modules specialized for one vision profile."""

    monolithic: nn.Module
    patch_embedder: nn.Module
    encoder_layers: tuple[nn.Module, ...]
    pooler: nn.Module
    post_projection: nn.Module
    attention_mask: torch.Tensor
    position_embeddings_cos: torch.Tensor
    position_embeddings_sin: torch.Tensor
    boundary_contracts: tuple[Gemma4VisionBoundaryContract, ...]


def build_gemma4_vision_split_export_bundle(
    wrapped_model: nn.Module,
    *,
    pixel_position_ids: torch.Tensor,
    output_dtype: torch.dtype = torch.float32,
    mode: str = "prefill",
) -> Gemma4VisionSplitExportBundle:
    """Build directly chainable Gemma4 vision modules for split export.

    The helper first prepares the existing monolithic static vision adapter. The
    prepared patch, encoder, and pooler modules are then reused to construct
    smaller stages. No observer is recalibrated and no profile-dependent tensor
    is recomputed with a different rule.

    Args:
        wrapped_model: Quantized Gemma4 multimodal wrapper containing the vision
            tower and ``embed_vision`` projection.
        pixel_position_ids: Fixed profile coordinates shaped ``(1, S, 2)``.
        output_dtype: Dtype used before the final vision projection.
        mode: Export mode. Only ``"prefill"`` is supported by the vision tower.

    Returns:
        A bundle containing the existing monolithic module, one module per
        encoder layer, and the shared external context tensors.
    """
    monolithic = build_gemma4_vision_prefill_export_module(
        wrapped_model,
        pixel_position_ids=pixel_position_ids,
        mode=mode,
    )

    vision_tower = getattr(wrapped_model, "vision_tower", None)
    if vision_tower is None:
        raise ValueError("Gemma4 split vision export requires a vision tower.")
    vision_model = _unwrap(vision_tower)
    vision_projection = getattr(wrapped_model, "embed_vision", None)
    if vision_projection is None:
        raise ValueError("Gemma4 split vision export requires embed_vision.")

    patch_embedder = getattr(vision_model, "patch_embedder_export", None)
    encoder_export = getattr(vision_model, "encoder_export", None)
    pooler_export = getattr(vision_model, "pooler_export", None)
    if patch_embedder is None or encoder_export is None or pooler_export is None:
        raise RuntimeError(
            "Gemma4 vision model did not materialize split export components."
        )

    encoder = _unwrap(encoder_export)
    pooler = _unwrap(pooler_export)
    encoder_layers = tuple(getattr(encoder, "layers", ()))
    if not encoder_layers:
        raise ValueError("Gemma4 split vision export requires encoder layers.")

    vision_mode = getattr(vision_model, "_mode", None)
    encoder_mode = getattr(encoder, "_mode", vision_mode)
    pooler_mode = getattr(pooler, "_mode", vision_mode)
    if not isinstance(vision_mode, Mode):
        raise TypeError("Gemma4 vision wrapper does not expose a valid export mode.")
    if encoder_mode is not vision_mode or pooler_mode is not vision_mode:
        raise RuntimeError(
            "Gemma4 vision split components must share the same quantization mode."
        )

    patch_stage = Gemma4VisionPatchStageExportAdapter(
        patch_embedder,
        mode=vision_mode,
        output_observer=encoder.obs_act_in,
    )

    layer_stages: list[nn.Module] = []
    num_layers = len(encoder_layers)
    for layer_index, layer in enumerate(encoder_layers):
        output_observers: list[ObserverBase] = []
        if layer_index == num_layers - 1:
            output_observers.extend(
                (
                    encoder.obs_encoder_out,
                    pooler.obs_act_in,
                )
            )
        else:
            next_layer = _unwrap(encoder_layers[layer_index + 1])
            output_observers.append(next_layer.obs_act_in)

        layer_stages.append(
            Gemma4VisionEncoderLayerExportAdapter(
                layer,
                mode=vision_mode,
                attention_mask_observer=encoder.obs_attention_mask,
                position_cos_observer=encoder.obs_position_cos,
                position_sin_observer=encoder.obs_position_sin,
                input_observer=(encoder.obs_act_in if layer_index == 0 else None),
                output_observers=output_observers,
            )
        )

    boundary_contracts: list[Gemma4VisionBoundaryContract] = [
        Gemma4VisionBoundaryContract(
            name="patch_embedder_to_encoder",
            producer="patch_embedder",
            consumer="encoder_layer_0",
            observer=encoder.obs_act_in,
        )
    ]
    for layer_index in range(num_layers - 1):
        next_layer = _unwrap(encoder_layers[layer_index + 1])
        boundary_contracts.append(
            Gemma4VisionBoundaryContract(
                name=f"encoder_layer_{layer_index}_to_{layer_index + 1}",
                producer=f"encoder_layer_{layer_index}",
                consumer=f"encoder_layer_{layer_index + 1}",
                observer=next_layer.obs_act_in,
            )
        )
    boundary_contracts.extend(
        (
            Gemma4VisionBoundaryContract(
                name="encoder_to_pooler",
                producer=f"encoder_layer_{num_layers - 1}",
                consumer="pooler",
                observer=pooler.obs_act_in,
            ),
            Gemma4VisionBoundaryContract(
                name="pooler_to_post_projection",
                producer="pooler",
                consumer="post_projection",
                observer=pooler.obs_pool_out,
            ),
        )
    )

    standardize = bool(getattr(vision_model.config, "standardize", False))
    post_projection = Gemma4VisionPostProjectionExportAdapter(
        mode=vision_mode,
        input_observer=pooler.obs_pool_out,
        strip_padding_observer=vision_model.obs_strip_padding,
        last_hidden_state_observer=vision_model.obs_last_hidden_state,
        num_valid_pool_outputs=int(vision_model.num_valid_pool_outputs),
        hidden_size=int(vision_model.config.hidden_size),
        output_dtype=output_dtype,
        vision_projection=vision_projection,
        standardize=standardize,
        minus_bias_observer=(vision_model.obs_minus_bias if standardize else None),
        std_bias_observer=(vision_model.obs_std_bias if standardize else None),
        std_scale_observer=(vision_model.obs_std_scale if standardize else None),
        std_bias=(vision_model.std_bias if standardize else None),
        std_scale=(vision_model.std_scale if standardize else None),
    )

    return Gemma4VisionSplitExportBundle(
        monolithic=monolithic,
        patch_embedder=patch_stage,
        encoder_layers=tuple(layer_stages),
        pooler=pooler_export,
        post_projection=post_projection,
        attention_mask=encoder.attention_mask_template.detach().clone(),
        position_embeddings_cos=(
            encoder.position_embeddings_cos_template.detach().clone()
        ),
        position_embeddings_sin=(
            encoder.position_embeddings_sin_template.detach().clone()
        ),
        boundary_contracts=tuple(boundary_contracts),
    )
