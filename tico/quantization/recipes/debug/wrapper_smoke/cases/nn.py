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

"""Smoke cases for standalone nn wrapper checks."""

from typing import Any, Mapping

import torch
import torch.nn as nn

from tico.quantization.recipes.debug.wrapper_smoke.case import (
    CaseAvailability,
    ForwardInput,
    WrapperSmokeCase,
)
from tico.quantization.recipes.debug.wrapper_smoke.utils import clone_module


class NNLinearCase(WrapperSmokeCase):
    """Smoke case for the nn Linear wrapper."""

    name = "nn_linear"
    description = "Quantize one torch.nn.Linear module."
    tags = ("nn", "linear")
    max_mean_abs_diff = 1.0

    def build(self, cfg: Mapping[str, Any]) -> tuple[nn.Module, nn.Module]:
        """Build the floating-point Linear module and reference copy."""
        torch.manual_seed(123)
        module = nn.Linear(16, 8, bias=False).eval()
        return module, clone_module(module)

    def calibration_inputs(
        self, prepared: nn.Module, cfg: Mapping[str, Any]
    ) -> list[ForwardInput]:
        """Create synthetic Linear calibration samples."""
        return [ForwardInput((torch.randn(4, 16),)) for _ in range(4)]

    def eval_input(self, prepared: nn.Module, cfg: Mapping[str, Any]) -> ForwardInput:
        """Create the synthetic Linear evaluation sample."""
        return ForwardInput((torch.randn(2, 16),))


class NNConv3dCase(WrapperSmokeCase):
    """Smoke case for the nn Conv3d wrapper."""

    name = "nn_conv3d"
    description = "Quantize a patch-embed-like torch.nn.Conv3d module."
    tags = ("nn", "conv3d")
    max_mean_abs_diff = 2.0
    inplace_prepare = True
    inplace_convert = True

    def build(self, cfg: Mapping[str, Any]) -> tuple[nn.Module, nn.Module]:
        """Build the floating-point Conv3d module and reference copy."""
        torch.manual_seed(123)
        module = nn.Conv3d(
            in_channels=3,
            out_channels=8,
            kernel_size=(2, 4, 4),
            stride=(2, 4, 4),
            bias=True,
        ).eval()
        return module, clone_module(module)

    def calibration_inputs(
        self, prepared: nn.Module, cfg: Mapping[str, Any]
    ) -> list[ForwardInput]:
        """Create synthetic Conv3d calibration samples."""
        return [ForwardInput((torch.randn(1, 3, 2, 8, 8),)) for _ in range(3)]


class NNConv3dSpecialCase(WrapperSmokeCase):
    """Smoke case for the patch-sized nn Conv3d wrapper path."""

    name = "nn_conv3d_special_case"
    description = "Quantize a Conv3d where kernel and stride match the patch volume."
    tags = ("nn", "conv3d")
    max_mean_abs_diff = 2.0
    inplace_prepare = True
    inplace_convert = True

    def build(self, cfg: Mapping[str, Any]) -> tuple[nn.Module, nn.Module]:
        """Build the special-case Conv3d module and reference copy."""
        torch.manual_seed(123)
        module = nn.Conv3d(
            in_channels=3,
            out_channels=16,
            kernel_size=(2, 8, 8),
            stride=(2, 8, 8),
            padding=0,
            bias=True,
            groups=1,
        ).eval()
        return module, clone_module(module)

    def calibration_inputs(
        self, prepared: nn.Module, cfg: Mapping[str, Any]
    ) -> list[ForwardInput]:
        """Create synthetic special-case Conv3d calibration samples."""
        return [ForwardInput((torch.randn(2, 3, 2, 8, 8),)) for _ in range(3)]


class NNLayerNormCase(WrapperSmokeCase):
    """Smoke case for the nn LayerNorm wrapper."""

    name = "nn_layernorm"
    description = "Quantize one torch.nn.LayerNorm module."
    tags = ("nn", "layernorm")
    max_mean_abs_diff = 1.0
    inplace_prepare = True
    inplace_convert = True

    def build(self, cfg: Mapping[str, Any]) -> tuple[nn.Module, nn.Module]:
        """Build the floating-point LayerNorm module and reference copy."""
        torch.manual_seed(123)
        module = nn.LayerNorm((32,), eps=1e-5, elementwise_affine=True).eval()
        return module, clone_module(module)

    def calibration_inputs(
        self, prepared: nn.Module, cfg: Mapping[str, Any]
    ) -> list[ForwardInput]:
        """Create synthetic LayerNorm calibration samples."""
        return [ForwardInput((torch.randn(4, 32),)) for _ in range(3)]


class TiedEmbeddingLM(nn.Module):
    """Tiny language-model-like module with tied embedding and LM-head weights."""

    def __init__(self, vocab_size: int = 16, hidden_size: int = 8) -> None:
        """Initialize a tied embedding and projection pair."""
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Run embedding lookup followed by the tied LM head."""
        return self.lm_head(self.embed(token_ids))


class NNTiedEmbeddingCase(WrapperSmokeCase):
    """Smoke case for tied Embedding and Linear wrapper sharing."""

    name = "nn_tied_embedding"
    description = (
        "Quantize tied embedding and LM-head modules while preserving sharing."
    )
    tags = ("nn", "embedding", "linear")
    max_mean_abs_diff = 2.0

    def build(self, cfg: Mapping[str, Any]) -> tuple[nn.Module, nn.Module]:
        """Build the tied-weight model and reference copy."""
        torch.manual_seed(123)
        module = TiedEmbeddingLM().eval()
        return module, clone_module(module)

    def prepare_model(self, model: nn.Module, cfg: Mapping[str, Any]) -> nn.Module:
        """Prepare the embedding and LM head independently to preserve tied weights."""
        from tico.quantization import prepare

        qcfg = self.ptq_config(cfg)
        model.embed = prepare(model.embed, qcfg)  # type: ignore[assignment]
        model.lm_head = prepare(model.lm_head, qcfg)  # type: ignore[assignment]
        return model

    def convert_model(self, prepared: nn.Module, cfg: Mapping[str, Any]) -> nn.Module:
        """Convert both tied submodules to quantized simulation mode."""
        from tico.quantization import convert

        prepared.embed = convert(prepared.embed)  # type: ignore[assignment]
        prepared.lm_head = convert(prepared.lm_head)  # type: ignore[assignment]
        return prepared

    def calibration_inputs(
        self, prepared: nn.Module, cfg: Mapping[str, Any]
    ) -> list[ForwardInput]:
        """Create synthetic token IDs for tied-embedding calibration."""
        return [
            ForwardInput((torch.randint(0, 16, (2, 5), dtype=torch.long),))
            for _ in range(3)
        ]

    def eval_input(self, prepared: nn.Module, cfg: Mapping[str, Any]) -> ForwardInput:
        """Create synthetic token IDs for tied-embedding evaluation."""
        return ForwardInput((torch.randint(0, 16, (1, 4), dtype=torch.long),))


NN_CASES: tuple[WrapperSmokeCase, ...] = (
    NNLinearCase(),
    NNConv3dCase(),
    NNConv3dSpecialCase(),
    NNLayerNormCase(),
    NNTiedEmbeddingCase(),
)
