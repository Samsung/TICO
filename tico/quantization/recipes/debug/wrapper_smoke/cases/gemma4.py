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

from typing import Any, Mapping

import torch

from tico.quantization.recipes.debug.wrapper_smoke.case import (
    CaseAvailability,
    ForwardInput,
    WrapperSmokeCase,
)
from tico.quantization.recipes.debug.wrapper_smoke.utils import clone_module


def _has_gemma4() -> CaseAvailability:
    """Return availability for Hugging Face Gemma4 modules."""
    try:
        from transformers.models.gemma4.modeling_gemma4 import (  # noqa: F401
            Gemma4TextConfig,
        )

        return CaseAvailability(True)
    except Exception as exc:
        return CaseAvailability(False, f"Gemma4 modules are unavailable: {exc}")


class Gemma4BaseCase(WrapperSmokeCase):
    """Base class for Gemma4 E2B wrapper smoke cases."""

    tags: tuple[str, ...] = ("gemma4", "e2b")

    def availability(self) -> CaseAvailability:
        """Return whether Gemma4 modules can be imported."""
        return _has_gemma4()


class Gemma4TextMLPCase(Gemma4BaseCase):
    """Smoke case for one tiny Gemma4 text MLP."""

    name = "gemma4_text_mlp"
    description = "Quantize one tiny dense Gemma4 text MLP module."
    tags = ("gemma4", "e2b", "text", "mlp")

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a tiny Gemma4 text MLP and reference copy."""
        from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextMLP

        text_cfg = Gemma4TextConfig(
            vocab_size=256,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=32,
            enable_moe_block=False,
        )
        self.text_cfg = text_cfg
        module = Gemma4TextMLP(text_cfg, layer_idx=0).eval()
        return module, clone_module(module)

    def calibration_inputs(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> list[ForwardInput]:
        """Create calibration samples."""
        return [
            ForwardInput((torch.randn(1, 8, self.text_cfg.hidden_size),))
            for _ in range(3)
        ]

    def eval_input(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create an evaluation sample."""
        return ForwardInput((torch.randn(1, 8, self.text_cfg.hidden_size),))
