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

"""Unit tests for the Gemma4 clippable linear PTQ wrapper."""

import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from tico.quantization.recipes.qparams import inject_gptq_qparams
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.wrappers.gemma4.quant_clippable_linear import (
    QuantGemma4ClippableLinear,
)
from tico.quantization.wrapq.wrappers.nn.quant_linear import QuantLinear
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper

from test.quantization.quant_spec_helpers import make_affine_ptq_config


class _DummyGemma4ClippableLinear(nn.Module):
    """Minimal Gemma4ClippableLinear-like module for dependency-free tests."""

    def __init__(self, in_features: int = 4, out_features: int = 3):
        super().__init__()
        self.use_clipped_linears = False
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the inner floating-point linear layer."""
        return self.linear(hidden_states)


class TestQuantGemma4ClippableLinear(unittest.TestCase):
    """Validate child config scoping and qualified inner linear names."""

    _FP_NAME = "model.vision_tower.encoder.layers.0.self_attn.q_proj"

    def test_inner_linear_uses_child_scope_and_qualified_fp_name(self):
        """The inner QuantLinear should consume `.linear` config and FQN scopes."""
        qcfg = make_affine_ptq_config(
            dtype=DType.uint(8),
            overrides={
                "linear": {
                    "weight": {"dtype": DType.uint(4)},
                }
            },
        )

        quant = QuantGemma4ClippableLinear(
            _DummyGemma4ClippableLinear(),
            qcfg=qcfg,
            fp_name=self._FP_NAME,
        )

        self.assertIsInstance(quant.linear, PTQWrapper)
        inner = quant.linear.wrapped
        self.assertIsInstance(inner, QuantLinear)
        self.assertEqual(inner.fp_name, f"{self._FP_NAME}.linear")
        self.assertEqual(inner.obs_weight.dtype, DType.uint(4))

    def test_gptq_qparams_match_qualified_inner_linear_name(self):
        """GPTQ qparams keyed by `<parent>.linear` should reach the weight observer."""
        quant = QuantGemma4ClippableLinear(
            _DummyGemma4ClippableLinear(),
            fp_name=self._FP_NAME,
        )

        inner = quant.linear.wrapped
        self.assertIsInstance(inner, QuantLinear)

        scale = torch.tensor([0.25, 0.5, 0.75])
        zero = torch.zeros(3, dtype=torch.int)
        inner_name = f"{self._FP_NAME}.linear"
        stats = inject_gptq_qparams(
            quant,
            {
                inner_name: SimpleNamespace(scale=scale, zero=zero),
            },
        )

        self.assertEqual(stats, {"matched": 1, "missed": 0, "unused": 0})
        self.assertTrue(torch.equal(inner.obs_weight._cached_scale, scale))
        self.assertTrue(torch.equal(inner.obs_weight._cached_zp, zero))
        self.assertFalse(inner.obs_weight.enabled)


if __name__ == "__main__":
    unittest.main()
