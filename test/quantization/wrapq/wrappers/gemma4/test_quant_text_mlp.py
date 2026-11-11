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

"""Dense Gemma4 shared-layer MLP shape and PTQ contracts."""

import unittest

import torch
from tico.quantization import convert, prepare
from tico.quantization.config.gemma4_builders import build_gemma4_ptq_config
from tico.quantization.config.specs import affine, no_quant
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.wrappers.gemma4.quant_text_mlp import QuantGemma4TextMLP

from test.quantization.wrapq.wrappers.gemma4.test_quant_text_decoder_layer import (
    _has_gemma4,
    _make_text_config,
)


@unittest.skipUnless(
    _has_gemma4(), "required transformers Gemma4 modules are not installed"
)
class Gemma4TextMLPVariantTest(unittest.TestCase):
    """Actual HF projection modules determine E2B and E4B shared MLP widths."""

    def test_shared_mlp_width_parity_and_ptq_overrides(self):
        """Shared classification doubles E2B widths only when HF enables it."""
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextMLP

        for double_wide, expected_width in ((True, 64), (False, 32)):
            with self.subTest(use_double_wide_mlp=double_wide):
                torch.manual_seed(0)
                config = _make_text_config(
                    layer_types=["full_attention", "full_attention"],
                    num_kv_shared_layers=1,
                    use_double_wide_mlp=double_wide,
                )
                fp_mlp = Gemma4TextMLP(config, layer_idx=1).eval()
                cfg = (
                    build_gemma4_ptq_config(
                        num_text_layers=2,
                        num_vision_layers=0,
                        activation=no_quant(),
                        linear_weight=affine(DType.uint(4)),
                    )
                    .child("layers")
                    .child("1")
                    .child("mlp")
                )
                wrapped = QuantGemma4TextMLP(fp_mlp, qcfg=cfg).eval()
                x = torch.randn(2, 3, config.hidden_size)
                with torch.no_grad():
                    torch.testing.assert_close(wrapped(x), fp_mlp(x), atol=0, rtol=0)
                for name, expected_shape in (
                    ("gate_proj", (expected_width, 16)),
                    ("up_proj", (expected_width, 16)),
                    ("down_proj", (16, expected_width)),
                ):
                    projection = getattr(wrapped, name).wrapped
                    self.assertIs(projection.module, getattr(fp_mlp, name))
                    self.assertEqual(
                        tuple(projection.module.weight.shape), expected_shape
                    )
                    self.assertEqual(projection.obs_weight.dtype, DType.uint(4))

                prepared = prepare(fp_mlp, cfg, inplace=False)
                with torch.no_grad():
                    prepared(x)
                    quantized = convert(prepared)
                    result = quantized(x)
                self.assertEqual(result.shape, x.shape)
                self.assertTrue(torch.isfinite(result).all())


if __name__ == "__main__":
    unittest.main()
