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

"""Unit tests for GPTQ qparam handoff to folded linear layers."""

import unittest

import torch
import torch.nn as nn
from tico.quantization.algorithm.gptq.quant import quantize, Quantizer
from tico.quantization.config import affine, PTQConfig
from tico.quantization.recipes.qparams import inject_gptq_qparams
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.utils.linear_folding import fold_input_affine_into_linear
from tico.quantization.wrapq.wrappers.nn.quant_linear import QuantLinear


class FoldedLinearQParamHandoffTest(unittest.TestCase):
    """Validate scale adjustment and conservative rejection behavior."""

    @staticmethod
    def _source():
        """Build deterministic GPTQ-grid weights and their real quantizer."""
        source = nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            source.weight.copy_(torch.tensor([[-1.0, 0.0, 2.0], [-0.5, 0.5, 1.0]]))
        quantizer = Quantizer()
        quantizer.configure(4, perchannel=True, sym=False)
        quantizer.find_params(source.weight, weight=True)
        with torch.no_grad():
            source.weight.copy_(
                quantize(source.weight, quantizer.scale, quantizer.zero, quantizer.maxq)
            )
        return source, quantizer

    @staticmethod
    def _wrapper(linear, name="vision_tower.patch_embedder.input_proj"):
        """Prepare a real weight observer for the post-fold linear layer."""
        wrapper = QuantLinear(
            linear, qcfg=PTQConfig(weight=affine(DType.uint(4))), fp_name=name
        )
        wrapper.enable_calibration()
        return wrapper

    def test_positive_uniform_fold_scales_reused_weight_qparams(self):
        """Multiply GPTQ scales by the fold while preserving quantized weights."""
        source, quantizer = self._source()
        folded = fold_input_affine_into_linear(source, scale=2.0, shift=-1.0)
        wrapper = self._wrapper(folded)

        result = inject_gptq_qparams(wrapper, {wrapper.fp_name: quantizer})

        observer = wrapper.obs_weight
        torch.testing.assert_close(
            observer._cached_scale, quantizer.scale.flatten() * 2.0, rtol=0, atol=0
        )
        torch.testing.assert_close(
            observer._cached_zp, quantizer.zero.flatten().to(torch.int), rtol=0, atol=0
        )
        torch.testing.assert_close(
            observer.fake_quant(folded.weight), folded.weight, rtol=0, atol=0
        )
        self.assertTrue(observer._qparams_locked)
        self.assertEqual(result, {"matched": 1, "missed": 0, "unused": 0})

    def test_unfolded_linear_keeps_reused_weight_qparams(self):
        """Preserve GPTQ scales and zero-points without folding metadata."""
        linear, quantizer = self._source()
        wrapper = self._wrapper(linear)

        result = inject_gptq_qparams(wrapper, {wrapper.fp_name: quantizer})

        torch.testing.assert_close(
            wrapper.obs_weight._cached_scale, quantizer.scale.flatten(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            wrapper.obs_weight.fake_quant(linear.weight), linear.weight, rtol=0, atol=0
        )
        self.assertTrue(wrapper.obs_weight._qparams_locked)
        self.assertEqual(result, {"matched": 1, "missed": 0, "unused": 0})

    def test_nonuniform_fold_requires_weight_qparam_recomputation(self):
        """Reject reuse when one scalar multiplier cannot describe the fold."""
        source, quantizer = self._source()
        folded = fold_input_affine_into_linear(
            source, scale=torch.tensor([1.0, 2.0, 3.0]), shift=0.0
        )
        wrapper = self._wrapper(folded)

        with self.assertRaisesRegex(ValueError, "Recompute"):
            inject_gptq_qparams(wrapper, {wrapper.fp_name: quantizer})
        self.assertFalse(wrapper.obs_weight.has_qparams)
        self.assertTrue(wrapper.obs_weight.enabled)

    def test_late_invalid_fold_does_not_partially_inject_qparams(self):
        """A later folding error must leave earlier weight observers untouched."""
        source, quantizer = self._source()
        first = self._wrapper(source, "first")
        folded = fold_input_affine_into_linear(
            source, scale=torch.tensor([1.0, 2.0, 3.0]), shift=0.0
        )
        last = self._wrapper(folded, "last")
        first.obs_weight.compute_qparams()
        previous_scale = first.obs_weight._cached_scale
        previous_zero = first.obs_weight._cached_zp

        with self.assertRaisesRegex(ValueError, "last"):
            inject_gptq_qparams(
                nn.Sequential(first, last), {"first": quantizer, "last": quantizer}
            )

        self.assertIs(first.obs_weight._cached_scale, previous_scale)
        self.assertIs(first.obs_weight._cached_zp, previous_zero)
        self.assertFalse(first.obs_weight._qparams_locked)
        self.assertTrue(first.obs_weight.enabled)
        self.assertFalse(last.obs_weight.has_qparams)


if __name__ == "__main__":
    unittest.main()
