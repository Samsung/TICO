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

import contextlib
import io
import unittest

import torch
from tico.quantization.algorithm.gptq.gptq import (
    convtranspose2d_weights_to_conv2d_weights,
)
from tico.quantization.algorithm.gptq.quant import quantize, Quantizer
from tico.quantization.config import affine, PTQConfig
from tico.quantization.recipes.qparams import (
    clear_gptq_quantizers,
    find_gptq_quantizers,
    inject_gptq_qparams,
)
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.qscheme import QScheme
from tico.quantization.wrapq.wrappers.nn.quant_conv_transpose2d import (
    QuantConvTranspose2d,
)
from tico.quantization.wrapq.wrappers.nn.quant_linear import QuantLinear


class TestRecipeQParams(unittest.TestCase):
    @staticmethod
    def _pair(gptq_bits=4, ptq_bits=4, *, name="linear", symmetric=False):
        """Prepare a tiny real affine GPTQ/PTQ weight pair."""
        linear = torch.nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            linear.weight.copy_(
                torch.tensor([[-8.0, -4.0, 0.0, 7.0], [-4.0, -2.0, 0.0, 3.5]])
            )
        quantizer = Quantizer()
        quantizer.configure(gptq_bits, perchannel=True, sym=symmetric)
        quantizer.find_params(linear.weight, weight=True)
        with torch.no_grad():
            linear.weight.copy_(
                quantize(linear.weight, quantizer.scale, quantizer.zero, quantizer.maxq)
            )
        wrapper = QuantLinear(
            linear,
            qcfg=PTQConfig(weight=affine(DType.uint(ptq_bits))),
            fp_name=name,
        )
        wrapper.enable_calibration()
        return wrapper, quantizer

    def test_find_gptq_quantizers_checks_common_owner_locations(self):
        """GPTQ quantizer lookup should search wrapper owner locations."""
        owner = torch.nn.Module()
        quantizers = {"linear": object()}
        setattr(owner, "quantizers", quantizers)
        wrapped = torch.nn.Module()
        wrapped.module = owner
        root = torch.nn.Module()
        root.wrapped = wrapped

        found_owner, found_quantizers = find_gptq_quantizers(root)

        self.assertIs(found_owner, owner)
        self.assertEqual(found_quantizers, quantizers)

    def test_clear_gptq_quantizers_removes_all_common_owner_attributes(self):
        """Quantizer cleanup should remove dictionaries from all candidates."""
        root = torch.nn.Module()
        root.wrapped = torch.nn.Module()
        root.wrapped.module = torch.nn.Module()
        for owner in (root, root.wrapped, root.wrapped.module):
            setattr(owner, "quantizers", {"linear": object()})

        clear_gptq_quantizers(root)

        self.assertFalse(hasattr(root, "quantizers"))
        self.assertFalse(hasattr(root.wrapped, "quantizers"))
        self.assertFalse(hasattr(root.wrapped.module, "quantizers"))

    def test_inject_gptq_qparams_reports_matched_missed_and_unused(self):
        """Only matched weights are frozen; unmatched observers keep collecting."""
        matched, quantizer = self._pair(name="matched")
        missed, _ = self._pair(name="missed")
        root = torch.nn.Sequential(matched, missed)

        with contextlib.redirect_stdout(io.StringIO()):
            stats = inject_gptq_qparams(
                root, {"matched": quantizer, "unused": object()}, verbose=True
            )

        self.assertEqual(stats, {"matched": 1, "missed": 1, "unused": 1})
        self.assertTrue(matched.obs_weight._qparams_locked)
        self.assertFalse(missed.obs_weight.has_qparams)
        self.assertTrue(missed.obs_weight.enabled)
        self.assertTrue(matched.obs_act_in.enabled)
        torch.testing.assert_close(
            matched.obs_weight._cached_scale, quantizer.scale.flatten(), rtol=0, atol=0
        )

    def test_matching_bits_preserve_weights_qparams_and_lock(self):
        """Matching 4/8-bit grids survive activation calibration and conversion."""
        for bits, symmetric in ((4, False), (8, False), (4, True), (8, True)):
            with self.subTest(bits=bits, symmetric=symmetric):
                wrapper, quantizer = self._pair(bits, bits, symmetric=symmetric)
                weight = wrapper.module.weight.detach().clone()
                inject_gptq_qparams(wrapper, {"linear": quantizer})
                wrapper(torch.tensor([[1.0, -1.0, 0.5, 2.0]]))
                wrapper.obs_weight.collect(torch.full_like(weight, 1000.0))
                wrapper.freeze_qparams()

                self.assertTrue(wrapper.obs_weight._qparams_locked)
                self.assertFalse(wrapper.obs_weight.enabled)
                self.assertTrue(wrapper.obs_act_in.has_qparams)
                self.assertFalse(wrapper.obs_act_in._qparams_locked)
                torch.testing.assert_close(
                    wrapper.obs_weight._cached_scale,
                    quantizer.scale.flatten(),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    wrapper.obs_weight._cached_zp,
                    quantizer.zero.flatten().to(torch.int),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    wrapper.obs_weight.fake_quant(weight), weight, rtol=0, atol=0
                )

    def test_bit_mismatch_rejected_in_both_directions(self):
        """A wider PTQ range must fail just as a narrower range does."""
        for source_bits, target_bits in ((4, 8), (8, 4)):
            with self.subTest(source_bits=source_bits, target_bits=target_bits):
                wrapper, quantizer = self._pair(source_bits, target_bits)
                with self.assertRaisesRegex(ValueError, "linear") as raised:
                    inject_gptq_qparams(wrapper, {"linear": quantizer})
                self.assertIn(f"uint{target_bits}", str(raised.exception))
                self.assertIn(str(2**source_bits - 1), str(raised.exception))
                self.assertFalse(wrapper.obs_weight.has_qparams)
                self.assertTrue(wrapper.obs_weight.enabled)

    def test_signed_target_with_same_bit_count_is_rejected(self):
        """Matching bit counts do not make signed/unsigned grids compatible."""
        wrapper, quantizer = self._pair()
        wrapper.obs_weight.dtype = DType.int(4)
        wrapper.obs_weight.qscheme = QScheme.PER_CHANNEL_SYMM
        with self.assertRaisesRegex(ValueError, "int4"):
            inject_gptq_qparams(wrapper, {"linear": quantizer})
        self.assertFalse(wrapper.obs_weight.has_qparams)

    def test_late_invalid_pair_does_not_mutate_earlier_observers(self):
        """Validate the whole handoff before replacing any existing qparams."""
        first, first_quantizer = self._pair(name="first")
        last, last_quantizer = self._pair(4, 8, name="last")
        first.obs_weight.compute_qparams()
        original_scale = first.obs_weight._cached_scale
        original_zero = first.obs_weight._cached_zp

        with self.assertRaisesRegex(ValueError, "last"):
            inject_gptq_qparams(
                torch.nn.Sequential(first, last),
                {"first": first_quantizer, "last": last_quantizer},
            )

        self.assertIs(first.obs_weight._cached_scale, original_scale)
        self.assertIs(first.obs_weight._cached_zp, original_zero)
        self.assertTrue(first.obs_weight.enabled)
        self.assertFalse(first.obs_weight._qparams_locked)
        self.assertFalse(last.obs_weight.has_qparams)

    def test_invalid_source_metadata_is_rejected(self):
        """Malformed metadata must not reach observer casting or reshaping."""
        cases = (
            ("maxq", torch.tensor([15, 15])),
            ("maxq", torch.tensor(float("nan"))),
            ("maxq", torch.tensor(15.5)),
            ("maxq", torch.tensor(-1)),
            ("perchannel", "true"),
            ("scale", torch.tensor([[0.0], [0.5]])),
            ("scale", torch.tensor([[-1.0], [0.5]])),
            ("scale", torch.tensor([[float("inf")], [0.5]])),
            ("scale", torch.tensor([[1], [2]])),
            ("scale", torch.ones(2, 1, dtype=torch.float16)),
            ("scale", torch.ones(2, 1, dtype=torch.bfloat16)),
            ("scale", torch.ones(2, 1, dtype=torch.float64)),
            ("zero", torch.tensor([[0.5], [1.0]])),
            ("zero", torch.tensor([[float("nan")], [1.0]])),
            ("zero", torch.tensor([[16.0], [1.0]])),
            ("zero", torch.tensor([[-1.0], [1.0]])),
            ("scale", torch.ones(1, 2)),
            ("scale", torch.ones(3, 1)),
            ("scale", torch.empty(0)),
        )
        for field, value in cases:
            with self.subTest(field=field, value=value):
                wrapper, quantizer = self._pair()
                setattr(quantizer, field, value)
                with self.assertRaisesRegex(ValueError, "linear"):
                    inject_gptq_qparams(wrapper, {"linear": quantizer})
                self.assertFalse(wrapper.obs_weight.has_qparams)
                self.assertTrue(wrapper.obs_weight.enabled)

    def test_missing_range_metadata_is_rejected(self):
        """Scale/zero-only metadata cannot prove the source integer domain."""
        wrapper, quantizer = self._pair()
        del quantizer.maxq
        with self.assertRaisesRegex(ValueError, "maxq"):
            inject_gptq_qparams(wrapper, {"linear": quantizer})

    def test_incompatible_channel_axis_is_rejected(self):
        """Linear weight qparams belong to output channels, not input channels."""
        for axis in (1, False, 0.0):
            with self.subTest(axis=axis):
                wrapper, quantizer = self._pair()
                wrapper.obs_weight.channel_axis = axis
                with self.assertRaisesRegex(ValueError, "axis"):
                    inject_gptq_qparams(wrapper, {"linear": quantizer})

    def test_nonuniform_per_channel_params_cannot_become_per_tensor(self):
        """A scalar observer cannot represent distinct output-channel scales."""
        wrapper, quantizer = self._pair()
        wrapper.obs_weight.channel_axis = None
        wrapper.obs_weight.qscheme = QScheme.PER_TENSOR_ASYMM
        with self.assertRaisesRegex(ValueError, "linear"):
            inject_gptq_qparams(wrapper, {"linear": quantizer})

    def test_equivalent_uniform_granularity_mappings_preserve_values(self):
        """Uniform affine grids may collapse or expand without changing values."""
        for source_perchannel in (False, True):
            for target_perchannel in (False, True):
                with self.subTest(source=source_perchannel, target=target_perchannel):
                    wrapper, quantizer = self._pair()
                    quantizer.perchannel = source_perchannel
                    quantizer.scale = torch.tensor([[0.5], [0.5]])
                    quantizer.zero = torch.tensor([[8.0], [8.0]])
                    if not target_perchannel:
                        wrapper.obs_weight.channel_axis = None
                        wrapper.obs_weight.qscheme = QScheme.PER_TENSOR_ASYMM
                    expected = quantize(
                        wrapper.module.weight,
                        quantizer.scale,
                        quantizer.zero,
                        quantizer.maxq,
                    )

                    inject_gptq_qparams(wrapper, {"linear": quantizer})

                    self.assertEqual(
                        wrapper.obs_weight._cached_scale.numel(),
                        2 if target_perchannel else 1,
                    )
                    torch.testing.assert_close(
                        wrapper.obs_weight.fake_quant(wrapper.module.weight),
                        expected,
                        rtol=0,
                        atol=0,
                    )

    def test_per_tensor_scalar_scales_keep_supported_float_dtypes(self):
        """Per-channel kernel constraints must not reject valid scalar scales."""
        for dtype in (torch.float16, torch.bfloat16, torch.float64):
            with self.subTest(dtype=dtype):
                wrapper, quantizer = self._pair()
                quantizer.perchannel = False
                quantizer.scale = torch.tensor(0.5, dtype=dtype)
                quantizer.zero = torch.tensor(8.0)
                wrapper.obs_weight.channel_axis = None
                wrapper.obs_weight.qscheme = QScheme.PER_TENSOR_ASYMM

                inject_gptq_qparams(wrapper, {"linear": quantizer})

                self.assertEqual(wrapper.obs_weight._cached_scale.dtype, dtype)
                torch.testing.assert_close(
                    wrapper.obs_weight.fake_quant(torch.tensor([-1.0, 0.0, 1.0])),
                    torch.tensor([-1.0, 0.0, 1.0]),
                    rtol=0,
                    atol=0,
                )

    def test_conv_transpose_uses_output_channel_axis_one(self):
        """ConvTranspose2d stores the GPTQ output-channel grid on axis one."""
        layer = torch.nn.ConvTranspose2d(2, 3, kernel_size=1, bias=False)
        with torch.no_grad():
            layer.weight.copy_(
                torch.tensor([[-1.0, -2.0, -3.0], [2.0, 4.0, 6.0]]).reshape(2, 3, 1, 1)
            )
        output_major = convtranspose2d_weights_to_conv2d_weights(layer, layer.weight)
        quantizer = Quantizer()
        quantizer.configure(4, perchannel=True, sym=False)
        quantizer.find_params(output_major, weight=True)
        expected = quantize(
            output_major, quantizer.scale, quantizer.zero, quantizer.maxq
        ).transpose(0, 1)
        wrapper = QuantConvTranspose2d(
            layer, qcfg=PTQConfig(weight=affine(DType.uint(4))), fp_name="deconv"
        )
        wrapper.enable_calibration()

        inject_gptq_qparams(wrapper, {"deconv": quantizer})

        self.assertEqual(wrapper.obs_weight.channel_axis, 1)
        torch.testing.assert_close(
            wrapper.obs_weight.fake_quant(layer.weight), expected, rtol=0, atol=0
        )

    def test_grouped_conv_transpose_rejects_distinct_unrepresentable_grids(self):
        """Grouped output grids cannot be flattened into the smaller stored axis."""
        layer = torch.nn.ConvTranspose2d(4, 4, kernel_size=1, groups=2, bias=False)
        with torch.no_grad():
            layer.weight.copy_(torch.arange(1.0, 9.0).reshape(4, 2, 1, 1))
        output_major = convtranspose2d_weights_to_conv2d_weights(layer, layer.weight)
        quantizer = Quantizer()
        quantizer.configure(4, perchannel=True, sym=False)
        quantizer.find_params(output_major, weight=True)
        wrapper = QuantConvTranspose2d(
            layer, qcfg=PTQConfig(weight=affine(DType.uint(4))), fp_name="grouped"
        )
        wrapper.enable_calibration()

        with self.assertRaisesRegex(ValueError, "grouped"):
            inject_gptq_qparams(wrapper, {"grouped": quantizer})
        self.assertFalse(wrapper.obs_weight.has_qparams)


if __name__ == "__main__":
    unittest.main()
