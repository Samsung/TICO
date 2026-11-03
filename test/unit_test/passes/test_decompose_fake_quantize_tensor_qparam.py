# Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

import tempfile
import unittest
from pathlib import Path
from typing import Dict, Union
from unittest import mock

import numpy as np
import tico
import tico.passes.decompose_fake_quantize_tensor_qparams as fq_tqparams_module
import torch
from tico.circle import CircleDocument
from tico.circle.runtime import CircleReferenceRuntime
from tico.passes.const_prop_pass import ConstPropPass
from tico.passes.decompose_fake_quantize import DecomposeFakeQuantize
from tico.passes.decompose_fake_quantize_tensor_qparams import (
    DecomposeFakeQuantizeTensorQParams,
    get_constant_from_tensor,
)
from tico.passes.fill_meta_val import FillMetaVal
from tico.quantization.passes.fold_quant_ops import FoldQuantOps
from tico.quantization.passes.remove_weight_dequant_op import RemoveWeightDequantOp
from tico.serialize.quant_param import QPARAM_KEY

from test.support.helper import num_of_ops
from test.support.pass_value_test import PassTest, SinglePassValueTest


class FakeQuantizeTensorQParamPerTensor(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        s = torch.ones([1])
        zp = torch.zeros([1])
        qmin = 0
        qmax = 255

        return torch.fake_quantize_per_tensor_affine(input, s, zp, qmin, qmax)

    def get_example_inputs(self):
        return (torch.randn(1, 3, 32, 32),), {}


class DecomposeFakeQuantizeTensorQParamPerTensor(SinglePassValueTest):
    def test_pass(self):
        self.setup(FakeQuantizeTensorQParamPerTensor())
        self.assertEqual(
            num_of_ops(
                self.exported_program(),
                [torch.ops.aten.fake_quantize_per_tensor_affine.tensor_qparams],
            ),
            1,
        )

        self.run_value_test(DecomposeFakeQuantizeTensorQParams())
        self.assertEqual(
            num_of_ops(
                self.exported_program(),
                [torch.ops.aten.fake_quantize_per_tensor_affine.tensor_qparams],
            ),
            0,
        )
        self.assertEqual(
            num_of_ops(
                self.exported_program(),
                [torch.ops.quantized_decomposed.quantize_per_tensor.default],
            ),
            1,
        )
        self.assertEqual(
            num_of_ops(
                self.exported_program(),
                [torch.ops.quantized_decomposed.dequantize_per_tensor.default],
            ),
            1,
        )


class FakeQuantizeTensorQParamUint4Dtype(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 10)
        self.w_scale = torch.tensor([1.0] * self.linear.out_features)
        self.w_zp = torch.zeros(self.linear.out_features)
        self.w_qmin = 0
        self.w_qmax = 15
        self.a_qmin = 0
        self.a_qmax = 255

    def forward(self, input):
        q_weight = torch.fake_quantize_per_channel_affine(
            self.linear.weight, self.w_scale, self.w_zp, 0, self.w_qmin, self.w_qmax
        )
        quantized_input = torch.fake_quantize_per_tensor_affine(
            input, torch.tensor(0.1), torch.tensor(0), self.a_qmin, self.a_qmax
        )
        linear = torch.nn.functional.linear(quantized_input, q_weight)
        q_linear = torch.fake_quantize_per_tensor_affine(
            linear, torch.tensor(0.1), torch.tensor(0), self.a_qmin, self.a_qmax
        )
        return q_linear

    def get_example_inputs(self):
        return (torch.randn(1, 5),), {}


class DecomposeFakeQuantizeTensorQParamUint4Dtype(PassTest):
    def test_pass(self):
        self.setup(FakeQuantizeTensorQParamUint4Dtype())
        self.run_pass(DecomposeFakeQuantize())
        self.run_pass(DecomposeFakeQuantizeTensorQParams())
        self.run_pass(ConstPropPass())
        self.run_pass(FillMetaVal())
        self.run_pass(FoldQuantOps())
        self.run_pass(RemoveWeightDequantOp())
        for n in self.ep.graph.nodes:
            if QPARAM_KEY not in n.meta:
                continue
            if n.op == "placeholder":
                if n.target == "input":
                    self.assertEqual(n.meta[QPARAM_KEY].dtype, "uint8")
                else:  # linear weight
                    self.assertEqual(n.meta[QPARAM_KEY].dtype, "uint4")
            else:
                self.assertEqual(n.target, torch.ops.aten.linear.default)
                self.assertEqual(n.meta[QPARAM_KEY].dtype, "uint8")


class FakeQuantizeTensorQParamsCacheMaskModel(torch.nn.Module):
    def __init__(self, scale_val, zp_val, quant_min, quant_max):
        super().__init__()
        self.scale_val = scale_val
        self.zp_val = zp_val
        self.quant_min = quant_min
        self.quant_max = quant_max
        # These will become lifted tensor constants
        self.register_buffer("scale_tensor", torch.tensor(scale_val))
        self.register_buffer("zp_tensor", torch.tensor(zp_val))
        self.register_buffer("fq_enabled_tensor", torch.tensor(True))

    def forward(self, x):
        res = torch.ops.aten._fake_quantize_per_tensor_affine_cachemask_tensor_qparams.default(
            x,
            self.scale_tensor,
            self.zp_tensor,
            self.fq_enabled_tensor,
            self.quant_min,
            self.quant_max,
        )
        return res[0]

    def get_example_inputs(self):
        return ((torch.randn(2, 3),), {})


class DecomposeFakeQuantizeTensorQParamsCacheMaskTest(SinglePassValueTest):
    def test_decompose_uint8(self):
        model = FakeQuantizeTensorQParamsCacheMaskModel(0.1, 0, 0, 255)
        self.setup(model)

        ep_before = self.exported_program()
        self.assertEqual(
            num_of_ops(
                ep_before,
                [
                    torch.ops.aten._fake_quantize_per_tensor_affine_cachemask_tensor_qparams.default
                ],
            ),
            1,
        )
        self.assertEqual(
            num_of_ops(
                ep_before, [torch.ops.quantized_decomposed.quantize_per_tensor.default]
            ),
            0,
        )
        self.assertEqual(
            num_of_ops(
                ep_before,
                [torch.ops.quantized_decomposed.dequantize_per_tensor.default],
            ),
            0,
        )

        self.run_value_test(DecomposeFakeQuantizeTensorQParams())

        ep_after = self.exported_program()
        self.assertEqual(
            num_of_ops(
                ep_after,
                [
                    torch.ops.aten._fake_quantize_per_tensor_affine_cachemask_tensor_qparams.default
                ],
            ),
            0,
        )
        self.assertEqual(
            num_of_ops(
                ep_after, [torch.ops.quantized_decomposed.quantize_per_tensor.default]
            ),
            1,
        )
        self.assertEqual(
            num_of_ops(
                ep_after, [torch.ops.quantized_decomposed.dequantize_per_tensor.default]
            ),
            1,
        )
        # Check dtype inference
        quant_node = None
        for n in ep_after.graph_module.graph.nodes:
            if n.target == torch.ops.quantized_decomposed.quantize_per_tensor.default:
                quant_node = n
                break
        self.assertIsNotNone(quant_node)
        self.assertEqual(quant_node.kwargs["dtype"], torch.uint8)  # type: ignore[union-attr]

    def test_decompose_int16(self):
        model = FakeQuantizeTensorQParamsCacheMaskModel(0.05, 0, -32768, 32767)
        self.setup(model)
        self.run_value_test(DecomposeFakeQuantizeTensorQParams())

        ep_after = self.exported_program()
        self.assertEqual(
            num_of_ops(
                ep_after,
                [
                    torch.ops.aten._fake_quantize_per_tensor_affine_cachemask_tensor_qparams.default
                ],
            ),
            0,
        )
        quant_node = None
        for n in ep_after.graph_module.graph.nodes:
            if n.target == torch.ops.quantized_decomposed.quantize_per_tensor.default:
                quant_node = n
                break
        self.assertIsNotNone(quant_node)
        self.assertEqual(quant_node.kwargs["dtype"], torch.int16)  # type: ignore[union-attr]


class _CacheMask(torch.nn.Module):
    enabled_tensor: torch.Tensor

    def __init__(self, enabled, indices=(0,), *, lifted=False):
        super().__init__()
        self.enabled = enabled
        self.indices = indices
        self.lifted = lifted
        self.register_buffer("scale", torch.tensor(0.1))
        self.register_buffer("zero_point", torch.tensor(0, dtype=torch.int32))
        self.register_buffer("enabled_tensor", torch.tensor(enabled))

    def forward(self, x):
        enabled = torch.tensor(self.enabled) if self.lifted else self.enabled_tensor
        result = torch.ops.aten._fake_quantize_per_tensor_affine_cachemask_tensor_qparams.default(
            x, self.scale, self.zero_point, enabled, 0, 255
        )
        return tuple(result[index] for index in self.indices)


class _RuntimeCacheMask(torch.nn.Module):
    def forward(self, x, scale, zero_point, enabled):
        return torch.ops.aten._fake_quantize_per_tensor_affine_cachemask_tensor_qparams.default(
            x, scale, zero_point, enabled, 0, 255
        )[
            0
        ]


class DecomposeFakeQuantizeCacheMaskContractTest(unittest.TestCase):
    def setUp(self):
        # Both saturation and rounding expose accidental enabled quantization.
        self.x = torch.tensor([-0.15, 0.123, 30.0])

    def _lower(self, ep):
        inputs = tuple(ep.graph_signature.input_specs)
        with ep.graph_module._set_replace_hook(ep.graph_signature.get_replace_hook()):
            self.assertTrue(DecomposeFakeQuantizeTensorQParams().call(ep).modified)
        self.assertEqual(tuple(ep.graph_signature.input_specs), inputs)
        FillMetaVal().call(ep)
        ep.graph.lint()
        ep.validate()
        self.assertFalse(DecomposeFakeQuantizeTensorQParams().call(ep).modified)

    def test_disabled_preserves_values_masks_and_output_order(self):
        for lifted in (False, True):
            for indices in ((0,), (1,), (0, 1), (1, 0), (-1, -2), (0, 0, 1)):
                with self.subTest(lifted=lifted, indices=indices):
                    model = _CacheMask(False, indices, lifted=lifted).eval()
                    ep = torch.export.export(model, (self.x,))
                    expected = model(self.x)
                    self._lower(ep)
                    actual = ep.module()(self.x)
                    self.assertEqual(len(actual), len(expected))
                    for result, reference in zip(actual, expected):
                        torch.testing.assert_close(result, reference, rtol=0, atol=0)
                    self.assertEqual(
                        num_of_ops(
                            ep,
                            [
                                torch.ops.quantized_decomposed.quantize_per_tensor.default
                            ],
                        ),
                        0,
                    )

    def test_disabled_does_not_require_constant_qparams(self):
        class Disabled(torch.nn.Module):
            def forward(self, x, scale, zero_point):
                result = torch.ops.aten._fake_quantize_per_tensor_affine_cachemask_tensor_qparams.default(
                    x, scale, zero_point, torch.tensor(False), 0, 255
                )
                return result[0]

        args = (self.x, torch.tensor(0.1), torch.tensor(0, dtype=torch.int32))
        ep = torch.export.export(Disabled().eval(), args)
        self._lower(ep)
        torch.testing.assert_close(ep.module()(*args), self.x, rtol=0, atol=0)

    def test_disabled_preserves_scalar_empty_and_non_aliasing_outputs(self):
        for x in (torch.tensor(0.123), torch.empty(0), self.x.clone()):
            with self.subTest(shape=x.shape):
                model = _CacheMask(False, (0, 1)).eval()
                ep = torch.export.export(model, (x,))
                self._lower(ep)
                value, mask = ep.module()(x)
                torch.testing.assert_close(value, x, rtol=0, atol=0)
                torch.testing.assert_close(mask, torch.ones_like(x, dtype=torch.bool))
                saved = x.clone()
                value.fill_(7)
                torch.testing.assert_close(x, saved, rtol=0, atol=0)

    def test_constant_flags_follow_native_integer_conversion(self):
        for flag in (0, 1, 2, -1, 0.5, -0.5):
            with self.subTest(flag=flag):
                model = _CacheMask(flag).eval()
                ep = torch.export.export(model, (self.x,))
                expected = model(self.x)[0]
                self._lower(ep)
                torch.testing.assert_close(
                    ep.module()(self.x)[0], expected, rtol=0, atol=0
                )

    def test_constant_flag_cast_is_applied_before_integer_conversion(self):
        class CastFlag(_CacheMask):
            def __init__(self, use_copy):
                super().__init__(0.5)
                self.use_copy = use_copy

            def forward(self, x):
                if self.use_copy:
                    enabled = torch.ops.aten._to_copy.default(
                        self.enabled_tensor, dtype=torch.bool
                    )
                else:
                    enabled = self.enabled_tensor.to(dtype=torch.bool)
                return torch.ops.aten._fake_quantize_per_tensor_affine_cachemask_tensor_qparams.default(
                    x, self.scale, self.zero_point, enabled, 0, 255
                )[
                    0
                ]

        # .5 truncates to zero, but .5 cast to bool becomes True.
        for use_copy in (False, True):
            with self.subTest(use_copy=use_copy):
                model = CastFlag(use_copy).eval()
                ep = torch.export.export(model, (self.x,))
                self._lower(ep)
                torch.testing.assert_close(
                    ep.module()(self.x), model(self.x), rtol=0, atol=0
                )

    def test_constant_flag_arithmetic_preserves_tensor_dtype_and_rank(self):
        class ArithmeticFlag(_CacheMask):
            def forward(self, x):
                enabled = torch.ones([1], dtype=torch.uint8) * self.enabled_tensor
                result = torch.ops.aten._fake_quantize_per_tensor_affine_cachemask_tensor_qparams.default(
                    x, self.scale, self.zero_point, enabled, 0, 255
                )
                return result[0]

        # A uint8 vector times an int64 scalar wraps to uint8 zero; Python
        # multiplication or promotion to int64 would incorrectly enable Q/DQ.
        model = ArithmeticFlag(256).eval()
        ep = torch.export.export(model, (self.x,))
        self._lower(ep)
        torch.testing.assert_close(ep.module()(self.x), self.x, rtol=0, atol=0)

    def test_enabled_value_only_keeps_quantization(self):
        for lifted in (False, True):
            with self.subTest(lifted=lifted):
                model = _CacheMask(True, (-2,), lifted=lifted).eval()
                ep = torch.export.export(model, (self.x,))
                self._lower(ep)
                actual = ep.module()(self.x)[0]
                torch.testing.assert_close(
                    actual, torch.tensor([0.0, 0.1, 25.5]), rtol=0, atol=0
                )
                self.assertEqual(
                    num_of_ops(
                        ep, [torch.ops.quantized_decomposed.quantize_per_tensor.default]
                    ),
                    1,
                )

    def test_enabled_live_mask_rejects_without_changing_outputs(self):
        for indices in ((1,), (0, 1), (1, 0), (-1, -2)):
            with self.subTest(indices=indices):
                ep = torch.export.export(_CacheMask(True, indices).eval(), (self.x,))
                before = str(ep.graph)
                signature = str(ep.graph_signature)
                with self.assertRaisesRegex(RuntimeError, "mask"):
                    DecomposeFakeQuantizeTensorQParams().call(ep)
                self.assertEqual(str(ep.graph), before)
                self.assertEqual(str(ep.graph_signature), signature)

    def test_runtime_enable_flag_rejects_instead_of_specializing(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                args = (
                    self.x,
                    torch.tensor(0.1),
                    torch.tensor(0, dtype=torch.int32),
                    torch.tensor(enabled),
                )
                ep = torch.export.export(_RuntimeCacheMask().eval(), args)
                before = str(ep.graph)
                with self.assertRaisesRegex(
                    RuntimeError, "fake_quant_enabled.*constant"
                ):
                    DecomposeFakeQuantizeTensorQParams().call(ep)
                self.assertEqual(str(ep.graph), before)

    def test_direct_tuple_use_rejects_without_mutation(self):
        ep = torch.export.export(_CacheMask(False, (0, 1)).eval(), (self.x,))
        cachemask = next(
            node
            for node in ep.graph.nodes
            if node.target
            == torch.ops.aten._fake_quantize_per_tensor_affine_cachemask_tensor_qparams.default
        )
        output = next(node for node in ep.graph.nodes if node.op == "output")
        output.args = ((cachemask,),)
        ep.graph.eliminate_dead_code()
        ep.graph.lint()
        before = str(ep.graph)
        with self.assertRaisesRegex(RuntimeError, "tuple|getitem"):
            DecomposeFakeQuantizeTensorQParams().call(ep)
        self.assertEqual(str(ep.graph), before)

    def test_disabled_saved_circle_preserves_float_abi_and_boolean_mask(self):
        for indices in ((0,), (0, 1), (1, 0)):
            with (
                self.subTest(indices=indices),
                tempfile.TemporaryDirectory() as directory,
            ):
                model = _CacheMask(False, indices).eval()
                # Default conversion includes Circle O1; inspect the saved artifact.
                converted = tico.convert(model, (self.x,))
                path = Path(directory) / "disabled.circle"
                converted.save(path)
                document = CircleDocument.load(path)
                document.verify()
                runtime = CircleReferenceRuntime(document)
                self.assertEqual(len(runtime.input_tensors), 1)
                self.assertEqual(runtime.input_tensors[0].dtype, np.dtype("float32"))
                self.assertEqual(runtime.input_tensors[0].shape, (3,))
                self.assertEqual(len(runtime.output_tensors), len(indices))
                for tensor, index in zip(runtime.output_tensors, indices):
                    self.assertEqual(
                        tensor.dtype, np.dtype("bool" if index == 1 else "float32")
                    )
                    self.assertEqual(tensor.shape, (3,))
                    self.assertIsNone(tensor.quantization)
                self.assertIsNone(runtime.input_tensors[0].quantization)
                self.assertFalse(
                    any(
                        op.name in ("QUANTIZE", "DEQUANTIZE")
                        for op in runtime.program.operators
                    )
                )
                outputs = runtime.run([self.x.numpy()]).outputs
                for result, expected in zip(outputs, model(self.x)):
                    np.testing.assert_array_equal(result, expected.numpy())
                # The public call must continue to accept the original float input.
                public_outputs = converted(self.x)
                if isinstance(public_outputs, np.ndarray):
                    public_outputs = (public_outputs,)
                self.assertEqual(len(public_outputs), len(indices))
                for result, expected in zip(public_outputs, model(self.x)):
                    np.testing.assert_array_equal(result, expected.numpy())


class GetConstantFromTensorMemoTest(PassTest):
    def test_cached_wrapper_reuses_values_within_a_run(self):
        self.setup(FakeQuantizeTensorQParamPerTensor())
        ep = self.exported_program()
        fq_node = next(
            n
            for n in ep.graph.nodes
            if n.op == "call_function"
            and n.target == torch.ops.aten.fake_quantize_per_tensor_affine.tensor_qparams
        )
        scale_arg = fq_node.args[1]

        cache: Dict[Union[str, bool, int, float], Union[bool, int, float]] = {}
        with mock.patch.object(
            fq_tqparams_module,
            "_get_constant_from_tensor_impl",
            wraps=fq_tqparams_module._get_constant_from_tensor_impl,
        ) as impl_spy:
            cached_value = get_constant_from_tensor(scale_arg, ep, _cache=cache)
            calls_after_first = impl_spy.call_count
            cached_again = get_constant_from_tensor(scale_arg, ep, _cache=cache)

        self.assertEqual(cached_again, cached_value)
        self.assertEqual(impl_spy.call_count, calls_after_first)

        with mock.patch.object(
            fq_tqparams_module,
            "_get_constant_from_tensor_impl",
            wraps=fq_tqparams_module._get_constant_from_tensor_impl,
        ) as impl_spy_uncached:
            uncached_value = get_constant_from_tensor(scale_arg, ep)

        self.assertEqual(uncached_value, cached_value)
        self.assertGreaterEqual(impl_spy_uncached.call_count, 1)
