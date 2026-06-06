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

import unittest

import torch
from tico.passes.decompose_fake_quantize import DecomposeFakeQuantize
from tico.passes.decompose_fake_quantize_tensor_qparams import (
    DecomposeFakeQuantizeTensorQParams,
)
from tico.quantization import convert, prepare
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.passes.fold_quant_ops import FoldQuantOps
from tico.serialize.quant_param import QPARAM_KEY

from test.support.helper import num_of_ops


class S16ToU8Relu(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inp):
        i16_qmin = -32768
        i16_qmax = 32767
        u8_qmin = 0
        u8_qmax = 255
        quant_inp = torch.fake_quantize_per_tensor_affine(
            inp, 1.0, 0, i16_qmin, i16_qmax
        )
        output = torch.nn.functional.relu(quant_inp)
        output = torch.fake_quantize_per_tensor_affine(
            output, 1.0, 0, i16_qmin, i16_qmax
        )
        output = torch.fake_quantize_per_tensor_affine(output, 1.0, 0, u8_qmin, u8_qmax)

        return output

    def get_example_inputs(self):
        return (torch.randn(3, 3),), {}


class MXFakeQuantize(torch.nn.Module):
    """Module that produces an MX Q-DQ pattern after decomposition."""

    def __init__(self):
        super().__init__()

    def forward(self, inp):
        return torch.ops.circle_custom.mx_fake_quantize(inp, "int8", -1)

    def get_example_inputs(self):
        return (torch.randn(2, 32),), {}


class FoldQuantOpsTest(unittest.TestCase):
    def test_pass(self):
        m = torch.nn.SiLU().eval()

        q_m = prepare(m, PTQConfig())

        # Calibration
        for _ in range(10):
            cal_args = (torch.randn(2, 3),)
            q_m(*cal_args)

        q_m = convert(q_m)

        ep = torch.export.export(q_m, cal_args)

        DecomposeFakeQuantizeTensorQParams().call(ep)
        # input, other, sub
        self.assertEqual(
            num_of_ops(
                ep, [torch.ops.quantized_decomposed.quantize_per_tensor.default]
            ),
            3,
        )
        self.assertEqual(
            num_of_ops(
                ep, [torch.ops.quantized_decomposed.dequantize_per_tensor.default]
            ),
            3,
        )
        for node in ep.graph.nodes:
            if node.op != "call_function":
                continue
            if node.target == torch.ops.aten.sigmoid.default:
                self.assertFalse(QPARAM_KEY in node.meta)

        target_pass = FoldQuantOps()
        target_pass.call(ep)

        self.assertEqual(
            num_of_ops(
                ep, [torch.ops.quantized_decomposed.quantize_per_tensor.default]
            ),
            0,
        )
        self.assertEqual(
            num_of_ops(
                ep, [torch.ops.quantized_decomposed.dequantize_per_tensor.default]
            ),
            0,
        )
        for node in ep.graph.nodes:
            if node.op != "call_function":
                continue
            if node.target == torch.ops.aten.sigmoid.default:
                self.assertTrue(QPARAM_KEY in node.meta)

    def test_requantize(self):
        m = S16ToU8Relu()
        args, kwargs = m.get_example_inputs()
        ep = torch.export.export(m, args, kwargs)

        decomp = DecomposeFakeQuantize()
        decomp.call(ep)

        self.assertEqual(
            num_of_ops(
                ep, [torch.ops.quantized_decomposed.quantize_per_tensor.default]
            ),
            3,
        )
        self.assertEqual(
            num_of_ops(
                ep, [torch.ops.quantized_decomposed.dequantize_per_tensor.default]
            ),
            3,
        )

        target_pass = FoldQuantOps()
        target_pass.call(ep)

        self.assertEqual(
            num_of_ops(
                ep, [torch.ops.quantized_decomposed.quantize_per_tensor.default]
            ),
            1,
        )
        self.assertEqual(
            num_of_ops(
                ep, [torch.ops.quantized_decomposed.dequantize_per_tensor.default]
            ),
            0,
        )

        for node in ep.graph.nodes:
            if node.op != "call_function":
                continue
            if (
                node.target
                == torch.ops.quantized_decomposed.quantize_per_tensor.default
            ):
                self.assertTrue(QPARAM_KEY in node.meta)

    def test_fold_mx_qdq_to_producer_qparam(self):
        m = MXFakeQuantize().eval()
        args, kwargs = m.get_example_inputs()
        ep = torch.export.export(m, args, kwargs)

        DecomposeFakeQuantize().call(ep)
        self.assertEqual(
            num_of_ops(ep, [torch.ops.circle_custom.quantize_mx.default]), 1
        )
        self.assertEqual(
            num_of_ops(ep, [torch.ops.circle_custom.dequantize_mx.default]), 1
        )

        FoldQuantOps().call(ep)

        self.assertEqual(
            num_of_ops(ep, [torch.ops.circle_custom.quantize_mx.default]), 0
        )
        self.assertEqual(
            num_of_ops(ep, [torch.ops.circle_custom.dequantize_mx.default]), 0
        )

        found_input_qparam = False
        for node in ep.graph.nodes:
            if node.op == "placeholder" and QPARAM_KEY in node.meta:
                self.assertEqual(node.meta[QPARAM_KEY].dtype, "mxint8")
                self.assertEqual(node.meta[QPARAM_KEY].quantized_dimension, -1)
                found_input_qparam = True
        self.assertTrue(found_input_qparam)
