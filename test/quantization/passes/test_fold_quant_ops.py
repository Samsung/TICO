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

from test.modules.op.sub import SimpleSub
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

    def test_fold_per_channel_activation(self):
        """Per-channel qdq on an activation folds into a full per-channel QPARAM."""
        scales = torch.tensor([0.01, 0.02, 0.005, 0.04])
        zps = torch.zeros(4, dtype=torch.int32)

        class PerChannelActQuant(torch.nn.Module):
            def __init__(self, scales, zps, axis, qmin, qmax):
                super().__init__()
                self.register_buffer("scales", scales)
                self.register_buffer("zps", zps)
                self.axis = axis
                self.qmin = qmin
                self.qmax = qmax

            def forward(self, x):
                y = x * 2.0
                return torch.fake_quantize_per_channel_affine(
                    y, self.scales, self.zps, self.axis, self.qmin, self.qmax
                )

        m = PerChannelActQuant(scales, zps, axis=1, qmin=-32768, qmax=32767)
        ep = torch.export.export(m, (torch.randn(2, 4),))

        DecomposeFakeQuantize().call(ep)
        self.assertEqual(
            num_of_ops(
                ep, [torch.ops.quantized_decomposed.quantize_per_channel.default]
            ),
            1,
        )
        self.assertEqual(
            num_of_ops(
                ep, [torch.ops.quantized_decomposed.dequantize_per_channel.default]
            ),
            1,
        )

        FoldQuantOps().call(ep)

        self.assertEqual(
            num_of_ops(
                ep, [torch.ops.quantized_decomposed.quantize_per_channel.default]
            ),
            0,
        )
        self.assertEqual(
            num_of_ops(
                ep, [torch.ops.quantized_decomposed.dequantize_per_channel.default]
            ),
            0,
        )

        mul_nodes = [
            node
            for node in ep.graph.nodes
            if node.op == "call_function" and node.target == torch.ops.aten.mul.Tensor
        ]
        self.assertEqual(len(mul_nodes), 1)
        qparam = mul_nodes[0].meta[QPARAM_KEY]
        self.assertEqual(qparam.scale, scales.tolist())
        self.assertEqual(qparam.zero_point, [0, 0, 0, 0])
        self.assertEqual(qparam.quantized_dimension, 1)
        self.assertEqual(qparam.dtype, "int16")

    def test_fold_per_channel_activation_asym_negative_axis(self):
        """Asymmetric per-channel grid is kept in full; negative axis normalized."""
        scales = torch.tensor([0.2, 0.03])
        zps = torch.tensor([40, 130], dtype=torch.int32)

        class PerChannelActQuant(torch.nn.Module):
            def __init__(self, scales, zps, axis, qmin, qmax):
                super().__init__()
                self.register_buffer("scales", scales)
                self.register_buffer("zps", zps)
                self.axis = axis
                self.qmin = qmin
                self.qmax = qmax

            def forward(self, x):
                y = x * 2.0
                return torch.fake_quantize_per_channel_affine(
                    y, self.scales, self.zps, self.axis, self.qmin, self.qmax
                )

        m = PerChannelActQuant(scales, zps, axis=2, qmin=0, qmax=255)
        ep = torch.export.export(m, (torch.randn(2, 5, 2),))

        DecomposeFakeQuantize().call(ep)

        # Simulate a calibrated graph carrying axis=-1 (aten kernel requires a
        # non-negative axis at export time, but the fold must normalize -1).
        for node in ep.graph.nodes:
            if node.op != "call_function":
                continue
            if "per_channel" not in str(node.target):
                continue
            args = list(node.args)
            args[3] = -1
            node.args = tuple(args)

        FoldQuantOps().call(ep)

        self.assertEqual(
            num_of_ops(
                ep, [torch.ops.quantized_decomposed.dequantize_per_channel.default]
            ),
            0,
        )

        mul_nodes = [
            node
            for node in ep.graph.nodes
            if node.op == "call_function" and node.target == torch.ops.aten.mul.Tensor
        ]
        self.assertEqual(len(mul_nodes), 1)
        qparam = mul_nodes[0].meta[QPARAM_KEY]
        self.assertEqual(qparam.scale, scales.tolist())
        self.assertEqual(qparam.zero_point, zps.tolist())
        # axis=-1 over a 3D tensor must be normalized to 2
        self.assertEqual(qparam.quantized_dimension, 2)
        self.assertEqual(qparam.dtype, "uint8")
