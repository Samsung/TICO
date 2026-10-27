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


class MXRequantize(torch.nn.Module):
    def __init__(self, output_format="int8"):
        super().__init__()
        self.output_format = output_format

    def forward(self, inp):
        inp = torch.ops.circle_custom.mx_fake_quantize(inp, "int8", -1)
        return torch.ops.circle_custom.mx_fake_quantize(inp, self.output_format, 1)


class MXEmbedding(torch.nn.Module):
    weight: torch.Tensor

    def __init__(self):
        super().__init__()
        self.register_buffer("weight", torch.ones(4, 32))

    def forward(self, indices):
        value = torch.nn.functional.embedding(indices, self.weight)
        return torch.ops.circle_custom.mx_fake_quantize(value, "int8", -1)


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
                self.assertEqual(node.meta[QPARAM_KEY].quantized_dimension, 1)
                found_input_qparam = True
        self.assertTrue(found_input_qparam)

    def test_equivalent_positive_and_negative_mx_axes_are_one_domain(self):
        ep = torch.export.export(MXRequantize().eval(), (torch.ones(2, 32),))
        DecomposeFakeQuantize().call(ep)
        FoldQuantOps().call(ep)

        self.assertEqual(
            num_of_ops(ep, [torch.ops.circle_custom.quantize_mx.default]), 0
        )
        inp = next(node for node in ep.graph.nodes if node.op == "placeholder")
        self.assertEqual(inp.meta[QPARAM_KEY].quantized_dimension, 1)

    def test_different_mx_formats_preserve_requantization(self):
        ep = torch.export.export(
            MXRequantize(output_format="fp4").eval(), (torch.ones(2, 32),)
        )
        DecomposeFakeQuantize().call(ep)
        FoldQuantOps().call(ep)

        quantizes = [
            node
            for node in ep.graph.nodes
            if node.target == torch.ops.circle_custom.quantize_mx.default
        ]
        self.assertEqual(len(quantizes), 1)
        self.assertEqual(quantizes[0].args[0].meta[QPARAM_KEY].dtype, "mxint8")
        self.assertEqual(quantizes[0].meta[QPARAM_KEY].dtype, "mxfp4")
        self.assertEqual(quantizes[0].meta[QPARAM_KEY].quantized_dimension, 1)

    def test_mx_embedding_output_owns_first_activation_domain(self):
        ep = torch.export.export(MXEmbedding().eval(), (torch.tensor([0, 2]),))
        DecomposeFakeQuantize().call(ep)
        FoldQuantOps().call(ep)

        embedding = next(
            node
            for node in ep.graph.nodes
            if node.target == torch.ops.aten.embedding.default
        )
        self.assertEqual(embedding.meta[QPARAM_KEY].dtype, "mxint8")
        self.assertEqual(embedding.meta[QPARAM_KEY].quantized_dimension, 1)
        self.assertEqual(
            num_of_ops(ep, [torch.ops.circle_custom.quantize_mx.default]), 0
        )
        self.assertEqual(
            num_of_ops(ep, [torch.ops.circle_custom.dequantize_mx.default]), 0
        )

    def test_mx_shared_producer_owns_first_domain_and_requantizes_other_branch(self):
        for output_format, output_axis in (("int8", 0), ("fp4", 1)):
            with self.subTest(output_format=output_format, output_axis=output_axis):
                ep = torch.export.export(torch.nn.ReLU(), (torch.ones(2, 32),))
                producer = next(
                    node
                    for node in ep.graph.nodes
                    if node.target == torch.ops.aten.relu.default
                )
                output = next(node for node in ep.graph.nodes if node.op == "output")
                # Construct the two marker domains directly to isolate folding
                # from the MX fake operator's exported alias metadata.
                dequantizes = []
                with ep.graph.inserting_before(output):
                    for elem_format, axis in (
                        ("int8", -1),
                        (output_format, output_axis),
                    ):
                        q = ep.graph.call_function(
                            torch.ops.circle_custom.quantize_mx.default,
                            (producer, elem_format, axis),
                        )
                        dq = ep.graph.call_function(
                            torch.ops.circle_custom.dequantize_mx.default,
                            (q, elem_format, axis),
                        )
                        dequantizes.append(dq)
                output.args = ((producer, *dequantizes),)
                FoldQuantOps().call(ep)

                self.assertEqual(producer.meta[QPARAM_KEY].dtype, "mxint8")
                self.assertEqual(producer.meta[QPARAM_KEY].quantized_dimension, 1)
                quantizes = [
                    node
                    for node in ep.graph.nodes
                    if node.target == torch.ops.circle_custom.quantize_mx.default
                ]
                self.assertEqual(len(quantizes), 1)
                quantize = quantizes[0]
                self.assertIs(quantize.args[0], producer)
                self.assertEqual(
                    quantize.meta[QPARAM_KEY].dtype,
                    "mxint8" if output_format == "int8" else "mxfp4",
                )
                self.assertEqual(
                    quantize.meta[QPARAM_KEY].quantized_dimension, output_axis
                )
                self.assertEqual(
                    num_of_ops(ep, [torch.ops.circle_custom.dequantize_mx.default]), 0
                )
                output = next(node for node in ep.graph.nodes if node.op == "output")
                self.assertEqual(output.args[0], (producer, producer, quantize))
                FoldQuantOps().call(ep)
                self.assertEqual(output.args[0], (producer, producer, quantize))
