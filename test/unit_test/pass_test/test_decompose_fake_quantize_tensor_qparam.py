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

import torch
from tico.experimental.quantization.passes.fold_quant_ops import FoldQuantOps
from tico.experimental.quantization.passes.remove_weight_dequant_op import (
    RemoveWeightDequantOp,
)
from tico.passes.const_prop_pass import ConstPropPass
from tico.passes.decompose_fake_quantize import DecomposeFakeQuantize
from tico.passes.decompose_fake_quantize_tensor_qparams import (
    DecomposeFakeQuantizeTensorQParams,
)
from tico.passes.fill_meta_val import FillMetaVal
from tico.serialize.quant_param import QPARAM_KEY

from test.utils.helper import num_of_ops
from test.utils.pass_value_test import PassTest, SinglePassValueTest


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
        return (torch.randn(1, 3, 32, 32),)


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
        return (torch.randn(1, 5),)


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
