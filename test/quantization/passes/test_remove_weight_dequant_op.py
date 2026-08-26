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
from tico.passes.const_prop_pass import ConstPropPass
from tico.passes.decompose_fake_quantize import DecomposeFakeQuantize
from tico.quantization import convert, prepare
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.passes.remove_weight_dequant_op import RemoveWeightDequantOp
from tico.serialize.quant_param import QPARAM_KEY

from test.support.helper import num_of_ops


class _ConstantFakeQuantModule(torch.nn.Module):
    """Expose one constant tensor through a configurable affine fake quantizer."""

    def __init__(self, quant_max: int) -> None:
        super().__init__()
        self.quant_max = int(quant_max)
        self.register_buffer("constant", torch.tensor([0.0, 1.0, 15.0]))

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Add the fake-quantized constant to a runtime tensor."""
        constant = torch.fake_quantize_per_tensor_affine(
            self.constant,
            scale=1.0,
            zero_point=0,
            quant_min=0,
            quant_max=self.quant_max,
        )
        return input_ + constant


class RemoveWeightDequantOpTest(unittest.TestCase):
    """Validate dequantize removal and logical dtype recovery."""

    def _constant_qparam_dtype(self, quant_max: int) -> str:
        """Return the qparam dtype attached to one folded quantized constant."""
        module = _ConstantFakeQuantModule(quant_max).eval()
        exported = torch.export.export(module, (torch.zeros(3),))
        DecomposeFakeQuantize().call(exported)
        ConstPropPass().call(exported)
        RemoveWeightDequantOp().call(exported)

        qparams = [
            node.meta[QPARAM_KEY]
            for node in exported.graph.nodes
            if node.op == "placeholder" and QPARAM_KEY in node.meta
        ]
        self.assertEqual(len(qparams), 1)
        return qparams[0].dtype

    def test_pass(self):
        q_m = torch.nn.Linear(3, 3)
        assert isinstance(q_m, torch.nn.Module)

        q_m = prepare(q_m, PTQConfig())

        # Calibration
        for i in range(10):
            cal_args = (torch.randn(3, 3),)
            q_m(*cal_args)

        # Quantization
        q_m = convert(q_m)

        # 5. Export module
        ep = torch.export.export(q_m, (torch.randn(3, 3),))
        DecomposeFakeQuantize().call(ep)
        ConstPropPass().call(ep)
        # (weight - DQ)
        self.assertEqual(
            num_of_ops(
                ep, [torch.ops.quantized_decomposed.dequantize_per_channel.default]
            ),
            1,
        )

        target_pass = RemoveWeightDequantOp()
        target_pass.call(ep)
        self.assertEqual(
            num_of_ops(
                ep, [torch.ops.quantized_decomposed.dequantize_per_channel.default]
            ),
            0,
        )

    def test_declared_uint8_range_is_not_inferred_as_uint4(self):
        """Keep UINT8 when all stored codes happen to fit in four bits."""
        self.assertEqual(self._constant_qparam_dtype(255), "uint8")

    def test_declared_uint4_range_remains_uint4(self):
        """Recover UINT4 only when the fake-quant range explicitly declares it."""
        self.assertEqual(self._constant_qparam_dtype(15), "uint4")
