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

"""Regression tests for requantization immediately after RESHAPE."""

from __future__ import annotations

import unittest

import torch

from tico.passes.decompose_fake_quantize import DecomposeFakeQuantize
from tico.quantization.passes.fold_quant_ops import FoldQuantOps
from tico.quantization.passes.propagate_qparam_forward import PropagateQParamForward
from tico.serialize.quant_param import QPARAM_KEY

from test.support.helper import num_of_ops


class _ReshapeRequantize(torch.nn.Module):
    def __init__(
        self,
        *,
        target_scale: float,
        target_zero_point: int,
        target_qmin: int,
        target_qmax: int,
    ) -> None:
        super().__init__()
        self.target_scale = target_scale
        self.target_zero_point = target_zero_point
        self.target_qmin = target_qmin
        self.target_qmax = target_qmax

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        value = torch.fake_quantize_per_tensor_affine(
            input_,
            0.125,
            128,
            0,
            255,
        )
        value = value.reshape(1, 9)
        return torch.fake_quantize_per_tensor_affine(
            value,
            self.target_scale,
            self.target_zero_point,
            self.target_qmin,
            self.target_qmax,
        )


class FoldQuantOpsReshapeRequantizeTest(unittest.TestCase):
    def test_keeps_cross_dtype_requantize_after_reshape(self) -> None:
        self._check_case(
            target_scale=0.25,
            target_zero_point=0,
            target_qmin=-32768,
            target_qmax=32767,
            target_dtype="int16",
        )

    def test_keeps_same_dtype_different_qparam_after_reshape(self) -> None:
        self._check_case(
            target_scale=0.25,
            target_zero_point=127,
            target_qmin=0,
            target_qmax=255,
            target_dtype="uint8",
        )

    def _check_case(
        self,
        *,
        target_scale: float,
        target_zero_point: int,
        target_qmin: int,
        target_qmax: int,
        target_dtype: str,
    ) -> None:
        module = _ReshapeRequantize(
            target_scale=target_scale,
            target_zero_point=target_zero_point,
            target_qmin=target_qmin,
            target_qmax=target_qmax,
        ).eval()
        exported = torch.export.export(module, (torch.randn(3, 3),))
        DecomposeFakeQuantize().call(exported)

        self.assertEqual(
            num_of_ops(
                exported,
                [torch.ops.quantized_decomposed.quantize_per_tensor.default],
            ),
            2,
        )
        self.assertEqual(
            num_of_ops(
                exported,
                [torch.ops.quantized_decomposed.dequantize_per_tensor.default],
            ),
            2,
        )

        FoldQuantOps().call(exported)
        PropagateQParamForward().call(exported)

        self.assertEqual(
            num_of_ops(
                exported,
                [torch.ops.quantized_decomposed.quantize_per_tensor.default],
            ),
            1,
        )
        self.assertEqual(
            num_of_ops(
                exported,
                [torch.ops.quantized_decomposed.dequantize_per_tensor.default],
            ),
            0,
        )

        reshape = next(
            node
            for node in exported.graph.nodes
            if node.op == "call_function"
            and node.target == torch.ops.aten.reshape.default
        )
        reshape_qparam = reshape.meta[QPARAM_KEY]
        self.assertEqual(reshape_qparam.dtype, "uint8")
        self.assertEqual(reshape_qparam.scale, [0.125])
        self.assertEqual(reshape_qparam.zero_point, [128])

        quantize = next(
            node
            for node in exported.graph.nodes
            if node.op == "call_function"
            and node.target
            == torch.ops.quantized_decomposed.quantize_per_tensor.default
        )
        self.assertIs(quantize.args[0], reshape)
        target_qparam = quantize.meta[QPARAM_KEY]
        self.assertEqual(target_qparam.dtype, target_dtype)
        self.assertEqual(target_qparam.scale, [target_scale])
        self.assertEqual(target_qparam.zero_point, [target_zero_point])


if __name__ == "__main__":
    unittest.main()
