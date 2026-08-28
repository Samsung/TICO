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

"""Regression tests for requantization after a PERMUTE/RESHAPE chain."""

from __future__ import annotations

import unittest

import torch

from tico.passes.decompose_fake_quantize import DecomposeFakeQuantize
from tico.quantization.passes.fold_quant_ops import FoldQuantOps
from tico.serialize.quant_param import QPARAM_KEY


class _ViewChainRequantize(torch.nn.Module):
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

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = torch.fake_quantize_per_tensor_affine(
            value,
            0.5,
            127,
            0,
            255,
        )
        value = value.permute(0, 2, 3, 1)
        value = value.reshape(1, 4, 3)
        return torch.fake_quantize_per_tensor_affine(
            value,
            self.target_scale,
            self.target_zero_point,
            self.target_qmin,
            self.target_qmax,
        )


class FoldQuantOpsViewChainTest(unittest.TestCase):
    def test_cross_dtype_requantization_survives_view_chain(self) -> None:
        program = _fold(
            _ViewChainRequantize(
                target_scale=0.25,
                target_zero_point=0,
                target_qmin=-32768,
                target_qmax=32767,
            )
        )
        self._require_source_view_domain(program)
        quantize = _only_quantize(program)
        qparam = quantize.meta[QPARAM_KEY]
        self.assertEqual(qparam.dtype, "int16")
        self.assertEqual(qparam.scale, [0.25])
        self.assertEqual(qparam.zero_point, [0])
        self.assertEqual(quantize.args[0].target, torch.ops.aten.reshape.default)

    def test_same_dtype_different_qparams_survive_view_chain(self) -> None:
        program = _fold(
            _ViewChainRequantize(
                target_scale=0.25,
                target_zero_point=123,
                target_qmin=0,
                target_qmax=255,
            )
        )
        self._require_source_view_domain(program)
        quantize = _only_quantize(program)
        qparam = quantize.meta[QPARAM_KEY]
        self.assertEqual(qparam.dtype, "uint8")
        self.assertEqual(qparam.scale, [0.25])
        self.assertEqual(qparam.zero_point, [123])
        self.assertEqual(quantize.args[0].target, torch.ops.aten.reshape.default)

    def _require_source_view_domain(self, program) -> None:
        permute = _only_node(program, torch.ops.aten.permute.default)
        reshape = _only_node(program, torch.ops.aten.reshape.default)
        for node in (permute, reshape):
            self.assertIn(QPARAM_KEY, node.meta)
            qparam = node.meta[QPARAM_KEY]
            self.assertEqual(qparam.dtype, "uint8")
            self.assertEqual(qparam.scale, [0.5])
            self.assertEqual(qparam.zero_point, [127])

        dequantize_target = torch.ops.quantized_decomposed.dequantize_per_tensor.default
        self.assertFalse(
            any(
                node.op == "call_function" and node.target == dequantize_target
                for node in program.graph.nodes
            )
        )


def _fold(module: torch.nn.Module):
    args = (torch.randn(1, 3, 2, 2),)
    program = torch.export.export(module.eval(), args)
    DecomposeFakeQuantize().call(program)
    FoldQuantOps().call(program)
    return program


def _only_quantize(program):
    target = torch.ops.quantized_decomposed.quantize_per_tensor.default
    nodes = [
        node
        for node in program.graph.nodes
        if node.op == "call_function" and node.target == target
    ]
    if len(nodes) != 1:
        raise AssertionError(f"Expected one remaining Quantize, found {len(nodes)}.")
    node = nodes[0]
    if QPARAM_KEY not in node.meta:
        raise AssertionError("Remaining Quantize has no qparam metadata.")
    return node


def _only_node(program, target):
    nodes = [
        node
        for node in program.graph.nodes
        if node.op == "call_function" and node.target == target
    ]
    if len(nodes) != 1:
        raise AssertionError(f"Expected one {target}, found {len(nodes)}.")
    return nodes[0]


if __name__ == "__main__":
    unittest.main()
