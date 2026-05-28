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

import unittest

import torch
from tico.passes.convert_layout_op_to_reshape import ConvertLayoutOpToReshape
from tico.quantization.passes.qparam_safe_const_prop import QParamSafeConstPropPass
from tico.serialize.quant_param import QPARAM_KEY, QuantParam
from torch._export.utils import is_buffer
from torch.export import ExportedProgram

from test.support.helper import num_of_ops


def _make_qparam(
    *,
    scale: list[float],
    zero_point: list[int],
    dtype: str = "uint8",
    quantized_dimension: int | None = None,
) -> QuantParam:
    """Create quantization parameters for qparam-safe const-prop tests."""

    qparam = QuantParam()
    qparam.scale = scale
    qparam.zero_point = zero_point
    qparam.dtype = dtype
    qparam.quantized_dimension = quantized_dimension
    return qparam


def _get_buffer_placeholder(ep: ExportedProgram) -> torch.fx.Node:
    """Return the first buffer placeholder in an exported program."""

    for node in ep.graph.nodes:
        if node.op == "placeholder" and is_buffer(ep, node):
            return node

    raise AssertionError("Buffer placeholder is not found.")


def _get_qparam_placeholders(ep: ExportedProgram) -> list[torch.fx.Node]:
    """Return placeholders that carry qparam metadata."""

    return [
        node
        for node in ep.graph.nodes
        if node.op == "placeholder" and QPARAM_KEY in node.meta
    ]


class ConstReshapeModule(torch.nn.Module):
    """Test module with a constant reshape."""

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "weight",
            torch.arange(6, dtype=torch.uint8).reshape(2, 3),
        )

    def forward(self):
        return torch.ops.aten.reshape.default(self.weight, (3, 2))


class ConstPermuteModule(torch.nn.Module):
    """Test module with a constant permute."""

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "weight",
            torch.arange(24, dtype=torch.uint8).reshape(2, 3, 4),
        )

    def forward(self):
        return torch.ops.aten.permute.default(self.weight, (1, 0, 2))


class ConstAddModule(torch.nn.Module):
    """Test module with a value-changing constant op."""

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "weight",
            torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
        )

    def forward(self):
        return torch.ops.aten.add.Tensor(self.weight, self.weight)


class PerChannelReshapeModule(torch.nn.Module):
    """Test module with a per-channel constant reshape."""

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "weight",
            torch.arange(6, dtype=torch.uint8).reshape(2, 3),
        )

    def forward(self):
        return torch.ops.aten.reshape.default(self.weight, (2, 1, 3))


class QParamSafeConstPropPassTest(unittest.TestCase):
    """Unit tests for qparam-safe constant propagation."""

    def test_fold_per_tensor_reshape(self):
        ep = torch.export.export(ConstReshapeModule().eval(), ())
        # This is necessary for testing Reshape on torch 2.5
        ConvertLayoutOpToReshape().call(ep)

        weight = _get_buffer_placeholder(ep)
        weight.meta[QPARAM_KEY] = _make_qparam(
            scale=[0.5],
            zero_point=[3],
            dtype="uint8",
        )

        self.assertEqual(num_of_ops(ep, [torch.ops.aten.reshape.default]), 1)

        QParamSafeConstPropPass().call(ep)

        self.assertEqual(num_of_ops(ep, [torch.ops.aten.reshape.default]), 0)

        qparam_placeholders = _get_qparam_placeholders(ep)
        self.assertEqual(len(qparam_placeholders), 1)

        folded_qparam = qparam_placeholders[0].meta[QPARAM_KEY]
        self.assertEqual(folded_qparam.scale, [0.5])
        self.assertEqual(folded_qparam.zero_point, [3])
        self.assertEqual(folded_qparam.dtype, "uint8")
        self.assertIsNone(folded_qparam.quantized_dimension)

    def test_fold_per_channel_permute_updates_quantized_dimension(self):
        ep = torch.export.export(ConstPermuteModule().eval(), ())

        weight = _get_buffer_placeholder(ep)
        weight.meta[QPARAM_KEY] = _make_qparam(
            scale=[0.25, 0.5],
            zero_point=[1, 2],
            dtype="uint8",
            quantized_dimension=0,
        )

        self.assertEqual(num_of_ops(ep, [torch.ops.aten.permute.default]), 1)

        QParamSafeConstPropPass().call(ep)

        self.assertEqual(num_of_ops(ep, [torch.ops.aten.permute.default]), 0)

        qparam_placeholders = _get_qparam_placeholders(ep)
        self.assertEqual(len(qparam_placeholders), 1)

        folded_qparam = qparam_placeholders[0].meta[QPARAM_KEY]
        self.assertEqual(folded_qparam.scale, [0.25, 0.5])
        self.assertEqual(folded_qparam.zero_point, [1, 2])
        self.assertEqual(folded_qparam.dtype, "uint8")
        self.assertEqual(folded_qparam.quantized_dimension, 1)

    def test_do_not_fold_value_changing_op(self):
        ep = torch.export.export(ConstAddModule().eval(), ())

        weight = _get_buffer_placeholder(ep)
        weight.meta[QPARAM_KEY] = _make_qparam(
            scale=[1.0],
            zero_point=[0],
            dtype="uint8",
        )

        self.assertEqual(num_of_ops(ep, [torch.ops.aten.add.Tensor]), 1)

        QParamSafeConstPropPass().call(ep)

        self.assertEqual(num_of_ops(ep, [torch.ops.aten.add.Tensor]), 1)

    def test_do_not_fold_per_channel_reshape_without_axis_proof(self):
        ep = torch.export.export(PerChannelReshapeModule().eval(), ())
        # This is necessary for testing Reshape on torch 2.5
        ConvertLayoutOpToReshape().call(ep)

        weight = _get_buffer_placeholder(ep)
        weight.meta[QPARAM_KEY] = _make_qparam(
            scale=[0.25, 0.5],
            zero_point=[1, 2],
            dtype="uint8",
            quantized_dimension=0,
        )

        self.assertEqual(num_of_ops(ep, [torch.ops.aten.reshape.default]), 1)

        QParamSafeConstPropPass().call(ep)

        self.assertEqual(num_of_ops(ep, [torch.ops.aten.reshape.default]), 1)
