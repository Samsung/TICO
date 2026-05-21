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

import copy
import unittest

import torch
from tico.quantization.passes.remove_redundant_quantisers import RemoveRedundantQuantisers
from tico.serialize.quant_param import QPARAM_KEY, QuantParam
from tico.utils.graph import create_node
from tico.utils.utils import quant_min_max, set_new_meta_val

from test.utils.helper import num_of_ops


def _insert_quantize_per_tensor_after(graph, node, qparam):
    """Insert a quantize_per_tensor op after the given node with the given qparam."""
    assert qparam.scale is not None
    assert qparam.zero_point is not None
    scale = qparam.scale[0]
    zerop = qparam.zero_point[0]
    min_, max_ = quant_min_max(qparam.dtype)
    dtype = getattr(torch, qparam.dtype)

    with graph.inserting_after(node):
        q_args = (node, scale, zerop, min_, max_, dtype)
        quantize = create_node(
            graph,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            args=q_args,
        )

    node.replace_all_uses_with(quantize, propagate_meta=True)
    quantize.replace_input_with(quantize, node)

    quantize.meta[QPARAM_KEY] = copy.deepcopy(qparam)

    return quantize


def _insert_quantize_mx_after(graph, node, qparam):
    """Insert a quantize_mx op after the given node with the given qparam."""
    assert qparam.quantized_dimension is not None
    assert qparam.dtype is not None

    with graph.inserting_after(node):
        q_args = (node, qparam.dtype, qparam.quantized_dimension)
        quantize = create_node(
            graph,
            torch.ops.circle_custom.quantize_mx_decomposed.default,
            args=q_args,
        )

    node.replace_all_uses_with(quantize, propagate_meta=True)
    quantize.replace_input_with(quantize, node)

    quantize.meta[QPARAM_KEY] = copy.deepcopy(qparam)

    return quantize


class SimpleReshape(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.reshape(x.shape[0], -1)

    def get_example_inputs(self):
        return (torch.randn(2, 3, 4),), {}


class RemoveRedundantQuantisersTest(unittest.TestCase):
    """Test RemoveRedundantQuantisers pass for both round-trip patterns."""

    def _export_and_find_reshape(self):
        """Export a simple module and find the reshape node."""
        m = SimpleReshape().eval()
        args, kwargs = m.get_example_inputs()
        ep = torch.export.export(m, args, kwargs)

        reshape_node = None
        for node in ep.graph.nodes:
            if node.op == "call_function" and node.target == torch.ops.aten.reshape.default:
                reshape_node = node
                break

        assert reshape_node is not None, "Could not find reshape node in exported graph"
        return ep, reshape_node

    def test_pattern1_int16_mxint8_int16(self):
        """Test removal of int16 → quantize_mx(mxint8) → quantize_per_tensor(int16)."""
        ep, reshape_node = self._export_and_find_reshape()

        # Set int16 qparam on reshape output
        i16_qparam = QuantParam()
        i16_qparam.scale = [1.0]
        i16_qparam.zero_point = [0]
        i16_qparam.dtype = "int16"
        reshape_node.meta[QPARAM_KEY] = copy.deepcopy(i16_qparam)

        # Insert quantize_mx(mxint8) after reshape
        mx_qparam = QuantParam()
        mx_qparam.dtype = "mxint8"
        mx_qparam.quantized_dimension = -1
        q_mx = _insert_quantize_mx_after(ep.graph, reshape_node, mx_qparam)

        # Insert quantize_per_tensor(int16) after quantize_mx
        q_pt = _insert_quantize_per_tensor_after(ep.graph, q_mx, copy.deepcopy(i16_qparam))

        ep.graph.eliminate_dead_code()
        ep.graph.lint()
        ep.graph_module.recompile()

        # Before pass: there should be 1 quantize_mx and 1 quantize_per_tensor
        self.assertEqual(
            num_of_ops(ep, [torch.ops.circle_custom.quantize_mx_decomposed.default]),
            1,
        )
        self.assertEqual(
            num_of_ops(ep, [torch.ops.quantized_decomposed.quantize_per_tensor.default]),
            1,
        )

        # Run the pass
        result = RemoveRedundantQuantisers().call(ep)
        self.assertTrue(result.modified)

        # After pass: both quantisers should be removed
        self.assertEqual(
            num_of_ops(ep, [torch.ops.circle_custom.quantize_mx_decomposed.default]),
            0,
        )
        self.assertEqual(
            num_of_ops(ep, [torch.ops.quantized_decomposed.quantize_per_tensor.default]),
            0,
        )

        # The reshape node should still have int16 qparam
        self.assertEqual(reshape_node.meta[QPARAM_KEY].dtype, "int16")

    def test_pattern2_mxint8_int16_mxint8(self):
        """Test removal of mxint8 → quantize_per_tensor(int16) → quantize_mx(mxint8)."""
        ep, reshape_node = self._export_and_find_reshape()

        # Set mxint8 qparam on reshape output
        mx_qparam = QuantParam()
        mx_qparam.dtype = "mxint8"
        mx_qparam.quantized_dimension = -1
        reshape_node.meta[QPARAM_KEY] = copy.deepcopy(mx_qparam)

        # Insert quantize_per_tensor(int16) after reshape
        i16_qparam = QuantParam()
        i16_qparam.scale = [1.0]
        i16_qparam.zero_point = [0]
        i16_qparam.dtype = "int16"
        q_pt = _insert_quantize_per_tensor_after(ep.graph, reshape_node, copy.deepcopy(i16_qparam))

        # Insert quantize_mx(mxint8) after quantize_per_tensor
        q_mx = _insert_quantize_mx_after(ep.graph, q_pt, copy.deepcopy(mx_qparam))

        ep.graph.eliminate_dead_code()
        ep.graph.lint()
        ep.graph_module.recompile()

        # Before pass: there should be 1 quantize_per_tensor and 1 quantize_mx
        self.assertEqual(
            num_of_ops(ep, [torch.ops.quantized_decomposed.quantize_per_tensor.default]),
            1,
        )
        self.assertEqual(
            num_of_ops(ep, [torch.ops.circle_custom.quantize_mx_decomposed.default]),
            1,
        )

        # Run the pass
        result = RemoveRedundantQuantisers().call(ep)
        self.assertTrue(result.modified)

        # After pass: both quantisers should be removed
        self.assertEqual(
            num_of_ops(ep, [torch.ops.quantized_decomposed.quantize_per_tensor.default]),
            0,
        )
        self.assertEqual(
            num_of_ops(ep, [torch.ops.circle_custom.quantize_mx_decomposed.default]),
            0,
        )

        # The reshape node should still have mxint8 qparam
        self.assertEqual(reshape_node.meta[QPARAM_KEY].dtype, "mxint8")

    def test_no_redundant_quantisers(self):
        """Test that the pass does not modify the graph when there are no redundant quantisers."""
        ep, reshape_node = self._export_and_find_reshape()

        # Set int16 qparam on reshape output
        i16_qparam = QuantParam()
        i16_qparam.scale = [1.0]
        i16_qparam.zero_point = [0]
        i16_qparam.dtype = "int16"
        reshape_node.meta[QPARAM_KEY] = copy.deepcopy(i16_qparam)

        # Insert only quantize_mx(mxint8) — no round-trip
        mx_qparam = QuantParam()
        mx_qparam.dtype = "mxint8"
        mx_qparam.quantized_dimension = -1
        q_mx = _insert_quantize_mx_after(ep.graph, reshape_node, mx_qparam)

        ep.graph.eliminate_dead_code()
        ep.graph.lint()
        ep.graph_module.recompile()

        # Run the pass
        result = RemoveRedundantQuantisers().call(ep)
        self.assertFalse(result.modified)

        # quantize_mx should still be there
        self.assertEqual(
            num_of_ops(ep, [torch.ops.circle_custom.quantize_mx_decomposed.default]),
            1,
        )
