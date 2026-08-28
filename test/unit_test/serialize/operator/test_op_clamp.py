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

"""Tests for quantized Clamp Circle serialization."""

import copy
import unittest

import torch

from tico.serialize.circle_graph import CircleModel, CircleSubgraph
from tico.serialize.operators.op_clamp import ClampVisitor
from tico.serialize.quant_param import QPARAM_KEY, QuantParam


class _ClampModule(torch.nn.Module):
    """Expose one tensor-bound Clamp operator."""

    def forward(
        self,
        inputs: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
    ) -> torch.Tensor:
        return torch.clamp(inputs, lower, upper)


class TestClampVisitor(unittest.TestCase):
    """Validate the synthetic MINIMUM output tensor contract."""

    def test_intermediate_tensor_preserves_quantization_domain(self) -> None:
        """Both operators emitted for Clamp should share one INT16 qparam."""
        exported_program = torch.export.export(
            _ClampModule().eval(),
            (
                torch.randn(2, 3),
                torch.tensor(-1.0),
                torch.tensor(1.0),
            ),
        )
        nodes = list(exported_program.graph.nodes)
        clamp = next(node for node in nodes if node.op == "call_function")

        qparam = QuantParam(
            scale=[0.25],
            zero_point=[0],
            quantized_dimension=None,
            dtype="int16",
        )
        for node in nodes:
            if node.op in {"placeholder", "call_function"}:
                node.meta[QPARAM_KEY] = copy.deepcopy(qparam)

        model = CircleModel()
        graph = CircleSubgraph(model)
        for node in nodes:
            if node.op in {"placeholder", "call_function"}:
                graph.add_tensor_from_node(node)

        op_codes = {}
        visitor = ClampVisitor(op_codes=op_codes, graph=graph)
        maximum = visitor.define_node(clamp)

        self.assertEqual(len(graph.operators), 1)
        minimum = graph.operators[0]
        self.assertTrue(op_codes)
        self.assertIsNotNone(maximum)

        intermediate = graph.tensors[minimum.outputs[0]]
        self.assertEqual(intermediate.type, graph.get_tensor(clamp).type)
        self.assertIsNotNone(intermediate.quantization)
        self.assertEqual(intermediate.quantization.scale, [0.25])
        self.assertEqual(intermediate.quantization.zeroPoint, [0])


if __name__ == "__main__":
    unittest.main()
