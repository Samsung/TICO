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

"""Real serializer/O1 regressions for scalar select and zero-byte constants."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


@unittest.skipUnless(
    all(
        importlib.util.find_spec(name) is not None
        for name in ("circle_schema", "flatbuffers")
    ),
    "Generated Circle schema and FlatBuffers runtime are required.",
)
class TestEmptyConstantsIntegration(unittest.TestCase):
    @staticmethod
    def _inputs(copy=False):
        import torch

        class ScalarSelect(torch.nn.Module):
            def forward(self, x):
                if copy:
                    return torch.ops.aten.select_copy.int(x, 0, 1)
                return torch.select(x, 0, 1)

        return ScalarSelect().eval(), (torch.arange(4, dtype=torch.float32),)

    def _assert_scalar_interface(self, document):
        from tico.circle.graph import as_indices, as_list

        document.verify()
        graph = document.graph()
        self.assertEqual(len(graph.inputs), 1)
        self.assertEqual(len(graph.outputs), 1)
        output = as_list(graph.subgraph.tensors)[graph.outputs[0]]
        self.assertEqual(as_indices(output.shape), [])

    def _convert(self, copy=False):
        import tico
        from tico.circle.document import CircleDocument
        from tico.circle.export import _interface_snapshot
        from tico.config import CompileConfigV1

        model, args = self._inputs(copy)
        # This reproduces the reported failure with the default O1 path enabled.
        optimized = CircleDocument.from_bytes(tico.convert(model, args).circle_binary)
        original = CircleDocument.from_bytes(
            tico.convert(
                model, args, config=CompileConfigV1(circle_optimize=False)
            ).circle_binary
        )
        self._assert_scalar_interface(optimized)
        self.assertEqual(_interface_snapshot(original), _interface_snapshot(optimized))

    def test_scalar_select_default_o1(self):
        self._convert(copy=False)

    def test_scalar_select_copy_default_o1(self):
        self._convert(copy=True)

    def _convert_pt2(self, copy=False):
        import torch
        from tico.circle.document import CircleDocument
        from tico.pt2_to_circle import convert

        model, args = self._inputs(copy)
        exported = torch.export.export(model, args)
        with tempfile.TemporaryDirectory() as directory:
            pt2_path = Path(directory) / "select.pt2"
            circle_path = Path(directory) / "select.circle"
            torch.export.save(exported, pt2_path)
            convert(str(pt2_path), str(circle_path))
            self._assert_scalar_interface(CircleDocument.load(circle_path))

    def test_pt2_scalar_select_default_o1(self):
        self._convert_pt2(copy=False)

    def test_pt2_scalar_select_copy_default_o1(self):
        self._convert_pt2(copy=True)

    def test_empty_shape_binary_round_trip_and_full_o1(self):
        import tico
        from tico.circle.document import CircleDocument
        from tico.circle.export import optimize_for_export
        from tico.circle.graph import as_indices, as_list
        from tico.circle.value import TensorValueCodec
        from tico.config import CompileConfigV1

        model, args = self._inputs()
        raw = tico.convert(
            model, args, config=CompileConfigV1(circle_optimize=False)
        ).circle_binary
        for data in (None, np.empty(0, dtype=np.uint8)):
            with self.subTest(data_type=type(data).__name__):
                document = CircleDocument.from_bytes(raw)
                indices = [
                    index
                    for index, tensor in enumerate(as_list(document.subgraph().tensors))
                    if as_indices(tensor.shape) == [0] and int(tensor.buffer) > 0
                ]
                self.assertTrue(indices, "Scalar reshape must have an empty shape.")
                for index in indices:
                    tensor = document.subgraph().tensors[index]
                    document.model.buffers[int(tensor.buffer)].data = data
                repacked = document.to_bytes()
                reloaded = CircleDocument.from_bytes(repacked)
                reloaded.verify()
                for index in indices:
                    self.assertTrue(reloaded.graph().is_constant(index))
                    value = TensorValueCodec().decode_tensor(
                        reloaded.model, subgraph_index=0, tensor_index=index
                    )
                    self.assertEqual(value.shape, (0,))
                    self.assertEqual(value.data.size, 0)
                optimized = optimize_for_export(repacked)
                self._assert_scalar_interface(CircleDocument.from_bytes(optimized))
                self.assertEqual(optimize_for_export(optimized), optimized)

    def test_public_empty_constant_binary_round_trip_and_o1(self):
        from circle_schema import circle

        from tico.circle.document import CircleDocument
        from tico.circle.export import optimize_for_export
        from tico.circle.value import TensorValueCodec

        model = circle.Model.ModelT()
        model.version = 0
        model.operatorCodes = []
        model.buffers = [circle.Buffer.BufferT(), circle.Buffer.BufferT()]
        model.buffers[1].data = np.empty(0, dtype=np.uint8)
        graph = circle.SubGraph.SubGraphT()
        graph.name = "empty_output"
        graph.inputs, graph.outputs, graph.operators = [], [0], []
        tensor = circle.Tensor.TensorT()
        tensor.name = "empty_constant"
        tensor.shape = [0]
        tensor.type = circle.TensorType.TensorType.INT32
        tensor.buffer = 1
        graph.tensors = [tensor]
        model.subgraphs = [graph]
        optimized = optimize_for_export(CircleDocument(model).to_bytes())
        document = CircleDocument.from_bytes(optimized)
        document.verify()
        output_index = document.graph().outputs[0]
        self.assertTrue(document.graph().is_constant(output_index))
        value = TensorValueCodec().decode_tensor(
            document.model, subgraph_index=0, tensor_index=output_index
        )
        self.assertEqual(value.shape, (0,))
        self.assertEqual(value.data.size, 0)


if __name__ == "__main__":
    unittest.main()
