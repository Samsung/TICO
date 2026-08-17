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

from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from tico.circle import CircleBuilder, CircleDocument, TensorValue


@unittest.skipUnless(
    importlib.util.find_spec("circle_schema") is not None
    and importlib.util.find_spec("flatbuffers") is not None,
    "circle-schema and flatbuffers are required for generated integration tests",
)
class GeneratedInfrastructureTest(unittest.TestCase):
    """Check new infrastructure against generated Object API tables and binary IO."""

    def test_empty_generated_options_have_stable_fingerprint(self):
        """Fingerprint empty Object API option tables by their generated type."""

        from circle_schema import circle

        from tico.circle._object import freeze_object

        first = circle.TransposeOptions.TransposeOptionsT()
        second = circle.TransposeOptions.TransposeOptionsT()

        self.assertEqual(freeze_object(first), freeze_object(second))

    def test_constant_builder_survives_circle_binary_round_trip(self):
        """Create, serialize, restore, verify, and decode one generated constant."""

        from circle_schema import circle

        from tico.circle.value import TensorValueCodec

        model = circle.Model.ModelT()
        model.version = 0
        model.description = "infrastructure-round-trip"
        model.operatorCodes = []
        model.buffers = [circle.Buffer.BufferT()]
        model.signatureDefs = []
        model.metadataBuffer = []
        model.metadata = []

        subgraph = circle.SubGraph.SubGraphT()
        subgraph.name = "main"
        subgraph.tensors = []
        subgraph.inputs = []
        subgraph.outputs = []
        subgraph.operators = []
        model.subgraphs = [subgraph]

        document = CircleDocument(model)
        tensor_type = int(circle.TensorType.TensorType.FLOAT32)
        value = TensorValue(
            tensor_type=tensor_type,
            shape=(2,),
            data=np.array([1.0, -2.0], dtype=np.float32),
        )
        tensor_index = CircleBuilder(document).add_constant("constant", value)
        document.subgraph().outputs = [tensor_index]

        restored = CircleDocument.from_bytes(document.to_bytes())
        self.assertTrue(restored.verify(raise_on_error=False).ok)
        decoded = TensorValueCodec().decode_tensor(
            restored.model,
            subgraph_index=0,
            tensor_index=0,
        )
        np.testing.assert_array_equal(decoded.data, value.data)


if __name__ == "__main__":
    unittest.main()
