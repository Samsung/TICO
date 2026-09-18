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

"""Regression tests for zero-byte constants in scalar-select graphs.

These Object API tests inject a dtype registry and use no generated-schema
objects. The standalone runner can bypass package initialization when schema
packages are absent. Actual binary/export tests live alongside this module.
"""

import unittest
from types import SimpleNamespace as NS

import numpy as np

from tico.circle.document import CircleDocument
from tico.circle.errors import CircleValueError
from tico.circle.graph import has_buffer_payload, is_constant_tensor
from tico.circle.value import TensorTypeRegistry, TensorTypeSpec, TensorValueCodec
from tico.circle.verify import CircleVerificationError


def _tensor(name, shape, buffer=0, dtype=0):
    return NS(
        name=name,
        shape=list(shape),
        shapeSignature=None,
        type=dtype,
        buffer=buffer,
        isVariable=False,
        quantization=None,
    )


def _select_graph(data=None):
    """Model select(x[4], 0, 1) as SLICE then scalar RESHAPE.

    Tensor 6 is the empty INT32 shape vector, consumed by operator 1, matching
    the reported failure site. This is an explicit Object API fixture, not a
    captured CI binary.
    """
    tensors = [
        _tensor("x", [4]),
        _tensor("slice_begin", [1], buffer=1, dtype=2),
        _tensor("slice_size", [1], buffer=2, dtype=2),
        _tensor("slice", [1]),
        _tensor("scalar_output", []),
        _tensor("unused", [1]),
        _tensor("scalar_shape", [0], buffer=3, dtype=2),
    ]
    subgraph = NS(
        name="main",
        tensors=tensors,
        inputs=[0],
        outputs=[4],
        operators=[
            NS(opcodeIndex=0, inputs=[0, 1, 2], outputs=[3]),
            NS(opcodeIndex=1, inputs=[3, 6], outputs=[4]),
        ],
    )
    buffers = [
        NS(data=None, offset=0, size=0),
        NS(data=np.array([1], dtype="<i4").view(np.uint8), offset=0, size=0),
        NS(data=np.array([1], dtype="<i4").view(np.uint8), offset=0, size=0),
        NS(data=data, offset=0, size=0),
    ]
    model = NS(
        subgraphs=[subgraph],
        buffers=buffers,
        operatorCodes=[
            NS(builtinCode=65, deprecatedBuiltinCode=65, version=1),
            NS(builtinCode=22, deprecatedBuiltinCode=22, version=1),
        ],
        signatureDefs=[],
        metadata=[],
        metadataBuffer=[],
    )
    return CircleDocument(model)


def _codec():
    dtype = np.dtype("int32")
    return TensorValueCodec(
        TensorTypeRegistry([TensorTypeSpec("INT32", 2, dtype, dtype, 32)])
    )


class TestEmptyConstantClassification(unittest.TestCase):
    def test_empty_inline_and_omitted_data_are_constants(self):
        for data in (None, b"", [], np.empty(0, dtype=np.uint8)):
            with self.subTest(data_type=type(data).__name__):
                document = _select_graph(data)
                document.verify()
                self.assertTrue(document.graph().is_constant(6))
                self.assertFalse(has_buffer_payload(document.model.buffers[3]))

    def test_static_shapes_with_zero_extent(self):
        for shape in ([0], [2, 0, 3], [0, 0]):
            for signature in (None, [], shape):
                with self.subTest(shape=shape, signature=signature):
                    document = _select_graph()
                    tensor = document.subgraph().tensors[6]
                    tensor.shape, tensor.shapeSignature = shape, signature
                    self.assertTrue(document.graph().is_constant(6))

    def test_missing_scalar_and_nonempty_payload_stay_invalid(self):
        for shape in ([], [1], [2, 3]):
            with self.subTest(shape=shape):
                document = _select_graph()
                document.subgraph().tensors[6].shape = shape
                self.assertFalse(document.graph().is_constant(6))
                with self.assertRaises(CircleVerificationError) as error:
                    document.verify()
                self.assertIn("UNDEFINED_INPUT", str(error.exception))
                self.assertIn("Tensor 6", str(error.exception))

    def test_dynamic_or_inconsistent_shape_is_not_an_empty_constant(self):
        for shape, signature in (
            ([-1, 0], None),
            ([0], [-1]),
            ([2, 0], [-1, 0]),
            ([0], [1]),
            ([0], [0, 0]),
        ):
            with self.subTest(shape=shape, signature=signature):
                document = _select_graph()
                tensor = document.subgraph().tensors[6]
                tensor.shape, tensor.shapeSignature = shape, signature
                self.assertFalse(document.graph().is_constant(6))

    def test_unknown_shape_is_not_empty(self):
        document = _select_graph()
        document.subgraph().tensors[6].shape = None
        self.assertFalse(document.graph().is_constant(6))

    def test_reserved_and_invalid_buffer_indices_are_not_constants(self):
        for buffer in (0, -1, 999):
            with self.subTest(buffer=buffer):
                document = _select_graph()
                document.subgraph().tensors[6].buffer = buffer
                self.assertFalse(document.graph().is_constant(6))

    def test_missing_buffer_table_is_not_an_empty_constant(self):
        document = _select_graph()
        document.model.buffers[3] = None
        self.assertFalse(document.graph().is_constant(6))

    def test_variable_is_not_an_empty_constant(self):
        document = _select_graph()
        document.subgraph().tensors[6].isVariable = True
        self.assertFalse(document.graph().is_constant(6))

    def test_graph_input_is_not_an_empty_constant(self):
        document = _select_graph()
        document.subgraph().inputs.append(6)
        self.assertFalse(document.graph().is_constant(6))
        document.verify()

    def test_operator_output_is_not_an_empty_constant(self):
        document = _select_graph()
        document.subgraph().operators[0].outputs.append(6)
        self.assertFalse(document.graph().is_constant(6))
        document.verify()

    def test_empty_constant_can_be_a_public_output(self):
        document = _select_graph()
        document.subgraph().outputs.append(6)
        self.assertTrue(document.graph().is_constant(6))
        document.verify()

    def test_constant_shape_is_not_an_extracted_runtime_input(self):
        document = _select_graph()
        boundary = document.graph().region_boundary([1])
        self.assertEqual(boundary.inputs, (3,))
        self.assertEqual(boundary.outputs, (4,))

    def test_nonempty_and_external_storage_keep_existing_classification(self):
        document = _select_graph()
        self.assertTrue(document.graph().is_constant(1))
        document.model.buffers[3].offset = 1024
        self.assertTrue(document.graph().is_constant(6))

    def test_buffer_zero_is_still_required_to_be_empty(self):
        document = _select_graph()
        document.model.buffers[0].data = b"unexpected"
        with self.assertRaises(CircleVerificationError) as error:
            document.verify()
        self.assertIn("BUFFER_ZERO_NOT_EMPTY", str(error.exception))

    def test_invalid_tensor_indices_return_false(self):
        document = _select_graph()
        for index in (-1, 999):
            self.assertFalse(
                is_constant_tensor(document.model, document.subgraph(), index)
            )


class TestEmptyConstantCodec(unittest.TestCase):
    def test_decode_empty_and_omitted_data(self):
        for data in (None, b"", [], np.empty(0, dtype=np.uint8)):
            with self.subTest(data_type=type(data).__name__):
                document = _select_graph(data)
                value = _codec().decode_tensor(
                    document.model, subgraph_index=0, tensor_index=6
                )
                self.assertEqual(value.shape, (0,))
                self.assertEqual(value.data.dtype, np.dtype("int32"))
                self.assertEqual(value.data.size, 0)
                self.assertEqual(_codec().encode(value), b"")

    def test_empty_multidimensional_value(self):
        document = _select_graph()
        document.subgraph().tensors[6].shape = [2, 0, 3]
        value = _codec().decode_tensor(document.model, subgraph_index=0, tensor_index=6)
        self.assertEqual(value.shape, (2, 0, 3))
        self.assertEqual(value.data.size, 0)

    def test_missing_scalar_payload_is_not_replaced_with_empty_data(self):
        document = _select_graph()
        document.subgraph().tensors[6].shape = []
        with self.assertRaisesRegex(CircleValueError, "without inline data"):
            _codec().decode_tensor(document.model, subgraph_index=0, tensor_index=6)

    def test_explicit_empty_payload_does_not_satisfy_scalar_storage(self):
        document = _select_graph(b"")
        document.subgraph().tensors[6].shape = []
        with self.assertRaisesRegex(CircleValueError, "requires 4 bytes"):
            _codec().decode_tensor(document.model, subgraph_index=0, tensor_index=6)

    def test_missing_dynamic_payload_stays_invalid(self):
        document = _select_graph()
        document.subgraph().tensors[6].shapeSignature = [-1]
        with self.assertRaisesRegex(CircleValueError, "without inline data"):
            _codec().decode_tensor(document.model, subgraph_index=0, tensor_index=6)

    def test_external_storage_remains_unsupported(self):
        for data in (None, b""):
            for field in ("offset", "size"):
                with self.subTest(data=data, field=field):
                    document = _select_graph(data)
                    setattr(document.model.buffers[3], field, 1024)
                    with self.assertRaisesRegex(CircleValueError, "External Circle"):
                        _codec().decode_tensor(
                            document.model, subgraph_index=0, tensor_index=6
                        )

    def test_reserved_buffer_is_not_decoded_as_empty_constant(self):
        document = _select_graph()
        document.subgraph().tensors[6].buffer = 0
        with self.assertRaisesRegex(CircleValueError, "inline constant buffer"):
            _codec().decode_tensor(document.model, subgraph_index=0, tensor_index=6)

    def test_nonempty_constant_still_decodes(self):
        document = _select_graph()
        value = _codec().decode_tensor(document.model, subgraph_index=0, tensor_index=1)
        np.testing.assert_array_equal(value.data, np.array([1], dtype=np.int32))


if __name__ == "__main__":
    unittest.main()
