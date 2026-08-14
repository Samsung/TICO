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
from unittest.mock import patch

import numpy as np

from tico.circle.errors import CircleValueError
from tico.circle.value import (
    default_tensor_type_registry,
    TensorTypeSpec,
    TensorValue,
    TensorValueCodec,
)

from test.unit_test.circle.infrastructure_fixture import (
    FakeBuffer,
    FakeModel,
    FakeSubGraph,
    FakeTensor,
    FLOAT32,
    INT32,
    INT4,
    INT8,
    make_registry,
    UINT4,
    UINT8,
)


class TensorValueCodecTest(unittest.TestCase):
    """Check dense and packed inline Circle buffer encoding."""

    def setUp(self):
        """Create a schema-independent codec for every test."""

        self.codec = TensorValueCodec(make_registry())

    def test_default_registry_discovers_available_nested_schema_enums(self):
        """Build codecs lazily from enum members available in the active schema."""

        class FakeTensorType:
            """Expose only tensor types needed by this registry discovery test."""

            FLOAT32 = FLOAT32
            UINT4 = UINT4

        class FakeTensorTypeModule:
            """Mirror the nested generated enum module layout."""

            TensorType = FakeTensorType

        class FakeSchema:
            """Expose the generated TensorType module through a schema object."""

            TensorType = FakeTensorTypeModule

        default_tensor_type_registry.cache_clear()
        try:
            with patch(
                "tico.circle.value.dtype.circle_schema",
                return_value=FakeSchema,
            ):
                registry = default_tensor_type_registry()
            self.assertEqual(registry.by_name("FLOAT32").tensor_type, FLOAT32)
            self.assertEqual(registry.by_name("UINT4").tensor_type, UINT4)
            self.assertIsNone(registry.get(12345))
        finally:
            default_tensor_type_registry.cache_clear()

    def test_type_spec_rejects_dense_storage_conversion(self):
        """Keep dense encoding free from semantic dtype conversion."""

        with self.assertRaisesRegex(ValueError, "must match exactly"):
            TensorTypeSpec(
                "FLOAT32_AS_INT32",
                98,
                np.dtype(np.float32),
                np.dtype(np.int32),
                32,
            )

    def test_type_spec_rejects_unsupported_packed_width(self):
        """Reject packed formats that the four-bit codec would misinterpret."""

        with self.assertRaisesRegex(ValueError, "four-bit types only"):
            TensorTypeSpec(
                "INT2",
                99,
                np.dtype(np.int8),
                np.dtype(np.uint8),
                2,
                signed=True,
                packed=True,
            )

    def test_float32_round_trip_uses_little_endian_storage(self):
        """Encode dense values using Circle's little-endian FlatBuffer convention."""

        value = TensorValue(
            FLOAT32,
            (2,),
            np.array([1.0, -2.5], dtype=np.float32),
        )
        payload = self.codec.encode(value)
        self.assertEqual(payload, b"\x00\x00\x80?\x00\x00 \xc0")

        restored = self.codec.decode(
            payload,
            tensor_type=FLOAT32,
            shape=(2,),
        )
        np.testing.assert_array_equal(restored.data, value.data)

    def test_dense_integer_types_round_trip_without_value_conversion(self):
        """Preserve signed and unsigned dense integer values exactly."""

        cases = (
            (INT32, np.array([-2, 0, 5], dtype=np.int32)),
            (INT8, np.array([-128, 0, 127], dtype=np.int8)),
            (UINT8, np.array([0, 1, 255], dtype=np.uint8)),
        )
        for tensor_type, data in cases:
            with self.subTest(tensor_type=tensor_type):
                value = TensorValue(tensor_type, data.shape, data)
                restored = self.codec.decode(
                    self.codec.encode(value),
                    tensor_type=tensor_type,
                    shape=data.shape,
                )
                np.testing.assert_array_equal(restored.data, data)
                self.assertEqual(restored.data.dtype, data.dtype)

    def test_encoding_rejects_implicit_dtype_conversion(self):
        """Require the logical NumPy dtype to match the Circle tensor type exactly."""

        value = TensorValue(FLOAT32, (1,), np.array([1], dtype=np.int32))
        with self.assertRaisesRegex(CircleValueError, "requires NumPy dtype"):
            self.codec.encode(value)

    def test_uint4_round_trip_uses_low_nibble_first_for_odd_count(self):
        """Pack the first value into the low nibble and zero-fill an odd tail."""

        value = TensorValue(
            UINT4,
            (5,),
            np.array([1, 2, 3, 4, 15], dtype=np.uint8),
        )
        payload = self.codec.encode(value)
        self.assertEqual(payload, bytes((0x21, 0x43, 0x0F)))
        restored = self.codec.decode(
            payload,
            tensor_type=UINT4,
            shape=(5,),
        )
        np.testing.assert_array_equal(restored.data, value.data)

    def test_int4_round_trip_sign_extends_high_nibbles(self):
        """Preserve the complete signed four-bit range during packing."""

        value = TensorValue(
            INT4,
            (4,),
            np.array([-8, -1, 0, 7], dtype=np.int8),
        )
        payload = self.codec.encode(value)
        self.assertEqual(payload, bytes((0xF8, 0x70)))
        restored = self.codec.decode(
            payload,
            tensor_type=INT4,
            shape=(4,),
        )
        np.testing.assert_array_equal(restored.data, value.data)

    def test_decode_tensor_reads_inline_model_buffer(self):
        """Resolve tensor metadata and inline storage through model index spaces."""

        payload = self.codec.encode(
            TensorValue(FLOAT32, (1,), np.array([3.0], dtype=np.float32))
        )
        model = FakeModel(
            buffers=[FakeBuffer(), FakeBuffer(data=np.frombuffer(payload, np.uint8))],
            subgraphs=[
                FakeSubGraph(
                    tensors=[
                        FakeTensor(
                            name="constant",
                            buffer=1,
                            shape=[1],
                            shapeSignature=[1],
                            type=FLOAT32,
                        )
                    ]
                )
            ],
        )

        value = self.codec.decode_tensor(
            model,
            subgraph_index=0,
            tensor_index=0,
        )
        self.assertEqual(float(value.data[0]), 3.0)

    def test_decode_tensor_accepts_zero_element_inline_constant(self):
        """Treat an empty payload as valid when the concrete element count is zero."""

        model = FakeModel(
            buffers=[FakeBuffer(), FakeBuffer(data=np.array([], dtype=np.uint8))],
            subgraphs=[
                FakeSubGraph(
                    tensors=[
                        FakeTensor(
                            name="empty",
                            buffer=1,
                            shape=[0],
                            shapeSignature=[0],
                            type=FLOAT32,
                        )
                    ]
                )
            ],
        )

        value = self.codec.decode_tensor(
            model,
            subgraph_index=0,
            tensor_index=0,
        )
        self.assertEqual(value.shape, (0,))
        self.assertEqual(value.element_count, 0)

    def test_decode_tensor_rejects_unresolved_external_buffer(self):
        """Reject external payload references until a materializer is provided."""

        model = FakeModel(
            buffers=[FakeBuffer(), FakeBuffer(data=None, offset=64, size=4)],
            subgraphs=[
                FakeSubGraph(
                    tensors=[
                        FakeTensor(
                            name="external",
                            buffer=1,
                            shape=[1],
                            type=FLOAT32,
                        )
                    ]
                )
            ],
        )

        with self.assertRaisesRegex(CircleValueError, "External Circle buffers"):
            self.codec.decode_tensor(
                model,
                subgraph_index=0,
                tensor_index=0,
            )


if __name__ == "__main__":
    unittest.main()
