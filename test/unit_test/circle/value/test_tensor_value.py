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

import numpy as np

from tico.circle.errors import CircleValueError
from tico.circle.value import TensorQuantization, TensorValue

from test.unit_test.circle.infrastructure_fixture import (
    fake_object_factory,
    FakeDetails,
    FakeQuantizationParameters,
    FLOAT32,
)


class TensorValueTest(unittest.TestCase):
    """Check immutable tensor and quantization value semantics."""

    def test_tensor_value_owns_a_read_only_contiguous_copy(self):
        """Detach tensor values from mutable and non-contiguous input arrays."""

        source = np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::2]
        value = TensorValue(FLOAT32, (3, 2), source)
        source[0, 0] = 100.0

        self.assertTrue(value.data.flags.c_contiguous)
        self.assertFalse(value.data.flags.writeable)
        self.assertEqual(float(value.data[0, 0]), 0.0)
        with self.assertRaises(ValueError):
            value.data[0, 0] = 1.0

    def test_tensor_value_equality_and_hash_include_exact_payload_bits(self):
        """Distinguish signed zero while keeping equal values hash-compatible."""

        positive = TensorValue(FLOAT32, (1,), np.array([0.0], dtype=np.float32))
        same = TensorValue(FLOAT32, (1,), np.array([0.0], dtype=np.float32))
        negative = TensorValue(FLOAT32, (1,), np.array([-0.0], dtype=np.float32))

        self.assertEqual(positive, same)
        self.assertEqual(hash(positive), hash(same))
        self.assertNotEqual(positive, negative)

    def test_tensor_value_rejects_object_storage(self):
        """Reject object arrays that have no portable Circle byte representation."""

        with self.assertRaisesRegex(CircleValueError, "object or string"):
            TensorValue(
                FLOAT32,
                (1,),
                np.array(["value"], dtype=object),
            )

    def test_tensor_value_validates_shape_element_count(self):
        """Reject concrete shapes that do not match logical element count."""

        with self.assertRaisesRegex(CircleValueError, "requires 4 elements"):
            TensorValue(FLOAT32, (2, 2), np.array([1.0], dtype=np.float32))

    def test_quantization_round_trip_owns_nested_union_details(self):
        """Clone arbitrary details while preserving a stable structural fingerprint."""

        source = FakeQuantizationParameters(
            scale=[0.25, 0.5],
            zeroPoint=[0, 0],
            min=[-1.0, -2.0],
            max=[1.0, 2.0],
            quantizedDimension=1,
            detailsType=7,
            details=FakeDetails(),
        )
        value = TensorQuantization.from_object(source)
        self.assertIsNotNone(value)
        assert value is not None
        source.details.axes.append(2)

        restored = value.to_object(fake_object_factory)
        self.assertEqual(restored.scale, [0.25, 0.5])
        self.assertEqual(restored.zeroPoint, [0, 0])
        self.assertEqual(restored.details.axes, [0, 1])
        self.assertEqual(value, TensorQuantization.from_object(restored))

    def test_quantization_rejects_incomplete_affine_vectors(self):
        """Reject scale-only or zero-point-only affine quantization records."""

        with self.assertRaisesRegex(CircleValueError, "both be present"):
            TensorQuantization(scale=(0.5,))


if __name__ == "__main__":
    unittest.main()
