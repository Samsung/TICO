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

from tico.circle._object import clone_object, create_object, freeze_object
from tico.circle.errors import CircleValueError

from test.unit_test.circle.infrastructure_fixture import (
    fake_object_factory,
    FakeDetails,
)


class SlottedValue:
    """Provide public slot-backed fields for fingerprint tests."""

    __slots__ = ("first", "second")

    def __init__(self, first, second):
        """Store two values without an instance dictionary."""

        self.first = first
        self.second = second


class EmptyOptionsT:
    """Provide an empty generated-table substitute for fingerprint tests."""


class EmptyValue:
    """Provide an unsupported opaque empty object."""


class ObjectHelpersTest(unittest.TestCase):
    """Check generated-object creation, cloning, and structural fingerprints."""

    def test_create_object_uses_injected_factory(self):
        """Create a mutable Object API substitute without importing a schema."""

        tensor = create_object("Tensor", fake_object_factory)
        tensor.name = "value"
        self.assertEqual(tensor.name, "value")

    def test_clone_object_detaches_nested_mutable_fields(self):
        """Return an independently mutable deep copy of a generated value."""

        source = FakeDetails()
        cloned = clone_object(source)
        cloned.axes.append(2)

        self.assertEqual(source.axes, [0, 1])
        self.assertEqual(cloned.axes, [0, 1, 2])

    def test_freeze_object_handles_slot_backed_generated_values(self):
        """Fingerprint equivalent slot-backed objects independently of identity."""

        first = SlottedValue(1, [2, 3])
        second = SlottedValue(1, [2, 3])
        self.assertEqual(freeze_object(first), freeze_object(second))

    def test_freeze_object_handles_empty_generated_tables(self):
        """Fingerprint generated option tables that intentionally have no fields."""

        with patch(
            "tico.circle._object.object_api_type",
            return_value=EmptyOptionsT,
        ):
            self.assertEqual(
                freeze_object(EmptyOptionsT()),
                ("object", "EmptyOptionsT"),
            )

    def test_freeze_object_rejects_unrecognized_empty_objects(self):
        """Keep rejecting opaque empty values outside the generated Object API."""

        with self.assertRaisesRegex(CircleValueError, "Unsupported"):
            freeze_object(EmptyValue())

    def test_freeze_object_rejects_cycles_in_object_arrays(self):
        """Reject self-referential NumPy object arrays with a clear value error."""

        value = np.empty(1, dtype=object)
        value[0] = value
        with self.assertRaisesRegex(CircleValueError, "Cyclic"):
            freeze_object(value)


if __name__ == "__main__":
    unittest.main()
