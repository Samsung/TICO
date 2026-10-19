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

import gc
import io
import unittest
import weakref
from unittest.mock import patch

import numpy as np
from circle_schema import circle

from tico.circle.errors import CircleIOError
from tico.circle.io import (
    model_from_bytes,
    model_from_object,
    model_to_bytes,
    save_model,
)
from tico.serialize.circle_binary import CircleBinaryLayout

from test.support.circle.builder import CircleModelBuilder


def _model():
    builder = CircleModelBuilder()
    value = builder.const_f32("weight", np.arange(16, dtype=np.float32))
    builder.set_outputs(value)
    return builder.model


class ModelFromObjectTest(unittest.TestCase):
    def test_metadata_matches_binary_round_trip_without_joining_payloads(self):
        model = _model()
        model.construction_only = object()
        model.subgraphs[0].name_to_node = {"weight": object()}
        quantization = circle.QuantizationParameters.QuantizationParametersT()
        quantization.scale = [0.1]
        quantization.min = [-0.1]
        quantization.max = [0.3]
        quantization.zeroPoint = [0]
        model.subgraphs[0].tensors[0].quantization = quantization
        baseline = model_from_bytes(model_to_bytes(model))

        with (
            patch.object(
                CircleBinaryLayout,
                "to_bytes",
                side_effect=AssertionError("complete binary allocated"),
            ),
            patch(
                "tico.serialize.circle_binary._join_payloads",
                side_effect=AssertionError("payloads joined"),
            ),
        ):
            normalized = model_from_object(model)

        self.assertEqual(model_to_bytes(normalized), model_to_bytes(baseline))
        self.assertEqual(normalized.description, b"circle-value-test")
        self.assertEqual(normalized.subgraphs[0].tensors[0].name, b"weight")
        self.assertEqual(
            normalized.subgraphs[0].tensors[0].quantization.scale.dtype,
            np.dtype(np.float32),
        )
        self.assertEqual(
            float(normalized.subgraphs[0].tensors[0].quantization.scale[0]),
            float(np.float32(0.1)),
        )
        self.assertFalse(hasattr(normalized, "construction_only"))
        self.assertFalse(hasattr(normalized.subgraphs[0], "name_to_node"))
        normalized.subgraphs[0].tensors[0].shape[0] = 7
        self.assertEqual(model.subgraphs[0].tensors[0].shape, [16])
        self.assertEqual(quantization.scale, [0.1])

    def test_borrowed_payload_is_read_only_and_owns_its_lifetime(self):
        model = _model()
        source: np.ndarray = np.arange(64, dtype=np.uint8)
        model.buffers[1].data = source
        source_reference = weakref.ref(source)
        normalized = model_from_object(model)
        borrowed = normalized.buffers[1].data

        self.assertTrue(np.shares_memory(borrowed, source))
        self.assertTrue(source.flags.writeable)
        self.assertFalse(borrowed.flags.writeable)
        self.assertEqual(normalized.buffers[1].offset, 0)
        self.assertEqual(normalized.buffers[1].size, 0)
        self.assertEqual(model.buffers[1].offset, 0)
        with self.assertRaisesRegex(ValueError, "read-only"):
            borrowed[0] = 99
        del model, source
        gc.collect()
        self.assertIsNotNone(source_reference())
        np.testing.assert_array_equal(borrowed, np.arange(64, dtype=np.uint8))
        del normalized, borrowed
        gc.collect()
        self.assertIsNone(source_reference())

    def test_empty_and_one_byte_constants_keep_binary_semantics(self):
        builder = CircleModelBuilder()
        empty = builder.constant("empty", np.empty(0, dtype=np.uint8))
        first = builder.constant("first", np.array([7], dtype=np.uint8))
        second = builder.constant("second", np.array([11], dtype=np.uint8))
        builder.set_outputs(empty, first, second)
        original = model_to_bytes(builder.model)

        normalized = model_from_object(builder.model)

        self.assertEqual(model_to_bytes(normalized), original)
        self.assertEqual(normalized.buffers[1].data.size, 0)
        np.testing.assert_array_equal(normalized.buffers[2].data, [7])
        np.testing.assert_array_equal(normalized.buffers[3].data, [11])

    def test_invalid_model_and_serialization_failure_are_reported(self):
        with self.assertRaisesRegex(TypeError, "Object API"):
            model_from_object(None)
        with patch("tico.circle.io.plan_circle_layout", side_effect=MemoryError("oom")):
            with self.assertRaisesRegex(CircleIOError, "normalize.*oom") as raised:
                model_from_object(_model())
        self.assertIsInstance(raised.exception.__cause__, MemoryError)


class ExplicitInlineBudgetSaveTest(unittest.TestCase):
    def test_explicit_budget_selects_appended_storage_without_changing_default(self):
        model = _model()
        default = io.BytesIO()
        appended = io.BytesIO()

        save_model(model, default)
        save_model(model, appended, inline_budget=0)

        self.assertEqual(default.getvalue(), model_to_bytes(model))
        root = circle.Model.Model.GetRootAsModel(appended.getvalue(), 0)
        self.assertGreater(root.Buffers(1).Offset(), 0)
        self.assertEqual(
            model_to_bytes(model_from_bytes(appended.getvalue())), default.getvalue()
        )

    def test_invalid_budget_reports_error_before_destination_write(self):
        stream = io.BytesIO(b"existing")
        with self.assertRaisesRegex(
            CircleIOError, "inline_budget must not be negative"
        ):
            save_model(_model(), stream, inline_budget=-1)
        self.assertEqual(stream.getvalue(), b"existing")
