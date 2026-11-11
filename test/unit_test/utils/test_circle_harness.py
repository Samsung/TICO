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

"""Keep artifact verification, zero probing, and real inference distinct."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from circle_schema import circle
from tico.circle import CircleDocument
from tico.circle.runtime import CircleReferenceRuntime, CircleRuntimeValidationError
from tico.utils.model import CircleModel

from test.modules.base import TestModuleBase
from test.pt2_to_circle_test import builder as harness
from test.pt2_to_circle_test.test_pt2_to_circle import infer_circle, verify_circle
from test.support.circle.builder import CircleModelBuilder


class _AddOne(TestModuleBase):
    def forward(self, x):
        return x + 1

    def get_example_inputs(self):
        return (torch.ones(1, 2),), {}


class _DynamicAddOne(_AddOne):
    def get_dynamic_shapes(self):
        return {"x": {0: torch.export.Dim("batch")}}


class TestCircleHarness(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.path = str(self.directory / "fixture.circle")

    def _add_document(self, *, dynamic=False, shape=None, dtype=np.float32):
        fixture = CircleModelBuilder()
        signature = [-1, 2] if dynamic else None
        value = fixture.input("x", [1, 2], shape_signature=signature)
        one = fixture.const_f32("one", 1)
        out = fixture.activation(
            "out", shape or [1, 2], dtype=dtype, shape_signature=signature
        )
        options = circle.AddOptions.AddOptionsT()
        options.fusedActivationFunction = 0
        fixture.operator("ADD", [value, one], [out], options=options)
        fixture.set_outputs(out)
        return fixture.build()

    def _probe_spy(self):
        return patch.object(
            CircleReferenceRuntime,
            "probe_static_contracts",
            autospec=True,
            side_effect=CircleReferenceRuntime.probe_static_contracts,
        )

    def test_reference_keeps_structure_probe_public_loading_and_real_execution(self):
        self._add_document().save(self.path)
        with (
            patch.object(
                CircleDocument,
                "verify",
                autospec=True,
                side_effect=CircleDocument.verify,
            ) as verify,
            self._probe_spy() as probe,
            patch.object(CircleModel, "load", wraps=CircleModel.load) as load,
        ):
            verify_circle(self.path)
            probe.assert_called_once()
            verify.assert_called_once()
            result = infer_circle(self.path, (torch.tensor([[3.0, 4.0]]),), {})
            load.assert_called_once_with(self.path, runtime="reference")
            self.assertEqual(verify.call_count, 2)
            probe.assert_called_once()
        np.testing.assert_array_equal(result[0], [[4, 5]])

    def test_real_reference_still_rejects_wrong_operator_shape_and_dtype(self):
        for shape, dtype, error in [
            ([2, 1], np.float32, "operator 0 .* has shape"),
            (None, np.int32, "dtype float32.*requires INT32"),
        ]:
            with self.subTest(shape=shape, dtype=dtype):
                self._add_document(shape=shape, dtype=dtype).save(self.path)
                with self.assertRaisesRegex(CircleRuntimeValidationError, error):
                    verify_circle(self.path)
                with self.assertRaisesRegex(CircleRuntimeValidationError, error):
                    infer_circle(self.path, (torch.ones(1, 2),), {})

    def test_real_reference_failure_after_successful_probe_propagates(self):
        fixture = CircleModelBuilder()
        value = fixture.input("x", [2])
        index = fixture.input("index", [1], dtype=np.int32)
        out = fixture.activation("out", [1])
        options = circle.GatherOptions.GatherOptionsT()
        options.axis = 0
        options.batchDims = 0
        fixture.operator("GATHER", [value, index], [out], options=options)
        fixture.set_outputs(out)
        fixture.build().save(self.path)
        verify_circle(self.path)  # Zero is a valid index.
        with self.assertRaisesRegex(
            CircleRuntimeValidationError, "GATHER indices contain values outside"
        ):
            infer_circle(
                self.path,
                (torch.tensor([3.0, 4.0]), torch.tensor([2], dtype=torch.int32)),
                {},
            )

    def test_zero_probe_is_an_additional_data_case_even_with_static_shapes(self):
        fixture = CircleModelBuilder()
        value = fixture.input("x", [4])
        shape = fixture.input("shape", [2], dtype=np.int32)
        out = fixture.activation("out", [2, 2])
        fixture.operator(
            "RESHAPE",
            [value, shape],
            [out],
            options=circle.ReshapeOptions.ReshapeOptionsT(),
        )
        fixture.set_outputs(out)
        fixture.build().save(self.path)
        result = infer_circle(
            self.path,
            (
                torch.arange(4, dtype=torch.float32),
                torch.tensor([2, 2], dtype=torch.int32),
            ),
            {},
        )
        np.testing.assert_array_equal(result[0], [[0, 1], [2, 3]])
        # Actual reference success does not establish validity of zero shape values.
        with self.assertRaisesRegex(
            CircleRuntimeValidationError, r"RESHAPE target \[0, 0\] has 0 elements"
        ):
            verify_circle(self.path)

    def _conversion(self, document):
        def convert(*_args, circle_model_path, **_kwargs):
            document.save(circle_model_path)

        return patch.object(harness, "convert_nnmodule_to_circle", side_effect=convert)

    def test_conversion_only_keeps_reference_probe(self):
        runner = harness.NNModuleTest("conversion_only", _AddOne())
        runner.test_dir = self.directory
        with (
            self._conversion(self._add_document()),
            self._probe_spy() as probe,
            patch.object(harness, "infer_circle") as infer,
        ):
            runner._run(without_pt2=True, without_inference=True)
            probe.assert_called_once()
            infer.assert_not_called()

    def test_external_boundary_keeps_real_reference_prevalidation(self):
        runner = harness.NNModuleTest("external", _AddOne())
        runner.test_dir = self.directory
        with (
            self._conversion(self._add_document()),
            patch.object(runner, "runtime", return_value="onert"),
            self._probe_spy() as probe,
        ):

            def external(*_args, **_kwargs):
                probe.assert_called_once()
                return [np.array([[2, 2]], dtype=np.float32)]

            with patch.object(harness, "infer_circle", side_effect=external) as infer:
                runner._run(without_pt2=True)
                self.assertEqual(infer.call_args.kwargs["runtime"], "onert")
            probe.assert_called_once()

    def test_dynamic_shapes_keep_actual_and_alternative_reference_execution(self):
        runner = harness.NNModuleTest("dynamic", _DynamicAddOne())
        runner.test_dir = self.directory
        with (
            self._conversion(self._add_document(dynamic=True)),
            self._probe_spy() as probe,
            patch.object(harness, "infer_circle", wraps=infer_circle) as infer,
        ):
            runner._run(without_pt2=True)
            probe.assert_not_called()
            self.assertEqual(infer.call_count, 2)
            first, alternative = infer.call_args_list
            self.assertEqual(first.kwargs["forward_args"][0].shape, (1, 2))
            self.assertEqual(alternative.kwargs["forward_args"][0].shape, (3, 2))
