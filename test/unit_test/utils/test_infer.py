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

import unittest
from typing import ClassVar
from unittest.mock import patch

import numpy as np

import tico
import torch
from tico.interpreter import interpreter as interpreter_module
from tico.interpreter.backends import run_circle
from tico.interpreter.interpreter import Interpreter
from tico.utils.model import CircleModel

from test.modules.op.add import SimpleAdd
from test.modules.op.avg_pool2d import AvgPoolWithPaddingKwargs
from test.modules.op.cat import SimpleCatDefault, SimpleCatWithDim


class InferSimpleAddTest(unittest.TestCase):
    def setUp(self):
        # Input: torch.ones(1), torch.ones(1)
        m = SimpleAdd()
        self.torch_model = m
        args, kwargs = m.get_example_inputs()
        self.circle_model = tico.convert(m.eval(), args, kwargs)

    def test_add_float(self):
        x = torch.randn(1)
        y = torch.randn(1)
        with torch.no_grad():
            out_np = self.torch_model(x, y).numpy()
        np.testing.assert_allclose(
            actual=self.circle_model(x, y), desired=out_np, rtol=1e-4, atol=1e-4
        )

    def test_add_float_numpy(self):
        x = torch.randn(1).numpy()
        y = torch.randn(1).numpy()
        with torch.no_grad():
            out_np = self.torch_model(x, y)
        np.testing.assert_allclose(
            actual=self.circle_model(x, y), desired=out_np, rtol=1e-4, atol=1e-4
        )

    def test_add_float_builtin(self):
        x = torch.tensor([1.0])
        y = torch.tensor([2.0])
        np.testing.assert_allclose(
            actual=self.circle_model(x, y), desired=[10.0], rtol=1e-4, atol=1e-4
        )


class InferCatTest(unittest.TestCase):
    def test_concat(self):
        # convert
        m = SimpleCatDefault()
        torch_model = m
        args, kwargs = m.get_example_inputs()
        circle_model = tico.convert(m.eval(), args, kwargs)
        # test
        x = torch.randn(3)
        y = torch.randn(2)
        with torch.no_grad():
            out_np = torch_model((x, y)).numpy()
        np.testing.assert_allclose(
            actual=circle_model((x, y)), desired=out_np, rtol=1e-4, atol=1e-4
        )

    def test_concat_with_dim(self):
        # convert
        m = SimpleCatWithDim()
        torch_model = m
        args, kwargs = m.get_example_inputs()
        circle_model = tico.convert(m.eval(), args, kwargs)
        # test
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        with torch.no_grad():
            out_np = torch_model((x, y)).numpy()
        np.testing.assert_allclose(
            actual=circle_model((x, y)), desired=out_np, rtol=1e-4, atol=1e-4
        )


class InferAvgPoolReverseKwargsTest(unittest.TestCase):
    def test_avgpool_reverse_kwargs(self):
        # convert
        m = AvgPoolWithPaddingKwargs()
        torch_model = m
        args, kwargs = m.get_example_inputs()
        circle_model = tico.convert(m.eval(), args, kwargs)

        # test
        tensor0 = torch.randn(2, 4, 8, 16)
        tensor1 = torch.randn(2, 4, 4, 8)
        with torch.no_grad():
            out_np = torch_model(tensor1=tensor1, tensor0=tensor0).numpy()

        kwargs = {
            "tensor1": tensor1,
            "tensor0": tensor0,
        }
        np.testing.assert_allclose(
            actual=circle_model(**kwargs), desired=out_np, rtol=1e-4, atol=1e-4
        )


class FakeInterpreter:
    instances: ClassVar[list["FakeInterpreter"]] = []

    def __init__(self, circle_binary):
        self.circle_binary = circle_binary
        self.writes = []
        self.interpreted = False
        FakeInterpreter.instances.append(self)

    def writeInputTensor(self, input_idx, input_data):
        self.writes.append((input_idx, input_data.clone()))

    def interpret(self):
        self.interpreted = True

    def readOutputTensor(self, output_idx, output):
        output.fill(0)


class CircleInterpreterAdapterBindingTest(unittest.TestCase):
    """Check the call convention of the optional CFFI adapter without ONE."""

    def test_adapter_writes_inputs_in_model_input_order(self):
        m = AvgPoolWithPaddingKwargs()
        args, kwargs = m.get_example_inputs()
        circle_model = tico.convert(m.eval(), args, kwargs)

        tensor0 = torch.randn(2, 4, 8, 16)
        tensor1 = torch.randn(2, 4, 4, 8)
        FakeInterpreter.instances = []

        # ModelInputSpec orders the bound inputs; the adapter must forward them
        # positionally to writeInputTensor.
        model = CircleModel(circle_model.circle_binary, runtime="circle-interpreter")
        with patch.object(interpreter_module, "Interpreter", FakeInterpreter):
            model(tensor0=tensor0, tensor1=tensor1)

        fake = FakeInterpreter.instances[0]
        self.assertTrue(fake.interpreted)
        self.assertEqual(fake.writes[0][0], 0)
        self.assertEqual(fake.writes[0][1].shape, tensor1.shape)
        self.assertTrue(torch.equal(fake.writes[0][1], tensor1))
        self.assertEqual(fake.writes[1][0], 1)
        self.assertEqual(fake.writes[1][1].shape, tensor0.shape)
        self.assertTrue(torch.equal(fake.writes[1][1], tensor0))

    def test_adapter_is_not_used_by_the_default_runtime(self):
        m = SimpleAdd()
        args, kwargs = m.get_example_inputs()
        circle_model = tico.convert(m.eval(), args, kwargs)
        FakeInterpreter.instances = []

        with patch.object(interpreter_module, "Interpreter", FakeInterpreter):
            result = run_circle(circle_model.circle_binary, list(args))

        self.assertEqual(FakeInterpreter.instances, [])
        with torch.no_grad():
            expected = m(*args).numpy()
        np.testing.assert_allclose(result[0], expected, rtol=1e-6, atol=1e-6)


class RuntimeSelectionTest(unittest.TestCase):
    def test_unknown_runtime_name_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown Circle runtime"):
            CircleModel(b"", runtime="tflite")

    def test_default_runtime_is_reference(self):
        self.assertEqual(CircleModel(b"").runtime, "reference")


class InterpreterAvailabilityTest(unittest.TestCase):
    def test_missing_interpreter_library_does_not_raise_during_cleanup(self):
        with patch.object(
            Interpreter, "LIB_PATH", Interpreter.LIB_PATH.parent / "missing.so"
        ):
            with self.assertRaisesRegex(
                RuntimeError, "Please install one-compiler for circle inference"
            ):
                Interpreter(b"")
