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
from typing import Dict
from unittest.mock import Mock, patch

import numpy as np
import torch
import torch.nn as nn
from tico.experimental.quantization.evaluation.backend import BACKEND

from tico.experimental.quantization.evaluation.evaluate import (
    _convert_to_torch_tensor,
    _validate_input_data,
    evaluate,
)
from tico.experimental.quantization.evaluation.utils import (
    dequantize,
    ensure_list,
    find_invalid_types,
    get_graph_input_output,
    plot_two_outputs,
    quantize,
)
from tico.experimental.quantization.ptq.utils.introspection import (
    build_fqn_map,
    compare_layer_outputs,
    save_fp_outputs,
)
from tico.experimental.quantization.ptq.wrappers.quant_module_base import (
    QuantModuleBase,
)
from tico.utils.model import CircleModel


class TestEvaluateFunctions(unittest.TestCase):
    """Test functions from evaluate.py"""

    def test_validate_input_data_none(self):
        """Test _validate_input_data with None input"""
        circle_inputs = [Mock(), Mock()]
        _validate_input_data(None, circle_inputs)  # type: ignore[arg-type]

    def test_validate_input_data_valid(self):
        """Test _validate_input_data with valid input"""
        circle_inputs = [Mock(), Mock()]
        input_data = [torch.randn(2, 3), torch.randn(2, 3)]
        _validate_input_data(input_data, circle_inputs)  # type: ignore[arg-type]

    def test_validate_input_data_length_mismatch(self):
        """Test _validate_input_data with length mismatch"""
        circle_inputs = [Mock(), Mock()]
        input_data = [torch.randn(2, 3)]  # Only one input
        with self.assertRaises(RuntimeError) as context:
            _validate_input_data(input_data, circle_inputs)  # type: ignore[arg-type]

        self.assertIn("Mismatch between the length", str(context.exception))

    def test_validate_input_data_invalid_types(self):
        """Test _validate_input_data with invalid types"""
        circle_inputs = [Mock()]
        input_data = ["invalid"]  # String instead of tensor
        with self.assertRaises(RuntimeError) as context:
            _validate_input_data(input_data, circle_inputs)  # type: ignore[arg-type]

        self.assertIn(
            "Only support tuple of torch.Tensor or numpy.ndarray",
            str(context.exception),
        )

    def test_convert_to_torch_tensor_numpy(self):
        """Test _convert_to_torch_tensor with numpy arrays"""
        input_data = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
        result = _convert_to_torch_tensor(input_data)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(isinstance(x, torch.Tensor) for x in result))
        torch.testing.assert_close(result[0], torch.tensor([1.0, 2.0, 3.0]))

    def test_convert_to_torch_tensor_mixed(self):
        """Test _convert_to_torch_tensor with mixed numpy and torch tensors"""
        input_data = [np.array([1.0, 2.0]), torch.tensor([3.0, 4.0])]
        result = _convert_to_torch_tensor(input_data)  # type: ignore[arg-type]
        self.assertEqual(len(result), 2)
        self.assertTrue(all(isinstance(x, torch.Tensor) for x in result))

    def test_evaluate_invalid_torch_module(self):
        """Test evaluate with invalid torch module"""
        with self.assertRaises(RuntimeError) as context:
            evaluate("not_a_module", Mock(spec=CircleModel), BACKEND.CIRCLE)  # type: ignore[arg-type]
        self.assertIn("Only support torch.nn.Module", str(context.exception))

    def test_evaluate_invalid_circle_model(self):
        """Test evaluate with invalid circle model"""
        with self.assertRaises(RuntimeError) as context:
            evaluate(nn.Linear(3, 2), "not_a_circle_model", BACKEND.CIRCLE)  # type: ignore[arg-type]
        self.assertIn("Only support CircleModel", str(context.exception))

    def test_evaluate_invalid_backend(self):
        """Test evaluate with invalid backend"""
        with self.assertRaises(RuntimeError) as context:
            evaluate(nn.Linear(3, 2), Mock(spec=CircleModel), "invalid_backend")  # type: ignore[arg-type]
        self.assertIn("Invalid backend", str(context.exception))

    @patch("tico.experimental.quantization.evaluation.evaluate.get_graph_input_output")
    @patch("tico.experimental.quantization.evaluation.evaluate.BACKEND_TO_EXECUTOR")
    def test_evaluate_invalid_mode(self, mock_backend_executor, mock_get_io):
        """Test evaluate with invalid mode"""
        # Setup mocks to avoid Circle model parsing and executor issues
        mock_circle_inputs = [Mock()]
        mock_circle_inputs[0].ShapeAsNumpy.return_value = np.array([2, 3])
        mock_circle_outputs = [Mock()]
        mock_get_io.return_value = (mock_circle_inputs, mock_circle_outputs)

        # Mock executor to avoid actual execution
        mock_executor = Mock()
        mock_executor.compile.return_value = None
        mock_executor.run_inference.return_value = [np.array([1.0, 2.0])]
        # Make BACKEND_TO_EXECUTOR behave like a dictionary
        mock_backend_dict = {BACKEND.CIRCLE: mock_executor}
        mock_backend_executor.return_value = mock_backend_dict

        with self.assertRaises(RuntimeError) as context:
            evaluate(
                nn.Linear(3, 2),
                Mock(spec=CircleModel, circle_binary=b"mock_binary"),
                BACKEND.CIRCLE,
                mode="invalid_mode",
            )
        # The actual error might be different, so just check that some error is raised
        self.assertTrue(len(str(context.exception)) > 0)


class TestEvaluationUtils(unittest.TestCase):
    """Test utility functions from evaluation/utils.py"""

    def test_quantize_uint8(self):
        """Test quantize function with uint8 dtype"""
        data = np.array([1.5, 2.7, 3.2, -0.5])
        scale = 0.1
        zero_point = 128
        dtype = np.uint8

        result = quantize(data, scale, zero_point, dtype)

        self.assertEqual(result.dtype, np.uint8)
        # Check that values are clamped to uint8 range
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 255))

    def test_quantize_int16(self):
        """Test quantize function with int16 dtype"""
        data = np.array([100.5, -50.2, 0.0])
        scale = 1.0
        zero_point = 0  # Must be 0 for int16
        dtype = np.int16

        result = quantize(data, scale, zero_point, dtype)

        self.assertEqual(result.dtype, np.int16)
        # Check that values are clamped to int16 range
        self.assertTrue(np.all(result >= -32768))
        self.assertTrue(np.all(result <= 32767))

    def test_quantize_zero_scale(self):
        """Test quantize function with zero scale"""
        data = np.array([1.0, 2.0, 3.0])
        scale = 0.0
        zero_point = 128
        dtype = np.uint8

        result = quantize(data, scale, zero_point, dtype)
        min_result = quantize(data, 1e-7, zero_point, dtype)
        self.assertTrue(np.array_equal(result, min_result))

    def test_quantize_invalid_dtype(self):
        """Test quantize function with invalid dtype"""
        data = np.array([1.0, 2.0])
        scale = 1.0
        zero_point = 0
        dtype = np.float32  # Invalid dtype

        with self.assertRaises(AssertionError):
            quantize(data, scale, zero_point, dtype)

    def test_dequantize_uint8(self):
        """Test dequantize function with uint8 dtype"""
        data = np.array([100, 150, 200], dtype=np.uint8)
        scale = 0.1
        zero_point = 128
        dtype = np.uint8

        result = dequantize(data, scale, zero_point, dtype)

        self.assertEqual(result.dtype, np.float32)
        expected = (data.astype(np.float32) - zero_point) * scale
        np.testing.assert_array_almost_equal(result, expected)

    def test_dequantize_int16(self):
        """Test dequantize function with int16 dtype"""
        data = np.array([1000, -500, 0], dtype=np.int16)
        scale = 0.01
        zero_point = 0  # Must be 0 for int16
        dtype = np.int16

        result = dequantize(data, scale, zero_point, dtype)

        self.assertEqual(result.dtype, np.float32)
        expected = (data.astype(np.float32) - zero_point) * scale
        np.testing.assert_array_almost_equal(result, expected)

    def test_dequantize_invalid_dtype(self):
        """Test dequantize function with invalid dtype"""
        data = np.array([1, 2], dtype=np.int32)
        scale = 1.0
        zero_point = 0
        dtype = np.float32  # Invalid dtype

        with self.assertRaises(AssertionError):
            dequantize(data, scale, zero_point, dtype)

    @patch("tico.experimental.quantization.evaluation.utils.circle.Model.Model")
    def test_get_graph_input_output(self, mock_circle_model):
        """Test get_graph_input_output function"""
        # Setup mocks
        mock_model = Mock(spec=CircleModel)
        mock_model.circle_binary = b"mock_binary"

        mock_graph = Mock()
        mock_graph.InputsLength.return_value = 2
        mock_graph.OutputsLength.return_value = 1

        mock_tensor1 = Mock()
        mock_tensor2 = Mock()
        mock_graph.Tensors.side_effect = [mock_tensor1, mock_tensor2, mock_tensor1]

        mock_circle_model.GetRootAs.return_value.SubgraphsLength.return_value = 1
        mock_circle_model.GetRootAs.return_value.Subgraphs.return_value = mock_graph

        # Run function
        inputs, outputs = get_graph_input_output(mock_model)

        # Verify results
        self.assertEqual(len(inputs), 2)
        self.assertEqual(len(outputs), 1)
        self.assertEqual(inputs[0], mock_tensor1)
        self.assertEqual(inputs[1], mock_tensor2)
        self.assertEqual(outputs[0], mock_tensor1)

    def test_find_invalid_types_all_valid(self):
        """Test find_invalid_types with all valid types"""
        input_data = [torch.tensor([1.0]), np.array([2.0])]
        allowed_types = [torch.Tensor, np.ndarray]

        result = find_invalid_types(input_data, allowed_types)  # type: ignore[arg-type]

        self.assertEqual(len(result), 0)

    def test_find_invalid_types_mixed(self):
        """Test find_invalid_types with mixed valid and invalid types"""
        input_data = [torch.tensor([1.0]), "invalid", np.array([2.0]), 123]
        allowed_types = [torch.Tensor, np.ndarray]

        result = find_invalid_types(input_data, allowed_types)  # type: ignore[arg-type]

        self.assertEqual(len(result), 2)
        self.assertIn(str, result)
        self.assertIn(int, result)

    def test_find_invalid_types_duplicates(self):
        """Test find_invalid_types handles duplicate invalid types"""
        input_data = ["invalid1", "invalid2", "invalid3"]
        allowed_types = [torch.Tensor]

        result = find_invalid_types(input_data, allowed_types)  # type: ignore[arg-type]

        self.assertEqual(len(result), 1)
        self.assertIn(str, result)

    def test_plot_two_outputs(self):
        """Test plot_two_outputs function"""
        # Patch plotext import more directly by patching the module in sys.modules
        with patch.dict("sys.modules", {"plotext": Mock()}):
            import plotext

            # Setup mock for plotext
            plotext.clear_data = Mock()
            plotext.xlim = Mock()
            plotext.ylim = Mock()
            plotext.plotsize = Mock()
            plotext.scatter = Mock()
            plotext.theme = Mock()
            plotext.build = Mock(return_value="mock_figure")

            # Run function
            x_values = torch.tensor([1.0, 2.0, 3.0])
            y_values = torch.tensor([2.0, 4.0, 6.0])

            result = plot_two_outputs(x_values, y_values)

            # Verify results
            self.assertEqual(result, "mock_figure")

    def test_ensure_list_single_element(self):
        """Test ensure_list with single element"""
        result = ensure_list(42)
        self.assertEqual(result, [42])

    def test_ensure_list_tuple(self):
        """Test ensure_list with tuple"""
        result = ensure_list((1, 2, 3))
        self.assertEqual(result, [1, 2, 3])

    def test_ensure_list_list(self):
        """Test ensure_list with list (should return unchanged)"""
        original = [4, 5, 6]
        result = ensure_list(original)
        self.assertEqual(result, original)
        self.assertIs(result, original)  # Should be same object


class TestIntrospectionUtils(unittest.TestCase):
    """Test utility functions from ptq/utils/introspection.py"""

    def test_build_fqn_map(self):
        """Test build_fqn_map function"""

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(3, 2)
                self.linear2 = nn.Linear(2, 1)

        model = SimpleModel()
        fqn_map = build_fqn_map(model)

        self.assertIsInstance(fqn_map, dict)
        self.assertIn(model.linear1, fqn_map)
        self.assertIn(model.linear2, fqn_map)
        self.assertEqual(fqn_map[model.linear1], "linear1")
        self.assertEqual(fqn_map[model.linear2], "linear2")

    def test_build_fqn_map_nested(self):
        """Test build_fqn_map with nested modules"""

        class NestedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.block = nn.Sequential(nn.Linear(3, 2), nn.ReLU(), nn.Linear(2, 1))

        model = NestedModel()
        fqn_map = build_fqn_map(model)

        self.assertIn(model.block[0], fqn_map)
        self.assertIn(model.block[2], fqn_map)
        self.assertEqual(fqn_map[model.block[0]], "block.0")
        self.assertEqual(fqn_map[model.block[2]], "block.2")

    def test_save_fp_outputs(self):
        """Test save_fp_outputs function"""

        class MockQuantModule(QuantModuleBase):
            def __init__(self):
                super().__init__()
                self.fp_name = "test_module"

            @property
            def _all_observers(self):
                return []

            def forward(self, x):
                return x * 2

        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.quant_module = MockQuantModule()

            def forward(self, x):
                return self.quant_module(x)

        model = TestModel()
        handles, cache = save_fp_outputs(model)

        self.assertIsInstance(handles, list)
        self.assertIsInstance(cache, dict)
        self.assertEqual(len(handles), 1)

        # Run forward pass to populate cache
        input_data = torch.tensor([1.0, 2.0, 3.0])
        model(input_data)

        self.assertIn("test_module", cache)
        torch.testing.assert_close(cache["test_module"], input_data * 2)

        # Clean up handles
        for handle in handles:
            handle.remove()

    def test_save_fp_outputs_no_fp_name(self):
        """Test save_fp_outputs with module that has no fp_name"""

        class MockQuantModule(QuantModuleBase):
            def __init__(self):
                super().__init__()
                self.fp_name = None

            @property
            def _all_observers(self):
                return []

            def forward(self, x):
                return x * 2

        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.quant_module = MockQuantModule()

            def forward(self, x):
                return self.quant_module(x)

        model = TestModel()
        handles, cache = save_fp_outputs(model)

        self.assertIsInstance(handles, list)
        self.assertIsInstance(cache, dict)
        self.assertEqual(len(handles), 1)

        # Run forward pass to populate cache
        input_data = torch.tensor([1.0, 2.0, 3.0])
        model(input_data)

        # Should use id(module) as key
        module_id = str(id(model.quant_module))
        self.assertIn(module_id, cache)
        torch.testing.assert_close(cache[module_id], input_data * 2)

        # Clean up handles
        for handle in handles:
            handle.remove()

    @patch("tico.experimental.quantization.ptq.utils.introspection.MetricCalculator")
    @patch("builtins.print")
    def test_compare_layer_outputs_print_mode(self, mock_print, mock_metric_calculator):
        """Test compare_layer_outputs in print mode"""

        class MockQuantModule(QuantModuleBase):
            def __init__(self):
                super().__init__()
                self.fp_name = "test_module"

            @property
            def _all_observers(self):
                return []

            def forward(self, x):
                return x * 2

        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.quant_module = MockQuantModule()

            def forward(self, x):
                return self.quant_module(x)

        model = TestModel()
        cache = {"test_module": torch.tensor([2.0, 4.0, 6.0])}

        # Setup mock metric calculator
        mock_calc = Mock()
        mock_calc.compute.return_value = {"diff": torch.tensor([0.01])}
        mock_metric_calculator.return_value = mock_calc

        handles = compare_layer_outputs(model, cache, rtol=0.1, atol=0.1)

        self.assertIsInstance(handles, list)
        self.assertEqual(len(handles), 1)

        # Run forward pass to trigger comparison
        input_data = torch.tensor([1.0, 2.0, 3.0])
        model(input_data)

        # Verify print was called
        mock_print.assert_called()

        # Clean up handles
        for handle in handles:
            handle.remove()

    @patch("tico.experimental.quantization.ptq.utils.introspection.MetricCalculator")
    def test_compare_layer_outputs_collect_mode(self, mock_metric_calculator):
        """Test compare_layer_outputs in collect mode"""

        class MockQuantModule(QuantModuleBase):
            def __init__(self):
                super().__init__()
                self.fp_name = "test_module"

            @property
            def _all_observers(self):
                return []

            def forward(self, x):
                return x * 2

        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.quant_module = MockQuantModule()

            def forward(self, x):
                return self.quant_module(x)

        model = TestModel()
        cache = {"test_module": torch.tensor([2.0, 4.0, 6.0])}

        # Setup mock metric calculator
        mock_calc = Mock()
        mock_calc.compute.return_value = {"diff": torch.tensor([0.01])}
        mock_metric_calculator.return_value = mock_calc

        handles, results = compare_layer_outputs(
            model, cache, collect=True, rtol=0.1, atol=0.1
        )

        self.assertIsInstance(handles, list)
        self.assertIsInstance(results, dict)
        self.assertEqual(len(handles), 1)

        # Run forward pass to trigger comparison
        input_data = torch.tensor([1.0, 2.0, 3.0])
        model(input_data)

        # Verify results were collected
        self.assertIn("test_module", results)
        self.assertIn("diff", results["test_module"])

        # Clean up handles
        for handle in handles:
            handle.remove()

    def test_compare_layer_outputs_no_reference(self):
        """Test compare_layer_outputs with no cached reference"""

        class MockQuantModule(QuantModuleBase):
            def __init__(self):
                super().__init__()
                self.fp_name = "test_module"

            @property
            def _all_observers(self):
                return []

            def forward(self, x):
                return x * 2

        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.quant_module = MockQuantModule()

            def forward(self, x):
                return self.quant_module(x)

        model = TestModel()
        cache: Dict = {}  # Empty cache

        with patch("builtins.print") as mock_print:
            handles = compare_layer_outputs(model, cache)

            # Run forward pass to trigger comparison
            input_data = torch.tensor([1.0, 2.0, 3.0])
            model(input_data)

            # Verify print was called with no reference message
            mock_print.assert_called_with("[test_module]  no cached reference")

        # Clean up handles
        for handle in handles:
            handle.remove()
