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

import tico.utils.diff_graph as diff_graph
import torch
import torch.nn as nn
from torch.export import export


class TestDiffGraph(unittest.TestCase):
    """Test cases for tico.utils.diff_graph module"""

    def setUp(self):
        """Set up test fixtures"""
        # Create first test model and graph
        class TestModule1(nn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = nn.Linear(10, 5)
                self.relu = nn.ReLU()
                self.l2 = nn.Linear(5, 1)

            def forward(self, x):
                x = self.l1(x)
                x = self.relu(x)
                x = self.l2(x)
                return x

        # Create a second, slightly different test model and graph
        class TestModule2(nn.Module):  # Different class name for clarity
            def __init__(self):
                super().__init__()
                self.l1 = nn.Linear(10, 5)  # Same layer
                self.relu = nn.ReLU()
                # Add an extra layer to make the graph different
                self.extra_linear = nn.Linear(5, 5)
                self.l2 = nn.Linear(5, 1)

            def forward(self, x):
                x = self.l1(x)
                x = self.relu(x)
                x = self.extra_linear(x)  # Extra operation
                x = self.l2(x)
                return x

        self.sample_model1 = TestModule1()
        self.sample_model2 = TestModule2()
        self.sample_input = torch.randn(2, 10)

        # Create exported programs for testing
        self.exported_program1 = export(self.sample_model1, (self.sample_input,))
        self.exported_program2 = export(self.sample_model2, (self.sample_input,))

        # Create test graphs from different programs
        self.graph1 = self.exported_program1.graph_module.graph
        self.graph2 = self.exported_program2.graph_module.graph

    def test_strdiff_with_identical_strings(self):
        """Test strdiff function with identical strings"""
        str1 = "line1\nline2\nline3\n"
        str2 = "line1\nline2\nline3\n"

        result = diff_graph.strdiff(str1, str2)
        self.assertEqual(result, "")

    def test_strdiff_with_different_strings(self):
        """Test strdiff function with different strings"""
        str1 = "line1\nline2\nline3\n"
        str2 = "line1\nmodified_line2\nline3\nline4\n"

        result = diff_graph.strdiff(str1, str2)

        # Should show the differences
        # ndiff output includes a space after the symbol
        self.assertIn("- line2", result)
        self.assertIn("+ modified_line2", result)
        self.assertIn("+ line4", result)

    def test_strdiff_with_empty_strings(self):
        """Test strdiff function with empty strings"""
        result = diff_graph.strdiff("", "")
        self.assertEqual(result, "")

        result = diff_graph.strdiff("line1\n", "")
        # ndiff output includes a space after the symbol
        self.assertIn("- line1", result)

        result = diff_graph.strdiff("", "line1\n")
        # ndiff output includes a space after the symbol
        self.assertIn("+ line1", result)

    def test_strdiff_type_assertion(self):
        """Test strdiff function type assertions"""
        with self.assertRaises(AssertionError):
            diff_graph.strdiff(123, "string")  # type: ignore[arg-type]

        with self.assertRaises(AssertionError):
            diff_graph.strdiff("string", 123)  # type: ignore[arg-type]

    def test_disable_when_decorator_with_true_predicate(self):
        """Test disable_when decorator with True predicate"""

        @diff_graph.disable_when(True)
        def test_function():
            return "should_not_be_called"

        # Function should be disabled and return None
        result = test_function()
        self.assertIsNone(result)

    def test_disable_when_decorator_with_false_predicate(self):
        """Test disable_when decorator with False predicate"""

        @diff_graph.disable_when(False)
        def test_function():
            return "should_be_called"

        # Function should be enabled and return the expected value
        result = test_function()
        self.assertEqual(result, "should_be_called")

    def test_get_const_size_with_simple_model(self):
        """Test get_const_size function with simple model"""
        const_size = diff_graph.get_const_size(self.exported_program1)

        # Should return a non-negative integer
        self.assertIsInstance(const_size, int)
        self.assertGreaterEqual(const_size, 0)

    def test_get_const_size_with_model_having_state_dict(self):
        """Test get_const_size function with model having state_dict"""
        # Create a model with more complex state_dict
        complex_model = nn.Sequential(
            nn.Linear(10, 20), nn.BatchNorm1d(20), nn.ReLU(), nn.Linear(20, 5)
        )

        sample_input = torch.randn(4, 10)
        exported_program = export(complex_model, (sample_input,))

        const_size = diff_graph.get_const_size(exported_program)

        # Should return a positive integer for models with parameters
        self.assertIsInstance(const_size, int)
        self.assertGreater(const_size, 0)

    def test_get_const_size_with_scalar_tensors(self):
        """Test get_const_size function handling scalar tensors"""
        # Create a model with scalar constants
        model_with_scalar = nn.Sequential(nn.Linear(10, 5), nn.ReLU())

        # Add a scalar constant to the model
        model_with_scalar.register_buffer("scalar_const", torch.tensor(1.0))

        sample_input = torch.randn(2, 10)
        exported_program = export(model_with_scalar, (sample_input,))

        const_size = diff_graph.get_const_size(exported_program)

        # Should handle scalar tensors correctly
        self.assertIsInstance(const_size, int)
        self.assertGreaterEqual(const_size, 0)
