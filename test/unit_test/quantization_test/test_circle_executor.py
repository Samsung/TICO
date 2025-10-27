import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.export

from tico.quantization.evaluation.executor.circle_executor import CircleExecutor
from tico.serialize.circle_serializer import build_circle
from tico.utils.model import CircleModel


class AddModule(torch.nn.Module):
    def forward(self, x, y):
        return x + y


class TestCircleExecutor(unittest.TestCase):
    def _create_dummy_circle_model(self):
        mod = AddModule()
        ep = torch.export.export(mod, (torch.randn(1, 3), torch.randn(1, 3)))
        circle_bytes = build_circle(ep)
        return CircleModel(circle_bytes)

    @patch("pathlib.Path.is_file")
    def test_init_raises_runtime_error_if_compiler_not_found(self, mock_is_file):
        mock_is_file.return_value = False
        with self.assertRaises(RuntimeError):
            CircleExecutor()

    @patch("pathlib.Path.is_file")
    @patch("tico.quantization.evaluation.executor.circle_executor.run_bash_cmd")
    @patch("tico.quantization.evaluation.executor.circle_executor.CircleModel.load")
    def test_compile_and_run_inference(self, mock_load, mock_run_bash, mock_is_file):
        mock_is_file.return_value = True

        # Create a real CircleModel
        circle_model = self._create_dummy_circle_model()

        # Mock the load to return a model that returns a predictable value
        mock_loaded_model = MagicMock(spec=CircleModel)
        mock_loaded_model.return_value = [torch.tensor([1, 2, 3])]
        mock_load.return_value = mock_loaded_model

        executor = CircleExecutor()
        executor.compile(circle_model)

        # Check that the bash command was called
        self.assertTrue(mock_run_bash.called)

        # Run inference
        input_data = [torch.tensor([4, 5, 6])]
        result = executor.run_inference(input_data)

        # Check that the loaded model was called with the input data
        mock_loaded_model.assert_called_with(input_data[0])

        # Check the result
        self.assertEqual(len(result), 1)
        self.assertTrue(torch.equal(result[0], torch.tensor([1, 2, 3])))  # type: ignore[arg-type]

    @patch("pathlib.Path.is_file")
    def test_run_inference_before_compile_raises_error(self, mock_is_file):
        mock_is_file.return_value = True
        executor = CircleExecutor()
        with self.assertRaises(RuntimeError):
            executor.run_inference([])

    @patch("pathlib.Path.is_file")
    @patch("tico.quantization.evaluation.executor.circle_executor.CircleModel.load")
    def test_run_inference_with_single_tensor_output(self, mock_load, mock_is_file):
        mock_is_file.return_value = True

        # Create a real CircleModel
        circle_model = self._create_dummy_circle_model()

        # Mock the load to return a model that returns a single tensor
        mock_loaded_model = MagicMock(spec=CircleModel)
        mock_loaded_model.return_value = torch.tensor([1, 2, 3])
        mock_load.return_value = mock_loaded_model

        executor = CircleExecutor()
        executor.compile(circle_model)

        # Run inference
        input_data = [torch.tensor([4, 5, 6])]
        result = executor.run_inference(input_data)

        # Check the result
        self.assertEqual(len(result), 1)
        self.assertTrue(torch.equal(result[0], torch.tensor([1, 2, 3])))  # type: ignore[arg-type]
