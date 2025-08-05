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

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from circle_schema import circle
from torch.utils import _pytree as pytree

from tico.experimental.quantization.evaluation.backend import BACKEND
from tico.experimental.quantization.evaluation.executor.backend_executor import (
    BackendExecutor,
)
from tico.experimental.quantization.evaluation.executor.circle_executor import (
    CircleExecutor,
)
from tico.experimental.quantization.evaluation.executor.triv24_executor import (
    Triv24Executor,
)
from tico.experimental.quantization.evaluation.metric import MetricCalculator
from tico.experimental.quantization.evaluation.utils import (
    ensure_list,
    find_invalid_types,
    get_graph_input_output,
    plot_two_outputs,
)
from tico.utils.model import CircleModel

InputDataType = Union[
    None,
    np.ndarray,
    torch.Tensor,
    List[np.ndarray],
    List[torch.Tensor],
    Tuple[np.ndarray],
    Tuple[torch.Tensor],
]

BACKEND_TO_EXECUTOR: Dict[BACKEND, type[BackendExecutor]] = {
    BACKEND.CIRCLE: CircleExecutor,
    BACKEND.TRIV24: Triv24Executor,
}


def _validate_input_data(
    input_data: InputDataType, circle_inputs: List[circle.Tensor.Tensor]
) -> None:
    """
    Validate whether the given input data matches the shape of a list of circle input tensors.

    Parameters
    -----------
        input_data
            The input data to be checked.
        circle_inputs
            A list of circle.Tensor.Tensor to validate against.
    """
    if input_data is None:
        return
    assert isinstance(input_data, list)

    if len(input_data) != len(circle_inputs):
        raise RuntimeError(
            f"Mismatch between the length of input data and circle model: input_data({len(input_data)}) != circle_model({len(circle_inputs)})"
        )
    if invalid_type := find_invalid_types(input_data, [torch.Tensor, np.ndarray]):
        raise RuntimeError(
            f"Only support tuple of torch.Tensor or numpy.ndarray for input data. Invalid types: {invalid_type}"
        )


def _convert_to_torch_tensor(input_data: InputDataType) -> List[torch.Tensor]:
    """
    Convert the input data into a list of torch.Tensor.

    This function performs the following tasks:
        - Checks if `input_data` is a numpy array and converts it to a torch.Tensor.
        - If it is already a torch.Tensor, it is returned as is.

    Parameters
    -----------
        input_data
            The input data to be converted.
    """
    assert isinstance(input_data, list)

    # Cast to torch.Tensor to make the logic simpler
    for i, data in enumerate(input_data):
        if isinstance(data, np.ndarray):
            input_data[i] = torch.Tensor(data)  # type: ignore[call-overload]

    assert all(isinstance(input, torch.Tensor) for input in input_data)

    return input_data  # type: ignore[return-value]


def evaluate(
    torch_module: torch.nn.Module,
    circle_model: CircleModel,
    backend: BACKEND,
    input_data: InputDataType = None,
    *,
    mode="plot",
    custom_metrics: Dict[str, Callable] = dict(),
) -> Optional[Dict[str, Any]]:
    """
    Evaluate and compare a Pytorch module with a quantized circle model on a specific
     backend using a given metrics.

    It compiles the circle model using specified backend, run both the Pytorch module
     and the compiled model on the same input, and computes the comparison score based
    on the provided metrics.

    Parameters
    -----------
        torch_module
            A callable Pytorch module.
        circle_module
            The Circle model to be compiled and evaluated.
        backend
            The backend used to compile and execute the Circle model.
        input_data
            The input data to be used for evaluation. Should be compatible with both models.
            If None, random data will be generated.
        mode
            The mode of operation. Options are:
            - "plot": Plot the results (default)
            - "return": Return the results.
        custom_metrics
            A dictionary of metric names and corresponding callable functions for comparison.
            Example: {'mse': mean_squared_error, 'cosine_similarity': cosine_similarity_fn}
    # TODO Support options for backend optimizations.

    Returns
    --------
    dict or None
        If `mode` is "plot", plot the results and returns None.
        If `mode` is "return", returns a dictionary containing:
            - "peir": The computed PEIR value.
            - "<metric_name>": The computed value of the additional metric (if provided).
    """
    # Check if arguments are allowed types.
    if not isinstance(torch_module, torch.nn.Module):
        raise RuntimeError(
            f"Only support torch.nn.Module. Given module type: {type(torch_module)}"
        )
    if not isinstance(circle_model, CircleModel):
        raise RuntimeError(
            f"Only support CircleModel. Given module type: {type(circle_model)}"
        )
    if not isinstance(backend, BACKEND):
        raise RuntimeError(
            "Invalid backend. Please use tico.quantization.evaluate.BACKEND enum class"
        )
    # Make it a list for simpler logic.
    if input_data is not None:
        input_data = ensure_list(input_data)

    circle_inputs, _ = get_graph_input_output(circle_model)
    _validate_input_data(input_data, circle_inputs)

    if input_data:
        input_data = _convert_to_torch_tensor(input_data)
    else:
        # Make random inputs
        circle_input_shapes_np = [t.ShapeAsNumpy() for t in circle_inputs]
        input_data = [torch.randn(*shape) for shape in circle_input_shapes_np]

    assert isinstance(input_data, list)
    assert all(isinstance(data, torch.Tensor) for data in input_data)

    # Compile circle model and run inference.
    executor: BackendExecutor = BACKEND_TO_EXECUTOR[backend]()
    executor.compile(circle_model)
    circle_output = executor.run_inference(input_data)
    circle_output = [
        torch.from_numpy(out) for out in circle_output if isinstance(out, np.ndarray)
    ]

    # Run torch model.
    with torch.no_grad():
        torch_output = torch_module(*input_data)
    torch_output, _ = pytree.tree_flatten(torch_output)
    if isinstance(torch_output, torch.Tensor):
        torch_output = (torch_output,)
    if len(torch_output) != len(circle_output):
        raise RuntimeError(
            f"Mismatch between the length of torch output and circle output: torch_output({len(torch_output)}) != circle_output({len(circle_output)})"
        )

    # Computes the comparison score based on the provided metrics.
    metric_calculator = MetricCalculator(custom_metrics)
    results: Dict[str, Any] = metric_calculator.compute(torch_output, circle_output)

    if mode == "return":
        return results
    elif mode == "plot":
        for idx, (t_out, c_out) in enumerate(zip(torch_output, circle_output)):
            print(f"OUTPUT [{idx}]")
            fig = plot_two_outputs(t_out, c_out)
            print(fig)
            for metric_name, values in results.items():
                print(f"{metric_name}: {values[idx]}")
    else:
        raise RuntimeError("Invalid mode.")

    return None
