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

from typing import Any, List, Tuple

import numpy as np
import torch

from circle_schema import circle
from numpy.typing import DTypeLike

from tico.utils import logging
from tico.utils.model import CircleModel


def quantize(
    data: np.ndarray, scale: float, zero_point: int, dtype: DTypeLike
) -> np.ndarray:
    """
    Quantize the given data using the specified scale, zero point, and data type.
    This function takes input data and applies quantization using the formula:
        round(data / scale) + zero_point
    The result is clamped to the range of the specified data type.
    """
    logger = logging.getLogger(__name__)
    dtype = np.dtype(dtype)
    assert dtype == np.uint8 or dtype == np.int16, f"Invalid dtype: {dtype}"
    if dtype == np.int16:
        assert zero_point == 0

    # Convert input to Numpy array if necessary
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    # Perfrom quantization
    if not scale:
        logger.warning("WARNING: scale value is 0. 1e-7 will be used instead.")
        scale = 1e-7
    rescaled = np.round(data / scale) + zero_point
    # Clamp the values
    clipped = np.clip(rescaled, np.iinfo(dtype).min, np.iinfo(dtype).max)
    # Convert to the specified dtype
    return clipped.astype(dtype)


def dequantize(
    data: np.ndarray, scale: float, zero_point: int, dtype: DTypeLike
) -> np.ndarray:
    """
    Dequantize the given quantized data using the specified scale and zero point.
    This function reverses the quantization process by applying the formula:
        (quantized_value - zero_point) * scale
    """
    dtype = np.dtype(dtype)
    assert dtype == np.uint8 or dtype == np.int16, f"Invalid dtype: {dtype}"
    if dtype == np.int16:
        assert zero_point == 0

    # Convert input to Numpy array if necessary
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    # Perform dequantization
    ret = (data.astype(np.float32) - zero_point) * scale
    # np.float32 * np.int64 = np.float64
    return ret.astype(np.float32)


def get_graph_input_output(
    circle_model: CircleModel,
) -> Tuple[List[circle.Tensor.Tensor], List[circle.Tensor.Tensor]]:
    """
    Retrieve the inputs and the outputs from the circle model, and return them
     as two lists.
    """
    circle_buf: bytes = circle_model.circle_binary
    circle_fb: circle.Model.Model = circle.Model.Model.GetRootAs(circle_buf, 0)
    assert circle_fb.SubgraphsLength() == 1, "Only support single graph."
    circle_graph = circle_fb.Subgraphs(0)
    circle_inputs: List[circle.Tensor.Tensor] = [
        circle_graph.Tensors(circle_graph.Inputs(i))
        for i in range(circle_graph.InputsLength())
    ]
    circle_outputs: List[circle.Tensor.Tensor] = [
        circle_graph.Tensors(circle_graph.Outputs(o))
        for o in range(circle_graph.OutputsLength())
    ]

    return circle_inputs, circle_outputs


def find_invalid_types(
    input: List[torch.Tensor] | List[np.ndarray], allowed_types: List
) -> List:
    """
    Indentifies the types of items in a list that are not allowed and removes duplicates.

    Parameters
    -----------
        input
            List of itmes to check.
        allowed_types
            List of allowed types (e.g. [int, str])
    Returns
    --------
        A list of unique types that are not allowed in the input list.
    """
    # Use set comprehension for uniqueness
    invalid_types = {
        type(item) for item in input if not isinstance(item, tuple(allowed_types))
    }
    return list(invalid_types)


def plot_two_outputs(x_values: torch.Tensor, y_values: torch.Tensor):
    """
    Plot two values on a 2D graph using plotext.

    Returns
    --------
        A figure built from plotext.

    Example
    --------
        >>> x_values = torch.tensor([1, 2, 3, 4, 5])
        >>> y_values = torch.tensor([10, 20, 30, 40, 50])
        >>> fig = plot_two_outputs(x_values, y_values)
        >>> print(fig)
    """
    x_np = x_values.numpy().reshape(-1)
    y_np = y_values.numpy().reshape(-1)
    min_value = min([x_np.min(), y_np.min()])
    max_value = max([x_np.max(), y_np.max()])

    interval = max_value - min_value
    interval = 1.0 if interval == 0.0 else interval  # Avoid zero interval

    # Enlarge axis
    axis_min = min_value - interval * 0.05
    axis_max = max_value + interval * 0.05

    import plotext as plt

    plt.clear_data()
    plt.xlim(axis_min, axis_max)
    plt.ylim(axis_min, axis_max)
    plt.plotsize(width=50, height=25)
    plt.scatter(x_np, y_np, marker="dot")
    plt.theme("clear")

    return plt.build()


def ensure_list(inputs: Any | Tuple[Any] | List[Any]) -> List[Any]:
    """
    Ensures that the given inputs is converted into a list.

    - If the input is a single element, it wraps it into a list.
    - If the input is a tuple, it converts the tuple to a list.
    - If the input is already a list, it returns the input unchanged.

    Example
    --------
        >>> ensure_list(42)
        >>> [42]
        >>> ensure_list((1, 2, 3))
        >>> [1, 2, 3]
        >>> ensure_list([4, 5, 6])
        >>> [4, 5, 6]
    """
    if isinstance(inputs, list):
        return inputs
    elif isinstance(inputs, tuple):
        return list(inputs)
    else:
        return [inputs]
