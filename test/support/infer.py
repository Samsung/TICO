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

from typing import Any, List

import numpy as np

import tico.utils.model
from tico.utils.signature import ModelInputSpec

from test.support.runtime import Runtime


def infer_with_runtime(
    circle_path: str,
    forward_args: tuple,
    forward_kwargs: dict,
    runtime: Runtime,
) -> List[np.ndarray]:
    """
    Run inference on a .circle file with the selected runtime.

    The model is loaded from the serialized file, the user arguments are bound
    through ``ModelInputSpec`` exactly like ``CircleModel.__call__``, and the
    outputs are returned as a list of NumPy arrays in the serialized order.

    Parameters
    -----------
    circle_path
        Path to the .circle file to execute.
    forward_args
        Tuple of arguments for the model's forward function.
    forward_kwargs
        Dictionary of keyword arguments for the model's forward function.
    runtime
        ``"reference"`` (built-in), ``"circle-interpreter"`` (optional ONE
        luci-interpreter), or ``"onert"`` (optional onert package).
    """
    circle_model = tico.utils.model.CircleModel.load(circle_path, runtime=runtime)
    ispec = ModelInputSpec.load(circle_path)
    inputs = ispec.bind(forward_args, forward_kwargs, check=True)
    circle_result: Any = circle_model(*inputs)

    if not isinstance(circle_result, list):
        circle_result = [circle_result]

    return circle_result


def infer_with_reference(
    circle_path: str,
    forward_args: tuple,
    forward_kwargs: dict,
) -> List[np.ndarray]:
    """Run inference with the built-in reference runtime."""
    return infer_with_runtime(circle_path, forward_args, forward_kwargs, "reference")


def infer_with_circle_interpreter(
    circle_path: str,
    forward_args: tuple,
    forward_kwargs: dict,
) -> List[np.ndarray]:
    """Run inference with the optional ONE 'circle-interpreter' engine."""
    return infer_with_runtime(
        circle_path, forward_args, forward_kwargs, "circle-interpreter"
    )


def infer_with_onert(
    circle_path: str,
    forward_args: tuple,
    forward_kwargs: dict,
) -> List[np.ndarray]:
    """Run inference with the optional 'onert' package."""
    return infer_with_runtime(circle_path, forward_args, forward_kwargs, "onert")
