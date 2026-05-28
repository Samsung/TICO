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

from typing import Any

import tico.utils
import tico.utils.model
import torch
from tico.serialize.circle_mapping import (
    extract_circle_dtype,
    extract_circle_shape,
    str_to_circle_dtype,
    to_circle_dtype,
    to_circle_shape,
)
from tico.utils.signature import ModelInputSpec


def infer_with_circle_interpreter(
    circle_path: str,
    forward_args: tuple,
    forward_kwargs: dict,
) -> Any:
    """
    Run inference on a .circle model using the 'circle-interpreter' engine.

    Parameters
    -----------
    circle_path
        Path to the .circle file to execute.
    forward_args
        Tuple of arguments for the model's forward function.
    forward_kwargs
        Dictionary of keyword arguments for the model's forward function.

    Returns
    --------
    Any
        The output produced by the 'circle-interpreter'
    """
    circle_model = tico.utils.model.CircleModel.load(circle_path)
    ispec = ModelInputSpec.load(circle_path)
    inputs = ispec.bind(forward_args, forward_kwargs, check=True)
    circle_result = circle_model(*inputs)

    if not isinstance(circle_result, list):
        circle_result = [circle_result]

    return circle_result


def infer_with_onert(
    circle_path: str,
    forward_args: tuple,
    forward_kwargs: dict,
) -> Any:
    """
    Run inference on a .circle model using the 'onert' package.

    Parameters
    -----------
    circle_path
        Path to the .circle file to execute.
    forward_args
        Tuple of arguments for the model's forward function.
    forward_kwargs
        Dictionary of keyword arguments for the model's forward function.

    Returns
    --------
    Any
        The output produced by the 'onert' runtime.
    """
    try:
        from onert import infer
    except ImportError:
        raise RuntimeError("The 'onert' package is required to run this function.")

    ispec = ModelInputSpec.load(circle_path)
    inputs = ispec.bind(forward_args, forward_kwargs, check=True)

    session_float = infer.session(circle_path)

    # Handle dynamic shapes: onert cannot execute models with unspecified dimensions
    # Check if any input has dynamic dimensions (indicated by -1)
    input_tensorinfos = session_float.get_inputs_tensorinfo()
    has_dynamic_shapes = any(-1 in info.dims for info in input_tensorinfos)

    if has_dynamic_shapes:
        # Set concrete input shapes based on the actual input data
        from onert.native.libnnfw_api_pybind import tensorinfo

        for idx, (info, input_data) in enumerate(zip(input_tensorinfos, inputs)):
            if -1 in info.dims:
                # Create new tensorinfo with concrete shape from input data
                new_info = tensorinfo()
                new_info.rank = len(input_data.shape)
                new_info.dims = list(input_data.shape)

                assert input_data.dtype in [torch.float32, torch.float]
                new_info.dtype = "float32"

                try:
                    session_float.session.set_input_tensorinfo(idx, new_info)
                except Exception as e:
                    # If setting tensorinfo fails, try to continue anyway
                    # Some versions of onert might handle this differently
                    import warnings

                    warnings.warn(
                        f"Failed to set input tensorinfo for input {idx}: {e}. "
                        f"Attempting inference anyway."
                    )

    output = session_float.infer(inputs)

    return output
