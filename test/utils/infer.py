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
    output = session_float.infer(inputs)

    return output
