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

import os
from typing import Any, Tuple

import test.utils.helper as helper
import tico.utils
import tico.utils.model
import torch
from tico.interpreter.infer import flatten_and_convert


def infer_with_circle_interpreter(circle_path: str, example_inputs: Tuple) -> Any:
    """
    Run inference on a .circle model using the 'circle-interpreter' engine.

    Parameters
    -----------
    circle_path
        Path to the .circle file to execute.
    example_inputs
        Tuple of input tensors or values to feed into the model.

    Returns
    --------
    Any
        The output produced by the 'circle-interpreter'
    """
    circle_model = tico.utils.model.CircleModel.load(circle_path)
    _args, _kwargs = helper.get_args_kwargs(example_inputs)
    circle_result = circle_model(*_args, **_kwargs)

    if not isinstance(circle_result, list):
        circle_result = [circle_result]

    return circle_result


def infer_with_onert(circle_path: str, example_inputs: Tuple) -> Any:
    """
    Run inference on a .circle model using the 'onert' package.

    Parameters
    -----------
    circle_path
        Path to the .circle file to execute.
    example_inputs
        Tuple of input tensors or values to feed into the model.

    Returns
    --------
    Any
        The output produced by the 'onert' runtime.
    """
    try:
        from onert import infer
    except ImportError:
        raise RuntimeError("The 'onert' package is required to run this funciton.")

    _args, _kwargs = helper.get_args_kwargs(example_inputs)
    inputs = _args + tuple(_kwargs.values())
    inputs = flatten_and_convert(inputs)

    session_float = infer.session(circle_path)
    output = session_float.infer(inputs)

    return output
