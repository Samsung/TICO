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

from typing import Any, Sequence

import numpy as np
import torch
from circle_schema import circle

from tico.interpreter.interpreter import Interpreter
from tico.serialize.circle_mapping import np_dtype_from_circle_dtype, to_circle_dtype
from tico.utils.installed_packages import is_dynamic_cache_available


def flatten_and_convert(inputs: Sequence) -> tuple:
    result = []  # type: ignore[var-annotated]
    for item in inputs:
        if item is None:
            continue

        # 1. recursion on list and tuple
        if isinstance(item, (list, tuple)):
            result.extend(flatten_and_convert(item))
            continue

        # 2. handle DynamicCache
        if is_dynamic_cache_available():
            from transformers.cache_utils import DynamicCache

            if isinstance(item, DynamicCache):
                # NOTE The tensor order is: key_in → key_out → value_in → value_out
                #
                # Refer to https://github.com/huggingface/transformers/blob/3457e8e73e4f5532cc69059682b1ba4484d7e7e8/src/transformers/cache_utils.py#L557
                # ```py
                # self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                # self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
                # ```
                result.extend(item.key_cache)
                result.extend(item.value_cache)
                continue

        # 3. Convert to tensors
        result.append(item if isinstance(item, torch.Tensor) else torch.tensor(item))

    return tuple(result)


def infer(circle_binary: bytes, *args: Any, **kwargs: Any) -> Any:
    # When converting a model, it is assumed that the order of keyword arguments is maintained.
    raw_inputs = args + tuple(kwargs.values())
    user_inputs = flatten_and_convert(raw_inputs)

    # Get input spec from circle binary.
    model = circle.Model.Model.GetRootAsModel(circle_binary, 0)
    # assert model.SubgraphsLength() == 1
    graph = model.Subgraphs(0)
    model_input_tensors = [
        graph.Tensors(graph.Inputs(o)) for o in range(graph.InputsLength())
    ]
    model_input_shapes_np = [t.ShapeAsNumpy() for t in model_input_tensors]
    model_input_types_cm = [t.Type() for t in model_input_tensors]

    # Check if given inputs' dtype and shape from users match the inputs' from model binary.
    if len(model_input_shapes_np) != len(user_inputs):
        raise RuntimeError(
            f"Mismatch input length: input({len(user_inputs)}) != circle model({len(model_input_shapes_np)})"
        )
    for input_idx, user_input in enumerate(user_inputs):
        # Shape
        if list(user_input.shape) != list(model_input_shapes_np[input_idx]):
            raise RuntimeError(
                f"Mismatch input {input_idx} shape : input({user_input.shape}) != circle model({model_input_shapes_np[input_idx]})"
            )
        # Data type
        user_input_type_cm = to_circle_dtype(user_input.dtype)
        if user_input_type_cm != model_input_types_cm[input_idx]:
            raise RuntimeError(
                f"Mismatch input {input_idx} data type : input({user_input_type_cm}) != circle model({model_input_types_cm[input_idx]})"
            )

    # Initialize interpreter
    intp = Interpreter(circle_binary)

    # Set input
    for input_idx, user_input in enumerate(user_inputs):
        intp.writeInputTensor(input_idx, user_input)

    # Interpret
    intp.interpret()

    # Retrieve outputs' dtype and shape from circle model
    model_output_tensors = [
        graph.Tensors(graph.Outputs(o)) for o in range(graph.OutputsLength())
    ]
    model_output_shapes_np = [t.ShapeAsNumpy() for t in model_output_tensors]
    model_output_types_cm = [t.Type() for t in model_output_tensors]

    output = []
    # Get output
    for output_idx in range(len(model_output_tensors)):
        result: np.ndarray = np.empty(
            model_output_shapes_np[output_idx],
            dtype=np_dtype_from_circle_dtype(model_output_types_cm[output_idx]),
        )
        intp.readOutputTensor(output_idx, result)
        output.append(result)

    if len(output) == 1:
        return output[0]
    else:
        return output
