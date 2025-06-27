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

import numpy as np
import torch
from circle_schema import circle

from transformers.cache_utils import DynamicCache
from tico.interpreter.interpreter import Interpreter
from tico.serialize.circle_mapping import np_dtype_from_circle_dtype, to_circle_dtype


def preprocess_inputs(inputs: Any):
    """
    Preprocess user inputs for circle inference.

    1. None inputs are ignored.
    2. A list/tuple input is flatten when a torch module is exported.
      e.g. inputs = (torch.Tensor, [2,3,4]) -> inputs = (torch.Tensor, 2, 3, 4)
    """
    l = []
    for value in inputs:
        if value == None:
            continue
        if isinstance(value, (tuple, list)):
            for val in value:
                l.append(val)
        else:
            l.append(value)
    # Check if it is a list of a list.
    if any(isinstance(item, (tuple, list)) for item in l):
        l = preprocess_inputs(l)
    return tuple(l)

def flatten_and_convert(inputs: Any) -> tuple:
    """
    Recursively flatten nested lists/tuples and convert elements:
      1. Ignore None values
      2. Flatten nested lists/tuples
      3. Convert DynamicCache to expanded tensor tuple
      4. Non-tensor objects → torch.Tensor
    """
    result = []
    for item in inputs:
        if item is None:
            continue
            
        # 중첩 구조 재귀 처리
        if isinstance(item, (list, tuple)):
            result.extend(flatten_and_convert(item))
            
        # DynamicCache 처리 (key_cache + value_cache 확장)
        elif isinstance(item, DynamicCache):
            # print('DynamicCache in input!')
            # result.extend(item.key_cache + item.value_cache)
            num_layers = len(item.key_cache)
            half = num_layers // 2
            
            # 1. key out (첫 절반 레이어의 key)
            key_out = item.key_cache[:half]
            # 2. key in (나머지 레이어의 key)
            key_in = item.key_cache[half:]
            # 3. value out (첫 절반 레이어의 value)
            value_out = item.value_cache[:half]
            # 4. value in (나머지 레이어의 value)
            value_in = item.value_cache[half:]
            
            # 순서: key out → key in → value out → value in
            result.extend(key_out)
            result.extend(key_in)
            result.extend(value_out)
            result.extend(value_in)
        # 텐서 변환
        else:
            result.append(
                item if isinstance(item, torch.Tensor) 
                else torch.tensor(item)
            )
    return tuple(result)

def infer(circle_binary: bytes, *args: Any, **kwargs: Any) -> Any:
    # 입력 데이터 통합
    raw_inputs = args + tuple(kwargs.values())
    
    # 평탄화 및 변환
    
    # Get input spec from circle binary.
    model = circle.Model.Model.GetRootAsModel(circle_binary, 0)
    assert model.SubgraphsLength() == 1
    graph = model.Subgraphs(0)
    model_input_tensors = [graph.Tensors(graph.Inputs(o)) for o in range(graph.InputsLength())]
    model_input_shapes_np = [t.ShapeAsNumpy() for t in model_input_tensors]
    model_input_types_cm = [t.Type() for t in model_input_tensors]
    user_inputs = flatten_and_convert(raw_inputs)

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
