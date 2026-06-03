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
from circle_schema import circle

from tico.interpreter.interpreter import Interpreter
from tico.serialize.circle_mapping import np_dtype_from_circle_dtype
from tico.utils.signature import ModelInputSpec


def infer(circle_binary: bytes, *args: Any, **kwargs: Any) -> Any:
    input_spec = ModelInputSpec(circle_binary)
    user_inputs = input_spec.bind(args, kwargs, check=True)

    # Get input/output spec from circle binary.
    model = circle.Model.Model.GetRootAsModel(circle_binary, 0)
    assert model.SubgraphsLength() == 1
    graph = model.Subgraphs(0)

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
