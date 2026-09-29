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

from typing import List, Optional

import numpy as np
import torch

from tico.circle.runtime import CircleReferenceRuntime, ExecutionMode
from tico.quantization.evaluation.executor.backend_executor import BackendExecutor
from tico.utils.model import CircleModel


class CircleExecutor(BackendExecutor):
    """
    A class for running inference on quantized circle models with the built-in
    reference runtime in fake-quantize mode.

    Instead of leveraging the actual backend for quantized circle execution, the
    model is evaluated with the semantics of ONE's ``onecc quantize
    --fake_quantize`` conversion: quantized constants are dequantized, every
    operator computes in FLOAT32, quantized activations are rounded to their
    serialized quantization grid after each producing operator, and quantized
    graph inputs and outputs are exchanged as FLOAT32. See
    ``tico/circle/README.md`` for the exact rules and their differences from
    integer backend execution.
    """

    def __init__(self):
        self._runtime: Optional[CircleReferenceRuntime] = None

    def compile(self, circle_model: CircleModel) -> None:
        assert isinstance(circle_model, CircleModel)
        self._runtime = CircleReferenceRuntime(circle_model.circle_binary)

    def run_inference(self, input_data: List[torch.Tensor]) -> List[np.ndarray]:
        if self._runtime is None:
            raise RuntimeError("You must compile the model before running inference.")

        result = self._runtime.run(input_data, mode=ExecutionMode.FAKE_QUANTIZE)
        return list(result.outputs)
