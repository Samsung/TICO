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

import tempfile
from pathlib import Path
from typing import List

import numpy as np
import torch

from tico.quantization.evaluation.executor.backend_executor import BackendExecutor
from tico.utils.model import CircleModel
from tico.utils.utils import run_bash_cmd


class CircleExecutor(BackendExecutor):
    """
    A class for running inference on fake-quantized circle models.

    Instead of leveraging the actual backend for quantized circle execution,
     it applies fake quantization to the models and performs inference.
    """

    def __init__(self):
        self.compiler_path = Path("/usr/share/one/bin/onecc")
        self.interpreter_path = None  # Use circle-interpreter
        self.fq_circle_path = None

        # Check if the toolchain is installed.
        if not self.compiler_path.is_file():
            raise RuntimeError(
                "Not found one-compiler. Please install the one-compiler package first."
            )

        self.temp_dir = tempfile.TemporaryDirectory()

    def compile(self, circle_model: CircleModel) -> None:
        assert isinstance(circle_model, CircleModel)
        circle_path = Path(self.temp_dir.name) / "quantized.circle"
        circle_model.save(str(circle_path))
        self.fq_circle_path = Path(self.temp_dir.name) / "fake_quantized.circle"
        args = []
        args += ["quantize"]
        args += ["--fake_quantize"]
        args += ["-i", str(circle_path)]
        args += ["-o", str(self.fq_circle_path)]
        cmd = [str(self.compiler_path)] + args
        run_bash_cmd(cmd)

    def run_inference(self, input_data: List[torch.Tensor]) -> List[np.ndarray]:
        if not self.fq_circle_path:
            raise RuntimeError("You must compile the model before running inference.")

        fq_circle = CircleModel.load(self.fq_circle_path)
        assert isinstance(fq_circle, CircleModel)
        out = fq_circle(*input_data)
        if not isinstance(out, list):
            out = [out]
        return out

    def __del__(self):
        if hasattr(self, "temp_dir") and self.temp_dir:
            self.temp_dir.cleanup()
