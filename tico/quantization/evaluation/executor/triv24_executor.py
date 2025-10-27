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
from circle_schema import circle

from tico.quantization.evaluation.executor.backend_executor import BackendExecutor
from tico.quantization.evaluation.utils import (
    dequantize,
    get_graph_input_output,
    quantize,
)
from tico.serialize.circle_mapping import np_dtype_from_circle_dtype
from tico.utils.model import CircleModel
from tico.utils.utils import run_bash_cmd


class Triv24Executor(BackendExecutor):
    """
    Implementation for a TRIV24 backend.
    """

    def __init__(self):
        self.compiler_path = Path("/usr/share/one/backends/triv24/bin/triv24-compile")
        self.interpreter_path = Path(
            "/usr/share/one/backends/triv24/bin/triv24-ssinfer"
        )
        self.circle_model = None
        self.tvn_path = None

        # Check if triv24 toolchain is installed.
        if not self.compiler_path.is_file() or not self.interpreter_path.is_file():
            raise RuntimeError(
                "Not found triv24 toolchain. Please install the toolchain package first."
            )

        self.temp_dir = tempfile.TemporaryDirectory()

    def compile(self, circle_model: CircleModel) -> None:
        assert isinstance(circle_model, CircleModel)
        self.circle_model = circle_model
        circle_path = Path(self.temp_dir.name) / "quantized.circle"
        circle_model.save(str(circle_path))
        self.tvn_path = Path(self.temp_dir.name) / "compiled.tvn"
        args = []
        args += ["-o", str(self.tvn_path)]
        args += [str(circle_path)]
        cmd = [str(self.compiler_path)] + args
        run_bash_cmd(cmd)

    def run_inference(self, input_data: List[torch.Tensor]) -> List[np.ndarray]:
        if not self.tvn_path:
            raise RuntimeError("You must compile the model before running inference.")

        assert isinstance(self.circle_model, CircleModel)
        circle_inputs, circle_outputs = get_graph_input_output(self.circle_model)
        # Get input/output of scale/zp from quantized circle.
        # Note that qparams may be None because some of them are not quantized like indices, argmax.
        input_qparam: List[circle.QuantizationParameters.QuantizationParameters] = []
        for inp in circle_inputs:
            input_qparam.append(inp.Quantization())
        output_qparam: List[circle.QuantizationParameters.QuantizationParameters] = []
        for out in circle_outputs:
            output_qparam.append(out.Quantization())

        # Create input files for inference.
        for in_idx, data in enumerate(input_data):
            in_data_path = Path(self.temp_dir.name) / f"input.{in_idx}.tv2b"
            assert isinstance(data, torch.Tensor)
            if input_qparam[in_idx] is None:
                np_data = data.numpy()
            else:
                np_data = quantize(
                    data.numpy(),
                    input_qparam[in_idx].ScaleAsNumpy()[0],
                    input_qparam[in_idx].ZeroPointAsNumpy()[0],
                    np_dtype_from_circle_dtype(circle_inputs[in_idx].Type()),
                )
            np_data.tofile(in_data_path)
        args = []
        args += ["--loadable", str(self.tvn_path)]
        args += ["--input-spec", f"tv2b:{str(self.temp_dir.name)}/input"]
        args += ["--dump-output-as-tv2b", f"{str(self.temp_dir.name)}/output"]
        cmd = [str(self.interpreter_path)] + args

        # Run inference
        run_bash_cmd(cmd)

        # Load outputs from file
        circle_dequantized_output = []
        circle_output_shapes_np = [t.ShapeAsNumpy() for t in circle_outputs]
        for out_idx, out_shape in enumerate(circle_output_shapes_np):
            out_data_path = Path(self.temp_dir.name) / f"output.{out_idx}.tv2b"
            circle_output = np.fromfile(out_data_path, np.uint8).reshape(out_shape)
            if output_qparam[out_idx] is None:
                circle_dequantized_output.append(circle_output)
            else:
                circle_dequantized_output.append(
                    dequantize(
                        circle_output,
                        output_qparam[out_idx].ScaleAsNumpy()[0],
                        output_qparam[out_idx].ZeroPointAsNumpy()[0],
                        np_dtype_from_circle_dtype(circle_outputs[out_idx].Type()),
                    )
                )

        return circle_dequantized_output

    def __del__(self):
        self.temp_dir.cleanup()
