# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved
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

"""Runtime selection for executing serialized Circle models.

``reference`` is the default and only runtime that ships with TICO. It runs the
Circle model in-process with ``tico.circle.runtime`` and requires nothing beyond
the package dependencies. The remaining names are optional compatibility
adapters for external runtimes and are used only when selected explicitly:

- ``circle-interpreter``: ONE's ``luci-interpreter`` through its CFFI library.
- ``onert``: the ``onert`` Python package.

The selection never changes implicitly based on which external packages happen
to be installed.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable, Sequence

import numpy as np
import torch

DEFAULT_RUNTIME = "reference"
RUNTIME_CHOICES: tuple[str, ...] = ("reference", "circle-interpreter", "onert")

RuntimeCallable = Callable[[bytes, Sequence[torch.Tensor]], list[np.ndarray]]


def resolve_runtime_name(runtime: str | None) -> str:
    """Validate a runtime name and apply the default."""

    name = DEFAULT_RUNTIME if runtime is None else str(runtime)
    if name not in RUNTIME_CHOICES:
        raise ValueError(
            f"Unknown Circle runtime {name!r}. Choose one of {list(RUNTIME_CHOICES)}."
        )
    return name


def _run_with_reference(
    circle_binary: bytes,
    inputs: Sequence[torch.Tensor],
) -> list[np.ndarray]:
    from tico.circle.runtime import CircleReferenceRuntime

    runtime = CircleReferenceRuntime(circle_binary)
    return list(runtime.run(inputs).outputs)


def _run_with_circle_interpreter(
    circle_binary: bytes,
    inputs: Sequence[torch.Tensor],
) -> list[np.ndarray]:
    from circle_schema import circle

    from tico.interpreter.interpreter import Interpreter
    from tico.serialize.circle_mapping import np_dtype_from_circle_dtype

    model = circle.Model.Model.GetRootAsModel(circle_binary, 0)
    assert model.SubgraphsLength() == 1
    graph = model.Subgraphs(0)

    intp = Interpreter(circle_binary)
    for input_idx, user_input in enumerate(inputs):
        intp.writeInputTensor(input_idx, user_input)
    intp.interpret()

    outputs: list[np.ndarray] = []
    for output_idx in range(graph.OutputsLength()):
        tensor = graph.Tensors(graph.Outputs(output_idx))
        result: np.ndarray = np.empty(
            tensor.ShapeAsNumpy(),
            dtype=np_dtype_from_circle_dtype(tensor.Type()),
        )
        intp.readOutputTensor(output_idx, result)
        outputs.append(result)
    return outputs


def _run_with_onert(
    circle_binary: bytes,
    inputs: Sequence[torch.Tensor],
) -> list[np.ndarray]:
    try:
        from onert import infer as onert_infer
    except ImportError as error:
        raise RuntimeError(
            "The 'onert' package is required to run this function."
        ) from error

    with tempfile.TemporaryDirectory() as directory:
        circle_path = os.path.join(directory, "model.circle")
        with open(circle_path, "wb") as f:
            f.write(circle_binary)
        session = onert_infer.session(circle_path)

        # onert cannot execute models with unspecified dimensions; publish the
        # concrete input shapes first.
        input_tensorinfos = session.get_inputs_tensorinfo()
        if any(-1 in info.dims for info in input_tensorinfos):
            from onert.native.libnnfw_api_pybind import tensorinfo

            for idx, (info, input_data) in enumerate(zip(input_tensorinfos, inputs)):
                if -1 not in info.dims:
                    continue
                new_info = tensorinfo()
                new_info.rank = len(input_data.shape)
                new_info.dims = list(input_data.shape)
                if input_data.dtype not in (torch.float32, torch.float):
                    raise RuntimeError(
                        "The onert adapter only resizes FLOAT32 dynamic inputs; "
                        f"input {idx} has dtype {input_data.dtype}."
                    )
                new_info.dtype = "float32"
                session.session.set_input_tensorinfo(idx, new_info)

        output = session.infer(list(inputs))
        return list(output) if isinstance(output, (list, tuple)) else [output]


_RUNTIMES: dict[str, RuntimeCallable] = {
    "reference": _run_with_reference,
    "circle-interpreter": _run_with_circle_interpreter,
    "onert": _run_with_onert,
}


def run_circle(
    circle_binary: bytes,
    inputs: Sequence[torch.Tensor],
    *,
    runtime: str | None = None,
) -> list[np.ndarray]:
    """Execute a Circle model with the selected runtime and return NumPy outputs."""

    name = resolve_runtime_name(runtime)
    return _RUNTIMES[name](circle_binary, inputs)


def runtime_from_environment(variable: str = "CCEX_RUNTIME") -> str:
    """Return the runtime named by an environment variable or the default."""

    return resolve_runtime_name(os.environ.get(variable) or None)


__all__ = [
    "DEFAULT_RUNTIME",
    "RUNTIME_CHOICES",
    "resolve_runtime_name",
    "run_circle",
    "runtime_from_environment",
]
