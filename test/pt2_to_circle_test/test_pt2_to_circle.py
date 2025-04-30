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
import subprocess
from functools import wraps
from pathlib import Path
from typing import Any, List, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
import numpy as np
import tico.pt2_to_circle
import tico.utils
import tico.utils.model
import torch
from tico.utils.convert import convert_exported_module_to_circle
from tico.utils.utils import SuppressWarning
from torch.export import export
from torch.utils import _pytree as pytree

import test.utils.helper as helper
from test.utils.infer import infer_with_circle_interpreter, infer_with_onert
from test.utils.runtime import Runtime

# TODO Move this to utils or helper

__test_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "artifacts"
__circle2circle_path = "/usr/share/one/bin/circle2circle"

# Create empty test directories
if not os.path.exists(__test_dir):
    os.makedirs(__test_dir)


def print_name_on_exception(function):
    """
    Print its name on exception
    """

    @wraps(function)
    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception as e:
            print(f"\nTEST FAILED at '{str(function.__name__)} ...'\n", str(e))
            raise e

    return wrapper


@print_name_on_exception
def convert_nnmodule_to_pt2(
    model: torch.nn.Module, example_inputs: tuple, pt2_model_path: str
):
    # Create .pt2 model
    with torch.no_grad(), SuppressWarning(
        UserWarning, ".*quantize_per_tensor"
    ), SuppressWarning(
        UserWarning,
        ".*TF32 acceleration on top of oneDNN is available for Intel GPUs.*",
    ):
        # Warning details:
        #   ...site-packages/torch/_subclasses/functional_tensor.py:364
        #   UserWarning: At pre-dispatch tracing, we assume that any custom op marked with
        #     CompositeImplicitAutograd and have functional schema are safe to not decompose.
        _args, _kwargs = helper.get_args_kwargs(example_inputs)
        exported = export(model.eval(), args=_args, kwargs=_kwargs)
    torch.export.save(exported, pt2_model_path)


@print_name_on_exception
def convert_pt2_to_circle(pt2_model_path: str, circle_model_path: str):
    tico.pt2_to_circle.convert(pt2_model_path, circle_model_path)


@print_name_on_exception
def convert_nnmodule_to_circle(
    nnmodule: torch.nn.Module,
    example_inputs: tuple,
    circle_model_path: str,
):
    with torch.no_grad():
        _args, _kwargs = helper.get_args_kwargs(example_inputs)
        exported_program = export(nnmodule.eval(), args=_args, kwargs=_kwargs)
    circle_program = convert_exported_module_to_circle(exported_program)
    circle_binary = circle_program
    with open(circle_model_path, "wb") as f:
        f.write(circle_binary)


@print_name_on_exception
def verify_circle(circle_model_path: str, opt_circle_model_str: str):
    try:
        cmd = [
            __circle2circle_path,
            str(circle_model_path),
            str(opt_circle_model_str),
        ]

        # Test validity of circle model
        # This will check if the shape/dtype of the model is correct
        output = subprocess.run(cmd, check=True, text=True, capture_output=True)
        if output.stdout:
            print(output.stdout)
        if output.stderr:
            print(output.stderr)
    except subprocess.CalledProcessError as err:
        cmd_str = " ".join(err.cmd)
        msg = f"Error while running command:\n\n $ {cmd_str}"
        msg += "\n"
        msg += "[EXIT CODE]\n"
        msg += f"{err.returncode}\n"
        msg += "[STDOUT]\n"
        msg += err.stdout
        msg += "[STDERR]\n"
        msg += err.stderr
        raise RuntimeError(f"circle2circle failed.\n\n {msg}")


@print_name_on_exception
def infer_nnmodule(model: torch.nn.Module, example_inputs: tuple):
    with torch.no_grad():
        _args, _kwargs = helper.get_args_kwargs(example_inputs)
        torch_result = model.forward(*_args, **_kwargs)

        # Let's flatten torch output result.
        # The output of torch module can be a dictionary or a multi-dimensional tuple of tensors.
        # Circle only allows flattened (1-dim array of tensors) output.
        #
        # Q. Why use `pytree.tree_flatten`?
        # torch dynamo flattens torch input/output using pytree.tree_unflatten/flatten.
        # (See torch._dynamo.eval_frame.rewrite_signature)
        torch_result, _ = pytree.tree_flatten(torch_result)

        return torch_result


@print_name_on_exception
def infer_circle(
    circle_path: str, example_inputs: tuple, runtime: Runtime = "circle-interpreter"
) -> Any:
    """
    Run inference on a Circle model using the specified runtime.

    Parameters
    -----------
    circle_path
        Path to the .circle file.
    example_inputs
        Tuple of example inputs for the model.
    runtime
        Which runtime to use for execution.
        - 'circle-interpreter' (default)
        - 'onert'

    Returns
    --------
    Any
        The output produced by the chosen runtime.
    """
    if runtime == "circle-interpreter":
        return infer_with_circle_interpreter(circle_path, example_inputs)
    elif runtime == "onert":
        return infer_with_onert(circle_path, example_inputs)
    else:
        raise ValueError(f"Unknown runtime: {runtime!r}")


@print_name_on_exception
def validate_result(
    torch_result: List[torch.Tensor | int | float],
    circle_result: List[np.ndarray],
    rtol: float = 1e-5,
    atol: float = 1e-5,
):
    np.testing.assert_equal(
        actual=len(torch_result),
        desired=len(circle_result),
        err_msg=f"Number of outputs mismatches.\ntorch result: #{len(torch_result)}, circle result: #{len(circle_result)}",
    )
    for torch_res, circle_res in zip(torch_result, circle_result):
        if isinstance(torch_res, torch.Tensor):
            np.testing.assert_equal(
                actual=torch_res.shape,
                desired=circle_res.shape,
                err_msg=f"Shape mismatches.\ntorch result: {torch_res.shape}\ncircle result: {circle_res.shape}",
            )
        np.testing.assert_allclose(
            actual=torch_res,
            desired=circle_res,
            rtol=rtol,
            atol=atol,
            err_msg=f"Value mismatches.\ntorch result: {torch_res}\ncircle result: {circle_res}",
        )
        if isinstance(torch_res, torch.Tensor):
            assert (
                torch_res.dtype == torch.from_numpy(circle_res).dtype
            ), f"Type mismatches.\ntorch result: {torch_res.dtype}\ncircle result: {circle_res.dtype}"
        elif isinstance(torch_res, (int, float)):
            assert type(torch_res) == type(
                torch.from_numpy(circle_res).item()
            ), f"Type mismatches.\ntorch result: {type(torch_res)}\ncircle result: {circle_res.dtype}"
