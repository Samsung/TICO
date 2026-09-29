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
from functools import wraps
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

import tico.pt2_to_circle

import torch
from tico.circle import CircleDocument
from tico.circle.runtime import CircleReferenceRuntime
from tico.config.base import CompileConfigBase
from tico.utils.convert import convert_exported_module_to_circle
from tico.utils.utils import SuppressWarning
from torch.export import export
from torch.utils import _pytree as pytree

from test.support.infer import infer_with_runtime
from test.support.runtime import Runtime

# TODO Move this to utils or helper

__test_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "artifacts"

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
    model: torch.nn.Module,
    forward_args: tuple,
    forward_kwargs: dict,
    pt2_model_path: str,
    dynamic_shapes: dict | None = None,
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
        exported = export(
            model.eval(),
            args=forward_args,
            kwargs=forward_kwargs,
            dynamic_shapes=dynamic_shapes,
        )
    torch.export.save(exported, pt2_model_path)


@print_name_on_exception
def convert_pt2_to_circle(
    pt2_model_path: str,
    circle_model_path: str,
    config: Optional[CompileConfigBase] = None,
):
    with SuppressWarning(FutureWarning, ".*LeafSpec"):
        tico.pt2_to_circle.convert(pt2_model_path, circle_model_path, config=config)


@print_name_on_exception
def convert_nnmodule_to_circle(
    nnmodule: torch.nn.Module,
    forward_args: tuple,
    forward_kwargs: dict,
    circle_model_path: str,
    dynamic_shapes: Optional[dict] = None,
    config: Optional[CompileConfigBase] = None,
):
    with torch.no_grad():
        exported_program = export(
            nnmodule.eval(),
            args=forward_args,
            kwargs=forward_kwargs,
            dynamic_shapes=dynamic_shapes,
        )
    with SuppressWarning(FutureWarning, ".*LeafSpec"):
        circle_program = convert_exported_module_to_circle(exported_program, config)
    circle_binary = circle_program
    with open(circle_model_path, "wb") as f:
        f.write(circle_binary)


@print_name_on_exception
def verify_circle(circle_model_path: str):
    """
    Validate the serialized Circle file before it is executed.

    1. `CircleDocument.verify()` checks the structural consistency of the file:
       index ranges, dataflow, constant buffers, interfaces, and signatures.
    2. Preparing a `CircleReferenceRuntime` checks that every operator is a
       supported builtin with valid operand references and decodable constants.
    3. For models with static inputs, a zero-valued probe run validates that
       the shape and dtype computed by every operator match the serialized
       tensor contracts. Dynamic models receive the same contract validation
       during the real inference below.
    """
    document = CircleDocument.load(circle_model_path)
    document.verify(raise_on_error=True)
    runtime = CircleReferenceRuntime(document, verify=False)
    if not any(tensor.is_dynamic for tensor in runtime.input_tensors):
        runtime.probe_static_contracts()


@print_name_on_exception
def infer_nnmodule(
    model: torch.nn.Module,
    forward_args: tuple,
    forward_kwargs: dict,
):
    with torch.no_grad():
        # Model should be frozen to compare the result with others.
        # e.g. BatchNorm running_mean/running_var will be updated during training mode, thus changing the model behavior.
        model.eval()

        expected_result = model.forward(*forward_args, **forward_kwargs)

        # Let's flatten torch output result.
        # The output of torch module can be a dictionary or a multi-dimensional tuple of tensors.
        # Circle only allows flattened (1-dim array of tensors) output.
        #
        # Q. Why use `pytree.tree_flatten`?
        # torch dynamo flattens torch input/output using pytree.tree_unflatten/flatten.
        # (See torch._dynamo.eval_frame.rewrite_signature)
        expected_result, _ = pytree.tree_flatten(expected_result)

        return expected_result


@print_name_on_exception
def infer_circle(
    circle_path: str,
    forward_args: tuple,
    forward_kwargs: dict,
    runtime: Runtime = "reference",
) -> Any:
    """
    Run inference on a Circle model using the specified runtime.

    Parameters
    -----------
    circle_path
        Path to the .circle file.
    forward_args
        Tuple of arguments for the model's forward function.
    forward_kwargs
        Dictionary of keyword arguments for the model's forward function.
    runtime
        Which runtime to use for execution.
        - 'reference' (default, built into TICO)
        - 'circle-interpreter' (optional ONE luci-interpreter)
        - 'onert' (optional onert package)

    Returns
    --------
    Any
        The output produced by the chosen runtime.
    """
    if runtime not in ("reference", "circle-interpreter", "onert"):
        raise ValueError(f"Unknown runtime: {runtime!r}")
    return infer_with_runtime(circle_path, forward_args, forward_kwargs, runtime)


@print_name_on_exception
def validate_result(
    expected_result: List[torch.Tensor | int | float],
    circle_result: List[np.ndarray],
    rtol: float = 1e-5,
    atol: float = 1e-5,
):
    np.testing.assert_equal(
        actual=len(expected_result),
        desired=len(circle_result),
        err_msg=f"Number of outputs mismatches.\nexpected result: #{len(expected_result)}, circle result: #{len(circle_result)}",
    )
    for expected_res, circle_res in zip(expected_result, circle_result):
        if isinstance(expected_res, torch.Tensor):
            np.testing.assert_equal(
                actual=expected_res.shape,
                desired=circle_res.shape,
                err_msg=f"Shape mismatches.\nexpected result: {expected_res.shape}\ncircle result: {circle_res.shape}",
            )
            expected_tensor = expected_res
            circle_tensor = torch.from_numpy(circle_res.copy())
        elif isinstance(expected_res, (int, float)):
            # A Python scalar has no dtype or rank of its own. TICO serializes it
            # either as a scalar constant or as a one-element size tensor, so the
            # comparison adopts the serialized dtype (same kind) and shape while
            # still checking the value exactly.
            circle_tensor = torch.from_numpy(circle_res.copy())
            expected_kind = "f" if isinstance(expected_res, float) else "i"
            if circle_res.size != 1 or circle_res.dtype.kind not in (
                expected_kind,
                "u",
            ):
                raise AssertionError(
                    f"Scalar result {expected_res!r} cannot be compared with circle "
                    f"output {circle_res.dtype}{list(circle_res.shape)}."
                )
            expected_tensor = torch.full(
                circle_tensor.shape, expected_res, dtype=circle_tensor.dtype
            )
        else:
            raise TypeError("Expected result must be a tensor or scalar value.")

        # Check both dtype and value mismatch
        torch.testing.assert_close(
            actual=circle_tensor,
            expected=expected_tensor,
            equal_nan=True,
            check_dtype=True,
            rtol=rtol,
            atol=atol,
        )
