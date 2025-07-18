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

import inspect
import subprocess
import typing
import warnings
from functools import wraps
from typing import List

import torch
from circle_schema import circle
from packaging.version import Version
from torch._guards import detect_fake_mode
from torch.export import ExportedProgram
from torch.utils import _pytree as pytree

from tico.serialize.quant_param import QuantParam


HAS_TORCH_OVER_25 = Version(torch.__version__) >= Version("2.5.0")
HAS_TORCH_OVER_28_DEV = Version(torch.__version__) >= Version("2.8.0.dev")


def get_fake_mode(exported_program: ExportedProgram):
    fake_mode = detect_fake_mode(
        tuple(
            node.meta["val"]
            for node in exported_program.graph.nodes
            if node.op == "placeholder"
        )
    )
    assert fake_mode is not None
    return fake_mode


class SuppressWarning:
    def __init__(self, warning_category: type[Warning], regex):
        self.warning_category = warning_category
        self.regex = regex

    def __enter__(self):
        warnings.filterwarnings(
            "ignore", category=self.warning_category, message=self.regex
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        warnings.filterwarnings(
            "default", category=self.warning_category, message=self.regex
        )


class ArgTypeError(Exception):
    """
    Invalid argument type
    """

    pass


def enforce_type(callable):
    """Check types for your callable's signature

    NOTE Place this one above @dataclass decorator if you want to use it with dataclass initializer.
        Ex.
            @enforce_type
            @dataclass
            class Args:
                ...
    """
    spec = inspect.getfullargspec(callable)

    def check_types(*args, **kwargs):
        parameters = dict(zip(spec.args, args))
        parameters.update(kwargs)
        for name, value in parameters.items():
            if name == "self":
                # skip 'self' in spec.args
                continue

            assert (
                name in spec.annotations
            ), f"All parameter require type hints. {name} needs a type hint"

            type_hint = spec.annotations[name]

            # Return tuple of flattened types.
            # Q) What is flatten?
            # A) Optional/Union is not included. Below are included.
            # collections: List, Set, ...
            # primitive types: int, str, ...
            def _flatten_type(type_hint) -> tuple:
                # `get_origin` maps Union[...] and Optional[...] varieties to Union
                if typing.get_origin(type_hint) == typing.Union:
                    # ex. typing.Union[list, int] -> (list, int)
                    # ex. typing.Optional[torch.fx.Node] -> (torch.fx.Node, NoneType)
                    actual_type = tuple(
                        [_flatten_type(t) for t in typing.get_args(type_hint)]
                    )
                else:
                    actual_type = (type_hint,)
                return actual_type

            type_hint = _flatten_type(type_hint)

            # Return true if value matches with type_hint
            # Return false otherwise
            def _check_type(value, type_hint):
                if type_hint == typing.Any:
                    return True

                if isinstance(type_hint, tuple):
                    return any([_check_type(value, t) for t in type_hint])

                if typing.get_origin(type_hint) in (list, set):
                    if not isinstance(value, typing.get_origin(type_hint)):
                        return False

                    for v in value:
                        if not any(
                            [_check_type(v, t) for t in typing.get_args(type_hint)]
                        ):
                            return False

                    return True

                if typing.get_origin(type_hint) == dict:
                    if not isinstance(value, typing.get_origin(type_hint)):
                        return False

                    for k, v in value.items():
                        k_type, v_type = typing.get_args(type_hint)
                        if not _check_type(k, k_type):
                            return False
                        if not _check_type(v, v_type):
                            return False

                    return True

                # TODO: Support more type hints
                return isinstance(value, type_hint)

            type_check_result = _check_type(value, type_hint)
            if not type_check_result:
                raise ArgTypeError(
                    "Unexpected type for '{}' (expected {} but found {})".format(
                        name, type_hint, type(value)
                    )
                )

    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            check_types(*args, **kwargs)
            return func(*args, **kwargs)

        return wrapper

    if inspect.isclass(callable):
        callable.__init__ = decorate(callable.__init__)
        return callable

    return decorate(callable)


def fill_meta_val(exported_program: ExportedProgram):
    for node in exported_program.graph.nodes:
        assert hasattr(node, "meta"), f"{node.name} does not have meta attribute"

        if node.meta.get("val", None) is None:
            if node.op == "call_function":
                set_new_meta_val(node)


def set_new_meta_val(node: torch.fx.node.Node):
    """
    Set node.meta["val"].

    There are some cases when node.meta["val"] should be updated.
    - After creating new node
    - After updating node's args or kwargs
    """
    assert isinstance(node, torch.fx.node.Node)

    # `node.target()` needs only `Tensor` for its arguments.
    # Therefore, let's retrieve `FakeTensor` if it is `torch.fx.Node`.
    args, kwargs = pytree.tree_map_only(
        torch.fx.Node,
        lambda n: n.meta["val"],
        (node.args, node.kwargs),
    )
    new_val = node.target(*args, **kwargs)  # type: ignore[operator]
    node.meta["val"] = new_val


def unset_meta_val(node: torch.fx.node.Node):
    """
    Unset node.meta["val"].

    - When to use it?
        When we need to update a node's meta val
        but some precedent's meta value are not decided yet, (eg. newly created args)
        let's simply unset meta val and expect `FillMetaVal` do it.
    """
    assert isinstance(node, torch.fx.node.Node)

    if "val" in node.meta:
        del node.meta["val"]


def run_bash_cmd(command: typing.List[str]) -> subprocess.CompletedProcess[str]:
    """
    Executes a given bash command represented as a sequence of program arguments
     using subprocess and returns output.

    Args:
        command (List[str]): A sequence of program arguments.

    Returns:
        str: The standard output of the executed command.

    Example:
        >>> completed_process = run_bash_cmd(["echo", "Hello, World!"])
        print (completed_process.stdout)
        'Hello, World!\\n'

        >>> cp = run_bash_cmd(["ls", "-l"])
        print (cp.stdout)
        'drwxrwxr-x 8 user group 4096 12월  3 17:16 tico\\n'
    """
    if not isinstance(command, list) or not all(isinstance(c, str) for c in command):
        raise ValueError("Command must be a list of strings.")
    try:
        return subprocess.run(command, check=True, text=True, capture_output=True)
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
        raise RuntimeError(f"Failed.\n\n {msg}")


def has_quantization_ops(graph: torch.fx.Graph):
    """
    Checks whether the given fx graph contains any quantization-related operations.

    This function inspects the provided graph to determine if it includes operations associated
    with quantization (e.g., quantize, dequantize, fake quantize, etc.). The presence of such operations
    can be used to decide whether to run subsequent quantization-specific passes on the graph.

    Parameters:
        graph: The fx graph to be examined. It is expected that the graph supports
               iteration or traversal over its constituent operations.

    Returns:
        bool: True if the graph contains one or more quantization-related operations, False otherwise.
    """
    quantized_ops = [
        torch.ops.quantized_decomposed.quantize_per_tensor.default,
        torch.ops.quantized_decomposed.quantize_per_channel.default,
        torch.ops.quantized_decomposed.dequantize_per_tensor.default,
        torch.ops.quantized_decomposed.dequantize_per_channel.default,
    ]
    for node in graph.nodes:
        if node.op != "call_function":
            continue
        if node.target in quantized_ops:
            return True

    return False


def to_circle_qparam(qparam: QuantParam):
    circle_qparam = circle.QuantizationParameters.QuantizationParametersT()
    if qparam.scale is not None:
        circle_qparam.scale = qparam.scale

    if qparam.zero_point is not None:
        circle_qparam.zeroPoint = qparam.zero_point

    if qparam.quantized_dimension is not None:
        circle_qparam.quantizedDimension = qparam.quantized_dimension

    if qparam.min is not None:
        circle_qparam.min = qparam.min

    if qparam.max is not None:
        circle_qparam.max = qparam.max

    return circle_qparam


def quant_min_max(dtype: str):
    if dtype == "uint8":
        return (0, 255)
    elif dtype == "int16":
        return (-32768, 32767)
    else:
        raise NotImplementedError(f"NYI dtype: {dtype}")


def get_quant_dtype(qmin: int, qmax: int):
    """
    Returns the string representation of the quantized data type based on qmin and qmax.

    Args:
        qmin (int): Minimum quantized value.
        qmax (int): Maximum quantized value.

    Returns:
        str: A string representing the quantized data type, such as "int8", "uint4", etc.

    Raises:
        ValueError: If the (qmin, qmax) pair is not supported.
    """
    known_ranges = {
        (-32768, 32767): "int16",
        (-32767, 32767): "int16",
        (0, 65535): "uint16",
        (-128, 127): "int8",
        (0, 255): "uint8",
        (-8, 7): "int4",
        (0, 15): "uint4",
    }

    if (qmin, qmax) in known_ranges:
        return known_ranges[(qmin, qmax)]
    else:
        raise ValueError(f"Unsupported quantization range: ({qmin}, {qmax})")


def broadcastable(
    shape_a: List[int] | torch.Size, shape_b: List[int] | torch.Size
) -> bool:
    """
    Return **True** if two shapes are broadcast-compatible under the standard
    NumPy/PyTorch rules.

    Broadcasting rule
    --------------------------------
    - Align the shapes **right-to-left**.
    - For each aligned dimension `(a, b)` one of the following must hold
      - `a == b`  (sizes match)
      - `a == 1`  (shape-A can repeat along that dim)
      - `b == 1`  (shape-B can repeat along that dim)
    - When one shape is shorter, treat its missing leading dims as `1`.

    Examples
    --------
    >>> _broadcastable([8, 16, 32], [16, 32])
    True
    >>> _broadcastable([8, 16, 32], [1, 32])
    True
    >>> _broadcastable([8, 16, 32], [8, 32, 16])
    False
    """
    # Walk from the last dim to the front
    len_a, len_b = len(shape_a), len(shape_b)
    max_len = max(len_a, len_b)
    for i in range(1, max_len + 1):
        dim_a = shape_a[-i] if i <= len_a else 1
        dim_b = shape_b[-i] if i <= len_b else 1
        if dim_a != 1 and dim_b != 1 and dim_a != dim_b:
            return False
    return True


def is_target_node(
    node: torch.fx.Node, target_ops: list[torch._ops.OpOverload] | torch._ops.OpOverload
):
    """
    Check whether a given node is a `call_function` node that matches one of the specified targets.

    Args:
        node (torch.fx.Node): The node to check.
        target_ops (Iterable[Callable]): A list or set of target operations to match (e.g., ops.aten.reshape).

    Returns:
        bool: True if the node is a call_function, its target is in `target_ops`.
    """
    if not isinstance(target_ops, list):
        target_ops = [target_ops]
    assert all(isinstance(t, torch._ops.OpOverload) for t in target_ops), target_ops

    if node.op != "call_function":
        return False
    if node.target not in target_ops:
        return False

    return True
