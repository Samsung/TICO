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

from typing import Callable, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch.fx
import torch
from torch.ao.quantization.quantizer import QuantizationSpec
from torch.ao.quantization.quantizer.utils import _get_module_name_filter
from torch.utils import _pytree as pytree

from tico.experimental.quantization.algorithm.pt2e.annotation.config import (
    QuantizationConfig,
)


def get_module_type_filter(tp: Callable):
    """
    Get the module_type_filter function for a given module type.

    The filter accepts a node and checks if the node comes from a module
     that has certain module type.

    For example:
        node: linear_op = call_function[...](...)  # comes from a module with type Block -> Sub -> Linear


    >> module_type_filter = get_module_type_filter(Sub)  # submodule with type `Sub`, under the `Block` submodule
    >> print(module_type_filter(node))
    True  # the node is from the submodule `Sub`
    """

    tp_str = tp.__module__ + "." + tp.__qualname__

    def module_type_filter(n: torch.fx.Node) -> bool:
        # example: {
        #     'L__self___sub': ("L['self'].sub", <class '....Sub'>),
        #     'L__self___sub_linear': ("L['self'].sub.linear", <class 'torch.nn.modules.linear.Linear'>)
        # }
        nn_module_stack = n.meta.get("nn_module_stack", {})
        types = []
        for _, t in nn_module_stack.values():
            # export() returns str, but older APIs (e.g. capture_pre_autograd_graph)
            # return type. Handle both cases.
            if isinstance(t, type):
                t = t.__module__ + "." + t.__qualname__
            types.append(t)
        return tp_str in types

    return module_type_filter


def get_not_module_type_or_name_filter(
    tp_list: List[Callable], module_name_list: List[str]
) -> Callable[[torch.fx.Node], bool]:
    module_type_filters = [get_module_type_filter(tp) for tp in tp_list]
    module_name_list_filters = [_get_module_name_filter(m) for m in module_name_list]

    def not_module_type_or_name_filter(n: torch.fx.Node) -> bool:
        return not any(f(n) for f in module_type_filters + module_name_list_filters)

    return not_module_type_or_name_filter


def get_input_act_qspec(quantization_config: Optional[QuantizationConfig]):
    if quantization_config is None:
        return None
    if quantization_config.input_activation is None:
        return None
    quantization_spec: QuantizationSpec = quantization_config.input_activation
    assert quantization_spec.qscheme in [
        torch.per_tensor_affine,
    ]
    return quantization_spec


def get_output_act_qspec(quantization_config: Optional[QuantizationConfig]):
    if quantization_config is None:
        return None
    if quantization_config.output_activation is None:
        return None
    quantization_spec: QuantizationSpec = quantization_config.output_activation
    assert quantization_spec.qscheme in [
        torch.per_tensor_affine,
    ]
    return quantization_spec


def get_weight_qspec(quantization_config: Optional[QuantizationConfig]):
    if quantization_config is None:
        return None
    if quantization_config.weight is None:
        return None
    quantization_spec: QuantizationSpec = quantization_config.weight
    if quantization_spec.qscheme not in [
        torch.per_tensor_affine,
        torch.per_channel_affine,
    ]:
        raise ValueError(
            f"Unsupported quantization_spec {quantization_spec} for weight"
        )
    return quantization_spec


def get_bias_qspec(quantization_config: Optional[QuantizationConfig]):
    if quantization_config is None:
        return None
    if quantization_config.bias is None:
        return None
    quantization_spec: QuantizationSpec = quantization_config.bias
    return quantization_spec


def is_annotated(nodes: List[torch.fx.Node] | torch.fx.Node):
    """
    Check if any of the node in the given list is annotated.
    """
    annotated = False
    if isinstance(nodes, torch.fx.Node):
        nodes = [nodes]
    for node in nodes:
        annotated = annotated or (
            "quantization_annotation" in node.meta
            and node.meta["quantization_annotation"]._annotated
        )
    return annotated
