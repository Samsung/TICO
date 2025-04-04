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

from collections import defaultdict
from typing import Any, cast, Dict, final, List, Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:
    import torch.fx
import numpy as np
import torch
from circle_schema import circle
from torch._subclasses.fake_tensor import FakeTensor

from tico.serialize.circle_mapping import (
    extract_circle_dtype,
    extract_shape,
    str_to_circle_dtype,
    to_circle_dtype,
)
from tico.serialize.pack import pack_buffer
from tico.serialize.quant_param import QPARAM_KEY
from tico.utils.utils import to_circle_qparam

"""
Type alias for const
"""
_PRIMITIVE_TYPES = (
    float,
    int,
    bool,
    str,
    torch.Tensor,
    torch.device,
    torch.dtype,
    torch.layout,
)
ConstDataElement = Union[
    int, float, bool, str, torch.Tensor, torch.device, torch.dtype, torch.layout
]
ConstData = Union[ConstDataElement, List[ConstDataElement]]


def is_const(arg) -> bool:
    if isinstance(arg, FakeTensor):
        return False
    if isinstance(arg, _PRIMITIVE_TYPES):
        return True
    if isinstance(arg, (tuple, list)):
        return all(map(is_const, arg))
    if isinstance(arg, dict):
        return all(map(is_const, arg.values()))
    return False


@final
class CircleModel(circle.Model.ModelT):
    def __init__(self):
        super().__init__()
        self.subgraphs: List[circle.SubGraph.SubGraphT] = []
        self.buffers: List[circle.Buffer.BufferT] = []

    def add_subgraph(self, graph: circle.SubGraph.SubGraphT) -> None:
        self.subgraphs.append(graph)

    def add_buffer(self, buffer: circle.Buffer.BufferT) -> int:
        """Return buffer id"""
        self.buffers.append(buffer)
        buf_id = len(self.buffers) - 1  # last index
        return buf_id


@final
class CircleSubgraph(circle.SubGraph.SubGraphT):
    def __init__(self, model: CircleModel):
        super().__init__()
        self.model: CircleModel = model
        self.name: str = "subgraph"
        self.inputs: List[int] = []
        self.outputs: List[int] = []
        self.tensors: List[circle.Tensor.TensorT] = []
        self.operators: List[circle.Operator.OperatorT] = []
        self.name_to_tid: Dict[str, int] = {}
        self.counter: defaultdict = defaultdict(int)

    # Generate a unique name with prefix.
    # Naming rule
    # - If no tensor has the same name with prefix, return prefix
    # - Otherwise, add postfix f"_{idx}" where idx increases by 1 from 0
    # Example
    # If prefix = "add", this function will find a unique name in the following order.
    # "add", "add_0", "add_1", ...
    def _gen_unique_name_with_prefix(self, prefix: str):
        name = prefix
        while self.has_tensor(name):
            index = self.counter[prefix]
            name = f"{prefix}_{index}"
            self.counter[prefix] += 1

        return name

    def _add_tensor(self, tensor: circle.Tensor.TensorT) -> None:
        self.tensors.append(tensor)
        self.name_to_tid[tensor.name] = len(self.tensors) - 1

    def add_operator(self, op: circle.Operator.OperatorT) -> None:
        self.operators.append(op)

    def add_input(self, input_name: str) -> None:
        assert input_name in self.name_to_tid, f"{input_name}"
        tid = self.name_to_tid[input_name]
        self.inputs.append(tid)

    def add_output(self, output: Any) -> None:
        if isinstance(output, str):
            assert output in self.name_to_tid
            output_name = output
        elif isinstance(output, int | float):
            # output is built-in type.
            circle_tensor = self.add_const_tensor(output)
            output_name = circle_tensor.name
        else:
            raise NotImplementedError(f"Unsupported output dtype: {type(output)}")
        tid = self.name_to_tid[output_name]
        self.outputs.append(tid)

    def has_tensor(self, name: str):
        return name in self.name_to_tid

    def add_tensor_from_node(
        self, node: torch.fx.node.Node, data: Optional[np.ndarray] = None
    ) -> None:
        tensor = circle.Tensor.TensorT()
        tensor.name = self._gen_unique_name_with_prefix(node.name)
        assert node.meta.get("val") is not None
        tensor.type = extract_circle_dtype(node)
        tensor.shape = list(extract_shape(node))
        if QPARAM_KEY in node.meta:
            tensor.quantization = to_circle_qparam(node.meta[QPARAM_KEY])
            tensor.type = str_to_circle_dtype(node.meta[QPARAM_KEY].dtype)

        buffer = circle.Buffer.BufferT()
        if data is not None and isinstance(data, np.ndarray):
            data = data.flatten()

            if QPARAM_KEY in node.meta:
                if node.meta[QPARAM_KEY].dtype == "uint4":
                    data = pack_buffer(data, "uint4")

            # Packing np.ndarray is faster than packing bytes
            buffer.data = data.view(np.uint8)  # type: ignore[assignment]
        else:
            assert data is None
        bid = self.model.add_buffer(buffer)
        tensor.buffer = bid
        self._add_tensor(tensor)

    def add_const_tensor(self, data: ConstData) -> circle.Tensor.TensorT:
        assert is_const(data)
        tensor = circle.Tensor.TensorT()
        tensor.name = self._gen_unique_name_with_prefix("const_tensor")
        assert not self.has_tensor(tensor.name)
        torch_t = torch.as_tensor(data=data)
        torch_t_shape = list(torch_t.size())
        tensor.type = to_circle_dtype(torch_dtype=torch_t.dtype)
        tensor.shape = torch_t_shape

        buffer = circle.Buffer.BufferT()
        buffer.data = torch_t.flatten().cpu().numpy().view(np.uint8)  # type: ignore[assignment]
        bid = self.model.add_buffer(buffer)
        tensor.buffer = bid
        self._add_tensor(tensor)

        return tensor

    def add_tensor_from_scratch(
        self, prefix: str, shape: List[int], dtype: int
    ) -> circle.Tensor.TensorT:
        assert isinstance(dtype, int), f"{dtype} must be integer. Use to_circle_dtype."
        tensor = circle.Tensor.TensorT()
        tensor.name = self._gen_unique_name_with_prefix(prefix)
        tensor.type = dtype
        tensor.shape = shape

        buffer = circle.Buffer.BufferT()
        bid = self.model.add_buffer(buffer)
        tensor.buffer = bid
        self._add_tensor(tensor)

        return tensor

    # Some operators like `full`, `arange_start_step` or `scalar_tensor` needs buffers to be in-place updated.
    # TODO remove this function
    def update_tensor_buffer(
        self, data: ConstData, tensor_name: str = str()
    ) -> circle.Tensor.TensorT:
        assert is_const(data)
        assert self.has_tensor(tensor_name)
        data_tensor = torch.as_tensor(data=data)
        data_shape = list(data_tensor.size())
        op_tensor = self.tensors[self.name_to_tid[tensor_name]]
        assert op_tensor.type == to_circle_dtype(
            data_tensor.dtype
        ), f"{op_tensor.type}, {data_tensor.dtype}"
        assert op_tensor.shape == data_shape

        buffer = circle.Buffer.BufferT()
        # Packing np.ndarray is faster than packing bytes
        buffer.data = data_tensor.flatten().cpu().numpy().view(np.uint8)  # type: ignore[assignment]
        bid = self.model.add_buffer(buffer)
        op_tensor.buffer = bid

        return op_tensor

    def get_tid_registered(
        self, node: Union[torch.fx.node.Node, circle.Tensor.TensorT]
    ) -> int:
        assert hasattr(node, "name"), "FIX CALLER UNLESS"

        tid = self.name_to_tid.get(node.name, None)

        if tid is None:
            raise KeyError(f"{node}({node.name}) is not registered.")

        assert tid < len(self.tensors)

        return tid

    def get_tensor(self, node: torch.fx.node.Node) -> circle.Tensor.TensorT:
        tid = self.get_tid_registered(node)

        return self.tensors[tid]

    def get_buffer(self, node: torch.fx.Node) -> circle.Buffer.BufferT:
        buf_id = self.get_tensor(node).buffer
        return self.model.buffers[buf_id]

    # TODO Rename, it doesn't only get_tid but also possibly add a new const tensor
    def get_tid(
        self, node: Union[torch.fx.node.Node, circle.Tensor.TensorT, ConstData]
    ) -> int:
        # return -1 if node is None. This is for generating CircleOutputExclude
        if node == None:
            return -1

        if hasattr(node, "name") and node.name in self.name_to_tid:
            return self.name_to_tid[node.name]

        if is_const(node):
            node_name = self.add_const_tensor(cast(ConstData, node)).name
            return self.name_to_tid[node_name]

        # Unreachable
        raise RuntimeError("fx Node was not converted to tensor.")
