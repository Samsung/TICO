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

from typing import Dict, Type, TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    import torch.fx
import torch
from circle_schema import circle

from tico.serialize.circle_graph import CircleSubgraph
from tico.serialize.operators.hashable_opcode import OpCode


class NodeVisitor:
    """
    Node visitor for lowering edge IR to circle
    """

    def __init__(self, op_codes: Dict[OpCode, int], graph: CircleSubgraph):
        # For setting opcode index in circle model
        # This is updated during serialization
        self._op_codes = op_codes
        self.graph = graph

    # Define circle model operator
    def define_node(
        self,
        node: torch.fx.node.Node,
    ) -> circle.Operator.OperatorT:
        raise NotImplementedError("NodeVisitor must be extended.")


TorchOperator: TypeAlias = torch._ops.OpOverload | torch._ops.OpOverloadPacket


# container for all node visitors
_node_visitor_dict: Dict[TorchOperator, Type[NodeVisitor]] = {}


# Decorator for each visitor
def register_node_visitor(visitor):
    for target in visitor.target:
        _node_visitor_dict[target] = visitor
    return visitor


def get_node_visitor(target: TorchOperator) -> Type[NodeVisitor]:
    """
    Get a single node visitor (for unittest purpose)
    """
    _visitor = _node_visitor_dict.get(target, None)

    if not _visitor:
        raise LookupError(f"NodeVisitor for {target} is not registered")

    return _visitor


# Get all node visitors
def get_node_visitors(
    op_codes: Dict[OpCode, int], graph: CircleSubgraph
) -> Dict[TorchOperator, NodeVisitor]:
    node_visitors = {}
    for target, visitor in _node_visitor_dict.items():
        node_visitors[target] = visitor(op_codes, graph)

    return node_visitors


def get_support_targets():
    return _node_visitor_dict.keys()
