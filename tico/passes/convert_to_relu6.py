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

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    import torch.fx
import torch
from torch.export import ExportedProgram

from tico.utils import logging
from tico.utils.graph import create_node
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass
from tico.utils.validate_args_kwargs import ClampArgs, HardTanhArgs


class Converter:  # type: ignore[empty-body]
    def __init__(self):
        super().__init__()

    def match(self, node) -> bool:  # type: ignore[empty-body]
        return False

    def convert(self, exported_program, node) -> None:  # type: ignore[empty-body]
        pass


class ConvertHardTanhToReLU6(Converter):
    def __init__(self):
        super().__init__()

    def match(self, node) -> bool:
        if node.target == torch.ops.aten.hardtanh.default:
            args = HardTanhArgs(*node.args, **node.kwargs)
            min_val = args.min_val
            max_val = args.max_val

            # NOTE: int and float are both covered by pytorch implicit type conversion
            return min_val == 0.0 and max_val == 6.0

        return False

    def convert(self, exported_program, node):
        graph_module = exported_program.graph_module
        graph = graph_module.graph
        args = HardTanhArgs(*node.args, **node.kwargs)
        input = args.input

        with graph.inserting_after(node):
            relu_node = create_node(graph, torch.ops.aten.relu6.default, args=(input,))
            node.replace_all_uses_with(relu_node, propagate_meta=True)


class ConvertClampToReLU6(Converter):
    def __init__(self):
        super().__init__()

    def match(self, node) -> bool:
        if node.target == torch.ops.aten.clamp.default:
            args = ClampArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]
            min_val = args.min
            max_val = args.max

            # NOTE: int and float are both covered by pytorch implicit type conversion
            return min_val == 0 and max_val == 6

        return False

    def convert(self, exported_program, node):
        graph_module = exported_program.graph_module
        graph = graph_module.graph
        args = ClampArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]
        input = args.input

        with graph.inserting_after(node):
            relu_node = create_node(graph, torch.ops.aten.relu6.default, args=(input,))
            node.replace_all_uses_with(relu_node, propagate_meta=True)


class ConvertDoubleClampsToReLU6(Converter):
    def __init__(self):
        super().__init__()

    def match(self, node) -> bool:
        """
        This pass matches the pattern of two clamps where it equals to clamp which has a min value of 0 and a max value of 6.

                    (equivalent)
        input                        input
        |                            |
        node_prev (min, max)         node   (0, 6)
        |                            |
        node      (min', max')       |
        |                            |
        output                       output

        *where max(min, min') == 0 and min(max, max') == 6 so that it equivalents to clamp(input, 0, 6)

        TODO Make this step more generic. For now we only support the case above.
        """
        if not node.target == torch.ops.aten.clamp.default:
            return False

        args = ClampArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]
        node_prev = args.input
        min_val = args.min if args.min is not None else float("-inf")
        max_val = args.max if args.max is not None else float("inf")

        if not node_prev.target == torch.ops.aten.clamp.default:
            return False

        prev_args = ClampArgs(*node_prev.args, **node_prev.kwargs)  # type: ignore[arg-type]
        min_val_prev = prev_args.min if prev_args.min is not None else float("-inf")
        max_val_prev = prev_args.max if prev_args.max is not None else float("inf")

        # NOTE: int and float are both covered by pytorch implicit type conversion
        if max(min_val, min_val_prev) == 0 and min(max_val, max_val_prev) == 6:
            return True

        return False

    def convert(self, exported_program, node):
        graph_module = exported_program.graph_module
        graph = graph_module.graph
        args = ClampArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]

        prev_node = args.input
        prev_args = ClampArgs(*prev_node.args, **prev_node.kwargs)  # type: ignore[arg-type]
        input = prev_args.input

        with graph.inserting_after(node):
            relu_node = create_node(graph, torch.ops.aten.relu6.default, args=(input,))
            node.replace_all_uses_with(relu_node, propagate_meta=True)


@trace_graph_diff_on_pass
class ConvertToReLU6(PassBase):
    def __init__(self):
        super().__init__()
        self.converters: List[Converter] = [
            ConvertHardTanhToReLU6(),
            ConvertClampToReLU6(),
            ConvertDoubleClampsToReLU6(),
        ]

    def call(self, exported_program: ExportedProgram) -> PassResult:
        logger = logging.getLogger(__name__)

        graph_module = exported_program.graph_module
        graph = graph_module.graph
        modified = False
        for node in graph.nodes:
            if not node.op == "call_function":
                continue

            for converter in self.converters:
                if not converter.match(node):
                    continue

                converter.convert(exported_program, node)
                modified = True
                logger.debug(f"{node.name} is replaced with ReLU6 operator")
                break

        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()

        return PassResult(modified)
