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

from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch.fx
import torch
from torch.export import ExportedProgram

from tico.serialize.circle_mapping import extract_torch_dtype
from tico.utils import logging
from tico.utils.graph import create_node
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass
from tico.utils.utils import set_new_meta_val
from tico.utils.validate_args_kwargs import ClampArgs, HardTanhArgs


class Converter:
    def match_input(self, node: torch.fx.Node) -> Optional[torch.fx.Node]:
        """Return the input that can replace the matched operation with ReLU6."""
        return None


class ConvertHardTanhToReLU6(Converter):
    def match_input(self, node: torch.fx.Node) -> Optional[torch.fx.Node]:
        if node.target == torch.ops.aten.hardtanh.default:
            args = HardTanhArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]
            min_val = args.min_val
            max_val = args.max_val

            if min_val == 0.0 and max_val == 6.0:
                assert isinstance(args.input, torch.fx.Node)
                return args.input

        return None


class ConvertClampToReLU6(Converter):
    def match_input(self, node: torch.fx.Node) -> Optional[torch.fx.Node]:
        if node.target == torch.ops.aten.clamp.default:
            args = ClampArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]
            min_val = args.min
            max_val = args.max

            if min_val == 0 and max_val == 6:
                return args.input

        return None


class ConvertDoubleClampsToReLU6(Converter):
    def match_input(self, node: torch.fx.Node) -> Optional[torch.fx.Node]:
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
            return None

        args = ClampArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]
        node_prev = args.input
        min_val = args.min if args.min is not None else float("-inf")
        max_val = args.max if args.max is not None else float("inf")

        if not node_prev.target == torch.ops.aten.clamp.default:
            return None

        prev_args = ClampArgs(*node_prev.args, **node_prev.kwargs)  # type: ignore[arg-type]
        min_val_prev = prev_args.min if prev_args.min is not None else float("-inf")
        max_val_prev = prev_args.max if prev_args.max is not None else float("inf")

        if max(min_val, min_val_prev) == 0 and min(max_val, max_val_prev) == 6:
            return prev_args.input

        return None


@trace_graph_diff_on_pass
class ConvertToReLU6(PassBase):
    """Replace equivalent clamps with ReLU6 in their exported result dtype."""

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
                input = converter.match_input(node)
                if input is None:
                    continue

                # Clamp's scalar bounds may promote integer or boolean inputs.
                # The exported result owns that dtype, including fused clamps.
                result_dtype = extract_torch_dtype(node)
                with graph.inserting_before(node):
                    if extract_torch_dtype(input) != result_dtype:
                        input = create_node(
                            graph,
                            torch.ops.aten._to_copy.default,
                            args=(input,),
                            kwargs={"dtype": result_dtype},
                            origin=node,
                        )
                        set_new_meta_val(input)
                    relu_node = create_node(
                        graph, torch.ops.aten.relu6.default, args=(input,)
                    )
                node.replace_all_uses_with(relu_node, propagate_meta=True)
                set_new_meta_val(relu_node)
                modified = True
                logger.debug(f"{node.name} is replaced with ReLU6 operator")
                break

        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()

        return PassResult(modified)
