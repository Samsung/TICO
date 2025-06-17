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

from typing import Union

import torch
from torch.export import ExportedProgram

from tico.passes import ops
from tico.serialize.circle_mapping import extract_torch_dtype
from tico.utils import logging
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass
from tico.utils.utils import is_target_node
from tico.utils.validate_args_kwargs import ToCopyArgs, ToDtypeArgs, ToDtypeLayoutArgs


@trace_graph_diff_on_pass
class RemoveRedundantToCopy(PassBase):
    """
    This pass removes redundant `aten._to_copy` operators.
    """

    def __init__(self):
        super().__init__()

    def call(self, exported_program: ExportedProgram) -> PassResult:
        logger = logging.getLogger(__name__)

        graph_module = exported_program.graph_module
        graph = graph_module.graph
        modified = False
        for node in graph.nodes:
            if not is_target_node(node, ops.aten.to_copy):
                continue

            args: Union[ToCopyArgs, ToDtypeArgs, ToDtypeLayoutArgs]
            if node.target == torch.ops.aten._to_copy.default:
                args = ToCopyArgs(*node.args, **node.kwargs)
            elif node.target == torch.ops.aten.to.dtype:
                args = ToDtypeArgs(*node.args, **node.kwargs)
            elif node.target == torch.ops.aten.to.dtype_layout:
                args = ToDtypeLayoutArgs(*node.args, **node.kwargs)
            else:
                raise NotImplementedError(
                    f"Unsupported to_copy operator: {node.target}"
                )

            input_ = args.input
            # https://pytorch.org/docs/stable/tensor_attributes.html#torch-layout
            # layout is two types: torch.strided(dense Tensors), torch.sparse_coo(sparse COO Tensors)
            if hasattr(args, "layout") and args.layout is not None:
                if args.layout != input_.meta["val"].layout:
                    continue

            if hasattr(args, "dtype") and args.dtype is not None:
                target_dtype = args.dtype
                input_dtype = extract_torch_dtype(input_)
                if input_dtype != target_dtype:
                    continue

            if hasattr(args, "memory_format") and args.memory_format is not None:
                if args.memory_format != torch.contiguous_format:
                    continue

            node.replace_all_uses_with(input_, propagate_meta=False)

            modified = True
            logger.debug(f"{node.name} is replaced with {input_.name}")

        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()

        return PassResult(modified)
