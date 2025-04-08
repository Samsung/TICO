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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.fx
import torch
from torch.export import ExportedProgram

from tico.passes import ops
from tico.serialize.circle_mapping import extract_shape, extract_stride
from tico.utils import logging
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass


def passes():
    """
    Return a list of passes that remove redundant `aten.permute` operators.

    NOTE Both shape and stride of input/output should be same.
    """
    return [
        RemoveRedundantPermutePattern1(),
    ]


@trace_graph_diff_on_pass
class RemoveRedundantPermutePattern1(PassBase):
    def __init__(self):
        super().__init__()

    def call(self, exported_program: ExportedProgram) -> PassResult:
        """
        [BEFORE]
            (AxBxC) - aten.permute - aten.permute - (AxBxC)
        [AFTER]
            (AxBxC)
        """
        logger = logging.getLogger(__name__)

        graph_module = exported_program.graph_module
        graph = graph_module.graph
        modified = False
        for permute2 in graph.nodes:
            if not permute2.op == "call_function":
                continue
            if not permute2.target in ops.aten.permute:
                continue
            if len(permute2.users) != 1:
                continue
            assert len(permute2.args) == 2
            permute1, permute2_dims = permute2.args
            assert isinstance(permute1, torch.fx.Node), type(permute1)
            assert isinstance(permute2_dims, list), type(permute2_dims)
            for dim in permute2_dims:
                assert isinstance(dim, int), type(dim)

            if not permute1.target in ops.aten.permute:
                continue
            if len(permute1.users) != 1:
                continue
            assert len(permute1.args) == 2
            permute1_input, permute1_dims = permute1.args
            assert isinstance(permute1_input, torch.fx.Node), type(permute1_input)
            assert isinstance(permute1_dims, list), type(permute1_dims)
            for dim in permute1_dims:
                assert isinstance(dim, int), type(dim)

            # shape
            permute1_input_shape = extract_shape(permute1_input)
            permute2_shape = extract_shape(permute2)
            if permute1_input_shape != permute2_shape:
                continue
            # stride
            permute1_input_stride = extract_stride(permute1_input)
            permute2_stride = extract_stride(permute2)
            if permute1_input_stride != permute2_stride:
                continue

            permute2.replace_all_uses_with(permute1_input, propagate_meta=False)

            modified = True
            logger.debug(f"{permute1.name} and {permute2.name} are removed.")

        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()

        return PassResult(modified)
