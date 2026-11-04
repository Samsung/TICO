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

from tico.serialize.circle_mapping import extract_torch_dtype
from tico.utils import logging
from tico.utils.graph import create_node
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass
from tico.utils.utils import is_target_node, set_new_meta_val
from tico.utils.validate_args_kwargs import PowTensorScalarArgs


@trace_graph_diff_on_pass
class LowerPow2ToMul(PassBase):
    """
    Lower scalar powers equal to two to multiplication in the result dtype.

    The original output metadata owns scalar promotion: integer bases with a
    floating exponent and boolean bases may require conversion before squaring.
    Casting after multiplication would preserve integer overflow incorrectly.
    Other exponents and tensor-exponent overloads are unchanged.
    """

    def __init__(self):
        super().__init__()

    def call(self, exported_program: ExportedProgram) -> PassResult:
        logger = logging.getLogger(__name__)

        graph_module = exported_program.graph_module
        graph = graph_module.graph
        modified = False
        for node in graph.nodes:
            if not is_target_node(node, torch.ops.aten.pow.Tensor_Scalar):
                continue

            args = PowTensorScalarArgs(*node.args, **node.kwargs)  # type: ignore[arg-type]
            in_, exp = args.input, args.exponent

            if exp != 2:
                continue

            result_dtype = extract_torch_dtype(node)
            with graph.inserting_before(node):
                if extract_torch_dtype(in_) != result_dtype:
                    in_ = create_node(
                        graph,
                        torch.ops.aten._to_copy.default,
                        args=(in_,),
                        kwargs={"dtype": result_dtype},
                        origin=node,
                    )
                    set_new_meta_val(in_)
                new_mul = create_node(
                    graph,
                    torch.ops.aten.mul.Tensor,
                    args=(in_, in_),
                    kwargs={},
                )

            node.replace_all_uses_with(new_mul, propagate_meta=True)
            set_new_meta_val(new_mul)

            modified = True
            logger.debug(f"{node.name} is replaced with {new_mul.name}")

        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()

        return PassResult(modified)
