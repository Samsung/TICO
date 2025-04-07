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

from tico.passes import ops
from tico.serialize.circle_mapping import extract_shape
from tico.utils import logging
from tico.utils.passes import PassBase, PassResult
from tico.utils.trace_decorators import trace_graph_diff_on_pass
from tico.utils.utils import set_new_meta_val
from tico.utils.validate_args_kwargs import ReshapeArgs


def passes():
    """
    Return list of passes that remove redundant `aten.reshape` operators.
    """
    return [
        RemoveRedundantReshapePattern1(),
        RemoveRedundantReshapePattern2(),
        RemoveRedundantReshapePattern3(),
        RemoveRedundantReshapePattern4(),
        RemoveRedundantReshapePattern5(),
    ]


@trace_graph_diff_on_pass
class RemoveRedundantReshapePattern1(PassBase):
    mul_ops: List[torch._ops.OpOverload] = ops.aten.mul_scalar + ops.aten.mul_tensor

    def __init__(self):
        super().__init__()

    def call(self, exported_program: ExportedProgram) -> PassResult:
        """
        [BEFORE]
            `(AxBxC) - aten.reshape` - (1xAxBxC) - `aten.permute` - (1xAxCxB) - `aten.mul` - (1xAxCxB) - `aten.reshape - (AxCxB)`
        [AFTER]
            `(AxBxC) - `aten.permute` - (AxCxB) - `aten.mul` - (AxCxB)`
        """
        logger = logging.getLogger(__name__)

        graph_module = exported_program.graph_module
        graph = graph_module.graph
        modified = False
        for reshape1 in graph.nodes:
            if not reshape1.op == "call_function":
                continue

            ### first reshape
            if not reshape1.target in ops.aten.reshape:
                continue
            # Assumes that other node do not use ops in the pattern for simplisity.
            if len(reshape1.users) != 1:
                continue
            assert len(reshape1.args) == 2, len(reshape1.args)
            reshape1_input, reshape1_size = reshape1.args
            # `(AxBxC) - aten.reshape` - (1xAxBxC)
            if [1] + list(extract_shape(reshape1_input)) != list(
                extract_shape(reshape1)
            ):
                continue

            ### permute
            permute = next(iter(reshape1.users))
            if not permute.target in ops.aten.permute:
                continue
            if len(permute.users) != 1:
                continue
            assert len(permute.args) == 2, len(permute.args)
            permute_input, permute_dims = permute.args
            # (1xAxBxC) - `aten.permute` - (1xAxCxB)
            if permute_dims != [0, 1, 3, 2]:
                continue

            ### mul
            mul = next(iter(permute.users))
            if not mul.target in RemoveRedundantReshapePattern1.mul_ops:
                continue
            if len(mul.users) != 1:
                continue

            ### second reshape
            reshape2 = next(iter(mul.users))
            if not reshape2.target in ops.aten.reshape:
                continue
            if len(reshape2.users) != 1:
                continue
            reshape2_input, reshape2_size = reshape2.args
            # (1xAxCxB) - `aten.reshape - (AxCxB)
            if list(extract_shape(reshape2_input)) != [1] + list(
                extract_shape(reshape2)
            ):
                continue

            ### remove redundant reshapes
            # update permute (remove reshape1)
            permute.args = (reshape1_input, [0, 2, 1])
            set_new_meta_val(permute)
            set_new_meta_val(mul)
            # remove reshape2
            reshape2.replace_all_uses_with(mul, propagate_meta=False)

            modified = True
            logger.debug(f"{reshape1.name} and {reshape2.name} are removed.")

        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()

        return PassResult(modified)


@trace_graph_diff_on_pass
class RemoveRedundantReshapePattern2(PassBase):
    def __init__(self):
        super().__init__()

    def call(self, exported_program: ExportedProgram) -> PassResult:
        """
        [BEFORE]
            `(AxBxC) - aten.reshape` - (1xAxBxC) - `aten.permute` - (Bx1xAxC) - `aten.reshape - (Bx(A*C))`
        [AFTER]
            `(AxBxC) - `aten.permute` - (BxAxC) - `aten.reshape` - (Bx(A*C))`
        """
        logger = logging.getLogger(__name__)

        graph_module = exported_program.graph_module
        graph = graph_module.graph
        modified = False
        for reshape1 in graph.nodes:
            if not reshape1.op == "call_function":
                continue

            ### first reshape
            if not reshape1.target in ops.aten.reshape:
                continue
            if len(reshape1.users) != 1:
                continue
            assert len(reshape1.args) == 2, len(reshape1.args)
            reshape1_input, reshape1_size = reshape1.args
            # `(AxBxC) - aten.reshape` - (1xAxBxC)
            if [1] + list(extract_shape(reshape1_input)) != list(
                extract_shape(reshape1)
            ):
                continue

            ### permute
            permute = next(iter(reshape1.users))
            if not permute.target in ops.aten.permute:
                continue
            if len(permute.users) != 1:
                continue
            assert len(permute.args) == 2, len(permute.args)
            permute_input, permute_dims = permute.args
            # (1xAxBxC) - `aten.permute` - (Bx1xAxC)
            if permute_dims != [2, 0, 1, 3]:
                continue

            ### second reshape
            reshape2 = next(iter(permute.users))
            if not reshape2.target in ops.aten.reshape:
                continue
            if len(reshape2.users) != 1:
                continue
            reshape2_input, reshape2_size = reshape2.args
            # (Bx1xAxC) - `aten.reshape - (Bx(A*C))
            reshape2_input_shape = list(extract_shape(reshape2_input))
            assert len(reshape2_input_shape) == 4
            if list(extract_shape(reshape2)) != [
                reshape2_input_shape[0],
                (reshape2_input_shape[2] * reshape2_input_shape[3]),
            ]:
                continue

            ### remove redundant reshapes
            # update permute (remove reshape1)
            permute.args = (reshape1_input, [1, 0, 2])
            set_new_meta_val(permute)
            reshape1.replace_all_uses_with(permute, propagate_meta=False)
            # update reshape2 args
            assert permute == reshape2_input
            reshape2.args = (permute, reshape2_size)

            modified = True
            logger.debug(f"{reshape1.name} is removed.")

        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()

        return PassResult(modified)


@trace_graph_diff_on_pass
class RemoveRedundantReshapePattern3(PassBase):
    def __init__(self):
        super().__init__()

    def call(self, exported_program: ExportedProgram) -> PassResult:
        """
        [BEFORE]
            (AxBxC) - aten.reshape - (1xAxBxC) - aten.add - (1xAxBxC) - aten.softmax - (1xAxBxC) - aten.reshape - (AxBxC)
                      (reshape_2)                 (add)                 (softmax)                  (reshape_1)
            (AxBxC) - aten.reshape - (1xAxBxC) /
                      (reshape_3)
        [AFTER]
            (AxBxC) - aten.add - (AxBxC) - aten.softmax - (AxBxC)
            (AxBxC) /   (add)                (softmax)
        """
        logger = logging.getLogger(__name__)

        graph_module = exported_program.graph_module
        graph = graph_module.graph
        modified = False
        for reshape_1 in graph.nodes:
            assert isinstance(reshape_1, torch.fx.Node), type(reshape_1)
            # reshape_1
            if not reshape_1.op == "call_function":
                continue
            if not reshape_1.target in ops.aten.reshape:
                continue
            assert len(reshape_1.args) == 2, len(reshape_1.args)
            softmax, reshape_1_size = reshape_1.args

            # softmax
            assert isinstance(softmax, torch.fx.Node), type(softmax)
            if not softmax.op == "call_function":
                continue
            if not softmax.target in ops.aten.softmax:
                continue
            assert len(softmax.args) == 3, len(softmax.args)
            add, softmax_dim, softmax_half_to_float = softmax.args
            assert isinstance(add, torch.fx.Node), type(add)
            assert isinstance(softmax_dim, int), type(softmax_dim)
            assert isinstance(softmax_half_to_float, bool), type(softmax_half_to_float)
            softmax_shape = extract_shape(softmax)
            # TODO support other dimension
            if softmax_dim != -1 and softmax_dim != len(softmax_shape) - 1:
                continue

            # add
            if not add.target in ops.aten.add:
                continue
            assert len(add.args) == 2, len(add.args)
            reshape_2, reshape_3 = add.args
            assert isinstance(reshape_2, torch.fx.Node), type(reshape_2)
            assert isinstance(reshape_3, torch.fx.Node), type(reshape_3)

            # reshape_2
            if not reshape_2.op == "call_function":
                continue
            if not reshape_2.target in ops.aten.reshape:
                continue
            assert len(reshape_2.args) == 2, len(reshape_2.args)
            reshape_2_input, reshape_2_size = reshape_2.args
            assert isinstance(reshape_2_input, torch.fx.Node), type(reshape_2_input)
            # reshape_3
            if not reshape_3.op == "call_function":
                continue
            if not reshape_3.target in ops.aten.reshape:
                continue
            assert len(reshape_3.args) == 2, len(reshape_3.args)
            reshape_3_input, reshape_3_size = reshape_3.args
            assert isinstance(reshape_3_input, torch.fx.Node), type(reshape_3_input)

            # Check condition
            reshape_2_input_shape = extract_shape(reshape_2_input)
            reshape_3_input_shape = extract_shape(reshape_3_input)
            if reshape_2_input_shape != reshape_3_input_shape:
                continue
            reshape_1_shape = extract_shape(reshape_1)
            if reshape_2_input_shape != reshape_1_shape:
                continue
            # Assume `aten.add` and `aten.softmax` have only one user.
            if len(add.users) != 1:
                continue
            if len(softmax.users) != 1:
                continue

            # Update add
            add.args = (reshape_2_input, reshape_3_input)
            set_new_meta_val(add)
            # Update softmax
            if softmax_dim == len(softmax_shape) - 1:
                updated_dim = len(extract_shape(reshape_2_input)) - 1
                softmax.args = (add, updated_dim, softmax_half_to_float)
            set_new_meta_val(softmax)

            reshape_1.replace_all_uses_with(softmax, propagate_meta=False)
            modified = True
            logger.debug(
                f"{reshape_2.name}, {reshape_3.name} and {reshape_1.name} are removed."
            )

        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()

        return PassResult(modified)


@trace_graph_diff_on_pass
class RemoveRedundantReshapePattern4(PassBase):
    def __init__(self):
        super().__init__()

    def call(self, exported_program: ExportedProgram) -> PassResult:
        """
        NOTE: Below graph is just an example. This pattern matches not only for the 3D tensors.
        What this pattern aims to remove is that the consecutive `aten.reshape` ops.
        [BEFORE]
            (AxBxC) - aten.reshape - (AxB'xC') - aten.reshape - (A'xB''xC')
        [AFTER]
            (AxBxC) - aten.reshape - (A'xB''xC')
        """
        logger = logging.getLogger(__name__)

        graph_module = exported_program.graph_module
        graph = graph_module.graph
        modified = False
        for reshape1 in graph.nodes:
            # reshape_1
            if not reshape1.op == "call_function":
                continue
            if not reshape1.target in ops.aten.reshape:
                continue
            assert len(reshape1.args) == 2, len(reshape1.args)

            reshape1_input, size = list(reshape1.args)
            assert isinstance(reshape1_input, torch.fx.Node), type(reshape1_input)
            assert isinstance(size, list), type(size)
            for s in size:
                assert isinstance(s, int), type(s)

            if not len(reshape1.users) == 1:
                continue

            # reshape_2
            reshape2 = next(iter(reshape1.users))
            if not reshape2.op == "call_function":
                continue
            if not reshape2.target in ops.aten.reshape:
                continue
            assert len(reshape2.args) == 2, len(reshape2.args)

            reshape2_input, reshape2_size = list(reshape2.args)
            assert isinstance(reshape2_input, torch.fx.Node), type(reshape2_input)
            assert isinstance(reshape2_size, list), type(reshape2_size)
            for s in reshape2_size:
                assert isinstance(s, int), type(s)

            with graph.inserting_before(reshape1):
                fused_reshape = graph.call_function(
                    reshape1.target, (reshape1_input, reshape2_size)
                )

            reshape2.replace_all_uses_with(fused_reshape, propagate_meta=True)

            modified = True
            logger.debug(
                f"{reshape1.name} and {reshape2.name} are fused to {fused_reshape.name}"
            )

        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()

        return PassResult(modified)


@trace_graph_diff_on_pass
class RemoveRedundantReshapePattern5(PassBase):
    def __init__(self):
        super().__init__()

    def call(self, exported_program: ExportedProgram) -> PassResult:
        """
        [BEFORE]
            (AxBxC) - aten.reshape - (AxBxC)
        [AFTER]
            (AxBxC)
        """
        logger = logging.getLogger(__name__)

        graph_module = exported_program.graph_module
        graph = graph_module.graph
        modified = False

        for node in graph.nodes:
            if not node.op == "call_function":
                continue

            if not node.target in ops.aten.reshape:
                continue

            args = ReshapeArgs(*node.args)
            output_shape = args.size
            input_shape = list(extract_shape(args.input))

            if output_shape != input_shape:
                continue

            with graph.inserting_after(node):
                node.replace_all_uses_with(args.input, propagate_meta=False)

            modified = True
            logger.debug(f"{node.name} is replaced with {args.input}")

        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()

        return PassResult(modified)
