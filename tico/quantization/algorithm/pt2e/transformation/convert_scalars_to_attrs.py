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

import torch
from torch.ao.quantization.fx.utils import get_new_attr_name_with_prefix


def convert_scalars_to_attrs(model: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    Convert scalar values in the graph to `get_attr` nodes.

    This function identifies scalar constants in the graph and transforms them
     into `get_attr` nodes to ensure compatibility with quantization workflows.
    """
    for n in model.graph.nodes:
        if n.op != "call_function" or n.target not in [
            # The operators that have scalar parameters.
            torch.ops.aten.add.Tensor,
        ]:
            continue
        args = list(n.args)
        new_args = []
        for arg in args:
            if isinstance(arg, torch.fx.Node):
                new_args.append(arg)
                continue

            assert isinstance(arg, float)
            prefix = "_tensor_constant_"
            get_new_attr_name = get_new_attr_name_with_prefix(prefix)
            tensor_constant_name = get_new_attr_name(model)
            float_tensor = torch.tensor(float(arg))
            model.register_buffer(tensor_constant_name, float_tensor)

            fake_mode = n.meta["val"].fake_mode
            with model.graph.inserting_before(n):
                get_attr_node = model.graph.create_node(
                    "get_attr", tensor_constant_name, (), {}
                )
                get_attr_node.meta["val"] = fake_mode.from_tensor(
                    float_tensor, static_shapes=True
                )
                new_args.append(get_attr_node)
        n.args = tuple(new_args)
    model.recompile()

    return model
