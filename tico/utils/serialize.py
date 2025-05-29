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

from typing import Optional

import torch

from tico.serialize.circle_graph import CircleSubgraph
from tico.utils.graph import get_module_name_chain


def finalise_tensor_names(
    graph: CircleSubgraph,
) -> None:
    """
    Replace every `tensor.name` with the *readable* version
     **after** the graph is fully built.

    Why late?
    ---------
    - All intermediate steps (add_input, add_output, get_tid…) rely on the
       original technical names in ExportedProgram.

    The rewrite is *in-place* and touches **only** the `name` field of
    each tensor.
    """
    assert hasattr(graph, "name_to_node")

    for tensor in graph.tensors:
        if tensor.name in graph.name_to_node:
            tensor.name = f"{get_module_name_chain(graph.name_to_node[tensor.name])}::{tensor.name}"
