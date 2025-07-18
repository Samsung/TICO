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

"""
Runtime **capability-detection helpers** for the `torch.export` stack.

Instead of sprinkling version checks like `torch.__version__ >= "2.9"` throughout
the codebase, import these helpers once and branch on the feature you need.

Each probe executes only **once per process** thanks to `functools.lru_cache`,
so the overhead is negligible.
"""

import functools

import torch


@functools.lru_cache(maxsize=None)
def export_produces_slice() -> bool:
    """
    Compile a minimal model with `torch.export.export` and inspect its FX graph
    to see whether an `aten.slice.Tensor` node appears.

    Returns
    -------
    bool
        * ``True``  — downstream passes should expect redundant **slice** nodes.
        * ``False`` — downstream passes should expect only a **select** node.
    """

    class _Probe(torch.nn.Module):
        def forward(self, x):  # simple slice: keep all dims except 3rd
            return x[:, :, 1]

        def get_example_inputs(self):
            return (torch.randn(1, 4, 4),)

    m = _Probe()
    ep = torch.export.export(m, m.get_example_inputs())
    return any(n.target == torch.ops.aten.slice.Tensor for n in ep.graph.nodes)
