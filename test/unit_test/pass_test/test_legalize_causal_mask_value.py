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
from tico.passes import ops
from tico.passes.const_prop_pass import ConstPropPass

from tico.passes.legalize_causal_mask_value import LegalizeCausalMaskValue
from tico.utils.validate_args_kwargs import AddTensorArgs

from test.modules.op.add import AddWithCausalMaskFolded

from test.utils.pass_value_test import PassTest


def get_mask_value(exported_program):
    mask_node = None
    for node in exported_program.graph.nodes:
        if node.op == "call_function" and node.target in ops.aten.add:
            args = AddTensorArgs(*node.args, **node.kwargs)
            mask_node = args.input

    assert mask_node is not None
    assert isinstance(mask_node, torch.fx.Node)
    mask_name = exported_program.graph_signature.inputs_to_lifted_tensor_constants[
        mask_node.name
    ]

    mask_value = exported_program.constants[mask_name]
    return mask_value


class LegalizeCausalMaskValueTest(PassTest):
    def test_pass(self):
        self.setup(AddWithCausalMaskFolded())
        self.run_pass(ConstPropPass())
        self.run_pass(LegalizeCausalMaskValue(enabled=True))

        mask_data = get_mask_value(exported_program=self.exported_program())

        self.assertEqual(
            torch.all(torch.logical_or(mask_data == 0, mask_data == -120)), True
        )
