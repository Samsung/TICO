# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved
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

from tico.passes.fuse_rms_norm import FuseRmsNorm

from test.support.helper import num_of_ops
from test.support.pass_value_test import SinglePassValueTest


class RMSNormNet(torch.nn.Module):
    def __init__(self, normalized_shape=16, eps=1e-6):
        super().__init__()
        self.rms_norm = torch.nn.RMSNorm(normalized_shape, eps=eps)

    def forward(self, x):
        return self.rms_norm(x)

    def get_example_inputs(self):
        return (torch.randn(1, 8, 16),), {}


class FuseRmsNormTest(SinglePassValueTest):
    def test_pass(self):
        self.setup(RMSNormNet())
        self.assertEqual(
            num_of_ops(self.exported_program(), [torch.ops.aten.rms_norm.default]), 1
        )
        self.assertEqual(
            num_of_ops(
                self.exported_program(), [torch.ops.circle_custom.rms_norm.default]
            ),
            0,
        )

        self.run_value_test(FuseRmsNorm())

        self.assertEqual(
            num_of_ops(self.exported_program(), [torch.ops.aten.rms_norm.default]),
            0,
        )
        self.assertEqual(
            num_of_ops(
                self.exported_program(), [torch.ops.circle_custom.rms_norm.default]
            ),
            1,
        )
