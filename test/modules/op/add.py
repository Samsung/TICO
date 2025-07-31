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
from tico.config.v1 import CompileConfigV1

from test.modules.base import TestModuleBase

from test.utils import tag


class SimpleAdd(TestModuleBase):
    def __init__(self):
        super().__init__()

        self.quantized_peir_tolerance = 5

    def forward(self, x, y):
        z = x + y
        z = z + x
        z = z + x
        z = z + z
        return z

    def get_example_inputs(self):
        return (
            torch.ones(1),
            torch.ones(1),
        ), {}

    def get_calibration_data(self):
        calibration_data = [
            (
                (
                    torch.randn(
                        (1, 2),
                    ),
                    torch.randn(
                        (1, 2),
                    ),
                ),
                {},
            )
            for i in range(100)
        ]  # type: ignore[var-annotated]
        return calibration_data


@tag.test_without_pt2
class SimpleAddWithoutPt2(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        z = x + y
        z = z + x
        z = z + x
        z = z + z
        return z

    def get_example_inputs(self):
        return (
            torch.ones(1),
            torch.ones(1),
        ), {}


class SimpleAddWithDifferentMemoryFormat(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        z = x + y
        z = z + x
        z = z + x
        z = z + z
        return z

    def get_example_inputs(self):
        return (
            torch.rand(1, 2, 3, 4).clone(memory_format=torch.contiguous_format),
            torch.rand(1, 2, 3, 4).clone(memory_format=torch.channels_last),
        ), {}


class AddWithNonPersistentBuffer(TestModuleBase):
    def __init__(self):
        super().__init__()
        x = torch.Tensor([2.0])
        # buffer is not saved to the `state_dict` when persistent is False.
        self.register_buffer("x", x, persistent=False)

    def forward(self, y):
        z = self.x + y
        return z

    def get_example_inputs(self):
        return (torch.ones(1),), {}


class AddWithBuffer(TestModuleBase):
    def __init__(self):
        super().__init__()
        x = torch.Tensor([2.0])
        self.register_buffer("x", x, persistent=True)

    def forward(self, y):
        z = self.x + y
        return z

    def get_example_inputs(self):
        return (torch.ones(1),), {}


class AddWithBuiltinFloat(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        z = x + y
        return z

    def get_example_inputs(self):
        return (
            torch.ones(1),
            2.0,
        ), {}


class AddWithBuiltinInt(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        z = x + y
        return z

    def get_example_inputs(self):
        return (
            torch.ones(1).to(torch.int64),
            2,
        ), {}


class ScalarAddFloat(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.ops.aten.add.Scalar(x, 2.0)

    def get_example_inputs(self):
        return (torch.ones(1),), {}


class ScalarAddInt(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.ops.aten.add.Scalar(x, 2)

    def get_example_inputs(self):
        return (torch.ones(1).to(torch.int64),), {}


class AddWithCausalMaskFolded(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        min_dtype = torch.finfo(torch.float32).min
        causal_mask = torch.tensor(
            [[min_dtype, 0.0, 0.0], [min_dtype, 0.0, 0.0], [min_dtype, 0.0, 0.0]]
        )

        return causal_mask + x

    def get_example_inputs(self):
        return (torch.ones(3, 3, dtype=torch.float32),), {}


@tag.with_golden
class AddWithCausalMaskLegalized(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        min_dtype = torch.finfo(torch.float32).min
        causal_mask = torch.tensor(
            [[min_dtype, 0.0, 0.0], [min_dtype, 0.0, 0.0], [min_dtype, 0.0, 0.0]]
        )

        return causal_mask + x

    def get_example_inputs(self):
        return (torch.ones(3, 3, dtype=torch.float32),), {}

    def get_compile_config(self):
        return CompileConfigV1(legalize_causal_mask_value=True)

    def get_golden_outputs(self):
        # By 'legalize_causal_mask_value', min_dtype -> -120
        return (
            torch.tensor([[-119.0, 1.0, 1.0], [-119.0, 1.0, 1.0], [-119.0, 1.0, 1.0]]),
        )
