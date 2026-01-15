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

import unittest

import torch
from tico.passes.const_prop_pass import ConstPropPass
from tico.passes.decompose_fake_quantize import DecomposeFakeQuantize
from tico.quantization import convert, prepare
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.passes.remove_weight_dequant_op import RemoveWeightDequantOp

from test.utils.helper import num_of_ops


class RemoveWeightDequantOpTest(unittest.TestCase):
    def test_pass(self):
        q_m = torch.nn.Linear(3, 3)
        assert isinstance(q_m, torch.nn.Module)

        q_m = prepare(q_m, PTQConfig())

        # Calibration
        for i in range(10):
            cal_args = (torch.randn(3, 3),)
            q_m(*cal_args)

        # Quantization
        q_m = convert(q_m)

        # 5. Export module
        ep = torch.export.export(q_m, (torch.randn(3, 3),))
        DecomposeFakeQuantize().call(ep)
        ConstPropPass().call(ep)
        # (weight - DQ)
        self.assertEqual(
            num_of_ops(
                ep, [torch.ops.quantized_decomposed.dequantize_per_channel.default]
            ),
            1,
        )

        target_pass = RemoveWeightDequantOp()
        target_pass.call(ep)
        self.assertEqual(
            num_of_ops(
                ep, [torch.ops.quantized_decomposed.dequantize_per_channel.default]
            ),
            0,
        )
