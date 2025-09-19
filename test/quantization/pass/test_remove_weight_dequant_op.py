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
from tico.experimental.quantization import convert, prepare
from tico.experimental.quantization.config.pt2e import PT2EConfig
from tico.experimental.quantization.passes.remove_weight_dequant_op import (
    RemoveWeightDequantOp,
)

from test.modules.op.linear import SimpleLinear
from test.utils.helper import num_of_ops


class RemoveWeightDequantOpTest(unittest.TestCase):
    def test_pass(self):
        m: SimpleLinear | torch.nn.Module = SimpleLinear().eval()
        assert isinstance(m, SimpleLinear)
        args, kwargs = m.get_example_inputs()

        q_m = prepare(m, PT2EConfig(), args=args, kwargs=kwargs)

        # Calibration
        for i in range(10):
            cal_args, cal_kwargs = m.get_example_inputs()
            q_m(*cal_args, **cal_kwargs)

        # Quantization
        q_m = convert(q_m)

        # 5. Export module
        ep = torch.export.export(q_m, args)
        # (weight - DQ) and (bias - DQ)
        self.assertEqual(
            num_of_ops(
                ep, [torch.ops.quantized_decomposed.dequantize_per_channel.default]
            ),
            2,
        )

        target_pass = RemoveWeightDequantOp()
        target_pass.call(ep)
        self.assertEqual(
            num_of_ops(
                ep, [torch.ops.quantized_decomposed.dequantize_per_channel.default]
            ),
            0,
        )
