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

"""Tests for group-wise quantization sensitivity analysis."""

import unittest

import torch

from tico.quantization.analysis import (
    QuantizationGroup,
    QuantizationSensitivity,
    SensitivityMode,
    SiteSelector,
)
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.config.specs import affine
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.observers.minmax import MinMaxObserver
from tico.quantization.wrapq.qscheme import QScheme
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from torch import nn


class IdentityReference(nn.Module):
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        return input_


class TwoScaleQuantizer(QuantModuleBase):
    def __init__(self) -> None:
        config = PTQConfig(
            activation=affine(DType.uint(8)),
            weight=affine(DType.uint(8)),
            strict_wrap=False,
        )
        super().__init__(config)
        self.obs_act_in = MinMaxObserver(
            name="act_in",
            dtype=DType.uint(8),
            qscheme=QScheme.PER_TENSOR_ASYMM,
        )
        self.obs_act_out = MinMaxObserver(
            name="act_out",
            dtype=DType.uint(8),
            qscheme=QScheme.PER_TENSOR_ASYMM,
        )
        self.obs_act_in.min_val.fill_(0.0)
        self.obs_act_in.max_val.fill_(100.0)
        self.obs_act_out.min_val.fill_(0.0)
        self.obs_act_out.max_val.fill_(1.0)
        self.freeze_qparams()

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        output = self._fq(input_, self.obs_act_in)
        return self._fq(output, self.obs_act_out)

    def _all_observers(self):
        return self.obs_act_in, self.obs_act_out


class Candidate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = TwoScaleQuantizer()

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        return self.block(input_)


class QuantizationSensitivityTest(unittest.TestCase):
    def test_enable_one_ranks_the_coarser_quantizer_first(self) -> None:
        groups = (
            QuantizationGroup("coarse", SiteSelector.paths("block.act_in")),
            QuantizationGroup("fine", SiteSelector.paths("block.act_out")),
        )
        _, results = QuantizationSensitivity(IdentityReference(), Candidate(),).run(
            [torch.tensor([0.13, 0.57, 0.91])],
            groups,
            mode=SensitivityMode.ENABLE_ONE,
            score_output="output",
        )
        self.assertEqual(results[0].group, "coarse")
        self.assertEqual(results[0].matched_sites, ("block.act_in",))
        self.assertEqual(results[0].to_dict()["matched_site_count"], 1)
        self.assertGreater(results[0].sensitivity, results[1].sensitivity)

    def test_rejects_a_group_that_matches_no_site(self) -> None:
        groups = (QuantizationGroup("missing", SiteSelector.paths("block.missing")),)
        with self.assertRaisesRegex(ValueError, "matched no quantization sites"):
            QuantizationSensitivity(IdentityReference(), Candidate(),).run(
                [torch.tensor([0.13, 0.57, 0.91])],
                groups,
                mode=SensitivityMode.ENABLE_ONE,
                score_output="output",
            )


if __name__ == "__main__":
    unittest.main()
