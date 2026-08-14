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

"""Tests for cumulative and greedy quantization sensitivity paths."""

from __future__ import annotations

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


class _IdentityReference(nn.Module):
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        return input_


class _TwoScaleQuantizer(QuantModuleBase):
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


class _Candidate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = _TwoScaleQuantizer()

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        return self.block(input_)


class CumulativeSensitivityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.samples = [torch.tensor([0.13, 0.57, 0.91])]
        self.groups = (
            QuantizationGroup("fine", SiteSelector.paths("block.act_out")),
            QuantizationGroup("coarse", SiteSelector.paths("block.act_in")),
        )

    def test_cumulative_applies_groups_in_requested_order(self) -> None:
        baseline, steps = QuantizationSensitivity(
            _IdentityReference(),
            _Candidate(),
        ).run_cumulative(
            self.samples,
            self.groups,
            mode=SensitivityMode.LEAVE_ONE_FLOAT,
            score_output="output",
        )

        self.assertGreater(float(baseline["output"]["mae"]), 0.0)
        self.assertEqual(tuple(step.group for step in steps), ("fine", "coarse"))
        self.assertEqual(steps[0].selected_groups, ("fine",))
        self.assertEqual(steps[1].selected_groups, ("fine", "coarse"))
        self.assertEqual(steps[1].selected_site_count, 2)
        self.assertGreater(steps[1].incremental_sensitivity, 0.0)
        self.assertAlmostEqual(float(steps[1].outputs["output"]["mae"]), 0.0)

    def test_greedy_recomputes_and_selects_the_best_next_group(self) -> None:
        candidate = _Candidate()
        _, steps = QuantizationSensitivity(_IdentityReference(), candidate,).run_greedy(
            self.samples,
            self.groups,
            mode=SensitivityMode.LEAVE_ONE_FLOAT,
            score_output="output",
            max_steps=2,
        )

        self.assertEqual(tuple(step.group for step in steps), ("coarse", "fine"))
        self.assertEqual(steps[1].selected_groups, ("coarse", "fine"))
        self.assertGreater(steps[0].incremental_sensitivity, 0.0)
        self.assertGreater(steps[1].incremental_sensitivity, 0.0)
        self.assertAlmostEqual(float(steps[1].outputs["output"]["mae"]), 0.0)
        self.assertTrue(candidate.block.obs_act_in.fake_quant_enabled)
        self.assertTrue(candidate.block.obs_act_out.fake_quant_enabled)

    def test_greedy_zero_steps_evaluates_only_the_baseline(self) -> None:
        baseline, steps = QuantizationSensitivity(
            _IdentityReference(),
            _Candidate(),
        ).run_greedy(
            self.samples,
            self.groups,
            mode=SensitivityMode.LEAVE_ONE_FLOAT,
            score_output="output",
            max_steps=0,
        )

        self.assertGreater(float(baseline["output"]["mae"]), 0.0)
        self.assertEqual(steps, [])

    def test_greedy_stops_below_the_minimum_improvement(self) -> None:
        baseline, steps = QuantizationSensitivity(
            _IdentityReference(),
            _Candidate(),
        ).run_greedy(
            self.samples,
            self.groups,
            mode=SensitivityMode.LEAVE_ONE_FLOAT,
            score_output="output",
            minimum_improvement=1.0,
        )

        self.assertGreater(float(baseline["output"]["mae"]), 0.0)
        self.assertEqual(steps, [])

    def test_cumulative_rejects_a_group_that_adds_no_new_site(self) -> None:
        groups = (
            QuantizationGroup("coarse", SiteSelector.paths("block.act_in")),
            QuantizationGroup("coarse_again", SiteSelector.paths("block.act_in")),
        )
        with self.assertRaisesRegex(ValueError, "adds no new quantization sites"):
            QuantizationSensitivity(_IdentityReference(), _Candidate(),).run_cumulative(
                self.samples,
                groups,
                mode=SensitivityMode.LEAVE_ONE_FLOAT,
                score_output="output",
            )

    def test_path_result_serializes_selected_sites(self) -> None:
        _, steps = QuantizationSensitivity(
            _IdentityReference(),
            _Candidate(),
        ).run_cumulative(
            self.samples,
            self.groups,
            mode=SensitivityMode.LEAVE_ONE_FLOAT,
            score_output="output",
        )

        value = steps[-1].to_dict()
        self.assertEqual(value["selected_group_count"], 2)
        self.assertEqual(value["selected_site_count"], 2)
        self.assertEqual(value["selected_groups"], ["fine", "coarse"])


if __name__ == "__main__":
    unittest.main()
