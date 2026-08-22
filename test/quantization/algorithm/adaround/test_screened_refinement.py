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

"""Tests for finite-difference screened single-code refinement."""

from __future__ import annotations

import unittest

from types import SimpleNamespace
from unittest import mock

import torch

from tico.quantization.algorithm.adaround import discrete_refinement
from tico.quantization.algorithm.adaround.discrete_refinement import (
    DiscreteCodeWeightSet,
)
from tico.quantization.algorithm.adaround.joint import JointAdaRoundWeightGroup
from tico.quantization.algorithm.adaround.joint_runner import JointAdaRoundObjective
from tico.quantization.algorithm.adaround.screened_refinement import (
    _channel_topk_entries,
    _topk_masked_indices,
    ScreenedCodeRefinementConfig,
    ScreenedCodeRefinementRunner,
)
from tico.quantization.wrapq.control import SiteRole
from tico.quantization.wrapq.dtypes import UINT8
from tico.quantization.wrapq.observers.minmax import MinMaxObserver
from tico.quantization.wrapq.qscheme import QScheme
from torch import nn


def _observer(channels: int) -> MinMaxObserver:
    observer = MinMaxObserver(
        name="weight",
        dtype=UINT8,
        qscheme=QScheme.PER_CHANNEL_ASYMM,
        channel_axis=0,
    )
    observer.load_qparams(
        torch.full((channels,), 0.1),
        torch.full((channels,), 128, dtype=torch.int),
        lock=True,
    )
    observer.fake_quant_enabled = True
    return observer


class _Owner(nn.Module):
    def __init__(self, output_channels: int = 1, width: int = 1) -> None:
        super().__init__()
        self.module = nn.Conv2d(
            1,
            output_channels,
            (1, width),
            bias=False,
        )
        self.obs_weight = _observer(output_channels)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        weight = self.obs_weight.fake_quant(self.module.weight)
        return self.module._conv_forward(input_, weight, self.module.bias)


def _weight_set(
    source: torch.Tensor,
    effective: torch.Tensor,
) -> tuple[DiscreteCodeWeightSet, _Owner, JointAdaRoundWeightGroup]:
    owner = _Owner(source.shape[0], source.shape[-1])
    with torch.no_grad():
        owner.module.weight.copy_(effective)
    site = SimpleNamespace(
        path="block.weight",
        module_path="block",
        observer_name="weight",
        role=SiteRole.PARAMETER,
        module=owner,
        observer=owner.obs_weight,
    )
    group = JointAdaRoundWeightGroup(
        name="conv",
        site_path="block.weight",
        family="regular_conv",
    )
    with mock.patch.object(
        discrete_refinement,
        "iter_quantization_sites",
        return_value=(site,),
    ):
        weights = DiscreteCodeWeightSet(
            owner,
            (group,),
            {"block.weight": source},
        )
    return weights, owner, group


class _Reference(nn.Module):
    def __init__(self, weight: float) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1, bias=False)
        with torch.no_grad():
            self.conv.weight.fill_(weight)

    def forward(self, input_: torch.Tensor):
        output = self.conv(input_)
        return {
            "regressors": output,
            "classifiers": output * 0.25,
        }


class _Candidate(nn.Module):
    def __init__(self, weight: float) -> None:
        super().__init__()
        self.owner = _Owner()
        with torch.no_grad():
            self.owner.module.weight.fill_(weight)

    def forward(self, input_: torch.Tensor):
        output = self.owner(input_)
        return {
            "regressors": output,
            "classifiers": output * 0.25,
        }


def _metric_outputs(
    reference: nn.Module,
    candidate: nn.Module,
    sample: torch.Tensor,
):
    with torch.no_grad():
        expected = reference(sample)
        actual = candidate(sample)
    return {
        name: {
            "mae": float((actual[name] - expected[name]).abs().mean()),
            "mse": float(((actual[name] - expected[name]) ** 2).mean()),
        }
        for name in expected
    }


class ScreenedCodeHelperTest(unittest.TestCase):
    def test_topk_masked_indices_are_sorted_by_value(self) -> None:
        values = torch.tensor([3.0, -2.0, -5.0, -1.0])
        mask = torch.tensor([True, True, False, True])
        self.assertEqual(_topk_masked_indices(values, mask, 2), (1, 3))

    def test_channel_topk_returns_one_candidate_per_channel(self) -> None:
        values = torch.tensor([[3.0, -2.0], [-5.0, -1.0]])
        mask = torch.ones_like(values, dtype=torch.bool)
        self.assertEqual(
            _channel_topk_entries(values, mask, 1),
            ((-2.0, 1), (-5.0, 2)),
        )

    def test_config_requires_a_shortlist_source(self) -> None:
        config = ScreenedCodeRefinementConfig(
            global_shortlist_count=0,
            per_site_shortlist_count=0,
            per_channel_shortlist_count=0,
        )
        with self.assertRaisesRegex(ValueError, "shortlist source"):
            config.validate()

    def test_zero_channel_cap_disables_channel_shortlist_source(self) -> None:
        config = ScreenedCodeRefinementConfig(
            global_shortlist_count=0,
            per_site_shortlist_count=0,
            per_channel_shortlist_count=1,
            maximum_channel_candidates=0,
        )
        with self.assertRaisesRegex(ValueError, "shortlist source"):
            config.validate()


class DiversifiedShortlistTest(unittest.TestCase):
    def test_global_site_and_channel_sources_are_deduplicated(self) -> None:
        source = torch.tensor(
            [
                [[[0.17, 0.28]]],
                [[[0.14, 0.36]]],
            ]
        )
        effective = torch.tensor(
            [
                [[[0.1, 0.3]]],
                [[[0.1, 0.4]]],
            ]
        )
        weights, _, _ = _weight_set(source, effective)
        try:
            parameter = weights.gradient_parameters()[0]
            parameter.grad = torch.tensor(
                [
                    [[[-4.0, 1.0]]],
                    [[[-2.0, 0.5]]],
                ]
            )
            runner = ScreenedCodeRefinementRunner(
                ScreenedCodeRefinementConfig(
                    max_rounds=1,
                    global_shortlist_count=1,
                    per_site_shortlist_count=1,
                    per_channel_shortlist_count=1,
                    maximum_channel_candidates=2,
                    maximum_shortlist_count=8,
                    selection_candidate_count=1,
                )
            )
            with mock.patch.object(
                runner,
                "_collect_gradient",
                return_value=((0,), 0.0, 0.0, 0.0),
            ):
                statistics, shortlist = runner._collect_shortlist(
                    1,
                    mock.Mock(),
                    nn.Identity(),
                    weights,
                    lambda outputs: outputs,
                    torch.device("cpu"),
                )
            self.assertEqual(statistics.recorded_candidate_count, 2)
            first, second = shortlist
            self.assertEqual(
                set(first[1]),
                {"channel", "global", "site"},
            )
            self.assertEqual(second[1], ("channel",))
        finally:
            weights.restore()


class ScreenedCodeRunnerTest(unittest.TestCase):
    def test_commits_one_code_then_rejects_harmful_reversal(self) -> None:
        source = torch.tensor([[[[0.17]]]])
        reference = _Reference(0.17)
        candidate = _Candidate(0.1)
        sample = torch.ones(1, 1, 1, 1)
        site = SimpleNamespace(
            path="block.weight",
            module_path="block",
            observer_name="weight",
            role=SiteRole.PARAMETER,
            module=candidate.owner,
            observer=candidate.owner.obs_weight,
        )
        group = JointAdaRoundWeightGroup(
            name="conv",
            site_path="block.weight",
            family="regular_conv",
        )
        objective = JointAdaRoundObjective(
            primary_output="regressors",
            primary_metric="mae",
            minimum_improvement=0.0,
            absolute_output_limits={"classifiers": 0.1},
        )

        def evaluator():
            return _metric_outputs(reference, candidate, sample)

        runner = ScreenedCodeRefinementRunner(
            ScreenedCodeRefinementConfig(
                max_rounds=2,
                gradient_sample_count=0,
                screening_sample_count=1,
                global_shortlist_count=1,
                per_site_shortlist_count=0,
                per_channel_shortlist_count=0,
                maximum_channel_candidates=0,
                maximum_shortlist_count=1,
                selection_candidate_count=1,
                target_primary_score=None,
            )
        )
        with mock.patch.object(
            discrete_refinement,
            "iter_quantization_sites",
            return_value=(site,),
        ):
            result = runner.refine(
                reference_model=reference,
                candidate_model=candidate,
                training_samples=(sample,),
                weight_groups=(group,),
                source_weights={"block.weight": source},
                output_adapter=lambda outputs: outputs,
                selection_evaluator=evaluator,
                selection_objective=objective,
                acceptance_evaluator=evaluator,
                acceptance_objective=objective,
                evaluation_evaluator=evaluator,
                device="cpu",
            )
        self.assertEqual(result.accepted_rounds, 1)
        self.assertEqual(len(result.final_code_changes), 1)
        self.assertEqual(result.rounds[0].selected_candidate.new_code, 130)
        self.assertTrue(result.rounds[0].accepted)
        self.assertFalse(result.rounds[1].accepted)
        self.assertIn("screening", result.rounds[1].stop_reason)
        self.assertLess(
            result.final_evaluation_outputs["regressors"]["mae"],
            result.entry_evaluation_outputs["regressors"]["mae"],
        )


if __name__ == "__main__":
    unittest.main()
