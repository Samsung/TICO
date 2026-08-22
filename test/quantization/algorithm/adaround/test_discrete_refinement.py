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

"""Tests for explicit gradient-ranked fixed-scale code refinement."""

from __future__ import annotations

import unittest

from types import SimpleNamespace
from unittest import mock

import torch

from tico.quantization.algorithm.adaround import discrete_refinement as module
from tico.quantization.algorithm.adaround.discrete_refinement import (
    _gradient_sample_indices,
    _proposal_sizes,
    _unravel_index,
    DiscreteCodeRefinementConfig,
    DiscreteCodeWeightSet,
    FixedScaleCodeObserver,
)
from tico.quantization.algorithm.adaround.joint import JointAdaRoundWeightGroup
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
    def __init__(self) -> None:
        super().__init__()
        self.module = nn.Conv2d(1, 1, (1, 4), bias=False)
        self.obs_weight = _observer(1)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        weight = self.obs_weight.fake_quant(self.module.weight)
        return self.module._conv_forward(input_, weight, self.module.bias)


def _proxy() -> FixedScaleCodeObserver:
    source = torch.tensor([[[[-0.12, 0.17, 0.31, -0.45]]]])
    observer = _observer(1)
    scale, zero_point = observer.compute_qparams()
    shape = (1, 1, 1, 1)
    normalized = source / scale.reshape(shape) + zero_point.reshape(shape)
    floor = torch.floor(normalized)
    codes = floor + torch.tensor([[[[1.0, 0.0, 1.0, 0.0]]]])
    codes = codes.clamp(UINT8.qmin, UINT8.qmax)
    effective = (codes - zero_point.reshape(shape)) * scale.reshape(shape)
    return FixedScaleCodeObserver(observer, source, effective)


def _weight_set() -> DiscreteCodeWeightSet:
    owner = _Owner()
    source = torch.tensor([[[[0.13, -0.23, 0.81, -0.41]]]])
    scale, zero_point = owner.obs_weight.compute_qparams()
    shape = (1, 1, 1, 1)
    codes = torch.round(source / scale.reshape(shape) + zero_point.reshape(shape))
    codes = codes.clamp(UINT8.qmin, UINT8.qmax)
    with torch.no_grad():
        owner.module.weight.copy_(
            (codes - zero_point.reshape(shape)) * scale.reshape(shape)
        )
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
        module,
        "iter_quantization_sites",
        return_value=(site,),
    ):
        return DiscreteCodeWeightSet(
            owner,
            (group,),
            {"block.weight": source},
        )


class FixedScaleCodeObserverTest(unittest.TestCase):
    def test_entry_weight_and_codes_are_reproduced_exactly(self) -> None:
        proxy = _proxy()
        torch.testing.assert_close(
            proxy.fake_quant(proxy._reference_weight),
            proxy._entry_effective_weight,
        )
        self.assertTrue(torch.equal(proxy.current_codes(), proxy.entry_codes()))

    def test_alternative_codes_toggle_reachable_floor_ceil_decisions(self) -> None:
        proxy = _proxy()
        alternative, valid = proxy.alternative_codes()
        self.assertTrue(bool(valid.all()))
        self.assertTrue(bool((alternative != proxy.current_codes()).all()))
        proxy.set_codes(alternative)
        self.assertTrue(torch.equal(proxy.current_codes(), alternative))

    def test_rejects_unreachable_code_state(self) -> None:
        proxy = _proxy()
        invalid = proxy.current_codes() + 2
        with self.assertRaisesRegex(ValueError, "unreachable"):
            proxy.set_codes(invalid)


class DiscreteCodeWeightSetTest(unittest.TestCase):
    def test_ranking_records_exact_code_identity_and_score(self) -> None:
        weights = _weight_set()
        try:
            parameter = weights.gradient_parameters()[0]
            parameter.grad = torch.tensor([[[[-1.0, 1.0, -0.5, 0.25]]]])
            ranked, reachable, improving, _ = weights.rank_candidates(
                maximum_count=4,
                minimum_predicted_improvement=0.0,
            )
            self.assertEqual(reachable, 4)
            self.assertGreaterEqual(improving, 2)
            self.assertGreater(len(ranked), 0)
            first = ranked[0]
            self.assertEqual(first.site_path, "block.weight")
            self.assertEqual(len(first.tensor_index), 4)
            self.assertIn(first.direction, {-1, 1})
            expected = first.gradient * (
                first.alternative_weight - first.current_weight
            )
            self.assertAlmostEqual(first.predicted_loss_delta, expected, places=6)
            self.assertGreater(first.predicted_improvement, 0.0)
        finally:
            weights.restore()

    def test_snapshot_apply_and_restore_are_transactional(self) -> None:
        weights = _weight_set()
        try:
            parameter = weights.gradient_parameters()[0]
            parameter.grad = torch.tensor([[[[-1.0, 1.0, -0.5, 0.25]]]])
            ranked, _, _, _ = weights.rank_candidates(
                maximum_count=1,
                minimum_predicted_improvement=0.0,
            )
            snapshot = weights.state_snapshot()
            weights.apply_candidates(ranked)
            self.assertEqual(
                weights.transition_summary(ranked).net_changed_count,
                1,
            )
            weights.load_state_snapshot(snapshot)
            self.assertEqual(weights.transition_summary(()).net_changed_count, 0)
        finally:
            weights.restore()

    def test_second_round_can_revert_a_previous_change(self) -> None:
        weights = _weight_set()
        try:
            parameter = weights.gradient_parameters()[0]
            parameter.grad = torch.tensor([[[[-1.0, 1.0, -0.5, 0.25]]]])
            first, _, _, _ = weights.rank_candidates(
                maximum_count=1,
                minimum_predicted_improvement=0.0,
            )
            weights.apply_candidates(first)
            changed = first[0]
            gradient = torch.zeros_like(parameter)
            delta = changed.current_weight - changed.alternative_weight
            gradient.flatten()[changed.flat_index] = 1.0 if delta < 0 else -1.0
            parameter.grad = gradient
            second, _, _, _ = weights.rank_candidates(
                maximum_count=1,
                minimum_predicted_improvement=0.0,
            )
            self.assertEqual(second[0].transition_kind, "reverted")
            weights.apply_candidates(second)
            summary = weights.transition_summary(second)
            self.assertEqual(summary.reverted_count, 1)
            self.assertEqual(summary.net_changed_count, 0)
        finally:
            weights.restore()

    def test_final_change_log_contains_tensor_coordinate(self) -> None:
        weights = _weight_set()
        try:
            parameter = weights.gradient_parameters()[0]
            parameter.grad = torch.tensor([[[[-1.0, 1.0, -0.5, 0.25]]]])
            ranked, _, _, _ = weights.rank_candidates(
                maximum_count=1,
                minimum_predicted_improvement=0.0,
            )
            weights.apply_candidates(ranked)
            changes = weights.final_code_changes()
            self.assertEqual(len(changes), 1)
            self.assertEqual(changes[0].site_path, "block.weight")
            self.assertEqual(len(changes[0].tensor_index), 4)
        finally:
            weights.restore()


class DiscreteCodeHelperTest(unittest.TestCase):
    def test_nested_proposal_sizes_include_available_candidate_count(self) -> None:
        self.assertEqual(
            _proposal_sizes((2048, 1024, 512, 256, 128, 64), 300),
            (300, 256, 128, 64),
        )

    def test_gradient_sample_indices_are_deterministic(self) -> None:
        first = _gradient_sample_indices(20, 5, seed=7)
        second = _gradient_sample_indices(20, 5, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), 5)

    def test_unravel_index_matches_tensor_layout(self) -> None:
        self.assertEqual(_unravel_index(7, (1, 2, 2, 2)), (0, 1, 1, 1))

    def test_config_rejects_duplicate_proposal_sizes(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            DiscreteCodeRefinementConfig(proposal_sizes=(512, 512)).validate()


if __name__ == "__main__":
    unittest.main()
