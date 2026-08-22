# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved
# Licensed under the Apache License, Version 2.0

"""Tests for joint learnable-scale AdaRound primitives."""

from __future__ import annotations

import unittest

from types import SimpleNamespace
from unittest import mock

import torch

from tico.quantization.algorithm.adaround import (
    joint as joint_module,
    JointAdaRoundObjective,
    JointAdaRoundWeightGroup,
    JointAdaRoundWeightSet,
    LearnableScaleAdaRoundWeightQuantizer,
)
from tico.quantization.wrapq.control import SiteRole
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.observers.minmax import MinMaxObserver
from tico.quantization.wrapq.qscheme import QScheme
from torch import nn


def _observer(channels: int) -> MinMaxObserver:
    observer = MinMaxObserver(
        name="weight",
        dtype=DType.uint(8),
        qscheme=QScheme.PER_CHANNEL_ASYMM,
        channel_axis=0,
    )
    observer.load_qparams(
        torch.full((channels,), 0.01),
        torch.zeros(channels, dtype=torch.int),
        lock=True,
    )
    return observer


class _Owner(nn.Module):
    def __init__(self, module: nn.Conv2d) -> None:
        super().__init__()
        self.module = module
        self.obs_weight = _observer(module.out_channels)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        weight = self.obs_weight.fake_quant(self.module.weight)
        return self.module._conv_forward(input_, weight, self.module.bias)


class JointAdaRoundTest(unittest.TestCase):
    def test_soft_initialization_reconstructs_unclipped_weight(self) -> None:
        weight = torch.tensor([[[[0.13]]], [[[0.81]]]], dtype=torch.float32)
        proxy = LearnableScaleAdaRoundWeightQuantizer(
            _observer(2),
            weight,
        )
        torch.testing.assert_close(proxy.fake_quant(weight), weight)

    def test_initial_hard_state_matches_affine_nearest_rounding(self) -> None:
        weight = torch.tensor([[[[0.13]]], [[[0.81]]]], dtype=torch.float32)
        observer = _observer(2)
        expected = observer.fake_quant(weight)
        proxy = LearnableScaleAdaRoundWeightQuantizer(observer, weight)
        proxy.set_hard(True)
        torch.testing.assert_close(proxy.fake_quant(weight), expected)

    def test_alpha_and_scale_receive_gradients(self) -> None:
        weight = torch.tensor([[[[0.13]]], [[[0.81]]]], dtype=torch.float32)
        proxy = LearnableScaleAdaRoundWeightQuantizer(
            _observer(2),
            weight,
        )
        loss = proxy.fake_quant(weight).square().sum()
        loss.backward()
        self.assertIsNotNone(proxy.alpha.grad)
        self.assertIsNotNone(proxy.raw_log_scale_delta.grad)
        self.assertGreater(float(proxy.alpha.grad.abs().sum()), 0.0)
        self.assertGreater(float(proxy.raw_log_scale_delta.grad.abs().sum()), 0.0)

    def test_scale_ratio_is_positive_and_bounded(self) -> None:
        weight = torch.ones(2, 1, 1, 1)
        proxy = LearnableScaleAdaRoundWeightQuantizer(
            _observer(2),
            weight,
            max_scale_ratio=1.25,
        )
        proxy.raw_log_scale_delta.data.copy_(torch.tensor([100.0, -100.0]))
        ratio = proxy.scale_ratio()
        self.assertLessEqual(float(ratio.detach().max()), 1.25 + 1.0e-6)
        self.assertGreaterEqual(float(ratio.detach().min()), 1.0 / 1.25 - 1.0e-6)

    def test_finalize_commits_weight_and_scale_and_restores_observer(self) -> None:
        owner = _Owner(nn.Conv2d(2, 2, 1, bias=False))
        owner.module.weight.data.copy_(
            torch.tensor(
                [
                    [[[0.13]], [[0.23]]],
                    [[[0.81]], [[0.41]]],
                ],
                dtype=torch.float32,
            )
        )
        original_observer = owner.obs_weight
        original_scale, _ = original_observer.compute_qparams()
        site = SimpleNamespace(
            path="block.weight",
            module_path="block",
            observer_name="weight",
            role=SiteRole.PARAMETER,
            module=owner,
            observer=original_observer,
        )
        group = JointAdaRoundWeightGroup(
            "conv",
            "block.weight",
            "regular_conv",
        )
        with mock.patch.object(
            joint_module,
            "iter_quantization_sites",
            return_value=(site,),
        ):
            weights = JointAdaRoundWeightSet(
                owner,
                (group,),
                gamma=-0.1,
                zeta=1.1,
                initialization_epsilon=1.0e-6,
                max_scale_ratio=1.25,
            )
        proxy = weights.bindings[0].proxy
        proxy.raw_log_scale_delta.data.fill_(0.5)
        expected_scale = proxy.learned_scale().detach().clone()
        expected_weight = proxy.hard_weight().detach().clone()
        weights.finalize()
        self.assertIs(owner.obs_weight, original_observer)
        committed_scale, _ = original_observer.compute_qparams()
        torch.testing.assert_close(committed_scale, expected_scale)
        self.assertFalse(torch.equal(committed_scale, original_scale))
        torch.testing.assert_close(owner.module.weight, expected_weight)

    def test_restore_rolls_back_weight_and_qparams(self) -> None:
        owner = _Owner(nn.Conv2d(2, 2, 1, bias=False))
        original_weight = owner.module.weight.detach().clone()
        original_observer = owner.obs_weight
        original_scale, original_zp = original_observer.compute_qparams()
        site = SimpleNamespace(
            path="block.weight",
            module_path="block",
            observer_name="weight",
            role=SiteRole.PARAMETER,
            module=owner,
            observer=original_observer,
        )
        group = JointAdaRoundWeightGroup(
            "conv",
            "block.weight",
            "regular_conv",
        )
        with mock.patch.object(
            joint_module,
            "iter_quantization_sites",
            return_value=(site,),
        ):
            weights = JointAdaRoundWeightSet(
                owner,
                (group,),
                gamma=-0.1,
                zeta=1.1,
                initialization_epsilon=1.0e-6,
                max_scale_ratio=1.25,
            )
        weights.bindings[0].proxy.raw_log_scale_delta.data.fill_(0.8)
        weights.restore()
        scale, zp = original_observer.compute_qparams()
        torch.testing.assert_close(owner.module.weight, original_weight)
        torch.testing.assert_close(scale, original_scale)
        torch.testing.assert_close(zp, original_zp)
        self.assertIs(owner.obs_weight, original_observer)

    def test_progressive_absolute_limit_allows_improvement_above_target(self) -> None:
        objective = JointAdaRoundObjective(
            absolute_output_limits={"classifiers": 0.1},
        )
        entry = {
            "regressors": {"mae": 0.5},
            "classifiers": {"mae": 0.15},
        }
        improving = {
            "regressors": {"mae": 0.4},
            "classifiers": {"mae": 0.14},
        }
        regressing = {
            "regressors": {"mae": 0.3},
            "classifiers": {"mae": 0.151},
        }
        self.assertTrue(objective.admissible(improving, entry)[0])
        self.assertFalse(objective.admissible(regressing, entry)[0])

        below_target = {
            "regressors": {"mae": 0.2},
            "classifiers": {"mae": 0.08},
        }
        within_limit = {
            "regressors": {"mae": 0.1},
            "classifiers": {"mae": 0.095},
        }
        above_limit = {
            "regressors": {"mae": 0.1},
            "classifiers": {"mae": 0.101},
        }
        self.assertTrue(objective.admissible(within_limit, below_target)[0])
        self.assertFalse(objective.admissible(above_limit, below_target)[0])


if __name__ == "__main__":
    unittest.main()
