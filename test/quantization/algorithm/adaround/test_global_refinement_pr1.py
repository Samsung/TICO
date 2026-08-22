# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved
# Licensed under the Apache License, Version 2.0

"""Tests for global-refinement diagnostics and objective alignment."""

from __future__ import annotations

import unittest

from types import SimpleNamespace
from unittest import mock

import torch

from tico.quantization.algorithm.adaround import global_refinement as global_module
from tico.quantization.algorithm.adaround.global_refinement import (
    _absolute_tensor_histogram,
    _output_loss,
    CheckpointInitializedScaleAdaRoundQuantizer,
    GlobalAdaRoundWeightSet,
    GlobalWeightRefinementConfig,
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
    def __init__(self, module: nn.Conv2d) -> None:
        super().__init__()
        self.module = module
        self.obs_weight = _observer(module.out_channels)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        weight = self.obs_weight.fake_quant(self.module.weight)
        return self.module._conv_forward(input_, weight, self.module.bias)


def _quantizer(*, initialization: str = "constant", magnitude: float = 0.05):
    source = torch.tensor(
        [[[[-0.12, 0.17, 0.31, -0.45]]]],
        dtype=torch.float32,
    )
    observer = _observer(1)
    scale, zero_point = observer.compute_qparams()
    shape = (1, 1, 1, 1)
    normalized = source / scale.reshape(shape) + zero_point.reshape(shape)
    floor_codes = torch.floor(normalized)
    initial_codes = floor_codes + torch.tensor([[[[1.0, 0.0, 1.0, 0.0]]]])
    initial_codes = initial_codes.clamp(UINT8.qmin, UINT8.qmax)
    effective = (initial_codes - zero_point.reshape(shape)) * scale.reshape(shape)
    quantizer = CheckpointInitializedScaleAdaRoundQuantizer(
        observer,
        source,
        initial_codes,
        effective,
        gamma=-0.1,
        zeta=1.1,
        initialization_epsilon=1e-6,
        max_scale_ratio=1.25,
        checkpoint_alpha_minimum_magnitude=1.5,
        checkpoint_alpha_initialization=initialization,
        checkpoint_alpha_initial_magnitude=magnitude,
    )
    return source, initial_codes, quantizer


def _weight_set() -> GlobalAdaRoundWeightSet:
    source = torch.tensor(
        [[[[0.13, -0.23, 0.81, -0.41]]]],
        dtype=torch.float32,
    )
    owner = _Owner(nn.Conv2d(1, 1, (1, 4), bias=False))
    observer = owner.obs_weight
    scale, zero_point = observer.compute_qparams()
    shape = (1, 1, 1, 1)
    normalized = source / scale.reshape(shape) + zero_point.reshape(shape)
    codes = torch.round(normalized).clamp(UINT8.qmin, UINT8.qmax)
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
        observer=observer,
    )
    group = JointAdaRoundWeightGroup(
        "conv",
        "block.weight",
        "regular_conv",
    )
    with mock.patch.object(
        global_module,
        "iter_quantization_sites",
        return_value=(site,),
    ):
        return GlobalAdaRoundWeightSet(
            owner,
            (group,),
            {"block.weight": source},
            gamma=-0.1,
            zeta=1.1,
            initialization_epsilon=1e-6,
            max_scale_ratio=1.25,
            checkpoint_alpha_minimum_magnitude=1.5,
            checkpoint_alpha_initialization="constant",
            checkpoint_alpha_initial_magnitude=0.05,
        )


class _TinyReference(nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(weight)

    def forward(self, input_: torch.Tensor):
        output = self.conv(input_)
        return {
            "regressors": output,
            "classifiers": output * 0.25,
        }


class _TinyCandidate(nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.owner = _Owner(nn.Conv2d(1, 1, 1, bias=False))
        scale, zero_point = self.owner.obs_weight.compute_qparams()
        shape = (1, 1, 1, 1)
        codes = torch.round(weight / scale.reshape(shape))
        codes = (codes + zero_point.reshape(shape)).clamp(
            UINT8.qmin,
            UINT8.qmax,
        )
        with torch.no_grad():
            self.owner.module.weight.copy_(
                (codes - zero_point.reshape(shape)) * scale.reshape(shape)
            )

    def forward(self, input_: torch.Tensor):
        output = self.owner(input_)
        return {
            "regressors": output,
            "classifiers": output * 0.25,
        }


class _AlwaysAcceptObjective:
    def score(self, outputs) -> float:
        return float(outputs["regressors"]["mae"])

    def better(self, candidate, incumbent, reference):
        del reference
        improvement = self.score(incumbent) - self.score(candidate)
        return improvement > 0.0, f"improvement={improvement:.6e}"

    def accepted(self, candidate, reference):
        del candidate, reference
        return True, "accepted by test objective"


class GlobalRefinementDiagnosticsTest(unittest.TestCase):
    def test_constant_initialization_uses_exact_checkpoint_magnitude(self) -> None:
        _, initial_codes, quantizer = _quantizer(magnitude=0.05)
        mask = quantizer._checkpoint_decision_mask
        torch.testing.assert_close(
            quantizer.alpha.detach()[mask].abs(),
            torch.full_like(quantizer.alpha.detach()[mask], 0.05),
        )
        self.assertTrue(
            torch.equal(quantizer.quantized_codes(hard=True), initial_codes)
        )

    def test_freeze_scale_keeps_scale_out_of_autograd(self) -> None:
        source, _, quantizer = _quantizer()
        quantizer.raw_log_scale_delta.requires_grad_(False)
        quantizer.set_hard(False)
        quantizer.fake_quant(source).square().mean().backward()
        self.assertIsNotNone(quantizer.alpha.grad)
        self.assertIsNone(quantizer.raw_log_scale_delta.grad)
        torch.testing.assert_close(
            quantizer.scale_ratio(),
            torch.ones_like(quantizer.scale_ratio()),
        )

    def test_flip_budget_keeps_only_strongest_checkpoint_flips(self) -> None:
        weights = _weight_set()
        proxy = weights.bindings[0].proxy
        mask = proxy._checkpoint_decision_mask.flatten()
        self.assertEqual(int(mask.sum()), 4)
        confidence = proxy.alpha.new_tensor([0.1, 0.4, 0.2, 0.3])
        with torch.no_grad():
            sign = proxy._checkpoint_rounding_sign.flatten()
            proxy.alpha.flatten()[mask] = -sign[mask] * confidence
        result = weights.enforce_checkpoint_flip_budget(
            2,
            projection_margin=1e-4,
        )
        self.assertEqual(result.before_count, 4)
        self.assertEqual(result.after_count, 2)
        self.assertEqual(result.projected_count, 2)
        signed = proxy.checkpoint_signed_alpha().detach().flatten()[mask]
        self.assertEqual(int((signed < 0).sum()), 2)
        kept = torch.sort((-signed[signed < 0])).values
        torch.testing.assert_close(kept, torch.tensor([0.3, 0.4]))
        weights.restore()

    def test_histogram_and_raw_mae_match_expected_values(self) -> None:
        histogram = _absolute_tensor_histogram(
            (torch.tensor([-0.1, 0.0, 0.2, 1.0]),),
            edges=(0.05, 0.5),
        )
        self.assertEqual(histogram.count, 4)
        self.assertEqual(histogram.zero_count, 1)
        self.assertEqual(histogram.histogram_counts, (1, 2, 1))
        candidate = torch.tensor([1.0, 4.0])
        reference = torch.tensor([1.0, 2.0])
        loss = _output_loss(
            candidate,
            reference,
            kind="raw_mae",
            epsilon=1e-8,
        )
        self.assertAlmostEqual(float(loss), 1.0)

    def test_runner_records_raw_loss_histograms_and_frozen_scale(self) -> None:
        source_weight = torch.tensor([[[[0.137]]]], dtype=torch.float32)
        reference = _TinyReference(source_weight)
        candidate = _TinyCandidate(source_weight)
        observer = candidate.owner.obs_weight
        site = SimpleNamespace(
            path="block.weight",
            module_path="block",
            observer_name="weight",
            role=SiteRole.PARAMETER,
            module=candidate.owner,
            observer=observer,
        )
        group = JointAdaRoundWeightGroup(
            "conv",
            "block.weight",
            "regular_conv",
        )
        samples = (torch.tensor([[[[1.0]]]]),)

        def evaluate():
            with torch.no_grad():
                expected = reference(samples[0])
                actual = candidate(samples[0])
            return {
                name: {"mae": float((actual[name] - expected[name]).abs().mean())}
                for name in expected
            }

        config = GlobalWeightRefinementConfig(
            steps=1,
            gradient_accumulation_steps=1,
            evaluation_interval=1,
            alpha_learning_rate=1e-3,
            scale_learning_rate=0.0,
            freeze_scale=True,
            training_loss="raw_mae",
            checkpoint_alpha_initialization="constant",
            checkpoint_alpha_initial_magnitude=0.05,
            checkpoint_flip_budget=0,
            checkpoint_anchor_loss_weight=0.0,
            rounding_loss_weight=0.0,
            scale_loss_weight=0.0,
        )
        with mock.patch.object(
            global_module,
            "iter_quantization_sites",
            return_value=(site,),
        ):
            result = global_module.GlobalWeightRefinementRunner(config).refine(
                reference_model=reference,
                candidate_model=candidate,
                training_samples=samples,
                weight_groups=(group,),
                source_weights={"block.weight": source_weight},
                output_adapter=lambda outputs: outputs,
                selection_evaluator=evaluate,
                selection_objective=_AlwaysAcceptObjective(),
                acceptance_evaluator=evaluate,
                acceptance_objective=_AlwaysAcceptObjective(),
                evaluation_evaluator=evaluate,
            )
        self.assertTrue(result.accepted)
        self.assertTrue(result.freeze_scale)
        self.assertEqual(result.training_loss, "raw_mae")
        checkpoint = result.checkpoint_history[-1]
        self.assertIsNotNone(checkpoint.alpha_absolute_histogram)
        self.assertIsNotNone(checkpoint.alpha_gradient_absolute_histogram)
        self.assertEqual(checkpoint.flip_budget_statistics.after_count, 0)
        for statistics in result.weight_statistics:
            self.assertAlmostEqual(statistics.base.scale_ratio_minimum, 1.0)
            self.assertAlmostEqual(statistics.base.scale_ratio_maximum, 1.0)

    def test_runner_allows_zero_regularization_with_frozen_scale(self) -> None:
        source_weight = torch.tensor([[[[0.137]]]], dtype=torch.float32)
        reference = _TinyReference(source_weight)
        candidate = _TinyCandidate(source_weight)
        observer = candidate.owner.obs_weight
        site = SimpleNamespace(
            path="block.weight",
            module_path="block",
            observer_name="weight",
            role=SiteRole.PARAMETER,
            module=candidate.owner,
            observer=observer,
        )
        group = JointAdaRoundWeightGroup(
            "conv",
            "block.weight",
            "regular_conv",
        )
        samples = (torch.tensor([[[[1.0]]]]),)

        def evaluate():
            with torch.no_grad():
                expected = reference(samples[0])
                actual = candidate(samples[0])
            return {
                name: {"mae": float((actual[name] - expected[name]).abs().mean())}
                for name in expected
            }

        config = GlobalWeightRefinementConfig(
            steps=1,
            gradient_accumulation_steps=1,
            evaluation_interval=1,
            alpha_learning_rate=1e-3,
            scale_learning_rate=0.0,
            freeze_scale=True,
            training_loss="raw_mae",
            checkpoint_alpha_initialization="constant",
            checkpoint_alpha_initial_magnitude=0.05,
            checkpoint_flip_budget=0,
            checkpoint_anchor_loss_weight=0.0,
            checkpoint_anchor_fraction=0.0,
            rounding_loss_weight=0.0,
            scale_loss_weight=0.0,
            warmup_fraction=0.5,
        )
        with mock.patch.object(
            global_module,
            "iter_quantization_sites",
            return_value=(site,),
        ):
            result = global_module.GlobalWeightRefinementRunner(config).refine(
                reference_model=reference,
                candidate_model=candidate,
                training_samples=samples,
                weight_groups=(group,),
                source_weights={"block.weight": source_weight},
                output_adapter=lambda outputs: outputs,
                selection_evaluator=evaluate,
                selection_objective=_AlwaysAcceptObjective(),
                acceptance_evaluator=evaluate,
                acceptance_objective=_AlwaysAcceptObjective(),
                evaluation_evaluator=evaluate,
            )
        self.assertTrue(result.accepted)
        self.assertEqual(len(result.checkpoint_history), 2)

    def test_config_supports_true_scale_freeze_and_rejects_bad_budget(self) -> None:
        GlobalWeightRefinementConfig(
            freeze_scale=True,
            scale_learning_rate=0.0,
            training_loss="raw_mae",
            checkpoint_flip_budget=0,
        ).validate()
        with self.assertRaises(ValueError):
            GlobalWeightRefinementConfig(
                freeze_scale=False,
                scale_learning_rate=0.0,
            ).validate()
        with self.assertRaises(ValueError):
            GlobalWeightRefinementConfig(
                checkpoint_flip_budget=-1,
            ).validate()


if __name__ == "__main__":
    unittest.main()
