# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved
# Licensed under the Apache License, Version 2.0

"""Tests for global end-to-end scale and AdaRound primitives."""

from __future__ import annotations

import unittest

from types import SimpleNamespace
from unittest import mock

import torch

from tico.quantization.algorithm.adaround import global_refinement as global_module
from tico.quantization.algorithm.adaround.global_refinement import (
    _normalized_l1,
    _validate_batch_one,
    CheckpointInitializedScaleAdaRoundQuantizer,
    GlobalAdaRoundWeightSet,
    GlobalWeightRefinementCheckpoint,
    GlobalWeightRefinementConfig,
    GlobalWeightRefinementRunner,
)
from tico.quantization.algorithm.adaround.joint import JointAdaRoundWeightGroup
from tico.quantization.wrapq.control import SiteRole
from tico.quantization.wrapq.dtypes import UINT8
from tico.quantization.wrapq.observers.minmax import MinMaxObserver
from tico.quantization.wrapq.qscheme import QScheme
from torch import nn


class _Owner(nn.Module):
    def __init__(self, module: nn.Conv2d) -> None:
        super().__init__()
        self.module = module
        self.obs_weight = _observer(module.out_channels)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        weight = self.obs_weight.fake_quant(self.module.weight)
        return self.module._conv_forward(input_, weight, self.module.bias)


def _observer(channels: int = 2) -> MinMaxObserver:
    observer = MinMaxObserver(
        name="weight",
        dtype=UINT8,
        qscheme=QScheme.PER_CHANNEL_ASYMM,
        channel_axis=0,
    )
    observer.load_qparams(
        torch.linspace(0.1, 0.2, channels),
        torch.full((channels,), 128, dtype=torch.int),
        lock=True,
    )
    observer.fake_quant_enabled = True
    return observer


class _TinyReference(nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(weight)

    def forward(self, input_: torch.Tensor):
        value = self.conv(input_)
        return {
            "regressors": value,
            "classifiers": value * 0.25,
        }


class _TinyCandidate(nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.owner = _Owner(nn.Conv2d(1, 1, 1, bias=False))
        self.owner.obs_weight.load_qparams(
            torch.tensor([0.1]),
            torch.tensor([128], dtype=torch.int),
            lock=True,
        )
        scale, zero_point = self.owner.obs_weight.compute_qparams()
        code = torch.round(weight / scale.reshape(1, 1, 1, 1))
        code = code + zero_point.reshape(1, 1, 1, 1)
        code = code.clamp(UINT8.qmin, UINT8.qmax)
        with torch.no_grad():
            self.owner.module.weight.copy_(
                (code - zero_point.reshape(1, 1, 1, 1)) * scale.reshape(1, 1, 1, 1)
            )

    def forward(self, input_: torch.Tensor):
        value = self.owner(input_)
        return {
            "regressors": value,
            "classifiers": value * 0.25,
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


class _AlwaysRejectObjective(_AlwaysAcceptObjective):
    def accepted(self, candidate, reference):
        del candidate, reference
        return False, "rejected by test objective"


class GlobalWeightRefinementTest(unittest.TestCase):
    def _quantizer(self):
        source = torch.tensor(
            [
                [[[-0.12, 0.17]]],
                [[[0.31, -0.45]]],
            ],
            dtype=torch.float32,
        )
        initial_codes = torch.tensor(
            [
                [[[126.0, 130.0]]],
                [[[130.0, 125.0]]],
            ]
        )
        quantizer = CheckpointInitializedScaleAdaRoundQuantizer(
            _observer(),
            source,
            initial_codes,
            gamma=-0.1,
            zeta=1.1,
            initialization_epsilon=1e-6,
            max_scale_ratio=1.25,
        )
        return source, initial_codes, quantizer

    def _weight_set(self):
        owner = _Owner(nn.Conv2d(2, 2, 1, bias=False))
        source = torch.tensor(
            [
                [[[0.13]], [[-0.23]]],
                [[[0.81]], [[-0.41]]],
            ],
            dtype=torch.float32,
        )
        original_observer = owner.obs_weight
        scale, zero_point = original_observer.compute_qparams()
        shape = (-1, 1, 1, 1)
        codes = torch.round(
            source / scale.reshape(shape) + zero_point.reshape(shape)
        ).clamp(UINT8.qmin, UINT8.qmax)
        with torch.no_grad():
            owner.module.weight.copy_(
                (codes - zero_point.reshape(shape)) * scale.reshape(shape)
            )
        entry_weight = owner.module.weight.detach().clone()
        entry_scale = scale.detach().clone()
        entry_zero_point = zero_point.detach().clone()
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
            global_module,
            "iter_quantization_sites",
            return_value=(site,),
        ):
            weights = GlobalAdaRoundWeightSet(
                owner,
                (group,),
                {"block.weight": source},
                gamma=-0.1,
                zeta=1.1,
                initialization_epsilon=1e-6,
                max_scale_ratio=1.25,
            )
        return (
            owner,
            source,
            codes,
            weights,
            original_observer,
            entry_weight,
            entry_scale,
            entry_zero_point,
        )

    def test_initial_hard_state_reproduces_checkpoint_codes(self) -> None:
        _, initial_codes, quantizer = self._quantizer()
        quantizer.set_hard(True)
        self.assertTrue(torch.equal(quantizer.quantized_codes(), initial_codes))

    def test_initial_hard_state_replays_checkpoint_effective_weight_exactly(
        self,
    ) -> None:
        source = torch.tensor([[[[0.137]]]], dtype=torch.float32)
        observer = _observer(1)
        entry = observer.fake_quant(source)
        scale, zero_point = observer.compute_qparams()
        codes = torch.round(
            entry / scale.reshape(1, 1, 1, 1) + zero_point.reshape(1, 1, 1, 1)
        ).clamp(UINT8.qmin, UINT8.qmax)
        quantizer = CheckpointInitializedScaleAdaRoundQuantizer(
            observer,
            source,
            codes,
            entry,
            gamma=-0.1,
            zeta=1.1,
            initialization_epsilon=1e-6,
            max_scale_ratio=1.25,
        )
        quantizer.set_hard(True)
        self.assertTrue(torch.equal(quantizer.fake_quant(source), entry))
        self.assertTrue(torch.equal(quantizer.hard_weight(), entry))

    def test_nearest_override_uses_current_scale_rtn(self) -> None:
        source, initial_codes, quantizer = self._quantizer()
        quantizer.set_hard(True)
        quantizer.set_nearest_override(True)
        scale, zero_point = quantizer.compute_qparams()
        normalized = source / scale.reshape(-1, 1, 1, 1)
        normalized = normalized + zero_point.reshape(-1, 1, 1, 1)
        expected = torch.round(normalized).clamp(UINT8.qmin, UINT8.qmax)
        self.assertTrue(torch.equal(quantizer.quantized_codes(), expected))
        self.assertFalse(torch.equal(expected, initial_codes))

    def test_soft_path_provides_alpha_and_scale_gradients(self) -> None:
        source, _, quantizer = self._quantizer()
        quantizer.set_hard(False)
        output = quantizer.fake_quant(source)
        output.square().mean().backward()
        self.assertIsNotNone(quantizer.alpha.grad)
        self.assertIsNotNone(quantizer.raw_log_scale_delta.grad)
        self.assertTrue(torch.isfinite(quantizer.alpha.grad).all())
        self.assertTrue(torch.isfinite(quantizer.raw_log_scale_delta.grad).all())

    def test_finalize_commits_hard_weight_and_learned_scale(self) -> None:
        (
            owner,
            _,
            _,
            weights,
            original_observer,
            _,
            _,
            _,
        ) = self._weight_set()
        proxy = weights.bindings[0].proxy
        proxy.raw_log_scale_delta.data.fill_(0.2)
        expected_scale = proxy.learned_scale().detach().clone()
        expected_weight = proxy.hard_weight().detach().clone()
        weights.finalize()
        self.assertIs(owner.obs_weight, original_observer)
        scale, _ = original_observer.compute_qparams()
        torch.testing.assert_close(scale, expected_scale)
        torch.testing.assert_close(owner.module.weight, expected_weight)
        torch.testing.assert_close(
            original_observer.fake_quant(owner.module.weight),
            owner.module.weight,
        )

    def test_restore_returns_checkpoint_weight_and_qparams(self) -> None:
        (
            owner,
            _,
            _,
            weights,
            original_observer,
            entry_weight,
            entry_scale,
            entry_zero_point,
        ) = self._weight_set()
        proxy = weights.bindings[0].proxy
        proxy.raw_log_scale_delta.data.fill_(0.5)
        weights.restore()
        scale, zero_point = original_observer.compute_qparams()
        self.assertIs(owner.obs_weight, original_observer)
        torch.testing.assert_close(owner.module.weight, entry_weight)
        torch.testing.assert_close(scale, entry_scale)
        torch.testing.assert_close(zero_point, entry_zero_point)

    def test_runner_uses_b1_accumulation_and_commits_atomically(self) -> None:
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
        samples = (
            torch.tensor([[[[1.0]]]]),
            torch.tensor([[[[2.0]]]]),
        )

        def adapter(outputs):
            return outputs

        def evaluate():
            absolute = {"regressors": 0.0, "classifiers": 0.0}
            count = {"regressors": 0, "classifiers": 0}
            with torch.no_grad():
                for sample in samples:
                    expected = reference(sample)
                    actual = candidate(sample)
                    for name in absolute:
                        absolute[name] += float(
                            (actual[name] - expected[name]).abs().sum()
                        )
                        count[name] += actual[name].numel()
            return {name: {"mae": absolute[name] / count[name]} for name in absolute}

        checkpoints: list[GlobalWeightRefinementCheckpoint] = []
        config = GlobalWeightRefinementConfig(
            steps=2,
            gradient_accumulation_steps=2,
            evaluation_interval=1,
            alpha_learning_rate=1e-3,
            scale_learning_rate=1e-4,
            rounding_loss_weight=1e-3,
            scale_loss_weight=1e-4,
        )
        with mock.patch.object(
            global_module,
            "iter_quantization_sites",
            return_value=(site,),
        ):
            result = GlobalWeightRefinementRunner(config).refine(
                reference_model=reference,
                candidate_model=candidate,
                training_samples=samples,
                weight_groups=(group,),
                source_weights={"block.weight": source_weight},
                output_adapter=adapter,
                selection_evaluator=evaluate,
                selection_objective=_AlwaysAcceptObjective(),  # type: ignore[arg-type]
                acceptance_evaluator=evaluate,
                acceptance_objective=_AlwaysAcceptObjective(),  # type: ignore[arg-type]
                evaluation_evaluator=evaluate,
                progress_callback=checkpoints.append,
            )
        self.assertTrue(result.accepted)
        self.assertEqual(result.gradient_accumulation_steps, 2)
        self.assertEqual(len(checkpoints), 3)
        self.assertIs(candidate.owner.obs_weight, observer)
        scale, zero_point = observer.compute_qparams()
        shape = (1, 1, 1, 1)
        effective = observer.fake_quant(candidate.owner.module.weight)
        torch.testing.assert_close(
            effective,
            candidate.owner.module.weight,
        )
        self.assertEqual(scale.numel(), 1)
        self.assertEqual(zero_point.numel(), 1)

    def test_rejected_runner_restores_loaded_checkpoint_state(self) -> None:
        source_weight = torch.tensor([[[[0.137]]]], dtype=torch.float32)
        reference = _TinyReference(source_weight)
        candidate = _TinyCandidate(source_weight)
        observer = candidate.owner.obs_weight
        entry_weight = candidate.owner.module.weight.detach().clone()
        entry_scale, entry_zero_point = observer.compute_qparams()
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
        )
        with mock.patch.object(
            global_module,
            "iter_quantization_sites",
            return_value=(site,),
        ):
            result = GlobalWeightRefinementRunner(config).refine(
                reference_model=reference,
                candidate_model=candidate,
                training_samples=samples,
                weight_groups=(group,),
                source_weights={"block.weight": source_weight},
                output_adapter=lambda outputs: outputs,
                selection_evaluator=evaluate,
                selection_objective=_AlwaysAcceptObjective(),  # type: ignore[arg-type]
                acceptance_evaluator=evaluate,
                acceptance_objective=_AlwaysRejectObjective(),  # type: ignore[arg-type]
                evaluation_evaluator=evaluate,
            )
        self.assertFalse(result.accepted)
        self.assertIs(candidate.owner.obs_weight, observer)
        scale, zero_point = observer.compute_qparams()
        torch.testing.assert_close(candidate.owner.module.weight, entry_weight)
        torch.testing.assert_close(scale, entry_scale)
        torch.testing.assert_close(zero_point, entry_zero_point)
        self.assertEqual(
            result.final_evaluation_outputs,
            result.entry_evaluation_outputs,
        )

    def test_normalized_l1_uses_final_output_shape(self) -> None:
        candidate = torch.tensor([[1.0, 3.0]])
        reference = torch.tensor([[1.0, 1.0]])
        value = _normalized_l1(candidate, reference, epsilon=1e-8)
        self.assertAlmostEqual(float(value), 1.0)
        with self.assertRaises(ValueError):
            _normalized_l1(
                candidate,
                reference.reshape(2, 1),
                epsilon=1e-8,
            )

    def test_global_refinement_requires_b1_samples(self) -> None:
        _validate_batch_one((torch.zeros(1, 2, 2, 3),))
        with self.assertRaises(ValueError):
            _validate_batch_one((torch.zeros(2, 2, 2, 3),))

    def test_initialization_validation_uses_absolute_and_relative_tolerance(
        self,
    ) -> None:
        runner = GlobalWeightRefinementRunner(
            GlobalWeightRefinementConfig(
                initialization_metric_tolerance=1e-4,
                initialization_metric_relative_tolerance=1e-3,
            )
        )
        entry = {
            "regressors": {"mae": 0.2192610107},
            "classifiers": {"mae": 0.0735},
        }
        close = {
            "regressors": {"mae": 0.2191410800},
            "classifiers": {"mae": 0.07355},
        }
        runner._validate_initialization(entry, close)
        far = {
            "regressors": {"mae": 0.217},
            "classifiers": {"mae": 0.0735},
        }
        with self.assertRaises(RuntimeError):
            runner._validate_initialization(entry, far)

    def test_config_rejects_invalid_accumulation(self) -> None:
        with self.assertRaises(ValueError):
            GlobalWeightRefinementConfig(gradient_accumulation_steps=0).validate()


if __name__ == "__main__":
    unittest.main()
