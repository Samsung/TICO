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

"""Tests for held-out block reconstruction selection and rollback."""

from __future__ import annotations

import unittest

from types import SimpleNamespace
from unittest.mock import patch

import torch

from tico.quantization.algorithm.block_reconstruction import (
    AffineObserverGroup,
    BlockInvocation,
    BlockReconstructionConfig,
    BlockReconstructor,
    reconstruction_loss,
    ReconstructionCache,
    ReconstructionLoss,
    ReconstructionSample,
    ValidationObjective,
)
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.observers.minmax import MinMaxObserver
from tico.quantization.wrapq.qscheme import QScheme
from torch import nn


class _QuantizedIdentity(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.obs_act_out = MinMaxObserver(
            name="act_out",
            dtype=DType.uint(8),
            qscheme=QScheme.PER_TENSOR_ASYMM,
        )
        self.obs_act_out.load_qparams(
            torch.tensor(0.5),
            torch.tensor(0, dtype=torch.int),
            lock=True,
        )

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        return self.obs_act_out.fake_quant(input_) * self.weight


class ValidationAwareReconstructionTest(unittest.TestCase):
    def _cache(self) -> ReconstructionCache:
        values = (0.2, 0.7, 1.2, 1.7)
        return ReconstructionCache(
            tuple(
                ReconstructionSample(
                    float_input=BlockInvocation(args=(torch.tensor([[value]]),)),
                    quantized_input=BlockInvocation(args=(torch.tensor([[value]]),)),
                    target=torch.tensor([[value]]),
                )
                for value in values
            )
        )

    @staticmethod
    def _site(model: _QuantizedIdentity):
        return SimpleNamespace(
            path="block.act_out",
            module=model,
            observer=model.obs_act_out,
            observer_name="act_out",
        )

    def _run(self, model: _QuantizedIdentity, **kwargs):
        with patch(
            "tico.quantization.algorithm.block_reconstruction.observer."
            "iter_quantization_sites",
            return_value=(self._site(model),),
        ):
            return BlockReconstructor(kwargs.pop("config")).reconstruct(
                block_name="block",
                observer_model=model,
                block=model,
                cache=self._cache(),
                selection_cache=self._cache(),
                observer_groups=(AffineObserverGroup("tensor_0", ("block.act_out",)),),
                **kwargs,
            )

    def test_result_constructor_defaults_preserve_local_only_behavior(self) -> None:
        from tico.quantization.algorithm.block_reconstruction import (
            BlockReconstructionResult,
        )

        result = BlockReconstructionResult(
            block="block",
            steps=0,
            cache_samples=1,
            qparam_groups=("tensor_0",),
            initial_loss=1.0,
            final_loss=1.0,
            best_step=0,
            qparams={"tensor_0": {"scale": 0.5, "zero_point": 0}},
            training_loss_history=(),
            evaluation_loss_history=((0, 1.0),),
        )

        self.assertTrue(result.accepted)
        self.assertEqual(result.selected_qparams, {})

    def test_normalized_l1_and_mse_are_distinct(self) -> None:
        candidate = torch.tensor([1.0, 3.0])
        target = torch.tensor([1.0, 1.0])
        mse = reconstruction_loss(
            candidate,
            target,
            ReconstructionLoss.NORMALIZED_MSE,
        )
        l1 = reconstruction_loss(
            candidate,
            target,
            ReconstructionLoss.NORMALIZED_L1,
        )
        self.assertAlmostEqual(float(mse), 2.0)
        self.assertAlmostEqual(float(l1), 1.0)

    def test_classifier_regression_rejects_and_restores_entry_qparams(self) -> None:
        model = _QuantizedIdentity()
        original_scale = model.obs_act_out.compute_qparams()[0].clone()
        values = iter(((1.0, 0.0), (0.8, 0.1), (0.7, 0.2)))

        def evaluator():
            regressor, classifier = next(values)
            return {
                "regressors": {"mae": regressor},
                "classifiers": {"mae": classifier},
            }

        result = self._run(
            model,
            config=BlockReconstructionConfig(
                steps=2,
                batch_size=2,
                evaluation_batch_size=2,
                evaluation_interval=1,
                optimize_zero_point=False,
            ),
            selection_evaluator=evaluator,
            selection_objective=ValidationObjective(
                primary_output="regressors",
                output_tolerances={"classifiers": 0.0},
            ),
        )

        self.assertFalse(result.accepted)
        self.assertEqual(result.best_step, 0)
        torch.testing.assert_close(
            model.obs_act_out.compute_qparams()[0],
            original_scale,
        )
        self.assertEqual(result.qparams, result.selected_qparams)

    def test_globally_improved_state_is_committed(self) -> None:
        model = _QuantizedIdentity()
        weight = model.weight.detach().clone()
        requires_grad = model.weight.requires_grad
        values = iter((1.0, 0.8, 0.9))

        def evaluator():
            return {
                "regressors": {"mae": next(values)},
                "classifiers": {"mae": 0.0},
            }

        result = self._run(
            model,
            config=BlockReconstructionConfig(
                steps=2,
                batch_size=2,
                evaluation_batch_size=2,
                evaluation_interval=1,
                optimize_zero_point=False,
            ),
            selection_evaluator=evaluator,
            selection_objective=ValidationObjective(
                primary_output="regressors",
                output_tolerances={"classifiers": 0.0},
            ),
        )

        self.assertTrue(result.accepted)
        self.assertEqual(result.best_step, 1)
        torch.testing.assert_close(model.weight.detach(), weight)
        self.assertEqual(model.weight.requires_grad, requires_grad)
        self.assertEqual(result.qparams, result.selected_qparams)

    def test_step_zero_can_win_and_restore_original_state(self) -> None:
        model = _QuantizedIdentity()
        original_scale = model.obs_act_out.compute_qparams()[0].clone()
        values = iter((1.0, 1.1, 1.2))

        def evaluator():
            return {
                "regressors": {"mae": next(values)},
                "classifiers": {"mae": 0.0},
            }

        result = self._run(
            model,
            config=BlockReconstructionConfig(
                steps=2,
                batch_size=2,
                evaluation_batch_size=2,
                evaluation_interval=1,
                optimize_zero_point=False,
            ),
            selection_evaluator=evaluator,
            selection_objective=ValidationObjective(
                primary_output="regressors",
                output_tolerances={"classifiers": 0.0},
            ),
        )

        self.assertFalse(result.accepted)
        self.assertEqual(result.best_step, 0)
        torch.testing.assert_close(
            model.obs_act_out.compute_qparams()[0],
            original_scale,
        )

    def test_best_held_out_checkpoint_can_precede_last_step(self) -> None:
        model = _QuantizedIdentity()
        values = iter((1.0, 0.8, 0.9))

        def evaluator():
            return {
                "regressors": {"mae": next(values)},
                "classifiers": {"mae": 0.0},
            }

        result = self._run(
            model,
            config=BlockReconstructionConfig(
                steps=2,
                batch_size=2,
                evaluation_batch_size=2,
                evaluation_interval=1,
                optimize_zero_point=False,
            ),
            selection_evaluator=evaluator,
            selection_objective=ValidationObjective(
                primary_output="regressors",
                output_tolerances={"classifiers": 0.0},
            ),
        )

        self.assertTrue(result.accepted)
        self.assertEqual(result.best_step, 1)
        self.assertAlmostEqual(
            float(result.selected_outputs["regressors"]["mae"]),
            0.8,
        )

    def test_minimum_improvement_is_measured_from_entry_state(self) -> None:
        objective = ValidationObjective(
            primary_output="regressors",
            minimum_improvement=0.05,
            output_tolerances={"classifiers": 0.0},
        )
        reference = {
            "regressors": {"mae": 1.0},
            "classifiers": {"mae": 0.1},
        }
        candidate = {
            "regressors": {"mae": 0.96},
            "classifiers": {"mae": 0.1},
        }
        accepted, _ = objective.accepted(candidate, reference)
        self.assertFalse(accepted)


if __name__ == "__main__":
    unittest.main()
