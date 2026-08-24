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

"""Tests for per-window QDrop probability/seed multi-start selection."""

from __future__ import annotations

import unittest

from unittest.mock import patch

import torch

from tico.quantization.algorithm.block_reconstruction import (
    AffineObserverGroup,
    AffineQParamSnapshot,
    BlockReconstructionConfig,
    BlockReconstructionResult,
    build_qdrop_candidates,
    QDropMultiStartReconstructor,
    ValidationObjective,
)
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.observers.minmax import MinMaxObserver
from tico.quantization.wrapq.qscheme import QScheme
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from torch import nn


class _QuantizedIdentity(QuantModuleBase):
    def __init__(self) -> None:
        super().__init__()
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
        for observer in (self.obs_act_in, self.obs_act_out):
            observer.load_qparams(
                torch.tensor(0.25),
                torch.tensor(0, dtype=torch.int),
                lock=True,
            )

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        return self.obs_act_out.fake_quant(self.obs_act_in.fake_quant(input_))

    def _all_observers(self):
        return self.obs_act_in, self.obs_act_out


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = _QuantizedIdentity()

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        return self.block(input_)


def _outputs(value: float):
    return {
        "regressors": {"mae": value},
        "classifiers": {"mae": value},
    }


def _result(name: str) -> BlockReconstructionResult:
    return BlockReconstructionResult(
        block=name,
        steps=1,
        cache_samples=1,
        qparam_groups=("tensor_0",),
        initial_loss=1.0,
        final_loss=0.5,
        best_step=1,
        qparams={},
        training_loss_history=(0.5,),
        evaluation_loss_history=((0, 1.0), (1, 0.5)),
        accepted=True,
    )


class QDropMultiStartTest(unittest.TestCase):
    def test_candidate_product_deduplicates_control_seed(self) -> None:
        candidates = build_qdrop_candidates(
            (0.0, 0.25),
            (17, 27, 17),
        )
        self.assertEqual(
            tuple(candidate.name for candidate in candidates),
            (
                "qdrop_0",
                "qdrop_0_25_seed_17",
                "qdrop_0_25_seed_27",
            ),
        )

    def test_snapshot_restores_locked_qparams(self) -> None:
        model = _Model()
        group = AffineObserverGroup("tensor_0", ("block.act_in",))
        snapshot = AffineQParamSnapshot.capture(model, (group,))
        model.block.obs_act_in.load_qparams(
            torch.tensor(0.5),
            torch.tensor(3, dtype=torch.int),
            lock=True,
        )
        snapshot.restore(model)
        scale, zero_point = model.block.obs_act_in.compute_qparams()
        torch.testing.assert_close(scale, torch.tensor(0.25))
        torch.testing.assert_close(zero_point, torch.tensor(0, dtype=torch.int))

    def test_all_candidates_start_from_entry_and_best_acceptance_wins(self) -> None:
        model = _Model()
        group = AffineObserverGroup("tensor_0", ("block.act_in",))
        candidates = build_qdrop_candidates((0.0, 0.25), (17, 27))
        observed_entry_scales: list[float] = []

        def fake_reconstruct(reconstructor, **kwargs):
            observer = kwargs["observer_model"].block.obs_act_in
            scale, _ = observer.compute_qparams()
            observed_entry_scales.append(float(scale))
            probability = reconstructor.config.qdrop_probability
            seed = reconstructor.config.qdrop_seed
            if probability == 0.0:
                selected_scale = 0.20
            elif seed == 17:
                selected_scale = 0.10
            else:
                selected_scale = 0.15
            observer.load_qparams(
                torch.tensor(selected_scale),
                torch.tensor(0, dtype=torch.int),
                lock=True,
            )
            return _result(kwargs["block_name"])

        def evaluator():
            scale, _ = model.block.obs_act_in.compute_qparams()
            return _outputs(float(scale))

        objective = ValidationObjective(
            primary_output="regressors",
            output_tolerances={"classifiers": 0.0},
        )
        with patch(
            "tico.quantization.algorithm.block_reconstruction.multistart."
            "BlockReconstructor.reconstruct",
            autospec=True,
            side_effect=fake_reconstruct,
        ):
            result = QDropMultiStartReconstructor(
                BlockReconstructionConfig(steps=1),
                candidates,
                acceptance_objective=objective,
            ).reconstruct(
                block_name="identity",
                observer_model=model,
                block=model.block,
                # The mocked runner does not inspect caches.
                cache=object(),  # type: ignore[arg-type]
                selection_cache=object(),  # type: ignore[arg-type]
                observer_groups=(group,),
                selection_evaluator=evaluator,
                selection_objective=objective,
                acceptance_evaluator=evaluator,
            )

        self.assertEqual(observed_entry_scales, [0.25, 0.25, 0.25])
        self.assertIsNotNone(result.winner)
        assert result.winner is not None
        self.assertEqual(result.winner.name, "qdrop_0_25_seed_17")
        scale, _ = model.block.obs_act_in.compute_qparams()
        self.assertAlmostEqual(float(scale), 0.10)
        self.assertEqual(
            sum(candidate.selected_as_winner for candidate in result.candidates),
            1,
        )

    def test_entry_state_wins_when_acceptance_does_not_improve(self) -> None:
        model = _Model()
        group = AffineObserverGroup("tensor_0", ("block.act_in",))

        def fake_reconstruct(reconstructor, **kwargs):
            observer = kwargs["observer_model"].block.obs_act_in
            observer.load_qparams(
                torch.tensor(0.30 + reconstructor.config.qdrop_probability),
                torch.tensor(0, dtype=torch.int),
                lock=True,
            )
            return _result(kwargs["block_name"])

        def evaluator():
            scale, _ = model.block.obs_act_in.compute_qparams()
            return _outputs(float(scale))

        objective = ValidationObjective(
            primary_output="regressors",
            output_tolerances={"classifiers": 0.0},
        )
        with patch(
            "tico.quantization.algorithm.block_reconstruction.multistart."
            "BlockReconstructor.reconstruct",
            autospec=True,
            side_effect=fake_reconstruct,
        ):
            result = QDropMultiStartReconstructor(
                BlockReconstructionConfig(steps=1),
                build_qdrop_candidates((0.0, 0.25), (17,)),
                acceptance_objective=objective,
            ).reconstruct(
                block_name="identity",
                observer_model=model,
                block=model.block,
                cache=object(),  # type: ignore[arg-type]
                selection_cache=object(),  # type: ignore[arg-type]
                observer_groups=(group,),
                selection_evaluator=evaluator,
                selection_objective=objective,
                acceptance_evaluator=evaluator,
            )

        self.assertFalse(result.accepted)
        self.assertIsNone(result.winner)
        scale, _ = model.block.obs_act_in.compute_qparams()
        self.assertAlmostEqual(float(scale), 0.25)


if __name__ == "__main__":
    unittest.main()
