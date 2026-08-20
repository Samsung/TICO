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

"""Tests for fixed-qparam learnable Conv2d weight rounding."""

from __future__ import annotations

import copy
import unittest

import torch

from tico.quantization.algorithm.adaround import (
    AdaRoundConfig,
    AdaRoundRunner,
    AdaRoundWeightGroup,
    AdaRoundWeightQuantizer,
    AdaRoundWeightSet,
)
from tico.quantization.algorithm.block_reconstruction import (
    BlockInvocation,
    ReconstructionCache,
    ReconstructionLoss,
    ReconstructionSample,
    ValidationObjective,
)
from tico.quantization.wrapq.control import iter_quantization_sites
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.observers.minmax import MinMaxObserver
from tico.quantization.wrapq.qscheme import QScheme
from tico.quantization.wrapq.wrappers.nn.quant_conv2d import QuantConv2d
from torch import nn


class _Model(nn.Module):
    def __init__(self, wrapper: QuantConv2d) -> None:
        super().__init__()
        self.block = wrapper

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        return self.block(input_)


class AdaRoundTest(unittest.TestCase):
    def test_initial_hard_rounding_matches_existing_fake_quant(self) -> None:
        weight = torch.tensor(
            [
                [[[-1.2, -0.2, 0.7, 1.3]]],
                [[[-2.3, -0.4, 0.4, 2.1]]],
            ],
            dtype=torch.float32,
        )
        observer = MinMaxObserver(
            name="weight",
            dtype=DType.uint(8),
            qscheme=QScheme.PER_CHANNEL_ASYMM,
            channel_axis=0,
        )
        observer.collect(weight)
        observer.compute_qparams()
        expected = observer.fake_quant(weight)

        quantizer = AdaRoundWeightQuantizer(observer, weight)
        quantizer.set_hard(True)
        torch.testing.assert_close(quantizer.fake_quant(weight), expected)

    def test_exact_half_ties_preserve_existing_rounding(self) -> None:
        weight = torch.tensor([0.5, 1.5, 2.5, 3.5], dtype=torch.float32)
        observer = MinMaxObserver(
            name="weight",
            dtype=DType.int(8),
            qscheme=QScheme.PER_TENSOR_SYMM,
        )
        observer.load_qparams(
            torch.tensor(1.0),
            torch.tensor(0, dtype=torch.int),
            lock=True,
        )
        expected = observer.fake_quant(weight)
        quantizer = AdaRoundWeightQuantizer(observer, weight)
        quantizer.set_hard(True)
        torch.testing.assert_close(quantizer.fake_quant(weight), expected)

    def test_finalize_restores_observer_and_writes_grid_weight(self) -> None:
        fp = nn.Conv2d(1, 2, kernel_size=1, bias=False)
        fp.weight.data.copy_(torch.tensor([[[[0.31]]], [[[0.73]]]]))
        wrapper = QuantConv2d(fp)
        wrapper.enable_calibration()
        wrapper.obs_weight.compute_qparams()
        model = _Model(wrapper)
        site = next(
            site
            for site in iter_quantization_sites(model)
            if site.observer_name == "weight"
        )
        original_observer = wrapper.obs_weight
        original_weight = fp.weight.detach().clone()
        weights = AdaRoundWeightSet(
            model,
            (AdaRoundWeightGroup("block.weight", site.path),),
            gamma=-0.1,
            zeta=1.1,
            initialization_epsilon=1e-6,
        )
        with torch.no_grad():
            weights.bindings[0].proxy.alpha.mul_(-1.0)
        statistics = weights.finalize()

        self.assertIs(wrapper.obs_weight, original_observer)
        self.assertEqual(len(statistics), 1)
        self.assertFalse(torch.equal(fp.weight, original_weight))
        torch.testing.assert_close(
            original_observer.fake_quant(fp.weight),
            fp.weight,
        )

    def test_restore_preserves_original_weight_and_observer(self) -> None:
        fp = nn.Conv2d(1, 1, kernel_size=1, bias=False)
        fp.weight.data.fill_(0.37)
        wrapper = QuantConv2d(fp)
        wrapper.enable_calibration()
        wrapper.obs_weight.compute_qparams()
        model = _Model(wrapper)
        site = next(
            site
            for site in iter_quantization_sites(model)
            if site.observer_name == "weight"
        )
        original_observer = wrapper.obs_weight
        original_weight = fp.weight.detach().clone()
        weights = AdaRoundWeightSet(
            model,
            (AdaRoundWeightGroup("block.weight", site.path),),
            gamma=-0.1,
            zeta=1.1,
            initialization_epsilon=1e-6,
        )
        weights.restore()
        self.assertIs(wrapper.obs_weight, original_observer)
        torch.testing.assert_close(fp.weight, original_weight)

    def test_runner_accepts_a_better_hard_rounding_state(self) -> None:
        fp = nn.Conv2d(1, 1, kernel_size=1, bias=False)
        fp.weight.data.fill_(0.6)
        wrapper = QuantConv2d(copy.deepcopy(fp))
        wrapper.enable_calibration()
        sample = torch.ones(1, 1, 1, 1)
        wrapper(sample)
        wrapper.freeze_qparams()
        wrapper.obs_weight.load_qparams(
            torch.tensor([1.0]),
            torch.tensor([0], dtype=torch.int),
            lock=True,
        )
        wrapper.obs_act_in.disable_fake_quant()
        wrapper.obs_act_out.disable_fake_quant()
        model = _Model(wrapper)
        target = torch.zeros_like(sample)
        cache = ReconstructionCache(
            (
                ReconstructionSample(
                    float_input=BlockInvocation(args=(sample,)),
                    quantized_input=BlockInvocation(args=(sample,)),
                    target=target,
                ),
            )
        )
        site = next(
            site
            for site in iter_quantization_sites(model)
            if site.observer_name == "weight"
        )

        def evaluator():
            value = model(sample).detach().abs().mean().item()
            return {"output": {"mae": value}}

        result = AdaRoundRunner(
            AdaRoundConfig(
                steps=100,
                batch_size=1,
                evaluation_batch_size=1,
                evaluation_interval=5,
                alpha_learning_rate=0.1,
                rounding_loss_weight=1.0e-3,
                warmup_fraction=0.0,
                beta_start=2.0,
                beta_end=2.0,
                reconstruction_loss=ReconstructionLoss.NORMALIZED_L1,
            )
        ).reconstruct(
            block_name="block",
            observer_model=model,
            block=wrapper,
            cache=cache,
            selection_cache=cache,
            weight_groups=(AdaRoundWeightGroup("block.weight", site.path),),
            selection_evaluator=evaluator,
            selection_objective=ValidationObjective(
                primary_output="output",
                output_tolerances={},
            ),
            acceptance_evaluator=evaluator,
            acceptance_objective=ValidationObjective(
                primary_output="output",
                output_tolerances={},
            ),
        )
        self.assertTrue(result.accepted)
        self.assertGreater(result.best_step, 0)
        torch.testing.assert_close(
            wrapper.module.weight,
            torch.zeros_like(wrapper.module.weight),
        )

    def test_runner_rolls_back_when_acceptance_requires_large_gain(self) -> None:
        fp = nn.Conv2d(1, 1, kernel_size=1, bias=False)
        fp.weight.data.fill_(0.37)
        wrapper = QuantConv2d(copy.deepcopy(fp))
        wrapper.enable_calibration()
        sample = torch.tensor([[[[0.2, 0.7]]]], dtype=torch.float32)
        wrapper(sample)
        wrapper.freeze_qparams()
        model = _Model(wrapper)
        target = fp(sample).detach()
        cache = ReconstructionCache(
            (
                ReconstructionSample(
                    float_input=BlockInvocation(args=(sample,)),
                    quantized_input=BlockInvocation(args=(sample,)),
                    target=target,
                ),
            )
        )
        site = next(
            site
            for site in iter_quantization_sites(model)
            if site.observer_name == "weight"
        )
        original_weight = wrapper.module.weight.detach().clone()

        calls = {"selection": 0, "acceptance": 0}

        def selection_evaluator():
            calls["selection"] += 1
            value = 1.0 if calls["selection"] == 1 else 0.5
            return {"output": {"mae": value}}

        def acceptance_evaluator():
            calls["acceptance"] += 1
            value = 1.0 if calls["acceptance"] == 1 else 0.5
            return {"output": {"mae": value}}

        result = AdaRoundRunner(
            AdaRoundConfig(
                steps=1,
                batch_size=1,
                evaluation_batch_size=1,
                evaluation_interval=1,
                alpha_learning_rate=1e-3,
                rounding_loss_weight=1e-2,
                reconstruction_loss=ReconstructionLoss.NORMALIZED_L1,
            )
        ).reconstruct(
            block_name="block",
            observer_model=model,
            block=wrapper,
            cache=cache,
            selection_cache=cache,
            weight_groups=(AdaRoundWeightGroup("block.weight", site.path),),
            selection_evaluator=selection_evaluator,
            selection_objective=ValidationObjective(
                primary_output="output",
                output_tolerances={},
            ),
            acceptance_evaluator=acceptance_evaluator,
            acceptance_objective=ValidationObjective(
                primary_output="output",
                minimum_improvement=0.75,
                output_tolerances={},
            ),
        )
        self.assertFalse(result.accepted)
        torch.testing.assert_close(wrapper.module.weight, original_weight)


if __name__ == "__main__":
    unittest.main()
