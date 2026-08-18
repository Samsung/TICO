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

"""Tests for fixed-weight activation block reconstruction."""

from __future__ import annotations

import unittest

import torch

from tico.quantization.algorithm.block_reconstruction import (
    AffineObserverGroup,
    BlockInvocation,
    BlockReconstructionConfig,
    BlockReconstructor,
    normalized_mse_loss,
    ReconstructionCache,
    ReconstructionSample,
)
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.observers.minmax import MinMaxObserver
from tico.quantization.wrapq.qscheme import QScheme
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase

from torch import nn


class _QuantizedIdentity(QuantModuleBase):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
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
        value = self.obs_act_in.fake_quant(input_)
        value = value * self.weight
        return self.obs_act_out.fake_quant(value)

    def _all_observers(self):
        return self.obs_act_in, self.obs_act_out


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = _QuantizedIdentity()

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        return self.block(input_)


class BlockReconstructionTest(unittest.TestCase):
    def test_normalized_mse_supports_multiple_outputs(self) -> None:
        candidate = (torch.tensor([1.0, 2.0]), torch.tensor([3.0]))
        target = (torch.tensor([1.0, 1.0]), torch.tensor([1.0]))
        loss = normalized_mse_loss(candidate, target)
        self.assertAlmostEqual(float(loss), 5.0 / 3.0)

    def test_reconstruction_reduces_loss_and_keeps_weight_fixed(self) -> None:
        model = _Model()
        weight_before = model.block.weight.detach().clone()
        requires_grad_before = model.block.weight.requires_grad
        values = (
            torch.tensor([[0.07], [0.14], [0.31], [0.46]], dtype=torch.float32),
            torch.tensor([[0.11], [0.23], [0.37], [0.59]], dtype=torch.float32),
        )
        samples = tuple(
            ReconstructionSample(
                float_input=BlockInvocation(args=(value,)),
                quantized_input=BlockInvocation(args=(value,)),
                target=value,
            )
            for value in values
        )
        result = BlockReconstructor(
            BlockReconstructionConfig(
                steps=200,
                batch_size=2,
                evaluation_batch_size=2,
                evaluation_interval=10,
                scale_learning_rate=2.0e-2,
                zero_point_learning_rate=1.0e-2,
                seed=7,
            )
        ).reconstruct(
            block_name="identity",
            observer_model=model,
            block=model.block,
            cache=ReconstructionCache(samples),
            observer_groups=(
                AffineObserverGroup(
                    "tensor_0",
                    ("block.act_in", "block.act_out"),
                ),
            ),
        )

        self.assertLess(result.final_loss, result.initial_loss * 0.2)
        self.assertGreater(result.best_step, 0)
        self.assertAlmostEqual(
            result.final_loss,
            min(loss for _, loss in result.evaluation_loss_history),
        )
        torch.testing.assert_close(model.block.weight, weight_before)
        self.assertEqual(model.block.weight.requires_grad, requires_grad_before)
        in_scale, in_zero_point = model.block.obs_act_in.compute_qparams()
        out_scale, out_zero_point = model.block.obs_act_out.compute_qparams()
        torch.testing.assert_close(in_scale, out_scale)
        torch.testing.assert_close(in_zero_point, out_zero_point)
        self.assertNotAlmostEqual(float(in_scale), 0.25)

    def test_tied_group_rejects_inconsistent_initial_qparams(self) -> None:
        model = _Model()
        model.block.obs_act_out.load_qparams(
            torch.tensor(0.125),
            torch.tensor(0, dtype=torch.int),
            lock=True,
        )
        sample = ReconstructionSample(
            float_input=BlockInvocation(args=(torch.tensor([[0.1]]),)),
            quantized_input=BlockInvocation(args=(torch.tensor([[0.1]]),)),
            target=torch.tensor([[0.1]]),
        )
        with self.assertRaisesRegex(ValueError, "inconsistent scales"):
            BlockReconstructor(BlockReconstructionConfig(steps=1)).reconstruct(
                block_name="identity",
                observer_model=model,
                block=model.block,
                cache=ReconstructionCache((sample,)),
                observer_groups=(
                    AffineObserverGroup(
                        "tensor_0",
                        ("block.act_in", "block.act_out"),
                    ),
                ),
            )


if __name__ == "__main__":
    unittest.main()
