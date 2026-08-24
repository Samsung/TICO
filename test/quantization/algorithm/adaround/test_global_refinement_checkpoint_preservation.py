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

"""Tests for checkpoint-preserving global W8 refinement."""

from __future__ import annotations

import unittest

from types import SimpleNamespace
from unittest import mock

import torch

from tico.quantization.algorithm.adaround import global_refinement as global_module
from tico.quantization.algorithm.adaround.global_refinement import (
    CheckpointInitializedScaleAdaRoundQuantizer,
    GlobalAdaRoundWeightSet,
    GlobalWeightRefinementConfig,
    GlobalWeightRefinementRunner,
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


def _quantizer() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    CheckpointInitializedScaleAdaRoundQuantizer,
]:
    source = torch.tensor(
        [[[[-0.12, 0.17, 0.31, -0.45]]]],
        dtype=torch.float32,
    )
    observer = _observer(1)
    scale, zero_point = observer.compute_qparams()
    shape = (1, 1, 1, 1)
    normalized = source / scale.reshape(shape) + zero_point.reshape(shape)
    floor_codes = torch.floor(normalized)
    # Deliberately keep both nearest and non-nearest checkpoint decisions.
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
    )
    return source, initial_codes, effective, quantizer


class GlobalCheckpointPreservationTest(unittest.TestCase):
    def test_training_forward_uses_hard_codes_with_ste_gradients(self) -> None:
        source, initial_codes, effective, quantizer = _quantizer()
        quantizer.set_hard(False)
        training_codes = quantizer.quantized_codes()
        hard_codes = quantizer.quantized_codes(hard=True)
        torch.testing.assert_close(training_codes, hard_codes)
        self.assertTrue(torch.equal(hard_codes.detach(), initial_codes))

        output = quantizer.fake_quant(source)
        self.assertTrue(torch.equal(output.detach(), effective))
        output.square().mean().backward()
        self.assertIsNotNone(quantizer.alpha.grad)
        self.assertIsNotNone(quantizer.raw_log_scale_delta.grad)
        self.assertTrue(torch.isfinite(quantizer.alpha.grad).all())
        self.assertTrue(torch.isfinite(quantizer.raw_log_scale_delta.grad).all())
        self.assertGreater(float(quantizer.alpha.grad.abs().sum()), 0.0)
        self.assertGreater(
            float(quantizer.raw_log_scale_delta.grad.abs().sum()),
            0.0,
        )

    def test_checkpoint_margin_and_anchor_guard_initial_decisions(self) -> None:
        _, initial_codes, _, quantizer = _quantizer()
        mask = quantizer._checkpoint_decision_mask
        self.assertTrue(bool(mask.all()))
        self.assertGreaterEqual(
            float(quantizer.alpha.detach()[mask].abs().min()),
            1.5,
        )
        self.assertEqual(
            float(quantizer.checkpoint_anchor_regularizer(0.5)),
            0.0,
        )
        self.assertEqual(quantizer.checkpoint_sign_flip_count(), 0)

        first = int(torch.nonzero(mask.flatten(), as_tuple=False)[0].item())
        with torch.no_grad():
            flat = quantizer.alpha.flatten()
            flat[first] = -flat[first]
        self.assertGreater(
            float(quantizer.checkpoint_anchor_regularizer(0.5)),
            0.0,
        )
        self.assertEqual(quantizer.checkpoint_sign_flip_count(), 1)
        self.assertFalse(
            torch.equal(
                quantizer.quantized_codes(hard=True).detach(),
                initial_codes,
            )
        )

    def test_weight_set_reports_checkpoint_code_and_sign_drift(self) -> None:
        source = torch.tensor([[[[0.17]]]], dtype=torch.float32)
        owner = _Owner(nn.Conv2d(1, 1, 1, bias=False))
        observer = owner.obs_weight
        scale, zero_point = observer.compute_qparams()
        shape = (1, 1, 1, 1)
        code = torch.floor(source / scale.reshape(shape) + zero_point.reshape(shape))
        with torch.no_grad():
            owner.module.weight.copy_(
                (code - zero_point.reshape(shape)) * scale.reshape(shape)
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
            weights = GlobalAdaRoundWeightSet(
                owner,
                (group,),
                {"block.weight": source},
                gamma=-0.1,
                zeta=1.1,
                initialization_epsilon=1e-6,
                max_scale_ratio=1.25,
                checkpoint_alpha_minimum_magnitude=1.5,
            )
        initial = weights.hard_state_statistics()
        self.assertEqual(initial.changed_from_checkpoint_count, 0)
        self.assertEqual(initial.checkpoint_sign_flip_count, 0)
        self.assertAlmostEqual(initial.scale_ratio_minimum, 1.0)
        self.assertAlmostEqual(initial.scale_ratio_maximum, 1.0)

        proxy = weights.bindings[0].proxy
        with torch.no_grad():
            proxy.alpha.mul_(-1.0)
        changed = weights.hard_state_statistics()
        self.assertEqual(changed.checkpoint_sign_flip_count, 1)
        self.assertEqual(changed.changed_from_checkpoint_count, 1)
        weights.restore()

    def test_checkpoint_anchor_weight_decays_during_initial_fraction(self) -> None:
        runner = GlobalWeightRefinementRunner(
            GlobalWeightRefinementConfig(
                steps=100,
                checkpoint_anchor_loss_weight=1e-2,
                checkpoint_anchor_fraction=0.2,
            )
        )
        self.assertAlmostEqual(runner._checkpoint_anchor_weight(1), 1e-2)
        self.assertGreater(runner._checkpoint_anchor_weight(10), 0.0)
        self.assertAlmostEqual(runner._checkpoint_anchor_weight(20), 0.0)
        self.assertAlmostEqual(runner._checkpoint_anchor_weight(21), 0.0)

    def test_config_rejects_invalid_checkpoint_guard_values(self) -> None:
        with self.assertRaises(ValueError):
            GlobalWeightRefinementConfig(
                checkpoint_alpha_minimum_magnitude=0.0,
            ).validate()
        with self.assertRaises(ValueError):
            GlobalWeightRefinementConfig(
                checkpoint_anchor_fraction=1.1,
            ).validate()
        with self.assertRaises(ValueError):
            GlobalWeightRefinementConfig(
                checkpoint_anchor_margin=-0.1,
            ).validate()


if __name__ == "__main__":
    unittest.main()
