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

"""Tests for runtime A/B/C/D quantization ablation."""

import unittest

import torch

from tico.quantization.analysis import (
    QuantizationAblation,
    QuantizationBoundaries,
    QuantizationProfile,
    SiteSelector,
)
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.config.specs import affine
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.observers.minmax import MinMaxObserver
from tico.quantization.wrapq.qscheme import QScheme
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from torch import nn


class TinyQuantBlock(QuantModuleBase):
    def __init__(self) -> None:
        config = PTQConfig(
            activation=affine(
                DType.uint(8),
                qscheme=QScheme.PER_TENSOR_ASYMM,
            ),
            weight=affine(
                DType.uint(8),
                qscheme=QScheme.PER_TENSOR_ASYMM,
            ),
            strict_wrap=False,
        )
        super().__init__(config)
        self.weight = nn.Parameter(torch.tensor([0.37]))
        self.obs_weight = MinMaxObserver(
            name="weight",
            dtype=DType.uint(8),
            qscheme=QScheme.PER_TENSOR_ASYMM,
        )
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

    def enable_calibration(self) -> None:
        super().enable_calibration()
        self.obs_weight.collect(self.weight.detach())

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        input_q = self._fq(input_, self.obs_act_in)
        weight = self.weight
        if self._mode.name == "QUANT":
            weight = self.obs_weight.fake_quant(weight)
        output = input_q * weight
        return self._fq(output, self.obs_act_out)

    def _all_observers(self):
        return self.obs_weight, self.obs_act_in, self.obs_act_out


class TinyModel(nn.Module):
    def __init__(self, quantized: bool) -> None:
        super().__init__()
        self.block = TinyQuantBlock()
        if quantized:
            self.block.enable_calibration()
            self.block(torch.tensor([0.13, 0.91, 1.37]))
            self.block.freeze_qparams()

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        return self.block(input_)


class QuantizationAblationTest(unittest.TestCase):
    def test_profiles_toggle_expected_site_groups_and_restore_state(self) -> None:
        reference = TinyModel(quantized=False)
        candidate = TinyModel(quantized=True)
        reference.block.weight.data.copy_(candidate.block.weight.data)
        boundaries = QuantizationBoundaries(outputs=SiteSelector.paths("block.act_out"))
        original_states = {
            observer.name: observer.fake_quant_enabled
            for observer in candidate.block._all_observers()
        }
        report = QuantizationAblation(
            reference,
            candidate,
            boundaries=boundaries,
        ).run([torch.tensor([0.19, 0.73, 1.11])])

        self.assertEqual(
            report.profiles[QuantizationProfile.OUTPUT_ONLY].enabled_site_count,
            1,
        )
        self.assertEqual(
            report.profiles[QuantizationProfile.WEIGHT_ONLY].enabled_site_count,
            1,
        )
        self.assertEqual(
            report.profiles[QuantizationProfile.ACTIVATION_ONLY].enabled_site_count,
            1,
        )
        self.assertEqual(
            report.profiles[QuantizationProfile.FULL].enabled_site_count,
            3,
        )
        restored = {
            observer.name: observer.fake_quant_enabled
            for observer in candidate.block._all_observers()
        }
        self.assertEqual(restored, original_states)
        self.assertLess(float(report.float_parity["output"]["mae"]), 1e-7)

    def test_rejects_an_empty_output_boundary(self) -> None:
        reference = TinyModel(quantized=False)
        candidate = TinyModel(quantized=True)
        reference.block.weight.data.copy_(candidate.block.weight.data)
        runner = QuantizationAblation(
            reference,
            candidate,
            boundaries=QuantizationBoundaries(
                outputs=SiteSelector.paths("block.missing")
            ),
        )
        with self.assertRaisesRegex(ValueError, "output selector"):
            runner.run(
                [torch.tensor([0.19, 0.73, 1.11])],
                profiles=(QuantizationProfile.OUTPUT_ONLY,),
            )


if __name__ == "__main__":
    unittest.main()
