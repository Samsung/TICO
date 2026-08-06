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

"""Tests for WrapQ runtime fake-quantization control."""

import unittest

import torch
from tico.quantization.analysis import SiteSelector

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.config.specs import affine
from tico.quantization.wrapq.control import FakeQuantState, iter_quantization_sites
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.observers.minmax import MinMaxObserver
from tico.quantization.wrapq.qscheme import QScheme
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from torch import nn


class OneObserverModule(QuantModuleBase):
    def __init__(self) -> None:
        config = PTQConfig(
            activation=affine(DType.uint(8)),
            weight=affine(DType.uint(8)),
            strict_wrap=False,
        )
        super().__init__(config)
        self.obs_act_out = MinMaxObserver(
            name="act_out",
            dtype=DType.uint(8),
            qscheme=QScheme.PER_TENSOR_ASYMM,
        )

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        return self._fq(input_, self.obs_act_out)

    def _all_observers(self):
        return (self.obs_act_out,)


class FakeQuantControlTest(unittest.TestCase):
    def test_context_restores_original_switches(self) -> None:
        model = nn.Sequential(OneObserverModule())
        site = next(iter(iter_quantization_sites(model)))
        self.assertEqual(site.path, "0.act_out")
        with FakeQuantState(model) as state:
            state.set_all(False)
            self.assertFalse(site.observer.fake_quant_enabled)
        self.assertTrue(site.observer.fake_quant_enabled)

    def test_selector_can_match_owner_module_type(self) -> None:
        """Select a site through its owning quantized wrapper type."""
        model = nn.Sequential(OneObserverModule())
        site = next(iter(iter_quantization_sites(model)))
        self.assertTrue(SiteSelector.module_types(OneObserverModule)(site))

    def test_affine_observer_bypasses_qparams_when_disabled(self) -> None:
        observer = MinMaxObserver(
            name="act_out",
            dtype=DType.uint(8),
            qscheme=QScheme.PER_TENSOR_ASYMM,
        )
        value = torch.tensor([0.25])
        observer.disable_fake_quant()
        torch.testing.assert_close(observer.fake_quant(value), value)


if __name__ == "__main__":
    unittest.main()
