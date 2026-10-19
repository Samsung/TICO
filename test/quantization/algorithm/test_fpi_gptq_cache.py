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

import unittest
from types import SimpleNamespace
from typing import cast

import torch
import torch.nn as nn

from tico.quantization import convert, prepare
from tico.quantization.algorithm.fpi_gptq.quantizer import FPIGPTQQuantizer
from tico.quantization.algorithm.gptq.quant import Quantizer
from tico.quantization.config.fpi_gptq import FPIGPTQConfig
from tico.quantization.public_interface import QUANTIZER_ATTRIBUTE_NAME


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(use_cache=True)
        self.linear = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.diag(torch.tensor([1.0, 2.0])))

    def forward(self, x):
        return self.linear(x)


class TestFPIGPTQCacheLifecycle(unittest.TestCase):
    def setUp(self):
        rng_state = torch.random.get_rng_state()
        self.addCleanup(torch.random.set_rng_state, rng_state)
        self.model = _TinyModel().eval()

    def _assert_cache_state(self, expected):
        if hasattr(self.model, "config"):
            if expected is None:
                self.assertFalse(hasattr(self.model.config, "use_cache"))
            else:
                self.assertIs(self.model.config.use_cache, expected)
        else:
            self.assertIsNone(expected)

    def _assert_success(self, original):
        config = getattr(self.model, "config", None)
        weight = self.model.linear.weight.detach().clone()
        self.assertIs(
            prepare(self.model, FPIGPTQConfig(show_progress=False)), self.model
        )
        quantizer = cast(
            FPIGPTQQuantizer, getattr(self.model, QUANTIZER_ATTRIBUTE_NAME)
        )
        during_calibration = False if original is not None else None
        self._assert_cache_state(during_calibration)
        for inputs in torch.eye(2).split(1):
            self.assertIsNone(self.model(inputs))
        self._assert_cache_state(during_calibration)
        self.assertEqual(quantizer.num_batches, 2)

        self.assertIs(convert(self.model), self.model)
        self.assertTrue(torch.equal(self.model.linear.weight, weight))
        self.assertTrue(torch.equal(self.model(torch.eye(2)), weight))
        quantizers = cast(dict[str, Quantizer], self.model.quantizers)
        self.assertEqual(list(quantizers), ["0.linear"])
        qparams = quantizers["0.linear"]
        self.assertEqual(qparams.maxq.item(), 255)
        self.assertEqual(qparams.scale.dtype, torch.float32)
        self.assertEqual(qparams.zero.dtype, torch.float32)
        self.assertEqual(qparams.scale.shape, (2, 1))
        self.assertEqual(qparams.zero.shape, (2, 1))
        self.assertTrue(torch.equal(qparams.scale, torch.tensor([[1.0], [2.0]]) / 255))
        self.assertTrue(torch.equal(qparams.zero, torch.zeros(2, 1)))
        self.assertEqual(quantizer.cache_args, [])
        self.assertEqual(quantizer.cache_kwargs, {})
        self.assertEqual(quantizer.num_batches, 0)
        self.assertFalse(hasattr(self.model, QUANTIZER_ATTRIBUTE_NAME))
        self.assertIs(getattr(self.model, "config", None), config)
        self._assert_cache_state(original)

    def test_original_true_is_restored(self):
        self._assert_success(True)

    def test_original_false_is_preserved(self):
        self.model.config.use_cache = False
        self._assert_success(False)

    def test_missing_config_is_preserved(self):
        del self.model.config
        self._assert_success(None)
        self.assertFalse(hasattr(self.model, "config"))

    def test_missing_cache_flag_is_preserved(self):
        del self.model.config.use_cache
        self._assert_success(None)
