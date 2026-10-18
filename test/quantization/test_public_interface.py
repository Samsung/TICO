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

import torch
import torch.nn as nn

from tico.quantization import convert, prepare
from tico.quantization.config.gptq import GPTQConfig
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.public_interface import QUANTIZER_ATTRIBUTE_NAME
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase


class _LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(use_cache=True)
        self.linear = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.diag(torch.arange(1.0, 5.0)))

    def forward(self, x, *, residual):
        return self.linear(x + residual)


class PublicInterfaceTest(unittest.TestCase):
    def setUp(self):
        rng_state = torch.random.get_rng_state()
        self.addCleanup(torch.random.set_rng_state, rng_state)
        torch.manual_seed(0)

    @torch.no_grad()
    def test_rejected_gptq_out_of_place_convert_preserves_calibration_for_retry(self):
        model = _LinearModel().eval()
        original_forward = model.forward
        weights = model.linear.weight.clone()
        hook_calls = []
        handle = model.linear.register_forward_hook(
            lambda _module, _args, _output: hook_calls.append(True)
        )
        self.addCleanup(handle.remove)
        original_hooks = dict(model.linear._forward_hooks)
        prepared = prepare(model, GPTQConfig(weight_bits=4, show_progress=False))
        self.assertIs(prepared, model)
        quantizer = getattr(model, QUANTIZER_ATTRIBUTE_NAME)
        prepared_forward = model.forward
        inputs = torch.eye(4).unsqueeze(0)
        residual = torch.zeros_like(inputs)
        input_snapshot = inputs.clone()
        residual_snapshot = residual.clone()
        self.assertIsNone(prepared(inputs, residual=residual))
        cached_args = quantizer.cache_args
        cached_kwargs = quantizer.cache_kwargs
        cached_input = cached_args[0][0]
        cached_residual = cached_kwargs["residual"][0]
        self.assertEqual(quantizer.num_batches, 1)
        self.assertFalse(model.config.use_cache)

        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError, "only supports `in-place=True`"):
                convert(prepared, inplace=False)
            self.assertIs(getattr(prepared, QUANTIZER_ATTRIBUTE_NAME, None), quantizer)
            self.assertIs(model.forward, prepared_forward)
            self.assertIs(quantizer.cache_args, cached_args)
            self.assertIs(quantizer.cache_kwargs, cached_kwargs)
            self.assertIs(cached_args[0][0], cached_input)
            self.assertIs(cached_kwargs["residual"][0], cached_residual)
            self.assertTrue(torch.equal(cached_input, input_snapshot))
            self.assertTrue(torch.equal(cached_residual, residual_snapshot))
            self.assertEqual(quantizer.num_batches, 1)
            self.assertFalse(model.config.use_cache)
            self.assertEqual(dict(model.linear._forward_hooks), original_hooks)
            self.assertEqual(hook_calls, [])
            self.assertTrue(torch.equal(model.linear.weight, weights))

        self.assertIs(convert(prepared, inplace=True), model)
        self.assertFalse(hasattr(model, QUANTIZER_ATTRIBUTE_NAME))
        self.assertEqual(model.forward, original_forward)
        self.assertTrue(model.config.use_cache)
        self.assertEqual(quantizer.cache_args, [])
        self.assertEqual(quantizer.cache_kwargs, {})
        self.assertEqual(quantizer.num_batches, 0)
        self.assertEqual(dict(model.linear._forward_hooks), original_hooks)
        self.assertEqual(list(model.quantizers), ["linear"])
        self.assertEqual(model.quantizers["linear"].maxq.item(), 15)
        self.assertTrue(
            torch.equal(model(inputs, residual=residual), weights.unsqueeze(0))
        )
        with self.assertRaisesRegex(RuntimeError, "Call prepare\\(\\) function first"):
            convert(model)

    def test_convert_requires_prepare(self):
        for inplace in (False, True):
            with self.subTest(inplace=inplace):
                model = nn.Linear(4, 4, bias=False)
                weights = model.weight.detach().clone()
                with self.assertRaisesRegex(
                    RuntimeError, "Call prepare\\(\\) function first"
                ):
                    convert(model, inplace=inplace)
                self.assertTrue(torch.equal(model.weight, weights))
                self.assertFalse(hasattr(model, QUANTIZER_ATTRIBUTE_NAME))

    @torch.no_grad()
    def test_ptq_out_of_place_convert_consumes_quantizer_before_copy(self):
        prepared = prepare(nn.Linear(4, 4, bias=False).eval(), PTQConfig())
        self.assertIsInstance(prepared, QuantModuleBase)
        inputs = torch.eye(4)
        calibrated_output = prepared(inputs)
        prepared_weight = next(prepared.parameters())
        weights = prepared_weight.clone()
        quantized = convert(prepared, inplace=False)
        quantized_weight = next(quantized.parameters())

        self.assertIsNot(quantized, prepared)
        self.assertFalse(hasattr(prepared, QUANTIZER_ATTRIBUTE_NAME))
        self.assertFalse(hasattr(quantized, QUANTIZER_ATTRIBUTE_NAME))
        self.assertEqual(prepared._mode, Mode.CALIB)
        self.assertEqual(quantized._mode, Mode.QUANT)
        self.assertTrue(torch.equal(prepared_weight, weights))
        self.assertTrue(torch.equal(quantized_weight, weights))
        self.assertNotEqual(prepared_weight.data_ptr(), quantized_weight.data_ptr())
        self.assertTrue(torch.equal(prepared(inputs), calibrated_output))
