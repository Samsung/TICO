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
from collections import OrderedDict
from types import SimpleNamespace
from typing import cast

import torch
import torch.nn as nn

from tico.quantization import convert, prepare
from tico.quantization.algorithm.fpi_gptq.quantizer import FPIGPTQQuantizer
from tico.quantization.algorithm.gptq.quant import Quantizer
from tico.quantization.config.fpi_gptq import FPIGPTQConfig
from tico.quantization.public_interface import QUANTIZER_ATTRIBUTE_NAME


class _DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(2, 2, bias=False)
        self.second = nn.Linear(2, 2, bias=False)
        self.replay_error: RuntimeError | None = None

    def forward(self, x):
        x = self.first(x)
        if self.replay_error is not None:
            raise self.replay_error
        return self.second(x)


class _TinyDecoder(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.config = SimpleNamespace(use_cache=True)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([layer])

    def forward(self, x):
        return cast(nn.ModuleList, self.model.layers)[0](x)


class _RejectNewHooks(OrderedDict):
    error: RuntimeError | None = None

    def __setitem__(self, key, value):
        if self.error is not None:
            raise self.error
        super().__setitem__(key, value)


class TestFPIGPTQStatisticsHooks(unittest.TestCase):
    def setUp(self):
        rng_state = torch.random.get_rng_state()
        self.addCleanup(torch.random.set_rng_state, rng_state)
        torch.manual_seed(0)
        self.layer = _DecoderLayer()
        self.model = _TinyDecoder(self.layer).eval()
        self.modules = [self.layer.first, self.layer.second]
        self.events: list[tuple[int, int]] = []
        self.original_hooks = []
        for index, module in enumerate(self.modules):
            with torch.no_grad():
                module.weight.copy_(torch.diag(torch.tensor([1.0, 2.0]) + index))
            handle = module.register_forward_hook(self._user_hook(index))
            self.addCleanup(handle.remove)
            self.original_hooks.append(dict(module._forward_hooks))
        self.weights = [module.weight.detach().clone() for module in self.modules]
        self.assertIs(
            prepare(self.model, FPIGPTQConfig(show_progress=False)), self.model
        )
        self.quantizer = cast(
            FPIGPTQQuantizer, getattr(self.model, QUANTIZER_ATTRIBUTE_NAME)
        )
        for inputs in torch.eye(2).split(1):
            self.assertIsNone(self.model(inputs))
        self.assertEqual(self.events, [])
        self.assertEqual(self.quantizer.num_batches, 2)

    def _user_hook(self, index):
        def record(module, _inputs, _output):
            self.events.append((index, len(module._forward_hooks)))

        return record

    def _assert_only_user_hooks(self):
        for module, original in zip(self.modules, self.original_hooks):
            self.assertEqual(dict(module._forward_hooks), original)

    def _assert_collection_error(self, error):
        with self.assertRaisesRegex(RuntimeError, str(error)) as raised:
            convert(self.model)
        self.assertIs(raised.exception, error)
        self.assertFalse(hasattr(self.model, "quantizers"))
        for module, weight in zip(self.modules, self.weights):
            self.assertTrue(torch.equal(module.weight, weight))
        self._assert_only_user_hooks()

    def test_replay_error_removes_only_statistics_hooks(self):
        error = RuntimeError("FPI replay failed")
        self.layer.replay_error = error
        self._assert_collection_error(error)
        self.assertEqual(self.events, [(0, 2)])

        self.layer.replay_error = None
        self.events.clear()
        expected = torch.diag(torch.tensor([2.0, 6.0]))
        self.assertTrue(torch.equal(self.model(torch.eye(2)), expected))
        self.assertEqual(self.events, [(0, 1), (1, 1)])
        self._assert_only_user_hooks()

    def test_partial_registration_removes_previous_statistics_hooks(self):
        error = RuntimeError("FPI second hook registration failed")
        # Keep exact nn.Linear types so the real traversal selects both modules.
        original_registry = self.layer.second._forward_hooks
        rejecting_registry = _RejectNewHooks(original_registry)
        rejecting_registry.error = error
        self.layer.second._forward_hooks = rejecting_registry
        self.addCleanup(setattr, self.layer.second, "_forward_hooks", original_registry)
        self._assert_collection_error(error)
        self.assertEqual(self.events, [])

    def test_success_preserves_weights_qparams_and_hook_order(self):
        self.assertIs(convert(self.model), self.model)
        self._assert_only_user_hooks()
        self.assertEqual(self.events, [(0, 2), (1, 2)] * 2 + [(0, 1), (1, 1)] * 2)
        quantizers = cast(dict[str, Quantizer], self.model.quantizers)
        names = ["model.layers.0.first", "model.layers.0.second"]
        self.assertEqual(list(quantizers), names)
        for index, (name, module, weight) in enumerate(
            zip(names, self.modules, self.weights)
        ):
            self.assertTrue(torch.equal(module.weight, weight))
            qparams = quantizers[name]
            self.assertEqual(qparams.maxq.item(), 255)
            self.assertEqual(qparams.scale.dtype, torch.float32)
            self.assertEqual(qparams.zero.dtype, torch.float32)
            self.assertEqual(qparams.scale.shape, (2, 1))
            self.assertEqual(qparams.zero.shape, (2, 1))
            self.assertTrue(
                torch.equal(
                    qparams.scale, (torch.tensor([1.0, 2.0]) + index)[:, None] / 255
                )
            )
            self.assertTrue(torch.equal(qparams.zero, torch.zeros(2, 1)))
        self.assertTrue(
            torch.equal(self.model(torch.eye(2)), torch.diag(torch.tensor([2.0, 6.0])))
        )
        self.assertEqual(self.quantizer.cache_args, [])
        self.assertEqual(self.quantizer.cache_kwargs, {})
        self.assertEqual(self.quantizer.num_batches, 0)
        self.assertFalse(hasattr(self.model, QUANTIZER_ATTRIBUTE_NAME))
