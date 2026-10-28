# Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

import torch
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.config.specs import affine
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.observers.ema import EMAObserver
from tico.quantization.wrapq.qscheme import QScheme
from tico.quantization.wrapq.quantizer import PTQQuantizer


class TestEMAObserver(unittest.TestCase):
    def test_ema_updates(self):
        torch.manual_seed(0)
        obs = EMAObserver(
            name="dummy",
            dtype=DType.uint(8),
            qscheme=QScheme.PER_TENSOR_ASYMM,
            momentum=0.5,
        )

        # First batch initialises stats hard
        obs.collect(torch.tensor([-1.0, 2.0, 3.0]))
        self.assertAlmostEqual(obs.min_val.item(), -1.0)
        self.assertAlmostEqual(obs.max_val.item(), 3.0)

        # Second batch updates via EMA
        obs.collect(torch.tensor([-9.0, 9.0]))
        # new_min = 0.5 * (-1) + 0.5 * (-9) = -5
        # new_max = 0.5 * 3   + 0.5 * 9     =  6
        self.assertAlmostEqual(obs.min_val.item(), -5.0)
        self.assertAlmostEqual(obs.max_val.item(), 6.0)
        scale, zero = obs.compute_qparams()
        self.assertTrue(torch.equal(scale, torch.tensor(11.0) / 255))
        self.assertTrue(torch.equal(zero, torch.tensor(116, dtype=torch.int)))
        self.assertEqual(scale.dtype, torch.float32)
        self.assertEqual(zero.dtype, torch.int32)

    def test_per_channel_ema(self):
        x1 = torch.tensor([[1.0, -2.0], [3.0, -4.0]])  # shape (C=2, N=2)
        x2 = torch.tensor([[10.0, -10.0], [5.0, -5.0]])
        obs = EMAObserver(
            name="dummy",
            dtype=DType.int(4),
            qscheme=QScheme.PER_CHANNEL_ASYMM,
            channel_axis=0,
            momentum=0.5,
        )

        obs.collect(x1)
        self.assertTrue(
            torch.equal(obs.min_val, torch.tensor([-2.0, -4.0])),
            f"obs.min_val: {obs.min_val}",
        )
        self.assertTrue(
            torch.equal(obs.max_val, torch.tensor([1.0, 3.0])),
            f"obs.max_val: {obs.max_val}",
        )

        obs.collect(x2)
        # Manual EMA check
        expected_min = 0.5 * torch.tensor([-2.0, -4.0]) + 0.5 * torch.tensor(
            [-10.0, -5.0]
        )
        expected_max = 0.5 * torch.tensor([1.0, 3.0]) + 0.5 * torch.tensor([10.0, 5.0])
        self.assertTrue(
            torch.allclose(obs.min_val, expected_min),
            f"{obs.min_val} != {expected_min}",
        )
        self.assertTrue(
            torch.allclose(obs.max_val, expected_max),
            f"{obs.max_val} != {expected_max}",
        )
        scale, zero = obs.compute_qparams()
        self.assertTrue(torch.equal(scale, torch.tensor([11.5, 8.5]) / 15))
        self.assertTrue(torch.equal(zero, torch.tensor([0, 0], dtype=torch.int)))
        self.assertEqual(scale.dtype, torch.float32)
        self.assertEqual(zero.dtype, torch.int32)

    def test_first_batch(self):
        x = torch.randn(10)
        obs = EMAObserver(name="dummy", momentum=0.9, channel_axis=None)
        obs.collect(x)
        self.assertFalse(torch.isinf(obs.min_val).any())
        self.assertFalse(torch.isinf(obs.max_val).any())

    def _per_channel_observer(self, channel_axis=0):
        return EMAObserver(
            name="weight",
            dtype=DType.int(8),
            qscheme=QScheme.PER_CHANNEL_SYMM,
            channel_axis=channel_axis,
            momentum=0.5,
        )

    def test_rank_one_stats_do_not_follow_input_mutation(self):
        for channel_axis in (0, -1):
            for dtype in (torch.float32, torch.float64):
                with self.subTest(channel_axis=channel_axis, dtype=dtype):
                    obs = self._per_channel_observer(channel_axis)
                    original = torch.tensor([-2.0, 1.0, 3.0], dtype=dtype)
                    x = original.clone()
                    obs.collect(x)
                    x.add_(10)
                    self.assertTrue(torch.equal(obs.min_val, original))
                    self.assertTrue(torch.equal(obs.max_val, original))
                    self.assertEqual(obs.min_val.dtype, dtype)
                    self.assertEqual(obs.max_val.dtype, dtype)
                    self.assertEqual(obs.min_val.device, x.device)
                    if dtype == torch.float64:
                        with self.assertRaisesRegex(
                            RuntimeError, "per-channel scale dtype"
                        ):
                            obs.compute_qparams()
                        self.assertTrue(torch.equal(obs.min_val, original))
                        self.assertTrue(torch.equal(obs.max_val, original))
                        self.assertEqual(obs.min_val.dtype, torch.float64)
                        self.assertIsNone(obs.frozen_state)
                        self.assertTrue(obs.enabled)
                        continue
                    scale, zero = obs.compute_qparams()
                    self.assertTrue(torch.equal(scale, original.abs() / 127))
                    self.assertTrue(torch.equal(zero, torch.zeros(3, dtype=torch.int)))

    def test_reset_preserves_rank_one_input(self):
        obs = self._per_channel_observer()
        original = torch.tensor([1.0, 2.0, 3.0])
        x = original.clone()
        obs.collect(x)
        obs.reset()
        self.assertTrue(torch.equal(x, original))
        self.assertTrue(torch.isposinf(obs.min_val).all())
        self.assertTrue(torch.isneginf(obs.max_val).all())
        self.assertFalse(obs.has_qparams)

    def test_rank_one_parameter_keeps_stats_as_buffers(self):
        obs = self._per_channel_observer()
        weight = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
        obs.collect(weight)
        buffers = dict(obs.named_buffers())
        self.assertIs(buffers.get("min_val"), obs.min_val)
        self.assertIs(buffers.get("max_val"), obs.max_val)
        self.assertEqual(list(obs.parameters()), [])
        self.assertFalse(obs.min_val.requires_grad)
        self.assertFalse(obs.max_val.requires_grad)

        obs.collect(torch.nn.Parameter(torch.tensor([5.0, 6.0, 7.0])))
        expected = torch.tensor([3.0, 4.0, 5.0])
        self.assertTrue(torch.equal(obs.min_val, expected))
        self.assertTrue(torch.equal(obs.max_val, expected))
        obs.reset()
        self.assertTrue(torch.equal(weight, torch.tensor([1.0, 2.0, 3.0])))

    def test_layernorm_prepare_preserves_affine_parameters(self):
        model = torch.nn.LayerNorm(3)
        qcfg = PTQConfig(
            weight=affine(
                DType.int(8),
                qscheme=QScheme.PER_CHANNEL_SYMM,
                observer=EMAObserver,
                channel_axis=0,
            )
        )
        prepared = PTQQuantizer(qcfg).prepare(model)
        self.assertTrue(torch.equal(model.weight, torch.ones(3)))
        self.assertTrue(torch.equal(model.bias, torch.zeros(3)))
        weight_obs = prepared.get_observer("weight")
        self.assertIsInstance(weight_obs, EMAObserver)
        scale, zero = weight_obs.compute_qparams()
        self.assertTrue(torch.equal(scale, torch.ones(3) / 127))
        self.assertTrue(torch.equal(zero, torch.zeros(3, dtype=torch.int)))
