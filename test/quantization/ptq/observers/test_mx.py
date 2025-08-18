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
from unittest.mock import patch

import torch

from tico.experimental.quantization.ptq.dtypes import DType
from tico.experimental.quantization.ptq.observers.mx import MXObserver
from tico.experimental.quantization.ptq.qscheme import QScheme


class TestMXObserver(unittest.TestCase):
    def test_compute_qparams_returns_none_and_collect_noop(self):
        """
        MXObserver does not produce affine qparams; compute_qparams() returns None.
        collect() is a no-op but must respect the 'enabled' flag (no crash).
        """
        obs = MXObserver(
            name="mx",
            elem_format="int8",
            axis=1,
            shared_exp_method="max",
            round="nearest",
        )

        # collect() should do nothing regardless of input; just smoke-test
        obs.collect(torch.randn(2, 3))
        obs.enabled = False
        obs.collect(torch.randn(2, 3))  # still no-op

        self.assertIsNone(obs.compute_qparams())

    def test_fake_quant_calls_quantize_mx_with_expected_args(self):
        """
        fake_quant(x) must delegate to quantize_mx with the configured arguments.
        """
        obs = MXObserver(
            name="mx",
            elem_format="int8",
            axis=1,
            shared_exp_method="max",
            round="nearest",
        )
        x = torch.randn(4, 5)

        patch_path = "tico.experimental.quantization.ptq.observers.mx.quantize_mx"
        with patch(patch_path) as qmx:
            # Return a distinctive tensor so we can assert passthrough
            qmx.side_effect = lambda input_, **kwargs: input_ * 2
            y = obs.fake_quant(x)

            self.assertTrue(torch.allclose(y, x * 2))

            # Verify it was called once with the configured kwargs
            self.assertEqual(qmx.call_count, 1)
            call = qmx.call_args
            # Positional arg 0 is the input tensor
            self.assertTrue(torch.equal(call.args[0], x))
            # Keyword args should match observer config
            self.assertEqual(call.kwargs["elem_format"], "int8")
            self.assertEqual(call.kwargs["axis"], 1)
            self.assertEqual(call.kwargs["shared_exp_method"], "max")
            self.assertEqual(call.kwargs["round"], "nearest")

    def test_fake_quant_still_runs_when_disabled(self):
        """
        Even when 'enabled' is False (no more stats collection), fake_quant should still run.
        """
        obs = MXObserver(name="mx", elem_format="int8", axis=0)
        obs.enabled = False
        x = torch.randn(3, 3)

        patch_path = "tico.experimental.quantization.ptq.observers.mx.quantize_mx"
        with patch(patch_path) as qmx:
            qmx.side_effect = lambda input_, **kwargs: input_ + 1.0
            y = obs.fake_quant(x)
            self.assertTrue(torch.allclose(y, x + 1.0))
            qmx.assert_called_once()

    def test_axis_is_independent_from_base_channel_axis(self):
        """
        MXObserver.axis must be used for shared-exponent grouping regardless of base.channel_axis.
        """
        # Intentionally pass a different base channel_axis; MX should use its own 'axis=2'.
        obs = MXObserver(
            name="mx",
            elem_format="int8",
            axis=2,  # expected to be passed to quantize_mx
        )
        x = torch.randn(2, 3, 4)

        patch_path = "tico.experimental.quantization.ptq.observers.mx.quantize_mx"
        with patch(patch_path) as qmx:
            qmx.side_effect = lambda input_, **kwargs: input_
            _ = obs.fake_quant(x)
            # Ensure 'axis' comes from MXObserver(axis=2), not base channel_axis=0
            self.assertEqual(qmx.call_args.kwargs["axis"], 2)

    def test_repr_smoke(self):
        """
        repr() should include class name and observer name for debugging.
        """
        obs = MXObserver(name="mx_debug", elem_format="int8", axis=0)
        s = repr(obs)
        self.assertIn("MXObserver", s)
        self.assertIn("mx_debug", s)
