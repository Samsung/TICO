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

"""Unit tests for the Gemma4 RMSNorm PTQ wrapper."""

import unittest
from unittest import mock

import torch
import torch.nn as nn

from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.observers.affine_base import AffineObserverBase
from tico.quantization.wrapq.wrappers.gemma4.quant_rmsnorm import QuantGemma4RMSNorm

from test.quantization.quant_spec_helpers import make_affine_ptq_config


def _ensure_circle_rms_norm_registered() -> None:
    """Register the Circle RMSNorm custom op if this process has not done so."""
    try:
        torch.ops.circle_custom.rms_norm
    except AttributeError:
        from tico.utils.register_custom_op import CircleRMSNorm

        CircleRMSNorm()


class _DummyGemma4RMSNorm(nn.Module):
    """Minimal Gemma4RMSNorm-like module with an optional scale parameter."""

    def __init__(self, hidden_size: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(hidden_size))


class _VarianceEpsilonRMSNorm(nn.Module):
    """Minimal RMSNorm-like module that exposes variance_epsilon instead of eps."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.variance_epsilon = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))


def _rmsnorm_reference(
    x: torch.Tensor, weight: torch.Tensor | None, eps: float
) -> torch.Tensor:
    """Return a PyTorch reference implementation of Gemma4 RMSNorm."""
    x_f = x.float()
    out = x_f * torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + eps)
    if weight is not None:
        out = out * weight.float()
    return out.to(dtype=x.dtype)


class TestQuantGemma4RMSNorm(unittest.TestCase):
    """Validate Gemma4 RMSNorm wrapper behavior."""

    def setUp(self):
        """Create deterministic inputs and register required custom ops."""
        _ensure_circle_rms_norm_registered()
        torch.manual_seed(2026)
        self.hidden = 16
        self.x = torch.randn(4, 7, self.hidden)

    def test_no_quant_matches_reference_with_scale_without_plus_one(self):
        """Check that the wrapper uses Gemma4 weight directly, not weight plus one."""
        fp = _DummyGemma4RMSNorm(self.hidden, eps=1e-6, with_scale=True)
        with torch.no_grad():
            fp.weight.copy_(torch.linspace(-0.5, 1.0, self.hidden))

        q_rms = QuantGemma4RMSNorm(fp)
        out = q_rms(self.x)
        ref = _rmsnorm_reference(self.x, fp.weight, fp.eps)
        plus_one_ref = _rmsnorm_reference(self.x, fp.weight + 1.0, fp.eps)

        self.assertIs(q_rms._mode, Mode.NO_QUANT)
        self.assertEqual(out.shape, self.x.shape)
        self.assertTrue(torch.allclose(out, ref, atol=1e-6, rtol=1e-6))
        self.assertFalse(torch.allclose(out, plus_one_ref, atol=1e-6, rtol=1e-6))

    def test_no_quant_matches_reference_without_scale(self):
        """Check that with_scale=False behaves like RMSNorm with unit scale."""
        fp = _DummyGemma4RMSNorm(self.hidden, eps=1e-6, with_scale=False)
        q_rms = QuantGemma4RMSNorm(fp)

        out = q_rms(self.x)
        ref = _rmsnorm_reference(self.x, None, fp.eps)

        self.assertFalse(q_rms.with_scale)
        self.assertIsNotNone(q_rms.obs_weight)
        self.assertEqual(out.shape, self.x.shape)
        self.assertTrue(torch.allclose(out, ref, atol=1e-6, rtol=1e-6))

    def test_variance_epsilon_fallback(self):
        """Check that wrappers still support modules exposing variance_epsilon."""
        fp = _VarianceEpsilonRMSNorm(self.hidden, eps=1e-5)
        with torch.no_grad():
            fp.weight.copy_(torch.linspace(0.25, 1.25, self.hidden))

        q_rms = QuantGemma4RMSNorm(fp)
        out = q_rms(self.x)
        ref = _rmsnorm_reference(self.x, fp.weight, fp.variance_epsilon)

        self.assertEqual(q_rms.eps, fp.variance_epsilon)
        self.assertTrue(torch.allclose(out, ref, atol=1e-6, rtol=1e-6))

    def test_mode_transitions_and_weight_observer(self):
        """Check the calibration lifecycle when a scale parameter exists."""
        fp = _DummyGemma4RMSNorm(self.hidden, eps=1e-6, with_scale=True)
        with torch.no_grad():
            fp.weight.copy_(torch.randn_like(fp.weight) * 0.5 + 0.75)
        q_rms = QuantGemma4RMSNorm(fp)

        self.assertIs(q_rms._mode, Mode.NO_QUANT)
        q_rms.enable_calibration()
        self.assertIs(q_rms._mode, Mode.CALIB)
        self.assertIsNotNone(q_rms.obs_weight)

        _ = q_rms(self.x)
        q_rms.freeze_qparams()

        self.assertIs(q_rms._mode, Mode.QUANT)
        self.assertTrue(hasattr(q_rms.obs_weight, "_cached_scale"))
        self.assertEqual(q_rms(self.x).shape, self.x.shape)

    def test_mode_transitions_without_scale_weight_observer(self):
        """Check the calibration lifecycle when Gemma4 RMSNorm has no scale."""
        fp = _DummyGemma4RMSNorm(self.hidden, eps=1e-6, with_scale=False)
        q_rms = QuantGemma4RMSNorm(fp)

        self.assertIsNotNone(q_rms.obs_weight)
        q_rms.enable_calibration()
        self.assertIs(q_rms._mode, Mode.CALIB)
        _ = q_rms(self.x)
        q_rms.freeze_qparams()

        self.assertIs(q_rms._mode, Mode.QUANT)
        self.assertTrue(hasattr(q_rms.obs_weight, "_cached_scale"))
        self.assertEqual(q_rms(self.x).shape, self.x.shape)

    def test_scale_weight_is_fake_quantized_in_quant_mode(self):
        """Check that an existing Gemma4 scale weight uses fake quantization."""
        fp = _DummyGemma4RMSNorm(self.hidden, eps=1e-6, with_scale=True)
        with torch.no_grad():
            fp.weight.copy_(torch.linspace(0.1, 1.7, self.hidden))
        q_rms = QuantGemma4RMSNorm(fp)

        q_rms.enable_calibration()
        _ = q_rms(self.x)
        q_rms.freeze_qparams()

        with mock.patch.object(
            q_rms.obs_weight,
            "fake_quant",
            wraps=q_rms.obs_weight.fake_quant,
        ) as fake_quant:
            _ = q_rms(self.x)

        fake_quant.assert_called_once()
        called_weight = fake_quant.call_args.args[0]
        self.assertTrue(torch.equal(called_weight, fp.weight))

    def test_unit_scale_weight_is_fake_quantized_in_quant_mode(self):
        """Check that synthetic unit-scale weight also uses fake quantization."""
        fp = _DummyGemma4RMSNorm(self.hidden, eps=1e-6, with_scale=False)
        q_rms = QuantGemma4RMSNorm(fp)

        q_rms.enable_calibration()
        _ = q_rms(self.x)
        q_rms.freeze_qparams()

        with mock.patch.object(
            q_rms.obs_weight,
            "fake_quant",
            wraps=q_rms.obs_weight.fake_quant,
        ) as fake_quant:
            _ = q_rms(self.x)

        fake_quant.assert_called_once()
        called_weight = fake_quant.call_args.args[0]
        self.assertEqual(tuple(called_weight.shape), (self.hidden,))
        self.assertTrue(torch.allclose(called_weight, torch.ones_like(called_weight)))

    def test_dtype_override(self):
        """Check that PTQConfig overrides propagate to Gemma4 RMSNorm observers."""
        cfg = make_affine_ptq_config(
            dtype=DType.uint(8),
            overrides={
                "act_in": {"dtype": DType.uint(4)},
                "act_out": {"dtype": DType.uint(4)},
                "weight": {"dtype": DType.uint(4)},
            },
        )
        fp = _DummyGemma4RMSNorm(self.hidden, eps=1e-6, with_scale=True)
        q_rms = QuantGemma4RMSNorm(fp, qcfg=cfg)

        self.assertIsInstance(q_rms.obs_weight, AffineObserverBase)
        self.assertIsInstance(q_rms.obs_act_in, AffineObserverBase)
        self.assertIsInstance(q_rms.obs_act_out, AffineObserverBase)
        self.assertEqual(q_rms.obs_weight.dtype, DType.uint(4))
        self.assertEqual(q_rms.obs_act_in.dtype, DType.uint(4))
        self.assertEqual(q_rms.obs_act_out.dtype, DType.uint(4))


if __name__ == "__main__":
    unittest.main()
