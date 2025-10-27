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
from typing import Optional

import torch
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.observers.affine_base import AffineObserverBase
from tico.quantization.wrapq.observers.identity import IdentityObserver
from tico.quantization.wrapq.wrappers.nn.quant_linear import QuantLinear
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from torch.utils.data import DataLoader, TensorDataset

from test.modules.op.linear import SimpleLinear


class TestPTQWrapper(unittest.TestCase):
    def _build(self, qcfg: Optional[PTQConfig] = None):
        torch.manual_seed(42)
        self.fp32 = torch.nn.Linear(4, 2)
        self.input = torch.randn(32, 4)
        self.qcfg = qcfg or PTQConfig(default_dtype=DType.uint(8))
        self.wrapper = PTQWrapper(self.fp32, qcfg=self.qcfg)

    def setUp(self):
        self._build()

    def test_default_mode_is_no_quant(self):
        self.assertIs(self.wrapper._mode, Mode.NO_QUANT)

    def test_mode_transitions(self):
        self.wrapper.enable_calibration()
        self.assertIs(self.wrapper._mode, Mode.CALIB)
        self.wrapper.freeze_qparams()
        self.assertIs(self.wrapper._mode, Mode.QUANT)

    def test_switch_activation_observer(self):
        # ----- pass #1: MinMax (default) ---------------------------
        self.wrapper.enable_calibration()
        _ = self.wrapper(self.input)
        self.wrapper.freeze_qparams()
        out_mm = self.wrapper(self.input)

        # ----- pass #2: Identity via PTQConfig override --------
        pct_cfg = PTQConfig(
            default_dtype=DType.uint(8),
            overrides={
                # LinearQuant uses act_in / act_out
                "act_in": {
                    "observer": IdentityObserver,
                    "dtype": DType.uint(8),
                },
                "act_out": {
                    "observer": IdentityObserver,
                    "dtype": DType.uint(8),
                },
            },
        )
        self._build(qcfg=pct_cfg)
        self.wrapper.enable_calibration()
        _ = self.wrapper(self.input)
        self.wrapper.freeze_qparams()
        out_pct = self.wrapper(self.input)

        diff = (out_mm - out_pct).abs().mean().item()
        self.assertGreater(diff, 0.0)
        self.assertLess(diff, 0.5)

    def test_weight_fake_quant_channelwise(self):
        self.wrapper.enable_calibration()  # collects weight stats now
        _ = self.wrapper(self.input)
        self.wrapper.freeze_qparams()

        assert isinstance(self.wrapper.wrapped, QuantLinear)
        assert isinstance(self.wrapper.wrapped.weight_obs, AffineObserverBase)
        w_obs = self.wrapper.wrapped.weight_obs
        w_fp = self.fp32.weight.data
        fq_w = w_obs.fake_quant(w_fp)

        scale, zp = w_obs.compute_qparams()
        ref = torch.empty_like(w_fp)
        for c in range(w_fp.size(0)):
            q = torch.round(w_fp[c] / scale[c]) + zp[c]
            q = q.clamp(w_obs.dtype.qmin, w_obs.dtype.qmax)
            ref[c] = scale[c] * (q - zp[c])

        self.assertTrue(torch.allclose(fq_w, ref, atol=1e-6))
        self.assertFalse(torch.allclose(fq_w, w_fp, atol=1e-6))


class TestPTQSmoke(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = SimpleLinear().eval()

        data = torch.randn(128, 3) * 2
        self.calib_loader = DataLoader(TensorDataset(data), batch_size=32)

        qcfg = PTQConfig(default_dtype=DType.uint(8))
        self.model.linear = PTQWrapper(self.model.linear, qcfg=qcfg)  # type: ignore[assignment]

    def test_smoke_forward_quantized(self):
        assert isinstance(self.model.linear, PTQWrapper)
        self.model.linear.enable_calibration()
        for (x,) in self.calib_loader:
            _ = self.model(x)
        self.model.linear.freeze_qparams()
        self.assertIs(self.model.linear._mode, Mode.QUANT)

        inp, _ = self.model.get_example_inputs()
        with torch.no_grad():
            assert hasattr(self.model.linear.wrapped, "module")
            assert isinstance(self.model.linear.wrapped.module, torch.nn.Linear)
            fp32_out = self.model.linear.wrapped.module(*inp)
            q_out = self.model(*inp)

        diff = (fp32_out - q_out).abs().mean().item()
        self.assertGreater(diff, 0.0)
        self.assertLess(diff, 0.5)
        self.assertEqual(fp32_out.shape, q_out.shape)


class TestPTQWrapperObserverSurface(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.fp = torch.nn.Linear(4, 2)
        self.qcfg = PTQConfig(default_dtype=DType.uint(8))
        self.wrapper = PTQWrapper(self.fp, qcfg=self.qcfg)

    def test_all_observers_is_empty_for_wrapper(self):
        # PTQWrapper itself must not expose local observers (prevents double-processing).
        self.assertEqual(list(self.wrapper._all_observers()), [])

    def test_named_and_get_observer_are_proxies(self):
        # PTQWrapper should proxy enumeration/lookup to the wrapped module.
        names_proxy = [name for name, _ in self.wrapper.named_observers()]
        names_wrapped = [name for name, _ in self.wrapper.wrapped.named_observers()]
        self.assertEqual(names_proxy, names_wrapped)

        if names_proxy:  # ensure lookup returns the same object
            sample = names_proxy[0]
            self.assertIs(
                self.wrapper.get_observer(sample),
                self.wrapper.wrapped.get_observer(sample),
            )


class TestPTQWrapperNoDoubleProcessing(unittest.TestCase):
    """
    Build a tiny parent QuantModuleBase that contains a PTQWrapper child.
    When `parent.enable_calibration()/freeze_qparams()` propagate into the child,
    each child's observer should be processed exactly once (no double hits).
    """

    class ParentQuant(QuantModuleBase):
        def __init__(self, fp_mod: torch.nn.Module, qcfg: Optional[PTQConfig] = None):
            super().__init__(qcfg or PTQConfig(default_dtype=DType.uint(8)))
            # PTQWrapper is a QuantModuleBase child -> visited by _child_quant_modules()
            self.child = PTQWrapper(fp_mod, qcfg=self.qcfg)

        def forward(self, x):
            return self.child(x)

        def _all_observers(self):
            # Parent itself owns no observers; we're testing child's observers only.
            return []

    def _instrument_counters(self, ptq: PTQWrapper):
        """
        Monkey-patch each child's observer to count reset/compute_qparams calls.
        Returns a dict: name -> obs (with .reset_calls / .compute_calls fields).
        """
        hooked = {}
        for name, obs in ptq.named_observers():
            # initialize counters
            obs.reset_calls = 0
            obs.compute_calls = 0

            # stash originals
            orig_reset = obs.reset
            orig_compute = obs.compute_qparams

            def make_reset(o=obs, f=orig_reset):
                def _r(*a, **k):
                    o.reset_calls += 1
                    return f(*a, **k)

                return _r

            def make_compute(o=obs, f=orig_compute):
                def _c(*a, **k):
                    o.compute_calls += 1
                    return f(*a, **k)

                return _c

            obs.reset = make_reset()
            obs.compute_qparams = make_compute()
            hooked[name] = obs
        return hooked

    def test_parent_propagation_calls_each_observer_once(self):
        torch.manual_seed(0)
        fp = torch.nn.Linear(4, 2)
        parent = self.ParentQuant(fp)
        x = torch.randn(32, 4)

        # Sanity: wrapper owns no local observers but proxies enumeration.
        self.assertEqual(list(parent.child._all_observers()), [])
        self.assertGreater(len(list(parent.child.named_observers())), 0)

        # Hook counters on child's observers
        obs_map = self._instrument_counters(parent.child)

        # 1) enable_calibration() on parent should reset each child observer ONCE
        parent.enable_calibration()
        for name, obs in obs_map.items():
            self.assertEqual(
                obs.reset_calls,
                1,
                msg=f"Observer `{name}` was reset {obs.reset_calls} times (expected 1).",
            )

        # Run a forward pass to allow observers to collect (if they need it)
        _ = parent(x)

        # 2) freeze_qparams() on parent should compute_qparams for each child observer ONCE
        parent.freeze_qparams()
        for name, obs in obs_map.items():
            self.assertEqual(
                obs.compute_calls,
                1,
                msg=f"Observer `{name}` computed {obs.compute_calls} times (expected 1).",
            )

        # Modes propagate
        self.assertIs(parent._mode, Mode.QUANT)
        self.assertIs(parent.child._mode, Mode.QUANT)
