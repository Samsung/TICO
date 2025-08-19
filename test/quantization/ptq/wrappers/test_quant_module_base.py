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

"""
Unit-tests for the abstract helper class QuantModuleBase.

Because the class is abstract, the tests create a tiny concrete subclass
(`DummyQM`) that:

1. owns exactly one observer (`obs`)
2. in `forward()` multiplies the input by 2.0 and passes it through `_fq`
   so we can verify collection / fake-quant behaviour.

The suite checks:

* default mode is NO_QUANT
* `enable_calibration()` resets the observer and switches mode
* `_fq()` really collects in CALIB and fake-quantises in QUANT
* `freeze_qparams()` disables the observer and populates cached q-params
* `_make_obs()` merges overrides from a `QuantConfig`
"""

import math, torch, unittest
from typing import Dict

from tico.experimental.quantization.ptq.dtypes import DType
from tico.experimental.quantization.ptq.mode import Mode
from tico.experimental.quantization.ptq.observers import MinMaxObserver
from tico.experimental.quantization.ptq.observers.affine_base import AffineObserverBase
from tico.experimental.quantization.ptq.observers.base import ObserverBase
from tico.experimental.quantization.ptq.observers.ema import EMAObserver
from tico.experimental.quantization.ptq.qscheme import QScheme
from tico.experimental.quantization.ptq.quant_config import QuantConfig
from tico.experimental.quantization.ptq.wrappers.quant_module_base import (
    QuantModuleBase,
)


# concrete toy subclass
class DummyQM(QuantModuleBase):
    def __init__(self, qcfg: QuantConfig | None = None):
        super().__init__(qcfg)
        self.obs = self._make_obs("act")

    def forward(self, x):
        return self._fq(x * 2.0, self.obs)

    def _all_observers(self):
        return (self.obs,)


class TestQuantModuleBase(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.randn(16, 4)
        self.qm = DummyQM()  # default uint8

    def test_mode_cycle(self):
        self.assertIs(self.qm._mode, Mode.NO_QUANT)

        self.qm.enable_calibration()
        self.assertIs(self.qm._mode, Mode.CALIB)
        # observer reset to +inf / -inf
        assert isinstance(self.qm.obs, AffineObserverBase)
        self.assertTrue(math.isinf(self.qm.obs.min_val.item()))
        self.assertTrue(math.isinf(self.qm.obs.max_val.item()))

        self.qm.freeze_qparams()
        self.assertIs(self.qm._mode, Mode.QUANT)
        self.assertTrue(self.qm.obs.has_qparams)

    def test_fq_collect_and_quantize(self):
        # CALIB pass – observer should collect
        self.qm.enable_calibration()
        _ = self.qm(self.x)
        assert isinstance(self.qm.obs, AffineObserverBase)
        lo = self.qm.obs.min_val.item()
        hi = self.qm.obs.max_val.item()
        self.assertLess(lo, hi)  # stats updated

        # QUANT pass – output must differ from FP32
        self.qm.freeze_qparams()
        q_out = self.qm(self.x)
        fp_out = self.x * 2.0
        self.assertFalse(torch.allclose(q_out, fp_out))

    def test_make_obs_override(self):
        cfg = QuantConfig(
            default_dtype=DType.uint(8),
            overrides={
                "act": {"dtype": DType.uint(4)},
            },
        )
        qm = DummyQM(qcfg=cfg)
        self.assertEqual(qm.obs.dtype, DType.uint(4))

    def test_named_observers(self):
        named: Dict[str, ObserverBase] = dict(self.qm.named_observers())
        self.assertIn("act", named)
        self.assertIs(named["act"], self.qm.obs)

    def test_get_observer(self):
        self.assertIs(self.qm.get_observer("act"), self.qm.obs)
        self.assertIsNone(self.qm.get_observer("not_exist"))

    def test_fp_name_storage(self):
        no_name = DummyQM()
        self.assertIsNone(no_name.fp_name)

        with_name = DummyQM()
        with_name.fp_name = "foo"
        self.assertEqual(with_name.fp_name, "foo")

    def test_extra_repr(self):
        self.assertEqual(self.qm.extra_repr(), "mode=no_quant")

        self.qm.enable_calibration()
        self.assertEqual(self.qm.extra_repr(), "mode=calib")

        self.qm.freeze_qparams()
        self.assertEqual(self.qm.extra_repr(), "mode=quant")


class TestQuantConfigDefaultObserver(unittest.TestCase):
    # 1) global change via default_observer -------------------------
    def test_global_default_observer(self):
        cfg = QuantConfig(default_dtype=DType.uint(8), default_observer=EMAObserver)
        qm = DummyQM(cfg)
        obs = qm.obs
        self.assertIsInstance(obs, EMAObserver)

    # 2) per-observer "observer" override beats default_observer -----
    def test_observer_override_precedence(self):
        cfg = QuantConfig(
            default_dtype=DType.uint(8),
            default_observer=EMAObserver,
            overrides={"act": {"observer": MinMaxObserver}},
        )
        qm = DummyQM(cfg)
        obs = qm.obs
        self.assertIsInstance(obs, MinMaxObserver)

    # 3) child() inherits parent default_observer -------------------
    def test_child_inherits_default_observer(self):
        parent = QuantConfig(
            default_dtype=DType.uint(8),
            default_observer=EMAObserver,
            overrides={"child_wrap": {"dtype": DType.uint(4)}},
        )
        child = parent.child("child_wrap")
        self.assertIs(child.default_observer, parent.default_observer)
        # and still works when materialised
        qm = DummyQM(child)
        obs = qm.obs
        self.assertIsInstance(obs, EMAObserver)


class DummyQMWrapperDefault(QuantModuleBase):
    def __init__(self, qcfg: QuantConfig | None = None):
        super().__init__(qcfg)
        # wrapper-level default value: per-channel asymm on axis 1
        self.obs = self._make_obs(
            "act",
            qscheme=QScheme.PER_CHANNEL_ASYMM,
            channel_axis=1,
        )

    def forward(self, x):
        return self._fq(x * 2.0, self.obs)

    def _all_observers(self):
        return (self.obs,)


class TestQuantModuleQScheme(unittest.TestCase):
    def test_config_default_qscheme(self):
        cfg = QuantConfig(default_qscheme=QScheme.PER_CHANNEL_SYMM)
        qm = DummyQM(cfg)
        self.assertEqual(qm.obs.qscheme, QScheme.PER_CHANNEL_SYMM)

    def test_wrapper_default_qscheme_applied(self):
        cfg = QuantConfig(default_qscheme=QScheme.PER_TENSOR_ASYMM)
        qm = DummyQMWrapperDefault(cfg)
        self.assertEqual(qm.obs.qscheme, QScheme.PER_CHANNEL_ASYMM)
        self.assertEqual(qm.obs.channel_axis, 1)

    def test_user_override_qscheme_wins(self):
        cfg = QuantConfig(
            default_qscheme=QScheme.PER_TENSOR_ASYMM,
            overrides={
                "act": {
                    "qscheme": QScheme.PER_CHANNEL_SYMM,
                    "channel_axis": 0,
                }
            },
        )
        qm = DummyQMWrapperDefault(cfg)
        self.assertEqual(qm.obs.qscheme, QScheme.PER_CHANNEL_SYMM)
        self.assertEqual(qm.obs.channel_axis, 0)
