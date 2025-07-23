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
Unit-tests for the abstract helper class **QuantModuleBase**.

Because the class is abstract, the tests create a tiny concrete subclass
(`DummyQM`) that:

1. owns exactly **one** observer (`obs`)
2. in `forward()` multiplies the input by 2.0 and passes it through `_fq`
   so we can verify collection / fake-quant behaviour.

The suite checks:

* default mode is **NO_QUANT**
* `enable_calibration()` resets the observer and switches mode
* `_fq()` really collects in CALIB and fake-quantises in QUANT
* `freeze_qparams()` disables the observer and populates cached q-params
* `_make_obs()` merges overrides from a `QuantConfig`
"""

import math, torch, unittest

from tico.experimental.quantization.custom.dtypes import DType
from tico.experimental.quantization.custom.mode import Mode
from tico.experimental.quantization.custom.observers import MinMaxObserver
from tico.experimental.quantization.custom.observers.percentile import (
    PercentileObserver,
)
from tico.experimental.quantization.custom.quant_config import QuantConfig
from tico.experimental.quantization.custom.wrappers.quant_module_base import (
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
        self.assertTrue(math.isinf(self.qm.obs.min_val.item()))

        self.qm.freeze_qparams()
        self.assertIs(self.qm._mode, Mode.QUANT)
        self.assertTrue(self.qm.obs.has_qparms)

    def test_fq_collect_and_quantise(self):
        # CALIB pass – observer should collect
        self.qm.enable_calibration()
        _ = self.qm(self.x)
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


class TestQuantConfigDefaultFactory(unittest.TestCase):
    # 1) global change via default_factory -------------------------
    def test_global_default_factory(self):
        cfg = QuantConfig(
            default_dtype=DType.uint(8), default_factory=PercentileObserver
        )
        qm = DummyQM(cfg)
        obs = qm.obs
        self.assertIsInstance(obs, PercentileObserver)

    # 2) per-observer "factory" override beats default_factory -----
    def test_factory_override_precedence(self):
        cfg = QuantConfig(
            default_dtype=DType.uint(8),
            default_factory=PercentileObserver,
            overrides={"act": {"factory": MinMaxObserver}},
        )
        qm = DummyQM(cfg)
        obs = qm.obs
        self.assertIsInstance(obs, MinMaxObserver)

    # 3) child() inherits parent default_factory -------------------
    def test_child_inherits_default_factory(self):
        parent = QuantConfig(
            default_dtype=DType.uint(8),
            default_factory=PercentileObserver,
            overrides={"child_wrap": {"dtype": DType.uint(4)}},
        )
        child = parent.child("child_wrap")
        self.assertIs(child.default_factory, parent.default_factory)
        # and still works when materialised
        qm = DummyQM(child)
        obs = qm.obs
        self.assertIsInstance(obs, PercentileObserver)
