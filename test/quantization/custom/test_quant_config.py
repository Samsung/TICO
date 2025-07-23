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

import torch.nn as nn
from tico.experimental.quantization.custom.dtypes import DType
from tico.experimental.quantization.custom.observers import (
    MinMaxObserver,
    PercentileObserver,
)
from tico.experimental.quantization.custom.quant_config import QuantConfig


# Dummy wrapper that consumes QuantConfig the "generic" way
class DummyWrapper(nn.Module):
    """
    Creates two observers ('a', 'b') using QuantConfig.
    Mimics pattern every real wrapper should copy.
    """

    def __init__(self, qcfg: QuantConfig | None = None):
        super().__init__()
        self.qcfg = qcfg or QuantConfig()

        def _make_obs(name: str, default_observer=MinMaxObserver):
            kw = self.qcfg.get_kwargs(name).copy()
            obs_cls = kw.pop("observer", default_observer)
            return obs_cls(**kw)

        self.obs_a = _make_obs("a")  # MinMax by default
        self.obs_b = _make_obs("b")


class TestQuantConfig(unittest.TestCase):
    def test_default_dtype_applied(self):
        cfg = QuantConfig(default_dtype=DType.uint(8))
        w = DummyWrapper(cfg)
        self.assertEqual(w.obs_a.dtype, DType.uint(8))
        self.assertEqual(w.obs_b.dtype, DType.uint(8))

    def test_per_observer_dtype_override(self):
        cfg = QuantConfig(
            default_dtype=DType.uint(8),
            overrides={"b": {"dtype": DType.uint(4)}},
        )
        w = DummyWrapper(cfg)
        self.assertEqual(w.obs_a.dtype, DType.uint(8))  # default
        self.assertEqual(w.obs_b.dtype, DType.uint(4))  # override

    def test_observer_override(self):
        cfg = QuantConfig(
            default_dtype=DType.uint(8),
            overrides={
                "a": {
                    "observer": PercentileObserver,
                    "dtype": DType.uint(8),
                    "percentile": 99.0,
                }
            },
        )
        w = DummyWrapper(cfg)
        self.assertIsInstance(w.obs_a, PercentileObserver)
        self.assertEqual(w.obs_a.dtype, DType.uint(8))
        self.assertIsInstance(w.obs_b, MinMaxObserver)  # unaffected


class TestQuantConfigChild(unittest.TestCase):
    def test_child_inherits_default_dtype(self):
        parent = QuantConfig(default_dtype=DType.uint(8))
        child = parent.child("gate_proj")
        self.assertEqual(child.default_dtype, DType.uint(8))
        self.assertEqual(child.get_kwargs("any")["dtype"], DType.uint(8))

    def test_child_override_applied(self):
        parent = QuantConfig(
            default_dtype=DType.uint(8),
            overrides={
                "gate_proj": {"act_in": {"dtype": DType.uint(4)}},
                "mul": {"dtype": DType.uint(4)},
            },
        )
        gate_cfg = parent.child("gate_proj")
        up_cfg = parent.child("up_proj")  # no specific override

        # gate_proj.act_in should pick up uint4
        self.assertEqual(gate_cfg.get_kwargs("act_in")["dtype"], DType.uint(4))
        # up_proj.act_in falls back to default uint8
        self.assertEqual(up_cfg.get_kwargs("act_in")["dtype"], DType.uint(8))
        # top-level override still visible to parent
        self.assertEqual(parent.get_kwargs("mul")["dtype"], DType.uint(4))

    def test_child_is_view_not_copy(self):
        parent = QuantConfig(default_dtype=DType.uint(8))
        child = parent.child("dummy")
        # mutate child's overrides → parent unaffected
        child.overrides["x"] = {"dtype": DType.int(8)}
        self.assertNotIn("x", parent.overrides)
