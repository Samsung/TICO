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

from tico.quantization.config.ptq import PTQConfig

from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.observers.affine_base import AffineObserverBase
from tico.quantization.wrapq.observers.ema import EMAObserver
from tico.quantization.wrapq.observers.minmax import MinMaxObserver
from tico.quantization.wrapq.qscheme import QScheme
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase


class DummyWrapper(QuantModuleBase):
    """Minimal wrapper to expose `_make_obs` and store the created observer."""

    def __init__(self, qcfg, **kwargs):
        super().__init__(qcfg)
        # kwargs here are wrapper-level defaults for _make_obs
        self.obs_act_in = self._make_obs("act_in", **kwargs)
        self.obs_act_out = self._make_obs("act_out", **kwargs)
        self.obs_weight = self._make_obs("weight", **kwargs)

    def _all_observers(self):
        # required by QuantModuleBase
        return (self.obs_act_in, self.obs_act_out, self.obs_weight)


class TestPTQConfig(unittest.TestCase):
    def test_default_dtype_applied(self):
        cfg = PTQConfig(default_dtype=DType.uint(8))
        w = DummyWrapper(cfg)
        self.assertEqual(w.obs_act_in.dtype, DType.uint(8))
        self.assertEqual(w.obs_act_out.dtype, DType.uint(8))

    def test_per_observer_dtype_override(self):
        cfg = PTQConfig(
            default_dtype=DType.uint(8),
            overrides={"act_out": {"dtype": DType.uint(4)}},
        )
        w = DummyWrapper(cfg)
        self.assertEqual(w.obs_act_in.dtype, DType.uint(8))  # default
        self.assertEqual(w.obs_act_out.dtype, DType.uint(4))  # override

    def test_observer_override(self):
        cfg = PTQConfig(
            default_dtype=DType.uint(8),
            overrides={
                "act_in": {
                    "observer": EMAObserver,
                    "dtype": DType.uint(8),
                }
            },
        )
        w = DummyWrapper(cfg)
        self.assertIsInstance(w.obs_act_in, EMAObserver)
        self.assertEqual(w.obs_act_in.dtype, DType.uint(8))
        self.assertIsInstance(w.obs_act_out, MinMaxObserver)  # unaffected


class TestPTQConfigChild(unittest.TestCase):
    def test_child_inherits_default_dtype(self):
        parent = PTQConfig(default_dtype=DType.uint(8))
        child = parent.child("gate_proj")
        self.assertEqual(child.default_dtype, DType.uint(8))
        self.assertEqual(child.default_dtype, DType.uint(8))

    def test_child_override_applied(self):
        parent = PTQConfig(
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
        # top-level override still visible to parent
        self.assertEqual(parent.get_kwargs("mul")["dtype"], DType.uint(4))

    def test_child_is_view_not_copy(self):
        parent = PTQConfig(default_dtype=DType.uint(8))
        child = parent.child("dummy")
        # mutate child's overrides → parent unaffected
        child.overrides["x"] = {"dtype": DType.int(8)}  # type: ignore[index]
        self.assertNotIn("x", parent.overrides)

    def test_child_inherits_default_qscheme(self):
        parent = PTQConfig(default_qscheme=QScheme.PER_CHANNEL_ASYMM)
        child = parent.child("gate_proj")
        self.assertEqual(child.default_qscheme, QScheme.PER_CHANNEL_ASYMM)

    def test_child_inherits_defaults(self):
        parent = PTQConfig(
            default_dtype=DType.uint(8),
            default_qscheme=None,
            wrapper_variant="decode",
        )
        child = parent.child("gate_proj")

        self.assertEqual(child.default_dtype, DType.uint(8))
        self.assertEqual(child.default_qscheme, QScheme.PER_TENSOR_ASYMM)
        self.assertEqual(child.wrapper_variant, "decode")

    def test_child_override_applied_and_normalized(self):
        parent = PTQConfig(
            default_dtype=DType.int(16),
            overrides={
                "gate_proj": {"act_in": {"dtype": DType.uint(4)}},
                "mul": {"dtype": DType.uint(4)},
            },
        )
        gate_cfg = parent.child("gate_proj")
        up_cfg = parent.child("up_proj")

        self.assertEqual(gate_cfg.get_kwargs("act_in")["dtype"], DType.uint(4))
        self.assertEqual(
            gate_cfg.get_kwargs("act_in")["qscheme"],
            QScheme.PER_TENSOR_ASYMM,
        )
        self.assertEqual(parent.get_kwargs("mul")["dtype"], DType.uint(4))
        self.assertEqual(parent.get_kwargs("mul")["qscheme"], QScheme.PER_TENSOR_ASYMM)
        self.assertEqual(up_cfg.get_kwargs("act_in"), {})

    def test_child_isolated_from_parent_mutation(self):
        parent = PTQConfig(default_dtype=DType.uint(8))
        child = parent.child("dummy")
        child.overrides["x"] = {"dtype": DType.int(8)}  # type: ignore[index]
        child.normalize_overrides()

        self.assertNotIn("x", parent.overrides)
        self.assertEqual(child.overrides["x"]["qscheme"], QScheme.PER_TENSOR_SYMM)


# ---- Dummy observers for testing (just to distinguish classes) ----
class DummyObserverA(AffineObserverBase):
    def _update_stats(self, x):
        return super()._update_stats(x)


class DummyObserverB(AffineObserverBase):
    def _update_stats(self, x):
        return super()._update_stats(x)


class TestObserverAndDTypePrecedence(unittest.TestCase):
    """
    Ensure `_make_obs()` applies 3-level precedence to dtype/observer:

        1) User override in PTQConfig.overrides[name]
        2) Wrapper default passed via `_make_obs(..., dtype=..., observer=...)`
        3) PTQConfig.default_dtype or default_observer

    And other kwargs follow:
        user override > wrapper default
    """

    def test_user_override_wins(self):
        """
        If user supplies both dtype and observer, they must override
        both wrapper defaults and PTQConfig defaults.
        """
        qcfg = PTQConfig(
            default_dtype=DType.uint(8),
            default_observer=MinMaxObserver,  # type: ignore[type-abstract]
            overrides={
                "act_in": {
                    "dtype": DType.uint(4),
                    "observer": DummyObserverA,
                    "qscheme": QScheme.PER_TENSOR_ASYMM,  # user override for another kw
                    "channel_axis": None,
                }
            },
        )

        # Wrapper defaults: dtype=6bit, observer=DummyObserverB, qscheme=PER_CHANNEL
        wrapper = DummyWrapper(
            qcfg,
            dtype=DType.uint(6),
            observer=DummyObserverB,
            qscheme=QScheme.PER_CHANNEL_ASYMM,
            channel_axis=0,
        )

        self.assertIsInstance(wrapper.obs_act_in, DummyObserverA)
        self.assertEqual(wrapper.obs_act_in.dtype, DType.uint(4))
        # user override wins for qscheme/channel_axis
        self.assertEqual(wrapper.obs_act_in.qscheme, QScheme.PER_TENSOR_ASYMM)
        self.assertIsNone(wrapper.obs_act_in.channel_axis)

    def test_wrapper_default_when_no_user_override(self):
        """
        If the user supplies nothing for a given name, wrapper defaults must
        override PTQConfig defaults.
        """
        qcfg = PTQConfig(
            default_dtype=DType.uint(8),
            default_observer=MinMaxObserver,  # type: ignore[type-abstract]
            overrides={
                # nothing for 'act_out'
            },
        )

        wrapper = DummyWrapper(
            qcfg,
            dtype=DType.uint(6),
            observer=DummyObserverB,
            qscheme=QScheme.PER_CHANNEL_ASYMM,
            channel_axis=1,
        )

        self.assertIsInstance(wrapper.obs_act_out, DummyObserverB)
        self.assertEqual(wrapper.obs_act_out.dtype, DType.uint(6))
        self.assertEqual(wrapper.obs_act_out.qscheme, QScheme.PER_CHANNEL_ASYMM)
        self.assertEqual(wrapper.obs_act_out.channel_axis, 1)

    def test_other_kwargs_user_override_precedence(self):
        """
        For keys without PTQConfig-level defaults (like qscheme/channel_axis),
        user overrides > wrapper defaults.
        """
        qcfg = PTQConfig(
            default_dtype=DType.uint(8),
            default_observer=MinMaxObserver,  # type: ignore[type-abstract]
            overrides={
                "act_in": {
                    "qscheme": QScheme.PER_TENSOR_ASYMM,
                    "channel_axis": None,
                }
            },
        )

        # wrapper defaults try to force a per-channel scheme
        wrapper = DummyWrapper(
            qcfg,
            qscheme=QScheme.PER_CHANNEL_ASYMM,
            channel_axis=2,
        )

        # user override must win
        self.assertEqual(wrapper.obs_act_in.qscheme, QScheme.PER_TENSOR_ASYMM)
        self.assertIsNone(wrapper.obs_act_in.channel_axis)

    def test_PTQConfig_get_kwargs_does_not_inject_dtype(self):
        """
        Ensure PTQConfig.get_kwargs() itself doesn't inject dtype anymore.
        It should return exactly the user override dict.
        """
        qcfg = PTQConfig(
            default_dtype=DType.uint(8),
            overrides={"bar": {"qscheme": QScheme.PER_TENSOR_ASYMM}},
        )
        kw = qcfg.get_kwargs("bar")
        self.assertIn("qscheme", kw)
        self.assertNotIn("dtype", kw)

    def test_config_default_when_neither_user_nor_wrapper(self):
        """
        If neither user nor wrapper provides dtype/qscheme, fallback to PTQConfig defaults.
        """
        qcfg = PTQConfig(
            default_dtype=DType.uint(8),
            default_qscheme=QScheme.PER_TENSOR_ASYMM,
            default_observer=MinMaxObserver,  # type: ignore[type-abstract]
            overrides={},
        )

        wrapper = DummyWrapper(qcfg)

        self.assertIsInstance(wrapper.obs_act_in, MinMaxObserver)
        self.assertEqual(wrapper.obs_act_in.dtype, DType.uint(8))
        self.assertEqual(wrapper.obs_act_in.qscheme, qcfg.default_qscheme)
        self.assertIsNone(wrapper.obs_act_in.channel_axis)


class TestPTQConfigQScheme(unittest.TestCase):
    def test_default_qscheme_applied(self):
        cfg = PTQConfig(
            default_dtype=DType.uint(8),
            default_qscheme=QScheme.PER_CHANNEL_ASYMM,
        )
        w = DummyWrapper(cfg)
        self.assertEqual(w.obs_act_in.qscheme, QScheme.PER_CHANNEL_ASYMM)
        self.assertEqual(w.obs_act_out.qscheme, QScheme.PER_CHANNEL_ASYMM)

    def test_per_observer_qscheme_override(self):
        cfg = PTQConfig(
            default_dtype=DType.uint(8),
            default_qscheme=QScheme.PER_CHANNEL_ASYMM,
            overrides={
                "act_out": {"dtype": DType.int(16)},
            },
        )
        w = DummyWrapper(cfg)
        self.assertEqual(w.obs_act_in.qscheme, QScheme.PER_CHANNEL_ASYMM)
        self.assertEqual(w.obs_act_out.qscheme, QScheme.PER_TENSOR_SYMM)


class TestPTQConfigDefaults(unittest.TestCase):
    def test_unsigned_default_auto_qscheme_for_activation(self):
        cfg = PTQConfig(default_dtype=DType.uint(8), default_qscheme=None)
        wrapper = DummyWrapper(cfg)

        self.assertEqual(cfg.default_qscheme, QScheme.PER_TENSOR_ASYMM)
        self.assertEqual(wrapper.obs_act_in.dtype, DType.uint(8))
        self.assertEqual(wrapper.obs_act_in.qscheme, QScheme.PER_TENSOR_ASYMM)
        self.assertEqual(wrapper.obs_act_out.qscheme, QScheme.PER_TENSOR_ASYMM)

    def test_signed_default_auto_qscheme(self):
        cfg = PTQConfig(default_dtype=DType.int(16), default_qscheme=None)
        wrapper = DummyWrapper(cfg)

        self.assertEqual(cfg.default_qscheme, QScheme.PER_TENSOR_SYMM)
        self.assertEqual(wrapper.obs_act_in.dtype, DType.int(16))
        self.assertEqual(wrapper.obs_act_in.qscheme, QScheme.PER_TENSOR_SYMM)

    def test_explicit_invalid_default_pair_raises(self):
        with self.assertRaises(ValueError):
            PTQConfig(
                default_dtype=DType.uint(8),
                default_qscheme=QScheme.PER_TENSOR_SYMM,
            )

    def test_default_weight_observer_uses_wrapper_default_when_provided(self):
        cfg = PTQConfig(default_dtype=DType.uint(8), default_qscheme=None)
        wrapper = DummyWrapper(
            cfg,
            dtype=DType.int(8),
            qscheme=QScheme.PER_CHANNEL_SYMM,
        )

        self.assertEqual(wrapper.obs_weight.dtype, DType.int(8))
        self.assertEqual(wrapper.obs_weight.qscheme, QScheme.PER_CHANNEL_SYMM)


class TestPTQConfigOverrides(unittest.TestCase):
    def test_per_observer_dtype_override_auto_infers_qscheme(self):
        cfg = PTQConfig(
            default_dtype=DType.int(16),
            overrides={"act_out": {"dtype": DType.uint(4)}},
        )
        wrapper = DummyWrapper(cfg)

        self.assertEqual(wrapper.obs_act_in.dtype, DType.int(16))
        self.assertEqual(wrapper.obs_act_in.qscheme, QScheme.PER_TENSOR_SYMM)
        self.assertEqual(wrapper.obs_act_out.dtype, DType.uint(4))
        self.assertEqual(wrapper.obs_act_out.qscheme, QScheme.PER_TENSOR_ASYMM)

    def test_weight_override_auto_infers_per_channel_asymmetric(self):
        cfg = PTQConfig(
            default_dtype=DType.int(16),
            overrides={"weight": {"dtype": DType.uint(8)}},
        )
        wrapper = DummyWrapper(cfg)

        self.assertEqual(wrapper.obs_weight.dtype, DType.uint(8))
        self.assertEqual(wrapper.obs_weight.qscheme, QScheme.PER_CHANNEL_ASYMM)

    def test_observer_override(self):
        cfg = PTQConfig(
            default_dtype=DType.uint(8),
            overrides={
                "act_in": {
                    "observer": EMAObserver,
                    "dtype": DType.uint(8),
                }
            },
        )
        wrapper = DummyWrapper(cfg)
        self.assertIsInstance(wrapper.obs_act_in, EMAObserver)
        self.assertEqual(wrapper.obs_act_in.dtype, DType.uint(8))
        self.assertEqual(wrapper.obs_act_in.qscheme, QScheme.PER_TENSOR_ASYMM)
        self.assertIsInstance(wrapper.obs_act_out, MinMaxObserver)

    def test_get_kwargs_returns_normalized_override(self):
        cfg = PTQConfig(
            default_dtype=DType.int(16),
            overrides={"weight": {"dtype": DType.uint(8)}},
        )
        kwargs = cfg.get_kwargs("weight")

        self.assertEqual(kwargs["dtype"], DType.uint(8))
        self.assertEqual(kwargs["qscheme"], QScheme.PER_CHANNEL_ASYMM)

    def test_explicit_invalid_override_pair_raises(self):
        with self.assertRaises(ValueError):
            PTQConfig(
                default_dtype=DType.int(16),
                overrides={
                    "weight": {
                        "dtype": DType.uint(8),
                        "qscheme": QScheme.PER_CHANNEL_SYMM,
                    }
                },
            )

    def test_other_kwargs_user_override_precedence(self):
        cfg = PTQConfig(
            default_dtype=DType.uint(8),
            overrides={
                "act_in": {
                    "qscheme": QScheme.PER_TENSOR_ASYMM,
                    "channel_axis": None,
                }
            },
        )

        wrapper = DummyWrapper(
            cfg,
            qscheme=QScheme.PER_CHANNEL_ASYMM,
            channel_axis=2,
        )

        self.assertEqual(wrapper.obs_act_in.qscheme, QScheme.PER_TENSOR_ASYMM)
        self.assertIsNone(wrapper.obs_act_in.channel_axis)

    def test_user_override_wins_over_wrapper_defaults(self):
        qcfg = PTQConfig(
            default_dtype=DType.uint(8),
            default_observer=MinMaxObserver,  # type: ignore[type-abstract]
            overrides={
                "act_in": {
                    "dtype": DType.uint(4),
                    "observer": DummyObserverA,
                    "qscheme": QScheme.PER_TENSOR_ASYMM,
                    "channel_axis": None,
                }
            },
        )

        wrapper = DummyWrapper(
            qcfg,
            dtype=DType.uint(6),
            observer=DummyObserverB,
            qscheme=QScheme.PER_CHANNEL_ASYMM,
            channel_axis=0,
        )

        self.assertIsInstance(wrapper.obs_act_in, DummyObserverA)
        self.assertEqual(wrapper.obs_act_in.dtype, DType.uint(4))
        self.assertEqual(wrapper.obs_act_in.qscheme, QScheme.PER_TENSOR_ASYMM)
        self.assertIsNone(wrapper.obs_act_in.channel_axis)

    def test_wrapper_default_when_no_user_override(self):
        qcfg = PTQConfig(
            default_dtype=DType.uint(8),
            default_observer=MinMaxObserver,  # type: ignore[type-abstract]
        )

        wrapper = DummyWrapper(
            qcfg,
            dtype=DType.uint(6),
            observer=DummyObserverB,
            qscheme=QScheme.PER_CHANNEL_ASYMM,
            channel_axis=1,
        )

        self.assertIsInstance(wrapper.obs_act_out, DummyObserverB)
        self.assertEqual(wrapper.obs_act_out.dtype, DType.uint(6))
        self.assertEqual(wrapper.obs_act_out.qscheme, QScheme.PER_CHANNEL_ASYMM)
        self.assertEqual(wrapper.obs_act_out.channel_axis, 1)


class TestPTQConfigMutationHelpers(unittest.TestCase):
    def test_set_override_for_activation_infers_per_tensor_asymmetric(self):
        cfg = PTQConfig(default_dtype=DType.int(16))
        cfg.set_override(("block", "act_in"), {"dtype": DType.uint(8)})

        self.assertEqual(
            cfg.overrides["block"]["act_in"]["qscheme"],
            QScheme.PER_TENSOR_ASYMM,
        )

    def test_set_override_for_weight_infers_per_channel_asymmetric(self):
        cfg = PTQConfig(default_dtype=DType.int(16))
        cfg.set_override(("block", "weight"), {"dtype": DType.uint(8)})

        self.assertEqual(
            cfg.overrides["block"]["weight"]["qscheme"],
            QScheme.PER_CHANNEL_ASYMM,
        )

    def test_set_override_empty_path_raises(self):
        cfg = PTQConfig(default_dtype=DType.int(16))
        with self.assertRaises(ValueError):
            cfg.set_override((), {"dtype": DType.uint(8)})

    def test_normalize_overrides_after_direct_mutation(self):
        cfg = PTQConfig(default_dtype=DType.int(16))
        cfg.overrides["block"] = {"weight": {"dtype": DType.uint(8)}}  # type: ignore[index]
        cfg.normalize_overrides()

        self.assertEqual(
            cfg.overrides["block"]["weight"]["qscheme"],
            QScheme.PER_CHANNEL_ASYMM,
        )

    def test_normalize_overrides_rejects_invalid_direct_mutation(self):
        cfg = PTQConfig(default_dtype=DType.int(16))
        cfg.overrides["block"] = {  # type: ignore[index]
            "weight": {
                "dtype": DType.uint(8),
                "qscheme": QScheme.PER_CHANNEL_SYMM,
            }
        }

        with self.assertRaises(ValueError):
            cfg.normalize_overrides()
