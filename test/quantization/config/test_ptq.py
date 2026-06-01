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
from tico.quantization.config.specs import affine, mx
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.observers.affine_base import AffineObserverBase
from tico.quantization.wrapq.observers.ema import EMAObserver
from tico.quantization.wrapq.observers.minmax import MinMaxObserver
from tico.quantization.wrapq.observers.mx import MXObserver
from tico.quantization.wrapq.qscheme import QScheme
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase


class DummyWrapper(QuantModuleBase):
    """Minimal wrapper that exposes _make_obs for config tests."""

    def __init__(self, qcfg, **kwargs):
        super().__init__(qcfg)
        self.obs_act_in = self._make_obs("act_in", **kwargs)
        self.obs_act_out = self._make_obs("act_out", **kwargs)
        self.obs_weight = self._make_obs("weight", **kwargs)

    def _all_observers(self):
        return (self.obs_act_in, self.obs_act_out, self.obs_weight)


class DummyObserverA(AffineObserverBase):
    """Observer used only to check precedence."""

    def _update_stats(self, x):
        return super()._update_stats(x)


class DummyObserverB(AffineObserverBase):
    """Observer used only to check precedence."""

    def _update_stats(self, x):
        return super()._update_stats(x)


class TestPTQConfigRoles(unittest.TestCase):
    def test_default_specs_are_applied(self):
        cfg = PTQConfig(
            activation=affine(DType.uint(8)),
            weight=affine(DType.uint(4)),
        )
        wrapper = DummyWrapper(cfg)

        self.assertEqual(wrapper.obs_act_in.dtype, DType.uint(8))
        self.assertEqual(wrapper.obs_act_out.dtype, DType.uint(8))
        self.assertEqual(wrapper.obs_weight.dtype, DType.uint(4))
        self.assertEqual(wrapper.obs_weight.qscheme, QScheme.PER_CHANNEL_ASYMM)

    def test_mx_activation_spec_does_not_change_weight_observer(self):
        cfg = PTQConfig(
            activation=mx("fp8_e4m3", axis=-1),
            weight=affine(DType.uint(4)),
        )
        wrapper = DummyWrapper(cfg)

        self.assertIsInstance(wrapper.obs_act_in, MXObserver)
        self.assertIsInstance(wrapper.obs_act_out, MXObserver)
        self.assertIsInstance(wrapper.obs_weight, MinMaxObserver)
        self.assertEqual(wrapper.obs_weight.dtype, DType.uint(4))

    def test_wrapper_defaults_override_role_defaults_when_user_does_not_override(self):
        cfg = PTQConfig(activation=affine(DType.uint(8)))
        wrapper = DummyWrapper(
            cfg,
            dtype=DType.uint(6),
            observer=DummyObserverB,
            qscheme=QScheme.PER_CHANNEL_ASYMM,
            channel_axis=1,
        )

        self.assertIsInstance(wrapper.obs_act_out, DummyObserverB)
        self.assertEqual(wrapper.obs_act_out.dtype, DType.uint(6))
        self.assertEqual(wrapper.obs_act_out.qscheme, QScheme.PER_CHANNEL_ASYMM)
        self.assertEqual(wrapper.obs_act_out.channel_axis, 1)

    def test_user_override_wins_over_role_and_wrapper_defaults(self):
        cfg = PTQConfig(
            activation=affine(DType.uint(8), observer=MinMaxObserver),  # type: ignore[type-abstract]
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
            cfg,
            dtype=DType.uint(6),  # type: ignore[type-abstract]
            observer=DummyObserverB,
            qscheme=QScheme.PER_CHANNEL_ASYMM,
            channel_axis=0,
        )

        self.assertIsInstance(wrapper.obs_act_in, DummyObserverA)
        self.assertEqual(wrapper.obs_act_in.dtype, DType.uint(4))
        self.assertEqual(wrapper.obs_act_in.qscheme, QScheme.PER_TENSOR_ASYMM)
        self.assertIsNone(wrapper.obs_act_in.channel_axis)


class TestPTQConfigOverrides(unittest.TestCase):
    def test_per_observer_dtype_override_infers_qscheme(self):
        cfg = PTQConfig(
            activation=affine(DType.int(16)),
            overrides={"act_out": {"dtype": DType.uint(4)}},
        )
        wrapper = DummyWrapper(cfg)

        self.assertEqual(wrapper.obs_act_in.dtype, DType.int(16))
        self.assertEqual(wrapper.obs_act_in.qscheme, QScheme.PER_TENSOR_SYMM)
        self.assertEqual(wrapper.obs_act_out.dtype, DType.uint(4))
        self.assertEqual(wrapper.obs_act_out.qscheme, QScheme.PER_TENSOR_ASYMM)

    def test_weight_override_infers_per_channel_asymmetric(self):
        cfg = PTQConfig(
            activation=affine(DType.int(16)),
            overrides={"weight": {"dtype": DType.uint(8)}},
        )
        wrapper = DummyWrapper(cfg)

        self.assertEqual(wrapper.obs_weight.dtype, DType.uint(8))
        self.assertEqual(wrapper.obs_weight.qscheme, QScheme.PER_CHANNEL_ASYMM)

    def test_quant_spec_override_replaces_role_spec(self):
        cfg = PTQConfig(
            activation=affine(DType.int(16)),
            overrides={"act_out": mx("int8", axis=-1)},
        )
        wrapper = DummyWrapper(cfg)

        self.assertIsInstance(wrapper.obs_act_out, MXObserver)
        self.assertNotIn("dtype", cfg.get_kwargs("act_out"))

    def test_dot_path_override_is_expanded(self):
        cfg = PTQConfig(
            activation=affine(DType.int(16)),
            overrides={"gate_proj.act_in": affine(DType.uint(4))},
        )
        child = cfg.child("gate_proj")

        self.assertEqual(child.get_kwargs("act_in")["dtype"], DType.uint(4))
        self.assertEqual(
            child.get_kwargs("act_in")["qscheme"], QScheme.PER_TENSOR_ASYMM
        )

    def test_set_override_accepts_tuple_and_dot_paths(self):
        cfg = PTQConfig(activation=affine(DType.int(16)))
        cfg.set_override(("model", "layers", "0", "act_in"), affine(DType.uint(4)))
        cfg.set_override("model.layers.0.act_out", affine(DType.uint(8)))

        layer_cfg = cfg.child("model").child("layers").child("0")
        self.assertEqual(layer_cfg.get_kwargs("act_in")["dtype"], DType.uint(4))
        self.assertEqual(layer_cfg.get_kwargs("act_out")["dtype"], DType.uint(8))

    def test_set_override_empty_path_raises(self):
        cfg = PTQConfig()
        with self.assertRaises(ValueError):
            cfg.set_override((), {"dtype": DType.uint(4)})

    def test_invalid_unsigned_symmetric_pair_raises(self):
        with self.assertRaises(ValueError):
            PTQConfig(activation=affine(DType.uint(8), qscheme=QScheme.PER_TENSOR_SYMM))

        with self.assertRaises(ValueError):
            PTQConfig(
                activation=affine(DType.int(16)),
                overrides={
                    "act_in": {
                        "dtype": DType.uint(8),
                        "qscheme": QScheme.PER_TENSOR_SYMM,
                    }
                },
            )


class TestPTQConfigChild(unittest.TestCase):
    def test_child_inherits_specs_and_model_args(self):
        parent = PTQConfig(
            activation=affine(DType.int(16)),
            weight=affine(DType.uint(4)),
            overrides={"gate_proj": {"act_in": {"dtype": DType.uint(8)}}},
            model_args={"profile": "reference_eval"},
            strict_wrap=False,
            attention_mask_fill_value=-100.0,
        )
        child = parent.child("gate_proj")

        self.assertEqual(child.activation, parent.activation)
        self.assertEqual(child.weight, parent.weight)
        self.assertEqual(child.get_kwargs("act_in")["dtype"], DType.uint(8))
        self.assertEqual(child.model_args, {"profile": "reference_eval"})
        self.assertFalse(child.strict_wrap)
        self.assertEqual(child.attention_mask_fill_value, -100.0)

    def test_child_mutation_is_isolated_from_parent(self):
        parent = PTQConfig(activation=affine(DType.uint(8)))
        child = parent.child("dummy")
        child.set_override("x", affine(DType.int(8)))

        self.assertNotIn("x", parent.overrides)
        self.assertEqual(child.get_kwargs("x")["qscheme"], QScheme.PER_TENSOR_SYMM)


if __name__ == "__main__":
    unittest.main()
