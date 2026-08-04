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

from tico.quantization.config.builders import build_llm_ptq_config
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.config.specs import affine, identity, mx
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.observers.base import ObserverBase
from tico.quantization.wrapq.observers.identity import IdentityObserver
from tico.quantization.wrapq.observers.minmax import MinMaxObserver
from tico.quantization.wrapq.observers.mx import MXObserver
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase


class DummyWrapper(QuantModuleBase):
    """Minimal wrapper used to exercise observer construction."""

    def __init__(self, qcfg, **kwargs):
        super().__init__(qcfg)
        self.obs_act_in = self._make_obs("act_in", **kwargs)
        self.obs_act_out = self._make_obs("act_out", **kwargs)
        self.obs_weight = self._make_obs("weight", **kwargs)

    def _all_observers(self):
        return (self.obs_act_in, self.obs_act_out, self.obs_weight)


class TestQuantSpecConfig(unittest.TestCase):
    def test_mx_activation_does_not_change_weight_observer(self):
        cfg = PTQConfig(activation=mx("fp8_e4m3", axis=-1))
        wrapper = DummyWrapper(cfg)

        self.assertIsInstance(wrapper.obs_act_in, MXObserver)
        self.assertIsInstance(wrapper.obs_act_out, MXObserver)
        self.assertIsInstance(wrapper.obs_weight, MinMaxObserver)
        self.assertEqual(wrapper.obs_act_in.elem_format, "fp8_e4m3")
        self.assertEqual(wrapper.obs_act_in.axis, -1)

    def test_affine_weight_spec_applies_to_weight_role(self):
        cfg = PTQConfig(weight=affine(DType.uint(4)))
        wrapper = DummyWrapper(cfg)

        self.assertIsInstance(wrapper.obs_weight, MinMaxObserver)
        self.assertEqual(wrapper.obs_weight.dtype, DType.uint(4))

    def test_dot_path_override_is_expanded(self):
        cfg = PTQConfig(
            overrides={
                "block.proj.act_out": mx("int8", axis=1),
            }
        )

        block_cfg = cfg.child("block")
        proj_cfg = block_cfg.child("proj")
        kwargs = proj_cfg.get_kwargs("act_out")

        self.assertEqual(kwargs["observer"], MXObserver)
        self.assertEqual(kwargs["elem_format"], "int8")
        self.assertEqual(kwargs["axis"], 1)

    def test_set_override_accepts_string_path(self):
        cfg = PTQConfig()
        cfg.set_override("block.proj.act_in", mx("fp8_e5m2", axis=0))

        kwargs = cfg.child("block").child("proj").get_kwargs("act_in")

        self.assertEqual(kwargs["observer"], MXObserver)
        self.assertEqual(kwargs["elem_format"], "fp8_e5m2")
        self.assertEqual(kwargs["axis"], 0)


class TestQuantSpecBuilders(unittest.TestCase):
    def test_llm_builder_accepts_quant_specs(self):
        cfg = build_llm_ptq_config(
            model_type="llama",
            num_hidden_layers=1,
            activation=mx("fp8_e4m3", axis=-1),
            linear_weight=affine(DType.uint(4)),
        )

        self.assertEqual(cfg.activation.observer, MXObserver)
        q_proj_weight = (
            cfg.child("model")
            .child("layers")
            .child("0")
            .child("self_attn")
            .child("q_proj")
            .get_kwargs("weight")
        )
        self.assertEqual(q_proj_weight["dtype"], DType.uint(4))

    def test_quant_module_base_role_precedence(self):
        class CustomObserver(MinMaxObserver):
            pass

        cfg = PTQConfig(
            activation=affine(DType.int(8), observer=CustomObserver),  # type: ignore[type-abstract]
            overrides={"act_out": mx("int8", axis=-1)},
        )
        wrapper = DummyWrapper(cfg)

        self.assertIsInstance(wrapper.obs_act_in, CustomObserver)
        self.assertIsInstance(wrapper.obs_act_out, MXObserver)


class TestIdentitySpec(unittest.TestCase):
    """Tests for the identity() spec — enables weights-only quantization."""

    def test_identity_returns_identity_observer(self):
        """identity() must produce a QuantSpec with IdentityObserver."""
        spec = identity()
        self.assertEqual(spec.observer, IdentityObserver)
        self.assertIsNone(spec.dtype)
        self.assertIsNone(spec.qscheme)

    def test_identity_to_kwargs_has_no_dtype_or_qscheme(self):
        """Identity spec must not emit dtype/qscheme (no quantization)."""
        spec = identity()
        kwargs = spec.to_kwargs(obs_name="act_in", context="test")
        self.assertEqual(kwargs["observer"], IdentityObserver)
        self.assertNotIn("dtype", kwargs)
        self.assertNotIn("qscheme", kwargs)

    def test_identity_to_kwargs_mark_replace(self):
        """mark_replace should add the internal replace marker."""
        spec = identity()
        kwargs = spec.to_kwargs(obs_name="act_in", mark_replace=True)
        self.assertTrue(kwargs.get("__quant_spec_replace_role__"))

    def test_identity_activation_weight_quantized(self):
        """Weight-only quantization: identity activations + affine weights."""
        cfg = PTQConfig(
            activation=identity(),
            weight=affine(DType.uint(4)),
        )
        wrapper = DummyWrapper(cfg)

        # Activations must be IdentityObserver (float32 passthrough)
        self.assertIsInstance(wrapper.obs_act_in, IdentityObserver)
        self.assertIsInstance(wrapper.obs_act_out, IdentityObserver)

        # Weights must still be quantized
        self.assertIsInstance(wrapper.obs_weight, MinMaxObserver)
        self.assertEqual(wrapper.obs_weight.dtype, DType.uint(4))

    def test_identity_observer_fake_quant_is_passthrough(self):
        """IdentityObserver.fake_quant() must return input unchanged."""
        import torch

        cfg = PTQConfig(activation=identity())
        wrapper = DummyWrapper(cfg)

        x = torch.randn(2, 3)
        out = wrapper.obs_act_in.fake_quant(x)
        self.assertTrue(torch.equal(x, out))


if __name__ == "__main__":
    unittest.main()
