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

"""Full-sequence MX V computation keeps independent affine cache boundaries."""

import copy
import unittest
from unittest import mock

import torch

from tico.quantization.config.specs import mx, no_quant
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.observers.mx import MXObserver
from tico.quantization.wrapq.wrappers.gemma4.quant_text_attention import (
    QuantGemma4TextAttention,
)

from test.quantization.quant_spec_helpers import make_affine_ptq_config
from test.quantization.wrapq.wrappers.gemma4.test_quant_text_attention import (
    _has_gemma4,
    _make_text_config,
)


@unittest.skipUnless(
    _has_gemma4(), "required transformers Gemma4 modules are not installed"
)
class TestGemma4PastKVMX(unittest.TestCase):
    """MX applies after concat, while read statistics cover only actual past."""

    @staticmethod
    def _make_attention(*, layout="unrolled", overrides=None):
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention

        torch.manual_seed(1077)
        config = _make_text_config(
            hidden_size=32,
            intermediate_size=64,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_global_key_value_heads=1,
            head_dim=32,
            global_head_dim=32,
        )
        policies = {
            "present_value": mx("int8", axis=-2),
            "softmax": mx("int8", axis=-1),
            "attn_weights": mx("int8", axis=-1),
        }
        policies.update({} if overrides is None else overrides)
        qcfg = make_affine_ptq_config(
            dtype=DType.int(16),
            overrides=policies,
            model_args={"attention": {"layout": layout}, "text": {"max_seq": 32}},
        )
        return QuantGemma4TextAttention(
            Gemma4TextAttention(config, layer_idx=0).eval(), qcfg=qcfg
        ).eval()

    @staticmethod
    def _forward(attention, length, past=None):
        hidden = torch.linspace(-1.4, 1.7, length * 32).reshape(1, length, 32)
        rope = (torch.ones(1, length, 32), torch.zeros(1, length, 32))
        return attention(
            hidden, rope, past_key_value=past, use_cache=True, cache_output_mode="full"
        )

    def _calibrate(self, attention):
        attention.enable_calibration()
        with torch.no_grad():
            cache = self._forward(attention, 3)[2]
            cache = self._forward(attention, 1, cache)[2]
            self._forward(attention, 1, cache)
        attention.freeze_qparams()

    def test_full_v_mx_blocks_follow_affine_read_conversion_and_concat(self):
        for layout in ("unrolled", "batched"):
            with self.subTest(layout=layout):
                attention = self._make_attention(layout=layout)
                self._calibrate(attention)
                # Deliberately different scale AND zero point. This case tests
                # conversion semantics separately from real cached calibration.
                attention.obs_new_v.load_qparams(torch.tensor(0.0625), torch.tensor(-3))
                attention.obs_past_value.load_qparams(
                    torch.tensor(0.125), torch.tensor(7)
                )
                past_v = torch.full((1, 1, 31, 32), 0.4375)
                new_v = torch.full((1, 1, 1, 32), 100.0)
                past_k = torch.zeros_like(past_v)
                new_k = torch.zeros_like(new_v)
                original_past = past_v.clone()
                original_new = new_v.clone()
                with mock.patch.object(
                    attention.obs_past_value,
                    "fake_quant",
                    wraps=attention.obs_past_value.fake_quant,
                ) as read:
                    compute, storage, cache = attention._build_present_key_value(
                        new_key_value=(new_k, new_v), past_key_value=(past_k, past_v)
                    )
                self.assertEqual(
                    [call.args[0].shape[-2] for call in read.call_args_list], [31, 1]
                )
                read_v = attention.obs_past_value.fake_quant(
                    attention.obs_new_v.fake_quant(past_v)
                )
                self.assertFalse(torch.equal(read_v, past_v))
                current_v = attention.obs_past_value.fake_quant(new_v)
                expected = attention.obs_present_value.fake_quant(
                    torch.cat((read_v, current_v), dim=-2)
                )
                per_part = torch.cat(
                    (
                        attention.obs_present_value.fake_quant(read_v),
                        attention.obs_present_value.fake_quant(current_v),
                    ),
                    dim=-2,
                )
                without_read_conversion = attention.obs_present_value.fake_quant(
                    torch.cat((past_v, new_v), dim=-2)
                )
                self.assertFalse(torch.equal(expected, without_read_conversion))
                self.assertFalse(torch.equal(expected, per_part))
                torch.testing.assert_close(compute[1], expected, atol=0, rtol=0)
                torch.testing.assert_close(
                    storage[1], torch.cat((past_v, new_v), dim=-2), atol=0, rtol=0
                )
                torch.testing.assert_close(past_v, original_past, atol=0, rtol=0)
                torch.testing.assert_close(new_v, original_new, atol=0, rtol=0)
                self.assertIsNone(cache)

    def test_prefill_unused_read_and_decode_only_existing_tokens(self):
        for layout in ("unrolled", "batched"):
            with self.subTest(layout=layout):
                attention = self._make_attention(layout=layout)
                attention.enable_calibration()
                with mock.patch.object(
                    attention.obs_past_value,
                    "collect",
                    wraps=attention.obs_past_value.collect,
                ) as collect:
                    cache = self._forward(attention, 3)[2]
                    collect.assert_not_called()
                    cache = self._forward(attention, 1, cache)[2]
                    self._forward(attention, 1, cache)
                self.assertEqual(
                    [call.args[0].shape[-2] for call in collect.call_args_list], [3, 4]
                )
                attention.freeze_qparams()
                with mock.patch.object(
                    attention.obs_past_value,
                    "fake_quant",
                    side_effect=AssertionError("prefill read"),
                ):
                    self._forward(attention, 3)
                attention.validate_export_cache("decode")
                self.assertIsNot(attention.obs_new_v, attention.obs_past_value)
                self.assertIsInstance(attention.obs_present_value, MXObserver)
                self.assertFalse(hasattr(attention, "obs_attn_value"))

    def test_mx_compute_state_round_trip_preserves_and_checks_policy(self):
        original = self._make_attention()
        self._calibrate(original)
        state = copy.deepcopy(original.state_dict())
        restored = self._make_attention()
        restored.enable_calibration()
        restored.load_state_dict(state)
        restored.freeze_qparams()
        self.assertEqual(restored.obs_present_value.elem_format, "int8")
        self.assertEqual(restored.obs_present_value.axis, -2)
        original_cache = self._forward(original, 3)[2]
        restored_cache = self._forward(restored, 3)[2]
        torch.testing.assert_close(
            self._forward(original, 1, original_cache)[0],
            self._forward(restored, 1, restored_cache)[0],
            atol=0,
            rtol=0,
        )
        for policy in (
            mx("int8", axis=-1),
            mx("fp4_e2m1", axis=-2),
            mx("int8", axis=-2, round="floor"),
        ):
            with self.subTest(policy=policy):
                incompatible = self._make_attention(overrides={"present_value": policy})
                with self.assertRaisesRegex(
                    RuntimeError, "cache observer policy differs"
                ):
                    incompatible.load_state_dict(state)

    def test_unsupported_mx_storage_read_and_compute_policies_are_rejected(self):
        policies = (
            {"new_v": mx("int8", axis=-2)},
            {"past_value": mx("int8", axis=-2)},
            {"present_key": mx("int8", axis=-1)},
            {"present_value": mx("int8", axis=-1)},
            {"present_value": mx("fp4_e2m1", axis=-2)},
            {"new_v": no_quant()},
            {"past_value": no_quant()},
        )
        for overrides in policies:
            with self.subTest(overrides=overrides):
                attention = self._make_attention(overrides=overrides)
                self._calibrate(attention)
                with self.assertRaisesRegex(
                    ValueError, "per-tensor affine|MX cache compute"
                ):
                    attention.validate_export_cache("decode")

    def test_mx_compute_requires_enabled_fake_quantization(self):
        attention = self._make_attention()
        self._calibrate(attention)
        attention.obs_present_value.disable_fake_quant()
        with self.assertRaisesRegex(ValueError, "enabled fake quantization"):
            attention.validate_export_cache("decode")

    def test_prefill_only_calibration_does_not_prepare_affine_read(self):
        attention = self._make_attention()
        attention.enable_calibration()
        self._forward(attention, 3)
        attention.freeze_qparams()
        self._forward(attention, 3)
        attention.validate_export_cache("prefill")
        with self.assertRaisesRegex(RuntimeError, "positive calibration.decode_steps"):
            attention.validate_export_cache("decode")
