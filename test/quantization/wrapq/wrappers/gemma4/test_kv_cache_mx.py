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

"""Affine storage and full-sequence MX computation for tiny Gemma4 attention."""

import copy
import unittest
from typing import Any
from unittest import mock

import torch

from tico.quantization.config.specs import affine, mx, no_quant
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.observers.ema import EMAObserver
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
class TestGemma4KVCacheMX(unittest.TestCase):
    """Keep S deltas independent from raw-current/full-V MX computation."""

    @staticmethod
    def _make_attention(*, layout="unrolled", overrides=None, config=None, layer_idx=0):
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention

        torch.manual_seed(1077)
        if config is None:
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
            Gemma4TextAttention(config, layer_idx=layer_idx).eval(), qcfg=qcfg
        ).eval()

    @staticmethod
    def _forward(attention, length, *, past=None, shared=None, mode="full"):
        hidden = torch.linspace(-1.4, 1.7, length * 32).reshape(1, length, 32)
        rope = (torch.ones(1, length, 32), torch.zeros(1, length, 32))
        return attention(
            hidden,
            rope,
            past_key_value=past,
            shared_kv_states=shared,
            use_cache=True,
            cache_output_mode=mode,
        )

    def _calibrate(self, attention):
        attention.enable_calibration()
        with torch.no_grad():
            self._forward(attention, 3)
        attention.freeze_qparams()

    def test_full_v_mx_quantizes_once_after_storage_past_and_raw_current_concat(self):
        for layout in ("unrolled", "batched"):
            with self.subTest(layout=layout):
                attention = self._make_attention(layout=layout)
                self._calibrate(attention)
                # Deliberately coarse S isolates the branch contract from the
                # real calibration above: the current raw value is off-grid.
                attention.obs_new_v.load_qparams(torch.tensor(0.5), torch.tensor(3))
                past_v = torch.full((1, 1, 31, 32), 0.5)
                raw_v = torch.full((1, 1, 1, 32), 100.375)
                new_v = attention.obs_new_v.fake_quant(raw_v)
                past_k, raw_k = torch.zeros_like(past_v), torch.zeros_like(raw_v)
                original_past, original_raw = past_v.clone(), raw_v.clone()
                with mock.patch.object(
                    attention.obs_present_value,
                    "fake_quant",
                    wraps=attention.obs_present_value.fake_quant,
                ) as compute_quantize:
                    compute, storage, cache = attention._build_present_key_value(
                        new_key_value=(raw_k, new_v),
                        current_key_value=(raw_k, raw_v),
                        past_key_value=(past_k, past_v),
                    )
                compute_quantize.assert_called_once()
                full = torch.cat((past_v, raw_v), dim=-2)
                torch.testing.assert_close(
                    compute_quantize.call_args.args[0], full, atol=0, rtol=0
                )
                expected = attention.obs_present_value.fake_quant(full)
                per_part = torch.cat(
                    (
                        attention.obs_present_value.fake_quant(past_v),
                        attention.obs_present_value.fake_quant(raw_v),
                    ),
                    dim=-2,
                )
                through_storage = attention.obs_present_value.fake_quant(
                    torch.cat((past_v, new_v), dim=-2)
                )
                self.assertFalse(torch.equal(expected, per_part))
                self.assertFalse(torch.equal(expected, through_storage))
                # The high current value changes the shared exponent for all
                # 32 tokens; separately quantized past would retain 0.5.
                self.assertEqual(float(expected[0, 0, 0, 0]), 1.0)
                self.assertEqual(float(expected[0, 0, -1, 0]), 100.0)
                torch.testing.assert_close(compute[1], expected, atol=0, rtol=0)
                torch.testing.assert_close(
                    storage[1], torch.cat((past_v, new_v), dim=-2), atol=0, rtol=0
                )
                torch.testing.assert_close(past_v, original_past, atol=0, rtol=0)
                torch.testing.assert_close(raw_v, original_raw, atol=0, rtol=0)
                self.assertIsNone(cache)

    def test_forward_current_branches_before_storage_rounding(self):
        for layout in ("unrolled", "batched"):
            for decode in (False, True):
                with self.subTest(layout=layout, decode=decode):
                    attention = self._make_attention(layout=layout)
                    self._calibrate(attention)
                    attention.obs_new_v.load_qparams(torch.tensor(0.5), torch.tensor(3))
                    raw_v = torch.full((1, 1, 1, 32), 0.22)
                    raw_k = torch.zeros_like(raw_v)
                    past_v = torch.full((1, 1, 31, 32), 0.5)
                    past = (torch.zeros_like(past_v), past_v) if decode else None
                    method = "_forward_" + layout
                    with (
                        mock.patch.object(
                            attention,
                            "_project_current_key_value",
                            return_value=(raw_k, raw_v),
                        ),
                        mock.patch.object(
                            attention, method, wraps=getattr(attention, method)
                        ) as compute,
                    ):
                        result = self._forward(attention, 1, past=past, mode="delta")
                    full = torch.cat((past_v, raw_v), dim=-2) if decode else raw_v
                    expected = attention.obs_present_value.fake_quant(full)
                    torch.testing.assert_close(
                        compute.call_args.kwargs["value_states"],
                        expected,
                        atol=0,
                        rtol=0,
                    )
                    self.assertGreater(float(expected[..., -1:, :].min()), 0)
                    torch.testing.assert_close(
                        result[2][1], torch.zeros_like(raw_v), atol=0, rtol=0
                    )

    def test_prefill_only_calibration_prepares_decode_without_past_observers(self):
        from transformers.cache_utils import DynamicCache, StaticCache

        for layout in ("unrolled", "batched"):
            for kind in ("tuple", "dynamic", "static"):
                with self.subTest(layout=layout, kind=kind):
                    attention = self._make_attention(layout=layout)
                    self._calibrate(attention)
                    attention.validate_export_cache("prefill")
                    attention.validate_export_cache("decode")
                    cache: Any
                    if kind == "dynamic":
                        cache = DynamicCache(config=attention.config)
                    elif kind == "static":
                        cache = StaticCache(attention.config, max_cache_len=32)
                    else:
                        cache = None
                    result = self._forward(attention, 3, past=cache)
                    cache = result[2]
                    for length in (4, 5):
                        previous_cache = cache
                        result = self._forward(attention, 1, past=cache)
                        cache = result[2]
                        if kind == "tuple":
                            self.assertEqual(cache[1].size(-2), length)
                        else:
                            self.assertIs(cache, previous_cache)
                            self.assertEqual(int(cache.get_seq_length()), length)
                    self.assertIsInstance(attention.obs_present_value, MXObserver)
                    for name in ("obs_past_key", "obs_past_value", "obs_attn_value"):
                        self.assertFalse(hasattr(attention, name))
                    self.assertFalse(
                        any("obs_past_" in k for k in attention.state_dict())
                    )

    def test_shared_mx_compute_preserves_producer_storage_and_lifecycle(self):
        config = _make_text_config(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            layer_types=["sliding_attention", "full_attention"] * 2,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_global_key_value_heads=1,
            num_kv_shared_layers=2,
            head_dim=32,
            global_head_dim=32,
        )
        for layout in ("unrolled", "batched"):
            with self.subTest(layout=layout):
                producer = self._make_attention(
                    layout=layout, config=config, layer_idx=1
                )
                consumer = self._make_attention(
                    layout=layout, config=config, layer_idx=3
                )
                consumer.bind_cache_source(producer)
                producer.enable_calibration()
                consumer.enable_calibration()
                shared: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
                self._forward(producer, 3, shared=shared)
                self._forward(consumer, 3, shared=shared)
                producer.freeze_qparams()
                consumer.freeze_qparams()
                self.assertIs(consumer.cache_storage_observers()[1], producer.obs_new_v)
                before = copy.deepcopy(producer.state_dict())
                cache = None
                for length in (3, 1, 1):
                    shared = {}
                    cache = self._forward(producer, length, past=cache, shared=shared)[
                        2
                    ]
                    stored = shared["full_attention"][1].clone()
                    with mock.patch.object(
                        consumer.obs_present_value,
                        "fake_quant",
                        wraps=consumer.obs_present_value.fake_quant,
                    ) as compute:
                        result = self._forward(consumer, length, shared=shared)
                    compute.assert_called_once()
                    torch.testing.assert_close(
                        compute.call_args.args[0], stored, atol=0, rtol=0
                    )
                    torch.testing.assert_close(
                        shared["full_attention"][1], stored, atol=0, rtol=0
                    )
                    self.assertIsNone(result[2])
                consumer.enable_calibration()
                self.assertIs(producer._mode, Mode.QUANT)
                for name, value in producer.state_dict().items():
                    if isinstance(value, torch.Tensor):
                        torch.testing.assert_close(value, before[name], atol=0, rtol=0)
                self.assertFalse(
                    any(module is producer for module in consumer.modules())
                )

    def test_mx_compute_state_round_trip_preserves_two_domain_policy(self):
        original = self._make_attention()
        self._calibrate(original)
        state = copy.deepcopy(original.state_dict())
        self.assertEqual(state["_extra_state"]["version"], 2)
        self.assertIn("new_v", state["_extra_state"]["qparams"])
        self.assertNotIn("present_value", state["_extra_state"]["qparams"])
        restored = self._make_attention()
        restored.enable_calibration()
        restored.load_state_dict(state)
        restored.freeze_qparams()
        self.assertEqual(restored.obs_present_value.elem_format, "int8")
        self.assertEqual(restored.obs_present_value.axis, -2)
        self.assertEqual(
            restored.get_extra_state()["policies"], state["_extra_state"]["policies"]
        )
        original_cache = self._forward(original, 3)[2]
        restored_cache = self._forward(restored, 3)[2]
        torch.testing.assert_close(
            self._forward(original, 1, past=original_cache)[0],
            self._forward(restored, 1, past=restored_cache)[0],
            atol=0,
            rtol=0,
        )
        for policy in (
            mx("int8", axis=-1),
            mx("fp4_e2m1", axis=-2),
            mx("int8", axis=-2, shared_exp_method="none"),
            mx("int8", axis=-2, round="floor"),
        ):
            with self.subTest(policy=policy):
                incompatible = self._make_attention(overrides={"present_value": policy})
                with self.assertRaisesRegex(
                    RuntimeError, "cache observer policy differs"
                ):
                    incompatible.load_state_dict(state)

    def test_mixed_policy_state_retains_affine_calibration_options(self):
        original = self._make_attention(
            overrides={
                "new_v": affine(DType.int(16), observer=EMAObserver, momentum=0.2)
            }
        )
        self._calibrate(original)
        incompatible = self._make_attention(
            overrides={
                "new_v": affine(DType.int(16), observer=EMAObserver, momentum=0.3)
            }
        )
        with self.assertRaisesRegex(RuntimeError, "cache observer policy differs"):
            incompatible.load_state_dict(copy.deepcopy(original.state_dict()))

    def test_unsupported_storage_and_compute_policies_are_rejected(self):
        for overrides in (
            {"new_v": mx("int8", axis=-2)},
            {"new_v": mx("int8", axis=-1)},
            {"new_v": affine(DType.uint(8))},
            {"present_key": mx("int8", axis=-1)},
            {"present_value": mx("int8", axis=-1)},
            {"present_value": mx("fp4_e2m1", axis=-2)},
            {"present_value": mx("int8", axis=-2, shared_exp_method="none")},
            {"present_value": mx("int8", axis=-2, round="floor")},
            {"new_v": no_quant()},
        ):
            with self.subTest(overrides=overrides):
                attention = self._make_attention(overrides=overrides)
                with self.assertRaisesRegex(ValueError, "per-tensor affine|MX compute"):
                    self._calibrate(attention)
                    attention.validate_export_cache("decode")

    def test_mx_compute_requires_enabled_fake_quantization(self):
        attention = self._make_attention()
        self._calibrate(attention)
        attention.obs_present_value.disable_fake_quant()
        with self.assertRaisesRegex(ValueError, "enabled fake quantization"):
            attention.validate_export_cache("decode")

    def test_stateless_mx_compute_does_not_bypass_storage_qparam_validation(self):
        attention = self._make_attention()
        self._calibrate(attention)
        state = copy.deepcopy(attention.state_dict())
        del state["_extra_state"]["qparams"]["new_v"]
        restored = self._make_attention()
        restored.enable_calibration()
        with self.assertRaisesRegex(RuntimeError, "lacks frozen new_v cache qparams"):
            restored.load_state_dict(state)
        for invalid_scale in (0.0, float("nan")):
            with self.subTest(scale=invalid_scale):
                attention.obs_new_v.load_qparams(
                    torch.tensor(invalid_scale), torch.tensor(0)
                )
                with self.assertRaisesRegex(
                    RuntimeError, "invalid frozen cache qparams"
                ):
                    attention.validate_export_cache("decode")
        attention.obs_new_v.reset()
        with self.assertRaisesRegex(
            RuntimeError, "storage/compute qparams are not ready"
        ):
            attention.validate_export_cache("decode")

    def test_mx_does_not_reintroduce_removed_past_policies_or_checkpoints(self):
        for name in ("past_key", "past_value"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "past.*removed"):
                    self._make_attention(overrides={name: mx("int8", axis=-2)})
        original = self._make_attention()
        self._calibrate(original)
        for strict in (False, True):
            for legacy_kind in ("version", "past_observer"):
                with self.subTest(strict=strict, legacy_kind=legacy_kind):
                    state = copy.deepcopy(original.state_dict())
                    if legacy_kind == "version":
                        state["_extra_state"]["version"] = 1
                    else:
                        state["obs_past_value.min_val"] = torch.tensor(-1.0)
                    with self.assertRaisesRegex(
                        RuntimeError, "incompatible cache quantization contract"
                    ):
                        self._make_attention().load_state_dict(state, strict=strict)
