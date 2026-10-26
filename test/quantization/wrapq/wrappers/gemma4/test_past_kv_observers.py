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

"""Storage and read boundaries for real, tiny Gemma4 attention wrappers."""

import copy
import unittest
from typing import Any
from unittest import mock

import torch

from tico.quantization.config.specs import affine, mx, no_quant
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.observers.ema import EMAObserver
from tico.quantization.wrapq.observers.identity import IdentityObserver
from tico.quantization.wrapq.qscheme import QScheme
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
class TestGemma4PastKVObservers(unittest.TestCase):
    """Separate current storage, existing cache reads, and attention inputs."""

    def _make_attention(
        self, *, layout="unrolled", config=None, layer_idx=0, overrides=None
    ):
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention

        torch.manual_seed(410)
        config = _make_text_config() if config is None else config
        qcfg = make_affine_ptq_config(
            dtype=DType.int(16),
            overrides={} if overrides is None else overrides,
            model_args={"attention": {"layout": layout}, "text": {"max_seq": 32}},
        )
        return QuantGemma4TextAttention(
            Gemma4TextAttention(config, layer_idx=layer_idx).eval(), qcfg=qcfg
        ).eval()

    @staticmethod
    def _inputs(length):
        hidden = torch.linspace(-1.4, 1.7, length * 16).reshape(1, length, 16)
        return hidden, (torch.ones(1, length, 4), torch.zeros(1, length, 4))

    def _forward(self, attention, length, *, past=None, mode="present", shared=None):
        hidden, rope = self._inputs(length)
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
            cache = self._forward(attention, 3)[2]
            cache = self._forward(attention, 1, past=cache)[2]
            self._forward(attention, 1, past=cache)
        attention.freeze_qparams()

    @staticmethod
    def _different_grids(attention):
        # These intentional qparam injections test incompatible read/storage grids;
        # actual decode calibration is tested separately without injected qparams.
        for observer in (attention.obs_new_k, attention.obs_new_v):
            observer.load_qparams(torch.tensor(0.125), torch.tensor(0))
        for observer in (attention.obs_past_key, attention.obs_past_value):
            observer.load_qparams(torch.tensor(0.7), torch.tensor(0))
        for observer in (attention.obs_present_key, attention.obs_present_value):
            observer.load_qparams(torch.tensor(0.3), torch.tensor(0))

    def test_independent_observers_and_explicit_overrides(self):
        attention = self._make_attention(
            overrides={
                "new_k": affine(DType.int(16)),
                "past_key": affine(DType.uint(8)),
                "present_key": affine(DType.uint(4)),
            }
        )
        roles = (attention.obs_new_k, attention.obs_past_key, attention.obs_present_key)
        self.assertEqual(len({id(observer) for observer in roles}), 3)
        self.assertEqual(
            [observer.dtype for observer in roles],
            [DType.int(16), DType.uint(8), DType.uint(4)],
        )
        for name in (
            "new_k",
            "new_v",
            "past_key",
            "past_value",
            "present_key",
            "present_value",
        ):
            self.assertIs(
                attention.get_observer(name), getattr(attention, "obs_" + name)
            )

    def test_default_read_policy_follows_storage_without_aliasing(self):
        attention = self._make_attention(overrides={"new_k": affine(DType.uint(8))})
        self.assertEqual(attention.obs_past_key.dtype, attention.obs_new_k.dtype)
        self.assertEqual(attention.obs_past_key.qscheme, attention.obs_new_k.qscheme)
        self.assertIsNot(attention.obs_past_key, attention.obs_new_k)

    def test_spec_replacement_and_partial_kwargs_keep_factory_semantics(self):
        attention = self._make_attention(
            overrides={
                "new_k": affine(DType.uint(8)),
                "past_key": no_quant(),
                "past_value": {"observer": EMAObserver, "momentum": 0.4},
            }
        )
        self.assertIsInstance(attention.obs_past_key, IdentityObserver)
        self.assertIsInstance(attention.obs_past_value, EMAObserver)
        self.assertEqual(attention.obs_past_value.momentum, 0.4)
        self.assertEqual(attention.obs_past_value.dtype, DType.int(16))

    def test_no_past_prefill_observes_new_and_present_only(self):
        from transformers.cache_utils import DynamicCache, StaticCache

        for layout in ("batched", "unrolled"):
            for cache_kind in ("none", "tuple", "dynamic", "static"):
                with self.subTest(layout=layout, cache_kind=cache_kind):
                    attention = self._make_attention(layout=layout)
                    cache: Any = None
                    if cache_kind == "tuple":
                        cache = (torch.empty(1, 2, 0, 4), torch.empty(1, 2, 0, 4))
                    elif cache_kind == "dynamic":
                        cache = DynamicCache(config=attention.config)
                    elif cache_kind == "static":
                        cache = StaticCache(attention.config, max_cache_len=32)
                    attention.enable_calibration()
                    with (
                        mock.patch.object(attention.obs_past_key, "collect") as past_k,
                        mock.patch.object(
                            attention.obs_past_value, "collect"
                        ) as past_v,
                    ):
                        self._forward(attention, 3, past=cache)
                    past_k.assert_not_called()
                    past_v.assert_not_called()
                    for observer in (
                        attention.obs_new_k,
                        attention.obs_new_v,
                        attention.obs_present_key,
                        attention.obs_present_value,
                    ):
                        self.assertTrue(torch.isfinite(observer.min_val).all())
                        self.assertTrue(torch.isfinite(observer.max_val).all())

    def test_quantized_prefill_never_uses_read_grid(self):
        from transformers.cache_utils import DynamicCache, StaticCache

        attention = self._make_attention()
        self._calibrate(attention)
        self._different_grids(attention)
        for cache in (
            None,
            (torch.empty(1, 2, 0, 4), torch.empty(1, 2, 0, 4)),
            DynamicCache(config=attention.config),
            StaticCache(attention.config, max_cache_len=32),
        ):
            with (
                mock.patch.object(
                    attention.obs_past_key,
                    "fake_quant",
                    side_effect=AssertionError("past K in prefill"),
                ),
                mock.patch.object(
                    attention.obs_past_value,
                    "fake_quant",
                    side_effect=AssertionError("past V in prefill"),
                ),
            ):
                self._forward(attention, 3, past=cache)

    def test_decode_observes_only_existing_tokens_even_without_cache_output(self):
        past = (
            torch.arange(24).reshape(1, 2, 3, 4).float() + 20,
            torch.arange(24).reshape(1, 2, 3, 4).float() - 30,
        )
        for layout in ("batched", "unrolled"):
            with self.subTest(layout=layout):
                attention = self._make_attention(layout=layout)
                attention.enable_calibration()
                with (
                    mock.patch.object(
                        attention.obs_past_key,
                        "collect",
                        wraps=attention.obs_past_key.collect,
                    ) as key,
                    mock.patch.object(
                        attention.obs_past_value,
                        "collect",
                        wraps=attention.obs_past_value.collect,
                    ) as value,
                ):
                    attention(*self._inputs(1), past_key_value=past, use_cache=False)
                self.assertEqual(key.call_count, 1)
                self.assertEqual(value.call_count, 1)
                torch.testing.assert_close(
                    key.call_args.args[0], past[0], atol=0, rtol=0
                )
                torch.testing.assert_close(
                    value.call_args.args[0], past[1], atol=0, rtol=0
                )

    def test_tuple_dynamic_and_static_preserve_storage_across_two_decodes(self):
        from transformers.cache_utils import DynamicCache, StaticCache

        reference = self._make_attention()
        self._calibrate(reference)
        self._different_grids(reference)
        expected_outputs = None
        for cache_kind in ("tuple", "dynamic", "static"):
            with self.subTest(cache_kind=cache_kind):
                attention = copy.deepcopy(reference)
                cache: Any = None
                if cache_kind == "dynamic":
                    cache = DynamicCache(config=attention.config)
                elif cache_kind == "static":
                    cache = StaticCache(attention.config, max_cache_len=32)
                outputs = []
                with torch.no_grad():
                    prefill = self._forward(attention, 3, past=cache)
                    cache = prefill[2]
                    outputs.append(prefill[0])
                    storage = (
                        cache
                        if isinstance(cache, tuple)
                        else (
                            cache.layers[0].keys[:, :, :3],
                            cache.layers[0].values[:, :, :3],
                        )
                    )
                    initial = tuple(tensor.clone() for tensor in storage)
                    self.assertFalse(
                        torch.equal(
                            initial[0], attention.obs_past_key.fake_quant(initial[0])
                        )
                    )
                    past_lengths = []

                    def record_read(tensor):
                        past_lengths.append(tensor.shape[-2])
                        torch.testing.assert_close(
                            tensor[:, :, :3], initial[0], atol=0, rtol=0
                        )
                        return original_read(tensor)

                    original_read = attention.obs_past_key.fake_quant
                    with mock.patch.object(
                        attention.obs_past_key, "fake_quant", side_effect=record_read
                    ):
                        for total in (4, 5):
                            result = self._forward(attention, 1, past=cache)
                            outputs.append(result[0])
                            cache = result[2]
                            storage = (
                                cache
                                if isinstance(cache, tuple)
                                else (
                                    cache.layers[0].keys[:, :, :total],
                                    cache.layers[0].values[:, :, :total],
                                )
                            )
                            for index, tensor in enumerate(storage):
                                torch.testing.assert_close(
                                    tensor[:, :, :3], initial[index], atol=0, rtol=0
                                )
                                observer = (attention.obs_new_k, attention.obs_new_v)[
                                    index
                                ]
                                torch.testing.assert_close(
                                    tensor, observer.fake_quant(tensor), atol=0, rtol=0
                                )
                    self.assertEqual(past_lengths, [3, 4])
                if expected_outputs is None:
                    expected_outputs = outputs
                else:
                    for actual, expected in zip(outputs, expected_outputs):
                        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_shared_consumer_reads_producer_storage_and_owns_no_producer_lifecycle(
        self,
    ):
        config = _make_text_config(
            num_hidden_layers=2,
            layer_types=["full_attention"] * 2,
            num_kv_shared_layers=1,
        )
        producer = self._make_attention(config=config)
        consumer = self._make_attention(
            config=config, layer_idx=1, overrides={"past_key": affine(DType.uint(8))}
        )
        consumer.bind_cache_source(producer)
        self._calibrate(producer)
        consumer.enable_calibration()
        shared: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        producer._mode = Mode.CALIB
        self._forward(producer, 3, shared=shared)
        with mock.patch.object(
            consumer.obs_past_key, "collect", wraps=consumer.obs_past_key.collect
        ) as read:
            self._forward(consumer, 3, shared=shared)
        torch.testing.assert_close(
            read.call_args.args[0], shared["full_attention"][0], atol=0, rtol=0
        )
        producer.freeze_qparams()
        consumer.freeze_qparams()
        self._different_grids(producer)
        shared = {}
        delta = self._forward(producer, 3, shared=shared, mode="delta")[2]
        for actual, expected in zip(shared["full_attention"], delta):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        consumer.obs_past_key.load_qparams(torch.tensor(0.6), torch.tensor(120))
        with mock.patch.object(
            consumer.obs_past_key, "fake_quant", wraps=consumer.obs_past_key.fake_quant
        ) as read:
            self._forward(consumer, 3, shared=shared)
        torch.testing.assert_close(read.call_args.args[0], delta[0], atol=0, rtol=0)
        before = copy.deepcopy(producer.state_dict())
        consumer.enable_calibration()
        self.assertIs(producer._mode, Mode.QUANT)
        for name, value in producer.state_dict().items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, before[name], atol=0, rtol=0)
        self.assertFalse(any(module is producer for module in consumer.modules()))
        self.assertFalse(
            any(name.startswith("_cache_source.") for name in consumer.state_dict())
        )

    def test_sliding_cache_rollover_reads_retained_past_without_padded_slots(self):
        from transformers.cache_utils import DynamicCache, StaticCache

        config = _make_text_config(
            num_hidden_layers=2,
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=4,
        )
        reference = self._make_attention(config=config)
        self._calibrate(reference)
        self._different_grids(reference)
        expected_outputs = None
        for cache_kind in ("tuple", "dynamic", "static"):
            with self.subTest(cache_kind=cache_kind):
                attention = copy.deepcopy(reference)
                cache: Any = None
                if cache_kind == "dynamic":
                    cache = DynamicCache(config=config)
                elif cache_kind == "static":
                    cache = StaticCache(config, max_cache_len=32)
                cache = self._forward(attention, 3, past=cache)[2]
                outputs = []
                lengths = []
                original_read = attention.obs_past_key.fake_quant

                def record_read(tensor):
                    lengths.append(tensor.shape[-2])
                    torch.testing.assert_close(
                        tensor, attention.obs_new_k.fake_quant(tensor), atol=0, rtol=0
                    )
                    return original_read(tensor)

                with mock.patch.object(
                    attention.obs_past_key, "fake_quant", side_effect=record_read
                ):
                    for step in range(3):
                        hidden, rope = self._inputs(1)
                        result = attention(
                            hidden + step * 0.13,
                            rope,
                            past_key_value=cache,
                            use_cache=True,
                            cache_output_mode="present",
                        )
                        outputs.append(result[0])
                        cache = result[2]
                        if isinstance(cache, tuple):
                            # The external tuple owner keeps the window's P=3
                            # retained tokens for the following one-token call.
                            cache = tuple(tensor[..., -3:, :] for tensor in cache)
                self.assertEqual(lengths, [3, 3, 3])
                if expected_outputs is None:
                    expected_outputs = outputs
                else:
                    for actual, expected in zip(outputs, expected_outputs):
                        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_invalid_tuple_lengths_are_rejected_explicitly(self):
        attention = self._make_attention()
        for key_length, value_length in ((0, 1), (2, 1), (1, 0)):
            with self.subTest(key_length=key_length, value_length=value_length):
                cache = (
                    torch.zeros(1, 2, key_length, 4),
                    torch.zeros(1, 2, value_length, 4),
                )
                with self.assertRaisesRegex(ValueError, "cache|key|value"):
                    self._forward(attention, 1, past=cache)

    def test_unsupported_read_policies_fail_before_cached_quantization(self):
        policies = (
            mx(),
            affine(DType.int(16), qscheme=QScheme.PER_CHANNEL_SYMM, channel_axis=1),
        )
        for policy in policies:
            with self.subTest(policy=policy):
                attention = self._make_attention(overrides={"past_key": policy})
                self._calibrate(attention)
                cache = self._forward(attention, 3)[2]
                with self.assertRaisesRegex(ValueError, "per-tensor affine"):
                    self._forward(attention, 1, past=cache)

    def test_stateless_read_policy_allows_prefill_only_calibration_then_decode(self):
        attention = self._make_attention(
            overrides={
                name: no_quant()
                for name in (
                    "new_k",
                    "new_v",
                    "past_key",
                    "past_value",
                    "present_key",
                    "present_value",
                )
            }
        )
        attention.enable_calibration()
        self._forward(attention, 3)
        attention.freeze_qparams()
        cache = self._forward(attention, 3)[2]
        self._forward(attention, 1, past=cache)

    def test_loaded_read_qparams_do_not_require_decode_statistics(self):
        attention = self._make_attention()
        attention.enable_calibration()
        self._forward(attention, 3)
        for observer in (attention.obs_past_key, attention.obs_past_value):
            self.assertTrue(torch.isinf(observer.min_val))
            observer.load_qparams(torch.tensor(0.25), torch.tensor(0))
        attention.freeze_qparams()
        cache = self._forward(attention, 3)[2]
        self._forward(attention, 1, past=cache)
        for observer in (attention.obs_past_key, attention.obs_past_value):
            torch.testing.assert_close(
                observer._cached_scale, torch.tensor(0.25), atol=0, rtol=0
            )

    def test_state_dict_round_trip_preserves_read_stats_qparams_and_binding(self):
        attention = self._make_attention()
        self._calibrate(attention)
        restored = self._make_attention()
        restored.enable_calibration()
        restored.load_state_dict(copy.deepcopy(attention.state_dict()))
        restored.freeze_qparams()
        for name in ("past_key", "past_value", "new_k", "new_v"):
            original = attention.get_observer(name)
            loaded = restored.get_observer(name)
            torch.testing.assert_close(loaded.min_val, original.min_val, atol=0, rtol=0)
            torch.testing.assert_close(loaded.max_val, original.max_val, atol=0, rtol=0)
            for actual, expected in zip(
                loaded.compute_qparams(), original.compute_qparams()
            ):
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        original_cache = self._forward(attention, 3)[2]
        loaded_cache = self._forward(restored, 3)[2]
        for actual, expected in zip(loaded_cache, original_cache):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(
            self._forward(restored, 1, past=loaded_cache)[0],
            self._forward(attention, 1, past=original_cache)[0],
            atol=0,
            rtol=0,
        )

    def test_consumer_state_rejects_changed_producer_storage_grid(self):
        config = _make_text_config(
            num_hidden_layers=2,
            layer_types=["full_attention"] * 2,
            num_kv_shared_layers=1,
        )
        producer = self._make_attention(config=config)
        consumer = self._make_attention(config=config, layer_idx=1)
        consumer.bind_cache_source(producer)
        self._calibrate(producer)
        consumer.enable_calibration()
        shared: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._forward(producer, 3, shared=shared)
        self._forward(consumer, 3, shared=shared)
        consumer.freeze_qparams()
        state = copy.deepcopy(consumer.state_dict())
        changed_producer = copy.deepcopy(producer)
        scale, zero_point = producer.obs_new_k.compute_qparams()
        changed_producer.obs_new_k.load_qparams(scale * 2, zero_point)
        restored = self._make_attention(config=config, layer_idx=1)
        restored.bind_cache_source(changed_producer)
        restored.enable_calibration()
        with self.assertRaisesRegex(RuntimeError, "producer storage qparams differ"):
            restored.load_state_dict(state)

    def test_state_rejects_changed_read_observer_class_or_dtype(self):
        original = self._make_attention()
        self._calibrate(original)
        state = copy.deepcopy(original.state_dict())
        policies = (
            affine(DType.uint(8)),
            affine(DType.int(16), observer=EMAObserver),
        )
        for policy in policies:
            with self.subTest(policy=policy):
                restored = self._make_attention(overrides={"past_key": policy})
                restored.enable_calibration()
                with self.assertRaisesRegex(
                    RuntimeError, "cache observer policy differs"
                ):
                    restored.load_state_dict(state)

    def test_old_state_and_old_pickled_wrapper_require_recalibration(self):
        original = self._make_attention()
        self._calibrate(original)
        old_state = {
            name: value
            for name, value in original.state_dict().items()
            if name != "_extra_state" and not name.startswith("obs_past_")
        }
        for strict in (True, False):
            with self.subTest(strict=strict):
                restored = self._make_attention()
                restored.enable_calibration()
                with self.assertRaisesRegex(RuntimeError, "reprepare and recalibrate"):
                    restored.load_state_dict(old_state, strict=strict)
        old_wrapper = self._make_attention()
        del old_wrapper.obs_past_key
        del old_wrapper.obs_past_value
        with self.assertRaisesRegex(RuntimeError, "reprepare and recalibrate"):
            old_wrapper.enable_calibration()
        with self.assertRaisesRegex(RuntimeError, "reprepare and recalibrate"):
            old_wrapper.as_export_module("decode")

    def test_invalid_loaded_read_scales_fail_cached_inference_and_export(self):
        original = self._make_attention()
        self._calibrate(original)
        cache = self._forward(original, 3)[2]
        for scale in (float("nan"), 0.0, -0.25):
            with self.subTest(scale=scale):
                attention = copy.deepcopy(original)
                attention.obs_past_key.load_qparams(
                    torch.tensor(scale), torch.tensor(0)
                )
                with self.assertRaisesRegex(
                    RuntimeError, "invalid frozen cache qparams"
                ):
                    self._forward(attention, 1, past=cache)
                with self.assertRaisesRegex(
                    RuntimeError, "invalid frozen cache qparams"
                ):
                    attention.as_export_module("decode")

    def test_freeze_recomputes_read_grid_after_additional_cached_calibration(self):
        attention = self._make_attention()
        attention.enable_calibration()
        cache = self._forward(attention, 3)[2]
        cache = self._forward(attention, 1, past=cache)[2]
        observers = (attention.obs_past_key, attention.obs_past_value)
        early_scales = [observer.compute_qparams()[0].clone() for observer in observers]
        # Feed a different real external cache through decode after an interim
        # qparam inspection. Never inject qparams or directly collect statistics.
        wider_cache = tuple(tensor * 32 for tensor in cache)
        self._forward(attention, 1, past=wider_cache)
        expected_scales = [
            torch.clamp(
                torch.maximum(observer.min_val.abs(), observer.max_val.abs())
                / observer.dtype.qmax,
                min=1e-8,
            )
            for observer in observers
        ]
        attention.freeze_qparams()
        for observer, early, expected in zip(observers, early_scales, expected_scales):
            self.assertGreater(observer._cached_scale.item(), early.item())
            torch.testing.assert_close(observer._cached_scale, expected, atol=0, rtol=0)

    def test_disabled_affine_cache_fake_quant_is_rejected(self):
        original = self._make_attention()
        self._calibrate(original)
        cache = self._forward(original, 3)[2]
        for name in ("new_k", "past_key", "present_key"):
            with self.subTest(name=name):
                attention = copy.deepcopy(original)
                attention.get_observer(name).disable_fake_quant()
                with self.assertRaisesRegex(ValueError, "fake.quant|disabled|enabled"):
                    self._forward(attention, 1, past=cache)
                with self.assertRaisesRegex(ValueError, "fake.quant|disabled|enabled"):
                    attention.as_export_module("decode")


if __name__ == "__main__":
    unittest.main()
