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

"""Independent storage and compute grids for tiny Gemma4 attention wrappers."""

import copy
import io
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
class TestGemma4KVCacheObservers(unittest.TestCase):
    """Keep persistent producer storage separate from attention computation."""

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
            self._forward(attention, 3)
        attention.freeze_qparams()

    @staticmethod
    def _different_grids(attention):
        # Deliberate frozen grids isolate storage/compute conversion numerics;
        # the calibration tests below establish real grids without injections.
        for observer in (attention.obs_new_k, attention.obs_new_v):
            observer.load_qparams(torch.tensor(0.125), torch.tensor(0))
        for observer in (attention.obs_present_key, attention.obs_present_value):
            observer.load_qparams(torch.tensor(0.3), torch.tensor(0))

    def test_only_storage_and_compute_observers_with_independent_policies(self):
        attention = self._make_attention(
            overrides={
                "new_k": affine(DType.int(16)),
                "present_key": affine(DType.uint(8)),
            }
        )
        self.assertIsNot(attention.obs_new_k, attention.obs_present_key)
        self.assertEqual(attention.obs_new_k.dtype, DType.int(16))
        self.assertEqual(attention.obs_present_key.dtype, DType.uint(8))
        for name in ("new_k", "new_v", "present_key", "present_value"):
            self.assertIs(
                attention.get_observer(name), getattr(attention, "obs_" + name)
            )
        for name in ("obs_past_key", "obs_past_value"):
            self.assertFalse(hasattr(attention, name))
            self.assertNotIn(name, dict(attention.named_modules()))
        self.assertFalse(any("obs_past_" in key for key in attention.state_dict()))

    def test_obsolete_past_overrides_require_explicit_migration(self):
        for name in ("past_key", "past_value"):
            for policy in (affine(DType.uint(8)), no_quant(), {"momentum": 0.4}):
                with self.subTest(name=name, policy=policy):
                    with self.assertRaisesRegex(
                        ValueError, "past.*(removed|obsolete)|no longer"
                    ):
                        self._make_attention(overrides={name: policy})

    def test_storage_and_compute_keep_factory_override_semantics(self):
        attention = self._make_attention(
            overrides={
                "new_k": no_quant(),
                "present_value": {"observer": EMAObserver, "momentum": 0.4},
            }
        )
        self.assertIsInstance(attention.obs_new_k, IdentityObserver)
        self.assertIsInstance(attention.obs_present_value, EMAObserver)
        self.assertEqual(attention.obs_present_value.momentum, 0.4)
        self.assertEqual(attention.obs_present_value.dtype, DType.int(16))

    def test_prefill_only_calibration_establishes_storage_and_compute(self):
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
                    self._forward(attention, 3, past=cache)
                    for observer in (
                        attention.obs_new_k,
                        attention.obs_new_v,
                        attention.obs_present_key,
                        attention.obs_present_value,
                    ):
                        self.assertTrue(torch.isfinite(observer.min_val).all())
                        self.assertTrue(torch.isfinite(observer.max_val).all())
                    attention.freeze_qparams()
                    attention.validate_export_cache("prefill")
                    attention.validate_export_cache("decode")
                    past = self._forward(attention, 3)[2]
                    self._forward(attention, 1, past=past)
                    attention.as_export_module("decode")

    def test_current_kv_branches_to_compute_before_storage_rounding(self):
        # For S=0.5 and C=0.3, raw=0.22 maps directly to C=0.3,
        # while the obsolete raw->S->C branch would produce zero.
        for storage_dtype in (DType.uint(8), DType.int(16)):
            for mode in ("delta", "present"):
                for decode in (False, True):
                    with self.subTest(
                        storage_dtype=storage_dtype, mode=mode, decode=decode
                    ):
                        attention = self._make_attention(
                            overrides={
                                "new_k": affine(storage_dtype),
                                "new_v": affine(storage_dtype),
                            }
                        )
                        self._calibrate(attention)
                        storage_zp = 128 if storage_dtype == DType.uint(8) else 0
                        for observer in (attention.obs_new_k, attention.obs_new_v):
                            observer.load_qparams(
                                torch.tensor(0.5), torch.tensor(storage_zp)
                            )
                        for observer in (
                            attention.obs_present_key,
                            attention.obs_present_value,
                        ):
                            observer.load_qparams(torch.tensor(0.3), torch.tensor(0))
                        raw = torch.full((1, 2, 1, 4), 0.22)
                        stored_past = (
                            torch.full_like(raw, 0.5),
                            torch.full_like(raw, -0.5),
                        )
                        past = stored_past if decode else None
                        captured = []
                        original_attention = attention._forward_unrolled

                        def record_attention(**kwargs):
                            captured.append(
                                (kwargs["key_states"], kwargs["value_states"])
                            )
                            return original_attention(**kwargs)

                        with (
                            mock.patch.object(
                                attention,
                                "_project_current_key_value",
                                return_value=(raw, -raw),
                            ),
                            mock.patch.object(
                                attention,
                                "_forward_unrolled",
                                side_effect=record_attention,
                            ),
                        ):
                            output = self._forward(attention, 1, past=past, mode=mode)
                        for index, sign in enumerate((1, -1)):
                            torch.testing.assert_close(
                                captured[0][index][..., -1:, :],
                                torch.full_like(raw, sign * 0.3),
                                atol=0,
                                rtol=0,
                            )
                            torch.testing.assert_close(
                                output[2][index][..., -1:, :],
                                torch.zeros_like(raw),
                                atol=0,
                                rtol=0,
                            )
                            if decode:
                                torch.testing.assert_close(
                                    captured[0][index][..., :1, :],
                                    torch.full_like(raw, sign * 0.6),
                                    atol=0,
                                    rtol=0,
                                )
                                if mode == "present":
                                    torch.testing.assert_close(
                                        output[2][index][..., :1, :],
                                        stored_past[index],
                                        atol=0,
                                        rtol=0,
                                    )

    def test_decode_calibration_collects_past_and_current_compute_statistics(self):
        past = (
            torch.arange(24).reshape(1, 2, 3, 4).float() + 20,
            torch.arange(24).reshape(1, 2, 3, 4).float() - 30,
        )
        for layout in ("batched", "unrolled"):
            with self.subTest(layout=layout):
                attention = self._make_attention(layout=layout)
                attention.enable_calibration()
                attention(*self._inputs(1), past_key_value=past, use_cache=False)
                for observer, values in zip(
                    (attention.obs_present_key, attention.obs_present_value), past
                ):
                    self.assertLessEqual(observer.min_val.item(), values.min().item())
                    self.assertGreaterEqual(
                        observer.max_val.item(), values.max().item()
                    )
                self.assertLess(
                    attention.obs_new_k.max_val.item(), past[0].min().item()
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
                            initial[0], attention.obs_present_key.fake_quant(initial[0])
                        )
                    )
                    past_lengths = []

                    def record_read(key_value):
                        past_lengths.append(key_value[0].shape[-2])
                        torch.testing.assert_close(
                            key_value[0][:, :, :3], initial[0], atol=0, rtol=0
                        )
                        return original_read(key_value)

                    original_read = attention._read_cache_key_value
                    with mock.patch.object(
                        attention, "_read_cache_key_value", side_effect=record_read
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

    def test_shared_consumer_inherits_storage_without_owning_producer_lifecycle(self):
        for layer_type in ("full_attention", "sliding_attention"):
            config = _make_text_config(
                num_hidden_layers=4,
                layer_types=["sliding_attention", "full_attention"] * 2,
                num_kv_shared_layers=2,
                sliding_window=4,
            )
            with self.subTest(layer_type=layer_type):
                producer_index = 0 if layer_type == "sliding_attention" else 1
                producer = self._make_attention(config=config, layer_idx=producer_index)
                consumer = self._make_attention(
                    config=config,
                    layer_idx=producer_index + 2,
                    overrides={
                        "present_key": affine(DType.uint(8)),
                        "present_value": affine(DType.uint(8)),
                    },
                )
                consumer.bind_cache_source(producer)
                producer.enable_calibration()
                consumer.enable_calibration()
                shared: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
                self._forward(producer, 3, shared=shared)
                self._forward(consumer, 3, shared=shared)
                producer.freeze_qparams()
                consumer.freeze_qparams()
                self._different_grids(producer)
                for observer in (consumer.obs_present_key, consumer.obs_present_value):
                    observer.load_qparams(torch.tensor(0.6), torch.tensor(120))
                storage = consumer.cache_storage_observers()
                self.assertIs(storage[0], producer.obs_new_k)
                self.assertIs(storage[1], producer.obs_new_v)
                before = copy.deepcopy(producer.state_dict())
                cache = None
                for length in (3, 1, 1):
                    shared = {}
                    result = self._forward(producer, length, shared=shared, past=cache)
                    cache = result[2]
                    stored = tuple(value.clone() for value in shared[layer_type])
                    with mock.patch.object(
                        consumer,
                        "_read_cache_key_value",
                        wraps=consumer._read_cache_key_value,
                    ) as read:
                        consumer_result = self._forward(consumer, length, shared=shared)
                    self.assertIsNone(consumer_result[2])
                    for actual, expected in zip(read.call_args.args[0], stored):
                        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    for actual, expected in zip(shared[layer_type], stored):
                        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                consumer.enable_calibration()
                self.assertIs(producer._mode, Mode.QUANT)
                for name, value in producer.state_dict().items():
                    if isinstance(value, torch.Tensor):
                        torch.testing.assert_close(value, before[name], atol=0, rtol=0)
                self.assertFalse(
                    any(module is producer for module in consumer.modules())
                )
                self.assertFalse(
                    any(
                        name.startswith("_cache_source.")
                        for name in consumer.state_dict()
                    )
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
                original_read = attention._read_cache_key_value

                def record_read(key_value):
                    lengths.append(key_value[0].shape[-2])
                    torch.testing.assert_close(
                        key_value[0],
                        attention.obs_new_k.fake_quant(key_value[0]),
                        atol=0,
                        rtol=0,
                    )
                    return original_read(key_value)

                with mock.patch.object(
                    attention, "_read_cache_key_value", side_effect=record_read
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
                            # The external tuple owner keeps the window's three
                            # retained tokens for the following one-token call.
                            cache = tuple(tensor[..., -3:, :] for tensor in cache)
                self.assertEqual(lengths, [3, 3, 3])
                if expected_outputs is None:
                    expected_outputs = outputs
                else:
                    for actual, expected in zip(outputs, expected_outputs):
                        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_hf_cache_delta_and_present_updates_only_storage(self):
        from transformers.cache_utils import DynamicCache, StaticCache

        for storage_dtype in (DType.uint(8), DType.int(16)):
            reference = self._make_attention(
                overrides={
                    "new_k": affine(storage_dtype),
                    "new_v": affine(storage_dtype),
                }
            )
            self._calibrate(reference)
            for observer in (reference.obs_new_k, reference.obs_new_v):
                observer.load_qparams(
                    torch.tensor(0.125),
                    torch.tensor(128 if storage_dtype == DType.uint(8) else 0),
                )
            for observer in (reference.obs_present_key, reference.obs_present_value):
                observer.load_qparams(torch.tensor(0.3), torch.tensor(0))
            for cache_type in (DynamicCache, StaticCache):
                for mode in ("delta", "present"):
                    with self.subTest(
                        storage_dtype=storage_dtype, cache_type=cache_type, mode=mode
                    ):
                        attention = copy.deepcopy(reference)
                        cache: Any = (
                            DynamicCache(config=attention.config)
                            if cache_type is DynamicCache
                            else StaticCache(attention.config, max_cache_len=16)
                        )
                        for length, total in ((3, 3), (1, 4), (1, 5)):
                            output = self._forward(
                                attention, length, past=cache, mode=mode
                            )
                            self.assertEqual(int(cache.get_seq_length()), total)
                            if mode == "present":
                                self.assertIs(output[2], cache)
                            for index, (tensor, observer) in enumerate(
                                zip(
                                    (cache.layers[0].keys, cache.layers[0].values),
                                    (attention.obs_new_k, attention.obs_new_v),
                                )
                            ):
                                occupied = tensor[..., :total, :]
                                torch.testing.assert_close(
                                    occupied,
                                    observer.fake_quant(occupied),
                                    atol=0,
                                    rtol=0,
                                )
                                if mode == "delta":
                                    self.assertEqual(output[2][index].shape[-2], length)
                                    torch.testing.assert_close(
                                        output[2][index],
                                        occupied[..., -length:, :],
                                        atol=0,
                                        rtol=0,
                                    )

    def test_use_cache_false_never_mutates_existing_cache(self):
        from transformers.cache_utils import DynamicCache, StaticCache

        for cache_kind in ("tuple", "dynamic", "static"):
            for quantized in (False, True):
                with self.subTest(cache_kind=cache_kind, quantized=quantized):
                    attention = self._make_attention()
                    if quantized:
                        self._calibrate(attention)
                        self._different_grids(attention)
                    cache: Any = None
                    if cache_kind == "dynamic":
                        cache = DynamicCache(config=attention.config)
                    elif cache_kind == "static":
                        cache = StaticCache(attention.config, max_cache_len=16)
                    cache = self._forward(attention, 3, past=cache)[2]
                    values = (
                        cache
                        if isinstance(cache, tuple)
                        else (cache.layers[0].keys, cache.layers[0].values)
                    )
                    before = tuple(value.clone() for value in values)
                    original_length = (
                        3 if isinstance(cache, tuple) else int(cache.get_seq_length())
                    )
                    result = attention(
                        *self._inputs(1), past_key_value=cache, use_cache=False
                    )
                    self.assertEqual(len(result), 2)
                    for actual, expected in zip(values, before):
                        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    if not isinstance(cache, tuple):
                        self.assertEqual(int(cache.get_seq_length()), original_length)

    def test_same_storage_compute_grid_preserves_values(self):
        attention = self._make_attention()
        self._calibrate(attention)
        for name in ("new_k", "new_v", "present_key", "present_value"):
            attention.get_observer(name).load_qparams(
                torch.tensor(0.125), torch.tensor(0)
            )
        cache = self._forward(attention, 3)[2]
        captured = []
        original_attention = attention._forward_unrolled

        def record_attention(**kwargs):
            captured.append((kwargs["key_states"], kwargs["value_states"]))
            return original_attention(**kwargs)

        with mock.patch.object(
            attention, "_forward_unrolled", side_effect=record_attention
        ):
            result = self._forward(attention, 1, past=cache)
        for actual, expected in zip(captured[0], result[2]):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_consumer_never_calibrates_or_freezes_a_local_storage_grid(self):
        config = _make_text_config(
            num_hidden_layers=2,
            layer_types=["full_attention"] * 2,
            num_kv_shared_layers=1,
        )
        producer = self._make_attention(config=config)
        consumer = self._make_attention(config=config, layer_idx=1)
        consumer.bind_cache_source(producer)
        producer.enable_calibration()
        consumer.enable_calibration()
        shared: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._forward(producer, 3, shared=shared)
        with (
            mock.patch.object(
                producer.obs_new_k,
                "collect",
                side_effect=AssertionError("consumer recalibrates producer"),
            ),
            mock.patch.object(
                producer.obs_new_v,
                "collect",
                side_effect=AssertionError("consumer recalibrates producer"),
            ),
        ):
            self._forward(consumer, 3, shared=shared)
            # A standalone consumer cannot freeze an unready external producer.
            with self.assertRaisesRegex(RuntimeError, "not frozen"):
                consumer.freeze_qparams()
        producer.freeze_qparams()
        consumer.freeze_qparams()
        consumer.validate_export_cache("prefill")
        consumer.validate_export_cache("decode")
        for observer in (consumer.obs_new_k, consumer.obs_new_v):
            self.assertTrue(torch.isposinf(observer.min_val).all())
            self.assertTrue(torch.isneginf(observer.max_val).all())
            self.assertFalse(observer.has_qparams)
        self.assertNotIn("qparams", consumer.get_extra_state())
        self.assertNotIn(
            consumer.obs_new_k, consumer.local_observer_groups()["required"]
        )

    def test_common_freeze_resolves_reverse_order_shared_source_plan(self):
        from tico.quantization.wrapq.lifecycle import freeze_qparams

        config = _make_text_config(
            num_hidden_layers=2,
            layer_types=["full_attention"] * 2,
            num_kv_shared_layers=1,
        )
        producer = self._make_attention(config=config)
        consumer = self._make_attention(config=config, layer_idx=1)
        consumer.bind_cache_source(producer)
        root = torch.nn.Module()
        root.add_module("consumer", consumer)
        root.add_module("producer", producer)
        producer.enable_calibration()
        consumer.enable_calibration()
        shared: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._forward(producer, 3, shared=shared)
        self._forward(consumer, 3, shared=shared)
        freeze_qparams(root)
        self.assertIs(consumer.cache_storage_observers()[0], producer.obs_new_k)
        self.assertIs(consumer._mode, Mode.QUANT)
        self.assertIs(producer._mode, Mode.QUANT)
        self.assertIsNone(consumer.obs_new_k.frozen_state)
        consumer.validate_export_cache("decode")

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

    def test_unsupported_storage_and_compute_policies_are_rejected(self):
        policies = (
            mx(),
            affine(DType.int(16), qscheme=QScheme.PER_CHANNEL_SYMM, channel_axis=1),
            affine(DType.uint(4)),
        )
        for name in ("new_k", "present_key"):
            for policy in policies:
                with self.subTest(name=name, policy=policy):
                    attention = self._make_attention(overrides={name: policy})
                    attention.enable_calibration()
                    self._forward(attention, 3)
                    with self.assertRaisesRegex(
                        ValueError, "per-tensor affine|UINT8 and INT16"
                    ):
                        attention.freeze_qparams()
                        attention.validate_export_cache("decode")

    def test_identity_storage_and_compute_allow_prefill_then_decode(self):
        attention = self._make_attention(
            overrides={
                name: no_quant()
                for name in ("new_k", "new_v", "present_key", "present_value")
            }
        )
        self._calibrate(attention)
        cache = self._forward(attention, 3)[2]
        self._forward(attention, 1, past=cache)
        attention.validate_export_cache("decode")

    def test_affine_storage_cannot_feed_identity_compute(self):
        attention = self._make_attention(overrides={"present_key": no_quant()})
        attention.enable_calibration()
        self._forward(attention, 3)
        with self.assertRaisesRegex(ValueError, "identity|Identity|float"):
            attention.freeze_qparams()
        self.assertIs(attention._mode, Mode.CALIB)

    def test_state_dict_round_trip_preserves_storage_and_compute_stats_and_qparams(
        self,
    ):
        attention = self._make_attention()
        self._calibrate(attention)
        restored = self._make_attention()
        restored.enable_calibration()
        restored.load_state_dict(copy.deepcopy(attention.state_dict()))
        restored.freeze_qparams()
        for name in ("present_key", "present_value", "new_k", "new_v"):
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

    def test_shared_source_binding_state_supports_bfloat16_buffers(self):
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
        # Exercise source fingerprints and restored per-tensor grids without
        # creating unsupported BF16 per-channel projection scales on CPU.
        for observer in (
            producer.obs_new_k,
            producer.obs_new_v,
            consumer.obs_present_key,
            consumer.obs_present_value,
        ):
            observer.to(dtype=torch.bfloat16)
        state = copy.deepcopy(consumer.state_dict())
        restored = self._make_attention(config=config, layer_idx=1)
        restored.bind_cache_source(producer)
        restored.load_state_dict(state)
        self.assertIs(restored.cache_storage_observers()[0], producer.obs_new_k)
        torch.testing.assert_close(
            restored.obs_present_key.require_frozen().scale,
            consumer.obs_present_key.require_frozen().scale,
            atol=0,
            rtol=0,
        )

    def test_state_rejects_changed_cache_observer_class_or_dtype(self):
        original = self._make_attention()
        self._calibrate(original)
        state = copy.deepcopy(original.state_dict())
        policies = (
            affine(DType.uint(8)),
            affine(DType.int(16), observer=EMAObserver),
        )
        for policy in policies:
            with self.subTest(policy=policy):
                restored = self._make_attention(overrides={"present_key": policy})
                restored.enable_calibration()
                with self.assertRaisesRegex(RuntimeError, "incompatible frozen policy"):
                    restored.load_state_dict(state)

    def test_three_domain_and_missing_schema_state_require_recalibration(self):
        original = self._make_attention()
        self._calibrate(original)
        state = copy.deepcopy(original.state_dict())
        old_state = copy.deepcopy(state)
        old_state["_extra_state"] = {"version": 2, "frozen": True, "qparams": {}}
        old_state["obs_past_key.min_val"] = torch.tensor(-1.0)
        missing_schema = {
            name: value for name, value in state.items() if name != "_extra_state"
        }
        leftover_past = copy.deepcopy(state)
        leftover_past["obs_past_value.min_val"] = torch.tensor(-1.0)
        for legacy in (old_state, missing_schema, leftover_past):
            for strict in (True, False):
                with self.subTest(strict=strict, keys=tuple(legacy)):
                    restored = self._make_attention()
                    with self.assertRaisesRegex(
                        RuntimeError, "reprepare and recalibrate"
                    ):
                        restored.load_state_dict(legacy, strict=strict)

    def test_state_rejects_changed_compute_observer_options(self):
        original = self._make_attention(
            overrides={"present_key": {"observer": EMAObserver, "momentum": 0.2}}
        )
        self._calibrate(original)
        restored = self._make_attention(
            overrides={"present_key": {"observer": EMAObserver, "momentum": 0.4}}
        )
        with self.assertRaisesRegex(RuntimeError, "incompatible frozen policy"):
            restored.load_state_dict(copy.deepcopy(original.state_dict()))

    def test_legacy_serialized_module_is_rejected(self):
        original = self._make_attention()
        self._calibrate(original)
        original.add_module("obs_past_key", copy.deepcopy(original.obs_new_k))
        original.add_module("obs_past_value", copy.deepcopy(original.obs_new_v))
        checkpoint = io.BytesIO()
        torch.save(original, checkpoint)
        checkpoint.seek(0)
        with self.assertRaisesRegex(RuntimeError, "reprepare and recalibrate"):
            torch.load(checkpoint, weights_only=False)
        with self.assertRaisesRegex(RuntimeError, "reprepare and recalibrate"):
            original.as_export_module("decode")

    def test_frozen_checkpoint_requires_both_storage_and_compute_qparams(self):
        original = self._make_attention()
        self._calibrate(original)
        for name in ("new_k", "new_v", "present_key", "present_value"):
            with self.subTest(name=name):
                state = copy.deepcopy(original.state_dict())
                del state[f"obs_{name}._extra_state"]
                for strict in (True, False):
                    with self.assertRaisesRegex(
                        RuntimeError, "frozen|legacy|recalibrat"
                    ):
                        self._make_attention().load_state_dict(state, strict=strict)

    def test_missing_prefill_statistics_are_never_fabricated(self):
        attention = self._make_attention()
        attention.enable_calibration()
        with self.assertRaisesRegex(RuntimeError, "missing.*calibration statistics"):
            attention.freeze_qparams()
        self.assertFalse(attention.obs_new_k.has_qparams)
        self.assertFalse(attention.obs_present_key.has_qparams)

    def test_module_round_trip_and_deepcopy_preserve_frozen_grids(self):
        original = self._make_attention()
        self._calibrate(original)
        self._different_grids(original)
        checkpoint = io.BytesIO()
        torch.save(original, checkpoint)
        checkpoint.seek(0)
        loaded = torch.load(checkpoint, weights_only=False)
        for restored in (loaded, copy.deepcopy(original)):
            self.assertEqual(restored.get_extra_state()["gemma_cache"]["version"], 3)
            self.assertNotIn("qparams", restored.get_extra_state())
            for name in ("new_k", "new_v", "present_key", "present_value"):
                before, after = original.get_observer(name), restored.get_observer(name)
                self.assertIsNot(before, after)
                for actual, expected in zip(
                    after.compute_qparams(), before.compute_qparams()
                ):
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            cache = self._forward(original, 3)[2]
            torch.testing.assert_close(
                self._forward(restored, 1, past=cache)[0],
                self._forward(original, 1, past=cache)[0],
                atol=0,
                rtol=0,
            )
            restored.as_export_module("decode")

    def test_invalid_imported_compute_scales_reject_before_mutation(self):
        original = self._make_attention()
        self._calibrate(original)
        cache = self._forward(original, 3)[2]
        expected = self._forward(original, 1, past=cache)[0]
        for scale in (float("nan"), 0.0, -0.25):
            with self.subTest(scale=scale):
                attention = copy.deepcopy(original)
                before = attention.obs_present_key.require_frozen()
                with self.assertRaisesRegex(
                    RuntimeError, "invalid scale|positive scale"
                ):
                    attention.obs_present_key.load_qparams(
                        torch.tensor(scale), torch.tensor(0)
                    )
                after = attention.obs_present_key.require_frozen()
                torch.testing.assert_close(after.scale, before.scale, atol=0, rtol=0)
                torch.testing.assert_close(
                    self._forward(attention, 1, past=cache)[0], expected, atol=0, rtol=0
                )
                attention.as_export_module("decode")

    def test_freeze_recomputes_compute_grid_after_optional_decode_calibration(self):
        attention = self._make_attention()
        attention.enable_calibration()
        cache = self._forward(attention, 3)[2]
        cache = self._forward(attention, 1, past=cache)[2]
        observers = (attention.obs_present_key, attention.obs_present_value)
        early_scales = [
            observer.derive_candidate().scale.clone() for observer in observers
        ]
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
        for name in ("new_k", "present_key"):
            with self.subTest(name=name):
                attention = copy.deepcopy(original)
                attention.get_observer(name).disable_fake_quant()
                with self.assertRaisesRegex(ValueError, "fake.quant|disabled|enabled"):
                    self._forward(attention, 1, past=cache)
                with self.assertRaisesRegex(ValueError, "fake.quant|disabled|enabled"):
                    attention.as_export_module("decode")


if __name__ == "__main__":
    unittest.main()
