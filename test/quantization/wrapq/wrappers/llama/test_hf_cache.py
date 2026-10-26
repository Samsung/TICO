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

"""Persistent Llama cache storage across empty prefill and two cached decodes."""

import unittest
from typing import Any

import torch

from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.observers.affine_base import AffineObserverBase
from tico.quantization.wrapq.utils.version import has_transformers_for
from tico.quantization.wrapq.wrappers.llama.quant_attention import QuantLlamaAttention

from test.quantization.quant_spec_helpers import make_affine_ptq_config


@unittest.skipUnless(has_transformers_for("llama"), "requires Transformers Llama")
class TestLlamaHFCache(unittest.TestCase):
    def _attention(self, *, profile="npu_export", dtype=DType.uint(8)):
        from transformers import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaAttention

        torch.manual_seed(43)
        config = LlamaConfig(
            hidden_size=8,
            intermediate_size=16,
            vocab_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            attention_dropout=0.0,
            max_position_embeddings=8,
        )
        config._attn_implementation = "eager"
        # Use a nonzero layer to detect reading layer zero's cache length.
        attention = QuantLlamaAttention(
            LlamaAttention(config, layer_idx=1),
            layer_idx=1,
            qcfg=make_affine_ptq_config(dtype=dtype, model_args={"profile": profile}),
        ).eval()
        attention.enable_calibration()
        with torch.no_grad():
            attention(*self._inputs(attention, 0, 3), use_cache=True)
        attention.freeze_qparams()
        if dtype == DType.uint(8):
            storage, compute, szp, czp = 0.03125, 0.0859375, 113, 91
        else:
            storage, compute, szp, czp = 0.000244140625, 0.00067138671875, 0, 0
        for observer in (attention.obs_new_k, attention.obs_new_v):
            assert isinstance(observer, AffineObserverBase)
            observer.load_qparams(torch.tensor(storage), torch.tensor(szp))
        for observer in (attention.obs_present_key, attention.obs_present_value):
            assert isinstance(observer, AffineObserverBase)
            observer.load_qparams(torch.tensor(compute), torch.tensor(czp))
        return attention, config

    @staticmethod
    def _inputs(attention, start, length):
        hidden = torch.sin(
            torch.arange(start * 8, (start + length) * 8).float() * 0.17
        ).reshape(1, length, 8)
        phase = (
            torch.arange(start, start + length).float().view(1, length, 1) + 0.4
        ) * torch.tensor([0.1, 0.7, 0.1, 0.7])
        cos, sin = phase.cos(), phase.sin()
        if attention.attn_options.rope == "pre_negated_sin":
            sin[..., :2] = -sin[..., :2]
        return hidden, (cos, sin)

    @staticmethod
    def _empty_cache(kind, config):
        from transformers.cache_utils import DynamicCache, StaticCache

        if kind == "tuple":
            return (torch.empty(1, 1, 0, 4), torch.empty(1, 1, 0, 4))
        if kind == "dynamic":
            return DynamicCache(config=config)
        return StaticCache(config=config, max_cache_len=8)

    @staticmethod
    def _storage(attention, cache):
        if isinstance(cache, tuple):
            return cache
        storage = attention._get_layer_kv_from_cache(cache)
        assert storage is not None
        return storage

    def test_tuple_dynamic_static_storage_survives_two_decodes(self):
        for profile in ("npu_export", "reference_eval"):
            for dtype in (DType.uint(8), DType.int(16)):
                attention, config = self._attention(profile=profile, dtype=dtype)
                for mode in ("delta", "present"):
                    for kind in ("tuple", "dynamic", "static"):
                        with self.subTest(
                            profile=profile, dtype=dtype, mode=mode, kind=kind
                        ):
                            cache: Any = self._empty_cache(kind, config)
                            expected_cache = self._empty_cache("tuple", config)
                            start = 0
                            differs_from_compute = False
                            for length in (3, 1, 1):
                                before = tuple(
                                    tensor.clone() for tensor in expected_cache
                                )
                                args = self._inputs(attention, start, length)
                                position = torch.arange(start, start + length)
                                # Reading the compute branch without an update must
                                # leave the persistent allocation and values intact.
                                if start:
                                    saved = tuple(
                                        tensor.clone()
                                        for tensor in self._storage(attention, cache)
                                    )
                                    with torch.no_grad():
                                        attention(
                                            *args,
                                            past_key_value=cache,
                                            use_cache=False,
                                            cache_position=position,
                                        )
                                    for actual, previous in zip(
                                        self._storage(attention, cache), saved
                                    ):
                                        torch.testing.assert_close(
                                            actual, previous, atol=0, rtol=0
                                        )
                                with torch.no_grad():
                                    expected = attention(
                                        *args,
                                        past_key_value=expected_cache,
                                        use_cache=True,
                                        cache_output_mode="delta",
                                        cache_position=position,
                                    )
                                    actual = attention(
                                        *args,
                                        past_key_value=cache,
                                        use_cache=True,
                                        cache_output_mode=mode,
                                        cache_position=position,
                                    )
                                for result, reference in zip(actual[:2], expected[:2]):
                                    torch.testing.assert_close(
                                        result, reference, atol=0, rtol=0
                                    )
                                delta = expected[2]
                                expected_cache = tuple(
                                    torch.cat((old, new), dim=2)
                                    for old, new in zip(expected_cache, delta)
                                )
                                if kind == "tuple":
                                    cache = (
                                        tuple(
                                            torch.cat((old, new), dim=2)
                                            for old, new in zip(cache, actual[2])
                                        )
                                        if mode == "delta"
                                        else actual[2]
                                    )
                                elif mode == "present":
                                    self.assertIs(actual[2], cache)
                                if mode == "delta":
                                    for result, reference in zip(actual[2], delta):
                                        torch.testing.assert_close(
                                            result, reference, atol=0, rtol=0
                                        )
                                storage = self._storage(attention, cache)
                                for result, reference, old, observer, compute in zip(
                                    storage,
                                    expected_cache,
                                    before,
                                    (attention.obs_new_k, attention.obs_new_v),
                                    (
                                        attention.obs_present_key,
                                        attention.obs_present_value,
                                    ),
                                ):
                                    self.assertEqual(result.shape[-2], start + length)
                                    torch.testing.assert_close(
                                        result, reference, atol=0, rtol=0
                                    )
                                    torch.testing.assert_close(
                                        result[..., :start, :], old, atol=0, rtol=0
                                    )
                                    torch.testing.assert_close(
                                        result,
                                        observer.fake_quant(result),
                                        atol=0,
                                        rtol=0,
                                    )
                                    differs_from_compute |= not torch.equal(
                                        result, compute.fake_quant(result)
                                    )
                                start += length
                            self.assertTrue(differs_from_compute)
                            self.assertEqual(start, 5)

    def test_calibration_cache_normalization_preserves_values(self):
        attention, config = self._attention()
        attention.enable_calibration()
        raw = (
            torch.arange(12).reshape(1, 1, 3, 4).float() * 0.013,
            torch.arange(12).reshape(1, 1, 3, 4).float() * -0.021,
        )
        for kind in ("tuple", "dynamic", "static"):
            with self.subTest(kind=kind):
                cache = raw if kind == "tuple" else self._empty_cache(kind, config)
                if kind != "tuple":
                    cache.update(
                        *raw, 1, cache_kwargs={"cache_position": torch.arange(3)}
                    )
                normalized = attention._normalize_past_key_value(cache)
                self.assertIsNotNone(normalized)
                assert normalized is not None
                for actual, expected in zip(normalized, raw):
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                for actual, expected in zip(self._storage(attention, cache), raw):
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_full_model_decode_adapter_returns_delta_and_preserves_hf_cache(self):
        from tico.quantization.wrapq.utils.static_export import prepared_static_export
        from tico.quantization.wrapq.wrappers.llama.quant_model_for_causal_lm import (
            QuantLlamaForCausalLM,
        )
        from transformers import LlamaForCausalLM

        _, config = self._attention()
        model = QuantLlamaForCausalLM(
            LlamaForCausalLM(config).eval(),
            qcfg=make_affine_ptq_config(model_args={"profile": "npu_export"}),
        ).eval()
        model.enable_calibration()
        with torch.no_grad():
            model(input_ids=torch.tensor([[1, 2, 3]]), use_cache=True)
        model.freeze_qparams()
        adapter = model.as_export_module("decode")
        text_model = model.model.wrapped
        for kind in ("tuple", "dynamic", "static"):
            with self.subTest(kind=kind), torch.no_grad():
                cache: Any = () if kind == "tuple" else self._empty_cache(kind, config)
                prefill = model(
                    input_ids=torch.tensor([[1, 2, 3]]),
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=True,
                )
                cache = prefill.past_key_values
                with prepared_static_export(adapter, 8):
                    for token, past_len in ((4, 3), (5, 4)):
                        full = (
                            cache
                            if kind == "tuple"
                            else text_model._cache_to_legacy(cache)
                        )
                        before = tuple(
                            (
                                key[:, :, :past_len].clone(),
                                value[:, :, :past_len].clone(),
                            )
                            for key, value in full
                        )
                        ids = torch.tensor([[token]])
                        expected = model(
                            input_ids=ids,
                            past_key_values=before,
                            use_cache=True,
                            return_dict=True,
                        )
                        logits, delta = adapter(ids, cache)
                        torch.testing.assert_close(
                            logits, expected.logits, atol=0, rtol=0
                        )
                        self.assertEqual(len(delta), config.num_hidden_layers)
                        for actual, full_layer in zip(delta, expected.past_key_values):
                            for result, reference in zip(actual, full_layer):
                                self.assertEqual(result.shape, (1, 1, 1, 4))
                                torch.testing.assert_close(
                                    result, reference[:, :, -1:], atol=0, rtol=0
                                )
                        if kind == "tuple":
                            cache = tuple(
                                tuple(
                                    torch.cat((old, new), dim=2)
                                    for old, new in zip(prefix, addition)
                                )
                                for prefix, addition in zip(before, delta)
                            )
                        full = (
                            cache
                            if kind == "tuple"
                            else text_model._cache_to_legacy(cache)
                        )
                        for current, prefix, expected_layer in zip(
                            full, before, expected.past_key_values
                        ):
                            for result, old, reference in zip(
                                current, prefix, expected_layer
                            ):
                                torch.testing.assert_close(
                                    result[:, :, :past_len], old, atol=0, rtol=0
                                )
                                torch.testing.assert_close(
                                    result[:, :, : past_len + 1],
                                    reference,
                                    atol=0,
                                    rtol=0,
                                )
