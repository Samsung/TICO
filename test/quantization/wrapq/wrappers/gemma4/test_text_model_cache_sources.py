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

"""Shared-KV producer ownership and lifecycle contracts in a prepared text model."""

import copy
import io
import unittest

import torch
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.config.specs import affine
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.wrappers.gemma4.quant_text_model import (
    QuantGemma4TextModel,
)

from test.quantization.wrapq.wrappers.gemma4.test_quant_text_model import (
    _has_gemma4,
    _make_text_config,
    _make_text_model,
    _SKIP_MSG,
)


@unittest.skipUnless(_has_gemma4(), _SKIP_MSG)
class TestGemma4TextModelCacheSources(unittest.TestCase):
    """Resolve producer storage by layer type without adding module ownership."""

    def setUp(self):
        torch.manual_seed(2026)

    @staticmethod
    def _model():
        layer_types = ["sliding_attention", "full_attention"] * 3
        config = _make_text_config(layer_types=layer_types, num_kv_shared_layers=2)
        # These explicit scalar policies isolate shared cache-source restoration
        # from differences in projection weight granularity.
        overrides = {"embed_tokens.weight": affine(DType.int(16))}
        for index in range(len(layer_types)):
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                overrides[f"layers.{index}.self_attn.{name}.weight"] = affine(
                    DType.int(16)
                )
            for name in ("gate_proj", "up_proj", "down_proj"):
                overrides[f"layers.{index}.mlp.{name}.weight"] = affine(DType.int(16))
        overrides["layers.4.self_attn.present_key"] = affine(DType.uint(8))
        qcfg = PTQConfig(
            model_args={"text": {"max_seq": 32}},
            overrides=overrides,
        )
        return QuantGemma4TextModel(_make_text_model(config), qcfg=qcfg).eval()

    @staticmethod
    def _attention(model, index):
        return model.layers[index].wrapped.self_attn.wrapped

    def test_binds_last_owning_layer_of_each_type_and_keeps_compute_override(self):
        model = self._model()
        for producer_index, consumer_index in ((2, 4), (3, 5)):
            producer = self._attention(model, producer_index)
            consumer = self._attention(model, consumer_index)
            storage_key, storage_value = consumer.cache_storage_observers()
            self.assertIs(storage_key, producer.obs_new_k)
            self.assertIs(storage_value, producer.obs_new_v)
            self.assertIsNot(consumer.obs_present_key, storage_key)
            self.assertIsNot(consumer.obs_present_value, storage_value)
            self.assertNotIn(producer, list(consumer.modules()))
            self.assertFalse(
                any("cache_source" in key for key in consumer.state_dict())
            )
        self.assertEqual(self._attention(model, 4).obs_present_key.dtype, DType.uint(8))

    def test_state_dict_round_trip_keeps_compute_qparams_and_local_producer_binding(
        self,
    ):
        model = self._model()
        model.enable_calibration()
        model(input_ids=torch.tensor([[1, 3, 5]]), use_cache=True)
        model.freeze_qparams()
        checkpoint = io.BytesIO()
        torch.save(model, checkpoint)
        checkpoint.seek(0)
        loaded = torch.load(checkpoint, weights_only=False)
        duplicated = copy.deepcopy(model)
        for copy_model in (loaded, duplicated):
            for producer_index, consumer_index in ((2, 4), (3, 5)):
                storage = self._attention(
                    copy_model, consumer_index
                ).cache_storage_observers()
                self.assertIs(
                    storage[0], self._attention(copy_model, producer_index).obs_new_k
                )
                self.assertIsNot(
                    storage[0], self._attention(model, producer_index).obs_new_k
                )
                self.assertTrue(
                    torch.equal(
                        self._attention(
                            copy_model, consumer_index
                        ).obs_present_key._cached_scale,
                        self._attention(
                            model, consumer_index
                        ).obs_present_key._cached_scale,
                    )
                )
        saved = copy.deepcopy(model.state_dict())
        restored = self._model()
        restored.load_state_dict(saved)

        for producer_index, consumer_index in ((2, 4), (3, 5)):
            producer = self._attention(restored, producer_index)
            consumer = self._attention(restored, consumer_index)
            original = self._attention(model, consumer_index)
            self.assertIs(consumer.cache_storage_observers()[0], producer.obs_new_k)
            self.assertIsNot(
                consumer.cache_storage_observers()[0],
                self._attention(model, producer_index).obs_new_k,
            )
            for name in ("obs_present_key", "obs_present_value"):
                before = getattr(original, name)
                after = getattr(consumer, name)
                self.assertTrue(after.has_qparams)
                self.assertTrue(torch.equal(after.min_val, before.min_val))
                self.assertTrue(torch.equal(after.max_val, before.max_val))
                self.assertTrue(torch.equal(after._cached_scale, before._cached_scale))
                self.assertTrue(torch.equal(after._cached_zp, before._cached_zp))

        restored.to(dtype=torch.float64)
        source = self._attention(restored, 2)
        consumer = self._attention(restored, 4)
        self.assertIs(consumer.cache_storage_observers()[0], source.obs_new_k)
        self.assertEqual(source.obs_new_k.min_val.dtype, torch.float64)
        self.assertEqual(consumer.obs_present_key.min_val.dtype, torch.float64)

    def test_complete_dense_variant_topologies(self):
        """Every E2B/E4B consumer references the final producer of its own type."""
        for variant, sliding_count, layers, shared, kv_heads, producers in (
            ("e2b", 4, 35, 20, 1, (13, 14)),
            ("e4b", 5, 42, 18, 2, (22, 23)),
        ):
            with self.subTest(variant=variant):
                layer_types = (
                    ["sliding_attention"] * sliding_count + ["full_attention"]
                ) * 7
                # Keep the complete production topology with tiny weights and
                # heterogeneous heads so ordinary CPU discovery stays cheap.
                config = _make_text_config(
                    layer_types=layer_types,
                    hidden_size=16,
                    intermediate_size=32,
                    num_attention_heads=8,
                    num_key_value_heads=kv_heads,
                    num_global_key_value_heads=kv_heads,
                    head_dim=4,
                    global_head_dim=8,
                    num_kv_shared_layers=shared,
                    use_double_wide_mlp=variant == "e2b",
                )
                qcfg = PTQConfig(
                    activation=affine(DType.int(16)),
                    weight=affine(DType.int(16)),
                    model_args={"text": {"max_seq": 16}},
                )
                model = QuantGemma4TextModel(_make_text_model(config), qcfg=qcfg).eval()
                first_shared = layers - shared
                self.assertEqual(first_shared, 15 if variant == "e2b" else 24)
                actual_producers = [
                    i
                    for i in range(layers)
                    if self._attention(model, i).store_full_length_kv
                ]
                self.assertEqual(actual_producers, list(producers))
                for index in range(first_shared, layers):
                    attention = self._attention(model, index)
                    source_index = producers[layer_types[index] == "full_attention"]
                    source = self._attention(model, source_index)
                    self.assertIs(attention._cache_source, source)
                    self.assertIs(
                        attention.cache_storage_observers()[0], source.obs_new_k
                    )
                    self.assertIs(
                        attention.cache_storage_observers()[1], source.obs_new_v
                    )
                    self.assertNotIn(source, list(attention.modules()))
                    self.assertNotIn("_cache_source", attention._modules)
                self.assertFalse(
                    any("cache_source" in key for key in model.state_dict())
                )

                model.enable_calibration()
                with torch.no_grad():
                    output = model(
                        input_ids=torch.tensor([[1, 3, 5]]),
                        use_cache=False,
                        return_shared_kv_states=True,
                    )
                for layer_type, head_dim in (
                    ("sliding_attention", 4),
                    ("full_attention", 8),
                ):
                    key, value = output.shared_kv_states[layer_type]
                    self.assertEqual(key.shape, (1, kv_heads, 3, head_dim))
                    self.assertEqual(value.shape, key.shape)
                model.freeze_qparams()
                for index in range(first_shared, layers):
                    consumer = self._attention(model, index)
                    source = consumer._cache_source
                    storage = consumer.cache_storage_observers()
                    self.assertIs(storage[0], source.obs_new_k)
                    self.assertTrue(storage[0].has_qparams)
                    scale = source.obs_new_k._cached_scale.clone()
                    zero_point = source.obs_new_k._cached_zp.clone()
                    # The source is a reference: resetting consumer statistics
                    # never traverses or resets the producer's frozen grid.
                    consumer.enable_calibration()
                    torch.testing.assert_close(
                        source.obs_new_k._cached_scale, scale, atol=0, rtol=0
                    )
                    torch.testing.assert_close(
                        source.obs_new_k._cached_zp, zero_point, atol=0, rtol=0
                    )


if __name__ == "__main__":
    unittest.main()
