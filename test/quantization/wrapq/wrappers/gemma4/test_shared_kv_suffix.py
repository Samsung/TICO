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

import types
import unittest
from typing import Any

import torch
import torch.nn as nn

from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.observers.minmax import MinMaxObserver
from tico.quantization.wrapq.observers.mx import MXObserver
from tico.quantization.wrapq.qscheme import QScheme
from tico.quantization.wrapq.wrappers.gemma4.utils import (
    gemma4_single_token_quantization_supported,
    resolve_gemma4_shared_kv_suffix,
    validate_gemma4_single_token_quantization,
)


def _layer(layer_type, *, shared=False, store=False, source=None):
    attention: Any = nn.Module()
    attention.layer_type = layer_type
    attention.is_kv_shared_layer = shared
    attention.store_full_length_kv = store
    attention.head_dim = 8
    attention.config = types.SimpleNamespace(num_key_value_heads=2)
    if shared:
        object.__setattr__(attention, "_cache_source", source)
    layer: Any = nn.Module()
    layer.self_attn = attention
    return layer


class Gemma4SharedKVSuffixTest(unittest.TestCase):
    """Derive topology from instantiated ownership and bound storage sources."""

    def test_distinct_producers_and_noncanonical_boundary(self):
        """Each suffix consumer reads its own attention type's storage owner."""
        sliding = _layer("sliding_attention", store=True)
        full = _layer("full_attention", store=True)
        layers = [
            _layer("full_attention"),
            sliding,
            full,
            _layer("full_attention", shared=True, source=full.self_attn),
            _layer("sliding_attention", shared=True, source=sliding.self_attn),
            _layer("full_attention", shared=True, source=full.self_attn),
        ]
        self.assertEqual(
            resolve_gemma4_shared_kv_suffix(layers), (3, {3: 2, 4: 1, 5: 2})
        )

    def test_no_suffix_and_empty_model(self):
        """A KV owner at the end preserves full prefill; zero layers are valid."""
        layers = [_layer("sliding_attention"), _layer("full_attention")]
        self.assertEqual(resolve_gemma4_shared_kv_suffix(layers), (2, {}))
        self.assertEqual(resolve_gemma4_shared_kv_suffix([]), (0, {}))

    def test_shared_layer_before_last_owner_stays_in_prefix(self):
        """Only the contiguous terminal consumer run can skip other queries."""
        source = _layer("full_attention", store=True)
        layers = [
            source,
            _layer("full_attention", shared=True, source=source.self_attn),
            _layer("sliding_attention", store=True),
            _layer("full_attention", shared=True, source=source.self_attn),
        ]
        self.assertEqual(resolve_gemma4_shared_kv_suffix(layers), (3, {1: 0, 3: 0}))

    def test_missing_or_wrong_producer_rejects(self):
        """An unmarked or wrong-type owner cannot substitute for storage KV."""
        for owner in (
            _layer("full_attention"),
            _layer("sliding_attention", store=True),
        ):
            with self.subTest(owner=owner.self_attn.layer_type):
                with self.assertRaisesRegex(
                    ValueError, "no preceding storage producer"
                ):
                    resolve_gemma4_shared_kv_suffix(
                        [owner, _layer("full_attention", shared=True)]
                    )
        source = _layer("full_attention", store=True)
        with self.assertRaisesRegex(ValueError, "cache binding does not match"):
            resolve_gemma4_shared_kv_suffix(
                [source, _layer("full_attention", shared=True)]
            )

    def test_geometry_and_consumer_ownership_reject(self):
        """Flag shape mismatches and a purported consumer that owns K/V."""
        source = _layer("full_attention", store=True)
        consumer = _layer("full_attention", shared=True, source=source.self_attn)
        consumer.self_attn.head_dim = 16
        with self.assertRaisesRegex(ValueError, "incompatible head dimensions"):
            resolve_gemma4_shared_kv_suffix([source, consumer])
        consumer.self_attn.head_dim = 8
        consumer.self_attn.config.num_key_value_heads = 1
        with self.assertRaisesRegex(ValueError, "incompatible KV head counts"):
            resolve_gemma4_shared_kv_suffix([source, consumer])
        consumer.self_attn.config.num_key_value_heads = 2
        consumer.self_attn.k_proj = nn.Identity()
        with self.assertRaisesRegex(ValueError, "must not produce K/V"):
            resolve_gemma4_shared_kv_suffix([source, consumer])

    def test_wrapped_layers_resolve_without_registering_source(self):
        """Wrapper shells do not change the actual producer identity."""
        source = _layer("full_attention", store=True)
        consumer = _layer("full_attention", shared=True, source=source.self_attn)
        wrapped = []
        for layer in (source, consumer):
            wrapper = nn.Module()
            wrapper.wrapped = layer
            attention_wrapper = nn.Module()
            attention_wrapper.wrapped = layer.self_attn
            layer.self_attn = attention_wrapper
            wrapped.append(wrapper)
        # Source references identify the underlying attention, not its wrapper.
        self.assertEqual(resolve_gemma4_shared_kv_suffix(wrapped), (1, {1: 0}))


class Gemma4SingleTokenQuantizationTest(unittest.TestCase):
    """Only token-independent activation policies may use suffix slicing."""

    def test_frozen_affine_grids_unchanged_and_weight_policy_independent(self):
        layer: Any = nn.Module()
        layer._mode = Mode.QUANT
        layer.obs_act_in = MinMaxObserver(name="act_in")
        layer.obs_act_in.collect(torch.tensor([-1.0, 3.0]))
        layer.obs_act_in.compute_qparams()
        frozen = layer.obs_act_in.require_frozen()
        layer.obs_weight = MXObserver(name="weight", axis=0)
        self.assertTrue(gemma4_single_token_quantization_supported([layer]))
        validate_gemma4_single_token_quantization([layer])
        after = layer.obs_act_in.require_frozen()
        self.assertEqual(after.policy, frozen.policy)
        torch.testing.assert_close(after.scale, frozen.scale, atol=0, rtol=0)
        torch.testing.assert_close(after.zero_point, frozen.zero_point, atol=0, rtol=0)

    def test_activation_axes_and_dynamic_mx_reject(self):
        layer: Any = nn.Module()
        layer._mode = Mode.QUANT
        for observer in (
            MinMaxObserver(
                name="act_in", channel_axis=1, qscheme=QScheme.PER_CHANNEL_SYMM
            ),
            MXObserver(name="act_in", axis=1),
        ):
            with self.subTest(policy=type(observer).__name__):
                layer.obs_act_in = observer
                self.assertFalse(gemma4_single_token_quantization_supported([layer]))
                with self.assertRaisesRegex(
                    ValueError, "per-tensor affine or identity"
                ):
                    validate_gemma4_single_token_quantization([layer])

    def test_ple_boundary_policy_is_validated(self):
        layer: Any = nn.Module()
        layer._mode = Mode.QUANT
        with self.assertRaisesRegex(ValueError, "per_layer_inputs"):
            validate_gemma4_single_token_quantization(
                [layer], per_layer_input_observer=MXObserver(name="ple", axis=1)
            )

    def test_float_execution_and_calibration_remain_distinct(self):
        layer: Any = nn.Module()
        layer._mode = Mode.NO_QUANT
        layer.obs_act_in = MXObserver(name="act_in", axis=1)
        validate_gemma4_single_token_quantization([layer])
        layer._mode = Mode.CALIB
        with self.assertRaisesRegex(ValueError, "during calibration"):
            validate_gemma4_single_token_quantization([layer])

    def test_training_dropout_rejects_even_without_quantization(self):
        layer = _layer("full_attention")
        layer.self_attn.attention_dropout = 0.1
        with self.assertRaisesRegex(ValueError, "dropout to be disabled"):
            validate_gemma4_single_token_quantization([layer])
        layer.eval()
        validate_gemma4_single_token_quantization([layer])


if __name__ == "__main__":
    unittest.main()
