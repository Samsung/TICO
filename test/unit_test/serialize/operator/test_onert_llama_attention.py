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

"""Cache decomposition contracts for the optional ONERT attention adapter."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import cast, TYPE_CHECKING
from unittest.mock import patch

import torch

from tico.quantization.wrapq.utils.version import has_transformers_for

if TYPE_CHECKING:
    from transformers.cache_utils import DynamicCache


@unittest.skipUnless(
    has_transformers_for("llama"), "Llama support requires Transformers"
)
class TestOnertLlamaAttentionCache(unittest.TestCase):
    """Validate cache selection before crossing the custom operator boundary."""

    def setUp(self):
        """Construct a tiny attention module and distinct per-layer cache tensors."""
        from transformers.models.llama.configuration_llama import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaAttention

        self.attention = LlamaAttention(
            LlamaConfig(
                hidden_size=8,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=4,
            ),
            layer_idx=1,
        )
        self.keys = [torch.zeros(1, 1, 2, 4), torch.ones(1, 1, 2, 4)]
        self.values = [torch.full((1, 1, 2, 4), 2.0), torch.full((1, 1, 2, 4), 3.0)]

    def _call_adapter(self, cache: object):
        """Call the adapter using its explicit legacy export signature."""
        from tico.serialize.operators.adapters.onert.llama_attention import (
            llama_attention_forward_adapter,
        )

        return llama_attention_forward_adapter(
            self.attention,
            hidden_states=torch.zeros(1, 1, 8),
            position_embeddings=[torch.ones(1, 1, 4), torch.zeros(1, 1, 4)],
            attention_mask=torch.zeros(1, 1, 1, 2),
            past_key_value=cast("DynamicCache", cache),
            cache_position=torch.tensor([1]),
        )

    def test_legacy_and_layer_caches_pass_selected_tensors_unchanged(self):
        """Both cache layouts must forward the exact layer's key and value tensors."""
        caches = {
            "legacy": SimpleNamespace(key_cache=self.keys, value_cache=self.values),
            "layers": SimpleNamespace(
                layers=[
                    SimpleNamespace(keys=key, values=value)
                    for key, value in zip(self.keys, self.values)
                ]
            ),
        }
        expected_output = torch.ones(1, 1, 8)
        for layout, cache in caches.items():
            with self.subTest(layout=layout):
                with patch.object(
                    torch.ops.circle_custom, "attention", return_value=expected_output
                ) as custom_attention:
                    output, weights = self._call_adapter(cache)

                self.assertIs(output, expected_output)
                self.assertIsNone(weights)
                custom_attention.assert_called_once()
                arguments = custom_attention.call_args.args
                self.assertIs(arguments[8], self.keys[1])
                self.assertIs(arguments[9], self.values[1])
                self.assertIs(arguments[1], self.attention.q_proj.weight)
                self.assertIs(arguments[4], self.attention.o_proj.weight)

    def test_missing_layer_rejects_before_custom_operator(self):
        """An absent selected layer must fail without calling the custom operator."""
        with patch.object(torch.ops.circle_custom, "attention") as custom_attention:
            with self.assertRaises(IndexError):
                self._call_adapter(SimpleNamespace(layers=[]))
        custom_attention.assert_not_called()

    def test_uninitialized_cache_rejects_before_custom_operator(self):
        """An unset key or value cannot cross the tensor-only operator boundary."""
        for key, value in ((None, self.values[1]), (self.keys[1], None)):
            with self.subTest(key_missing=key is None):
                cache = SimpleNamespace(
                    layers=[
                        SimpleNamespace(keys=self.keys[0], values=self.values[0]),
                        SimpleNamespace(keys=key, values=value),
                    ]
                )
                with patch.object(
                    torch.ops.circle_custom, "attention"
                ) as custom_attention:
                    with self.assertRaises(AssertionError):
                        self._call_adapter(cache)
                custom_attention.assert_not_called()
