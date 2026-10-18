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

"""Test the Llama benchmark adapter with a small CPU-only decoder layer."""

import unittest

import torch
from transformers import LlamaConfig

from test.performance.utils import Llama32LayerBase


class _TinyLlamaLayer(Llama32LayerBase):
    def get_config(self) -> LlamaConfig:
        return LlamaConfig(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            vocab_size=16,
            use_cache=True,
        )


class LlamaBenchmarkAdapterTest(unittest.TestCase):
    @torch.no_grad()
    def test_forward_preserves_batch_and_returns_stateless_legacy_cache(self):
        torch.manual_seed(0)
        model = _TinyLlamaLayer(seq_len=3).eval()
        for batch_size in (1, 2):
            with self.subTest(batch_size=batch_size):
                inputs = torch.randn(batch_size, 3, 8)
                hidden_states, cache = model(inputs)
                self.assertEqual(hidden_states.shape, inputs.shape)
                self.assertIsInstance(cache, tuple)
                self.assertEqual(len(cache), 1)
                keys, values = cache[0]
                self.assertEqual(keys.shape, (batch_size, 1, 3, 4))
                self.assertEqual(values.shape, keys.shape)
                repeated_states, repeated_cache = model(inputs)
                torch.testing.assert_close(repeated_states, hidden_states)
                torch.testing.assert_close(repeated_cache, cache)

    def test_decoder_and_cache_outputs_survive_export(self):
        torch.manual_seed(0)
        model = _TinyLlamaLayer(seq_len=3).eval()
        inputs = (torch.randn(1, 3, 8),)
        expected = model(*inputs)
        exported = torch.export.export(model, inputs)
        torch.testing.assert_close(exported.module()(*inputs), expected)


if __name__ == "__main__":
    unittest.main()
