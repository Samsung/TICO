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

import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import tico.quantization.recipes.export.llama as llama_export

import torch


class FakeLayerWrapper(torch.nn.Module):
    """Fake wrapped decoder layer used for per-layer export tests."""

    def __init__(self, max_seq=4):
        super().__init__()
        self.causal_mask_template = torch.zeros(1, 1, max_seq, max_seq)

    def _slice_rope(self, start, seq_len, device, dtype):
        """Return fake RoPE tensors with the requested sequence length."""
        return (
            torch.ones(1, seq_len, 4, device=device, dtype=dtype),
            torch.zeros(1, seq_len, 4, device=device, dtype=dtype),
        )

    def as_export_module(self, mode, return_kv=False):
        """Return a trivial module representing the export adapter."""
        return torch.nn.Identity()


class FakeWrappedModel:
    """Fake top-level PTQ-wrapped model."""

    def __init__(self, qmodel):
        self.wrapped = qmodel

    def eval(self):
        """Return self to mimic torch modules."""
        return self

    def cpu(self):
        """Return self to mimic torch modules."""
        return self


class TestLlamaExport(unittest.TestCase):
    def test_export_llama_per_layer_exports_embedding_layers_and_lm_head(self):
        """Per-layer LLaMA export should emit embedding, decoder, and LM-head artifacts."""
        calls = []
        layer = SimpleNamespace(wrapped=FakeLayerWrapper(max_seq=4))
        qmodel = SimpleNamespace(
            config=SimpleNamespace(
                hidden_size=8,
                num_attention_heads=2,
                num_key_value_heads=1,
            ),
            model=SimpleNamespace(wrapped=SimpleNamespace(layers=[layer])),
        )
        wrapped_model = FakeWrappedModel(qmodel)

        def fake_convert_and_save(module, example_inputs, save_path, **kwargs):
            calls.append(save_path.name)

        with tempfile.TemporaryDirectory() as tmpdir, patch.object(
            llama_export,
            "export_token_embedding",
            lambda qmodel, max_seq_len, output_dir: calls.append(
                "token_embedding.q.circle"
            ),
        ), patch.object(
            llama_export,
            "export_lm_head",
            lambda qmodel, output_dir: calls.append("lm_head.q.circle"),
        ), patch.object(
            llama_export, "_convert_and_save", fake_convert_and_save
        ):
            llama_export.export_llama_per_layer(
                q_model=wrapped_model,
                max_seq_len=4,
                output_dir=tmpdir,
                prefill_decode=True,
            )

        self.assertEqual(
            calls,
            [
                "token_embedding.q.circle",
                "decoder_layer_prefill_0.q.circle",
                "decoder_layer_decode_0.q.circle",
                "lm_head.q.circle",
            ],
        )
