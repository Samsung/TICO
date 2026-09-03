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

"""Tests for the host ``.pt`` artifact of the Gemma4 PLE embedding stage."""

import tempfile
import unittest
from pathlib import Path

import torch

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.mode import Mode


_SKIP_MSG = "required transformers Gemma4 modules are not installed"


def _has_gemma4() -> bool:
    """Return whether the installed transformers package provides Gemma4."""
    try:
        from transformers.models.gemma4.configuration_gemma4 import (  # noqa: F401
            Gemma4TextConfig,
        )
        from transformers.models.gemma4.modeling_gemma4 import (  # noqa: F401
            Gemma4TextModel,
        )
    except Exception:
        return False
    return True


def _make_ple_text_config(**overrides):
    """Create a tiny dense Gemma4 text config with Per-Layer Embeddings."""
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

    kwargs = dict(
        vocab_size=64,
        vocab_size_per_layer_input=48,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_global_key_value_heads=2,
        head_dim=4,
        global_head_dim=4,
        attention_bias=False,
        attention_dropout=0.0,
        max_position_embeddings=32,
        rms_norm_eps=1e-6,
        sliding_window=8,
        layer_types=["full_attention", "full_attention"],
        rope_parameters={
            "full_attention": {
                "rope_type": "proportional",
                "partial_rotary_factor": 0.25,
                "rope_theta": 1_000_000.0,
            }
        },
        hidden_size_per_layer_input=8,
        attention_k_eq_v=False,
        num_kv_shared_layers=0,
        enable_moe_block=False,
        use_cache=False,
    )
    kwargs.update(overrides)
    cfg = Gemma4TextConfig(**kwargs)
    cfg._attn_implementation = "eager"
    return cfg


def _make_adapter(cfg, *, quantized: bool, qcfg=None):
    """Return ``(qtext, ple_embedding_adapter)`` in NO_QUANT or frozen QUANT."""
    from tico.quantization.wrapq.wrappers.gemma4.export_adapters import (
        Gemma4PLEEmbeddingExportAdapter,
    )
    from tico.quantization.wrapq.wrappers.gemma4.quant_text_model import (
        QuantGemma4TextModel,
    )
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel

    fp_model = Gemma4TextModel(cfg).eval()
    qtext = QuantGemma4TextModel(fp_model, qcfg=qcfg or PTQConfig()).eval()
    if quantized:
        qtext.enable_calibration()
        with torch.no_grad():
            for _ in range(2):
                qtext(
                    input_ids=torch.randint(0, cfg.vocab_size_per_layer_input, (1, 5))
                )
        qtext.freeze_qparams()
        assert qtext._mode is Mode.QUANT
    return qtext, Gemma4PLEEmbeddingExportAdapter(qtext).eval()


@unittest.skipUnless(_has_gemma4(), _SKIP_MSG)
class TestGemma4PLEEmbeddingHostArtifact(unittest.TestCase):
    """Round-trip, parity, and size-estimate contracts of the ``.pt`` stage."""

    def setUp(self):
        torch.manual_seed(2026)
        self.cfg = _make_ple_text_config()

    def _ids(self, seq_len: int) -> torch.Tensor:
        return torch.randint(0, self.cfg.vocab_size_per_layer_input, (1, seq_len))

    def _roundtrip(self, adapter):
        from tico.quantization.wrapq.wrappers.gemma4.ple_embedding_host import (
            Gemma4PLEEmbeddingHostTable,
            save_gemma4_ple_embedding_artifact,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_gemma4_ple_embedding_artifact(
                adapter, Path(tmpdir) / "ple_embedding.test.pt"
            )
            self.assertTrue(path.exists())
            payload = torch.load(path, map_location="cpu", weights_only=True)
            host = Gemma4PLEEmbeddingHostTable.from_artifact(path)
        return payload, host

    def _assert_parity(self, adapter, host):
        for seq_len in (1, 7):
            ids = self._ids(seq_len)
            with torch.no_grad():
                expected = adapter(ids)
                actual = host(ids)
            self.assertEqual(
                tuple(actual.shape),
                (
                    1,
                    seq_len,
                    self.cfg.num_hidden_layers,
                    self.cfg.hidden_size_per_layer_input,
                ),
            )
            # Same table, same op order, same float32 kernels: bit-exact.
            self.assertTrue(torch.equal(actual, expected))

    def test_no_quant_artifact_roundtrip_matches_adapter(self):
        """Float artifact stores the raw table and replays the scaled lookup."""
        _qtext, adapter = _make_adapter(self.cfg, quantized=False)
        payload, host = self._roundtrip(adapter)

        self.assertFalse(payload["quantized"])
        self.assertEqual(payload["stage"], "ple_embedding")
        self.assertEqual(
            tuple(payload["weight"].shape),
            (
                self.cfg.vocab_size_per_layer_input,
                self.cfg.num_hidden_layers * self.cfg.hidden_size_per_layer_input,
            ),
        )
        self.assertNotIn("weight_int", payload)
        self.assertNotIn("observers", payload)
        self.assertEqual(payload["num_hidden_layers"], self.cfg.num_hidden_layers)
        self.assertEqual(
            payload["hidden_size_per_layer_input"], self.cfg.hidden_size_per_layer_input
        )
        self.assertAlmostEqual(
            float(payload["embed_scale"]),
            self.cfg.hidden_size_per_layer_input**0.5,
            places=5,
        )
        self._assert_parity(adapter, host)

    def test_quant_artifact_stores_integer_table_and_observers(self):
        """Quantized artifact keeps int weights plus every replayed observer."""
        qtext, adapter = _make_adapter(self.cfg, quantized=True)
        payload, host = self._roundtrip(adapter)

        self.assertTrue(payload["quantized"])
        self.assertNotIn("weight", payload)
        self.assertEqual(payload["weight_int"].dtype, torch.uint8)
        self.assertEqual(payload["weight_dtype"], "uint8")
        self.assertEqual(payload["weight_channel_axis"], 0)
        self.assertEqual(
            tuple(payload["weight_scale"].shape),
            (self.cfg.vocab_size_per_layer_input,),
        )
        self.assertEqual(
            set(payload["observers"]),
            {"embedding", "embed_scale", "act_out", "per_layer_token_inputs"},
        )
        token_observer = qtext.obs_per_layer_token_inputs
        self.assertTrue(
            torch.equal(
                payload["observers"]["per_layer_token_inputs"]["scale"],
                token_observer._cached_scale.cpu(),
            )
        )
        self.assertEqual(
            payload["observers"]["per_layer_token_inputs"]["quant_max"],
            token_observer.dtype.qmax,
        )
        # The dequantized table must equal the wrapper's fake-quantized weight.
        embedding = qtext.embed_tokens_per_layer.wrapped
        expected_weight = embedding.obs_weight.fake_quant(embedding.module.weight)
        self.assertTrue(torch.equal(host._dequantized_weight(), expected_weight))
        self._assert_parity(adapter, host)

    def test_quant_artifact_requires_frozen_qparams(self):
        """Saving before convert must fail loudly instead of writing junk."""
        from tico.quantization.wrapq.wrappers.gemma4.export_adapters import (
            Gemma4PLEEmbeddingExportAdapter,
        )
        from tico.quantization.wrapq.wrappers.gemma4.ple_embedding_host import (
            build_gemma4_ple_embedding_artifact,
        )

        qtext, _adapter = _make_adapter(self.cfg, quantized=False)
        # Pretend the model is quantized without ever computing qparams.
        adapter = Gemma4PLEEmbeddingExportAdapter(qtext, mode=Mode.QUANT)
        with self.assertRaisesRegex(RuntimeError, "no frozen qparams"):
            build_gemma4_ple_embedding_artifact(adapter)

    def test_loader_rejects_foreign_or_versioned_payloads(self):
        """The host table validates stage and schema before use."""
        from tico.quantization.wrapq.wrappers.gemma4.ple_embedding_host import (
            build_gemma4_ple_embedding_artifact,
            Gemma4PLEEmbeddingHostTable,
        )

        _qtext, adapter = _make_adapter(self.cfg, quantized=False)
        payload = build_gemma4_ple_embedding_artifact(adapter)

        with self.assertRaisesRegex(ValueError, "Expected a 'ple_embedding'"):
            Gemma4PLEEmbeddingHostTable({**payload, "stage": "token_embedding"})
        with self.assertRaisesRegex(ValueError, "schema_version"):
            Gemma4PLEEmbeddingHostTable({**payload, "schema_version": 99})

    def test_circle_size_estimate_follows_weight_storage_dtype(self):
        """Estimate counts the packed table bytes for float and quantized tables."""
        from tico.quantization.wrapq.wrappers.gemma4.ple_embedding_host import (
            CIRCLE_FLATBUFFER_LIMIT_BYTES,
            estimate_gemma4_ple_embedding_circle_bytes,
        )

        numel = self.cfg.vocab_size_per_layer_input * (
            self.cfg.num_hidden_layers * self.cfg.hidden_size_per_layer_input
        )
        _qtext, fp_adapter = _make_adapter(self.cfg, quantized=False)
        self.assertEqual(
            estimate_gemma4_ple_embedding_circle_bytes(fp_adapter), numel * 4
        )

        _qtext, q_adapter = _make_adapter(self.cfg, quantized=True)
        self.assertEqual(estimate_gemma4_ple_embedding_circle_bytes(q_adapter), numel)

        self.assertEqual(CIRCLE_FLATBUFFER_LIMIT_BYTES, 2**31)
        # E2B geometry: the table alone exceeds the limit in float32 and uint8.
        e2b_numel = 262144 * 35 * 256
        self.assertGreater(e2b_numel * 4, CIRCLE_FLATBUFFER_LIMIT_BYTES)
        self.assertGreater(e2b_numel, CIRCLE_FLATBUFFER_LIMIT_BYTES)
        self.assertLess(e2b_numel // 2, CIRCLE_FLATBUFFER_LIMIT_BYTES)


if __name__ == "__main__":
    unittest.main()
