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

"""Tests for the split Gemma4 PLE embedding and projection export adapters."""

import unittest

import torch

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.mode import Mode


_SKIP_MSG = "required transformers Gemma4 modules are not installed"

_OWNERSHIP_FORBIDDEN_PREFIXES = ("embed_tokens.", "layers.", "lm_head", "norm.")


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


def _make_quant_text_model(cfg, *, quantized: bool):
    """Return a wrapped tiny text model in NO_QUANT or frozen QUANT mode."""
    from tico.quantization.wrapq.wrappers.gemma4.quant_text_model import (
        QuantGemma4TextModel,
    )
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel

    fp_model = Gemma4TextModel(cfg).eval()
    qtext = QuantGemma4TextModel(fp_model, qcfg=PTQConfig()).eval()
    if quantized:
        qtext.enable_calibration()
        with torch.no_grad():
            for _ in range(2):
                qtext(
                    input_ids=torch.randint(0, cfg.vocab_size_per_layer_input, (1, 5))
                )
        qtext.freeze_qparams()
        assert qtext._mode is Mode.QUANT
    return fp_model, qtext


def _state_names(module: torch.nn.Module) -> list[str]:
    """Return every parameter, buffer, and state-dict key of a module."""
    names = [name for name, _ in module.named_parameters()]
    names += [name for name, _ in module.named_buffers()]
    names += list(module.state_dict().keys())
    return names


@unittest.skipUnless(_has_gemma4(), _SKIP_MSG)
class TestGemma4PLEExportAdapters(unittest.TestCase):
    """Ownership, observer, and parity contracts for the PLE stage adapters."""

    def setUp(self):
        """Create deterministic inputs."""
        torch.manual_seed(2026)
        self.cfg = _make_ple_text_config()

    def _adapters(self, qtext):
        from tico.quantization.wrapq.wrappers.gemma4.export_adapters import (
            Gemma4PLEEmbeddingExportAdapter,
            Gemma4PLEProjectionExportAdapter,
        )

        return (
            Gemma4PLEEmbeddingExportAdapter(qtext).eval(),
            Gemma4PLEProjectionExportAdapter(qtext).eval(),
        )

    def _ids(self, seq_len: int) -> torch.Tensor:
        return torch.randint(0, self.cfg.vocab_size_per_layer_input, (1, seq_len))

    # ------------------------------------------------------------------ ownership

    def test_projection_adapter_owns_only_projection_state(self):
        """The projection adapter must not register any embedding table or layer."""
        _fp_model, qtext = _make_quant_text_model(self.cfg, quantized=True)
        _embedding, projection = self._adapters(qtext)

        names = _state_names(projection)
        self.assertTrue(names)
        for name in names:
            self.assertNotIn("embed_tokens", name)
            self.assertNotIn("embed_tokens_per_layer", name)
            self.assertNotIn("lm_head", name)
            self.assertFalse(name.startswith("layers."), name)
            self.assertFalse(
                name.startswith(_OWNERSHIP_FORBIDDEN_PREFIXES),
                name,
            )
        parameter_names = [name for name, _ in projection.named_parameters()]
        self.assertEqual(
            sorted(parameter_names),
            [
                "per_layer_model_projection.wrapped.module.weight",
                "per_layer_projection_norm.wrapped.module.weight",
            ],
        )
        self.assertIs(
            projection.per_layer_model_projection, qtext.per_layer_model_projection
        )
        self.assertIs(
            projection.per_layer_projection_norm, qtext.per_layer_projection_norm
        )
        self.assertEqual(
            projection.per_layer_model_projection_scale,
            qtext.per_layer_model_projection_scale,
        )
        self.assertEqual(projection.per_layer_input_scale, qtext.per_layer_input_scale)

    def test_embedding_adapter_owns_only_the_per_layer_table(self):
        """The lookup adapter owns embed_tokens_per_layer and nothing else."""
        _fp_model, qtext = _make_quant_text_model(self.cfg, quantized=True)
        embedding, _projection = self._adapters(qtext)

        parameter_names = [name for name, _ in embedding.named_parameters()]
        self.assertEqual(
            parameter_names,
            ["embed_tokens_per_layer.wrapped.module.weight"],
        )
        for name in _state_names(embedding):
            self.assertTrue(
                name.startswith("embed_tokens_per_layer.")
                or name.startswith("per_layer_token_inputs_observer."),
                name,
            )
            self.assertNotIn("lm_head", name)
            self.assertFalse(name.startswith("layers."), name)
        self.assertIs(embedding.embed_tokens_per_layer, qtext.embed_tokens_per_layer)
        table = dict(embedding.named_parameters())[
            "embed_tokens_per_layer.wrapped.module.weight"
        ]
        self.assertEqual(
            tuple(table.shape),
            (
                self.cfg.vocab_size_per_layer_input,
                self.cfg.num_hidden_layers * self.cfg.hidden_size_per_layer_input,
            ),
        )

    # ------------------------------------------------------------------ observers

    def test_adapters_reuse_text_model_observers(self):
        """Both adapters must hold the original observer instances, not copies."""
        _fp_model, qtext = _make_quant_text_model(self.cfg, quantized=True)
        embedding, projection = self._adapters(qtext)

        self.assertIs(
            embedding.per_layer_token_inputs_observer,
            qtext.obs_per_layer_token_inputs,
        )
        self.assertIs(
            projection.per_layer_token_inputs_observer,
            qtext.obs_per_layer_token_inputs,
        )
        self.assertIs(
            projection.per_layer_projection_observer,
            qtext.obs_per_layer_projection,
        )
        self.assertIs(projection.per_layer_inputs_observer, qtext.obs_per_layer_inputs)
        self.assertTrue(qtext.obs_per_layer_token_inputs.has_qparams)
        self.assertTrue(qtext.obs_per_layer_inputs.has_qparams)

    def test_shared_token_input_boundary_is_idempotent(self):
        """Re-applying the producer observer at the consumer must not change values."""
        _fp_model, qtext = _make_quant_text_model(self.cfg, quantized=True)
        observer = qtext.obs_per_layer_token_inputs
        raw = (
            torch.randn(
                1, 5, self.cfg.num_hidden_layers, self.cfg.hidden_size_per_layer_input
            )
            * 4.0
        )

        once = observer.fake_quant(raw)
        twice = observer.fake_quant(once)
        self.assertTrue(torch.equal(once, twice))

    def test_calibration_mode_is_rejected(self):
        """Adapters are export boundaries and must reject CALIB mode."""
        from tico.quantization.wrapq.wrappers.gemma4.export_adapters import (
            Gemma4PLEEmbeddingExportAdapter,
            Gemma4PLEProjectionExportAdapter,
        )

        _fp_model, qtext = _make_quant_text_model(self.cfg, quantized=False)
        qtext.enable_calibration()
        with self.assertRaisesRegex(RuntimeError, "NO_QUANT or QUANT"):
            Gemma4PLEEmbeddingExportAdapter(qtext)
        with self.assertRaisesRegex(RuntimeError, "NO_QUANT or QUANT"):
            Gemma4PLEProjectionExportAdapter(qtext)

    def test_adapters_reject_ple_disabled_text_model(self):
        """A model without PLE has no lookup or projection stage to export."""
        from tico.quantization.wrapq.wrappers.gemma4.export_adapters import (
            Gemma4PLEEmbeddingExportAdapter,
            Gemma4PLEProjectionExportAdapter,
        )

        cfg = _make_ple_text_config(hidden_size_per_layer_input=0)
        _fp_model, qtext = _make_quant_text_model(cfg, quantized=False)
        with self.assertRaisesRegex(RuntimeError, "hidden_size_per_layer_input"):
            Gemma4PLEEmbeddingExportAdapter(qtext)
        with self.assertRaisesRegex(RuntimeError, "hidden_size_per_layer_input"):
            Gemma4PLEProjectionExportAdapter(qtext)

    def test_no_quant_adapters_never_call_fake_quant(self):
        """NO_QUANT export must bypass every observer boundary."""
        from tico.quantization.wrapq.wrappers.gemma4.export_adapters import (
            Gemma4PLEEmbeddingExportAdapter,
            Gemma4PLEProjectionExportAdapter,
        )

        _fp_model, qtext = _make_quant_text_model(self.cfg, quantized=False)
        embedding = Gemma4PLEEmbeddingExportAdapter(qtext)
        projection = Gemma4PLEProjectionExportAdapter(qtext)

        def _fail(_tensor):
            raise AssertionError("NO_QUANT export must not call fake_quant().")

        for observer in (
            qtext.obs_per_layer_token_inputs,
            qtext.obs_per_layer_projection,
            qtext.obs_per_layer_inputs,
        ):
            observer.fake_quant = _fail  # type: ignore[method-assign]

        ids = self._ids(3)
        with torch.no_grad():
            token_inputs = embedding(ids)
            projection(qtext.embed_tokens(ids), token_inputs)

    # --------------------------------------------------------------------- parity

    def _assert_parity(self, *, quantized: bool, seq_len: int) -> None:
        fp_model, qtext = _make_quant_text_model(self.cfg, quantized=quantized)
        embedding, projection = self._adapters(qtext)
        ids = self._ids(seq_len)
        inputs_embeds = fp_model.embed_tokens(ids)

        with torch.no_grad():
            token_inputs = embedding(ids)
            ref_token_inputs = qtext.get_per_layer_inputs(ids, None)
            per_layer_inputs = projection(inputs_embeds, token_inputs)
            ref_per_layer_inputs = qtext.project_per_layer_inputs(
                inputs_embeds, ref_token_inputs
            )

        expected_shape = (
            1,
            seq_len,
            self.cfg.num_hidden_layers,
            self.cfg.hidden_size_per_layer_input,
        )
        self.assertEqual(tuple(token_inputs.shape), expected_shape)
        self.assertEqual(tuple(per_layer_inputs.shape), expected_shape)
        # Same modules, same observers, same op order: bit-exact.
        self.assertTrue(torch.equal(token_inputs, ref_token_inputs))
        self.assertTrue(torch.equal(per_layer_inputs, ref_per_layer_inputs))

        if not quantized:
            # Floating-point path must also match the Hugging Face reference.
            ref_hf = fp_model.project_per_layer_inputs(
                inputs_embeds, fp_model.get_per_layer_inputs(ids, inputs_embeds)
            )
            torch.testing.assert_close(per_layer_inputs, ref_hf, atol=1e-5, rtol=1e-5)

    def test_parity_no_quant_prefill_like(self):
        """NO_QUANT S>1 outputs match the authoritative text-model methods."""
        self._assert_parity(quantized=False, seq_len=6)

    def test_parity_no_quant_decode_like(self):
        """NO_QUANT S=1 outputs match the authoritative text-model methods."""
        self._assert_parity(quantized=False, seq_len=1)

    def test_parity_quant_prefill_like(self):
        """QUANT S>1 outputs match the frozen text-model methods exactly."""
        self._assert_parity(quantized=True, seq_len=6)

    def test_parity_quant_decode_like(self):
        """QUANT S=1 outputs match the frozen text-model methods exactly."""
        self._assert_parity(quantized=True, seq_len=1)

    def test_token_inputs_bypass_projection(self):
        """Only inputs_embeds is projected; the lookup is added after the norm."""
        fp_model, qtext = _make_quant_text_model(self.cfg, quantized=False)
        _embedding, projection = self._adapters(qtext)
        ids = self._ids(4)
        inputs_embeds = fp_model.embed_tokens(ids)
        zeros = torch.zeros(
            1, 4, self.cfg.num_hidden_layers, self.cfg.hidden_size_per_layer_input
        )
        delta = torch.full_like(zeros, 0.5)

        with torch.no_grad():
            base = projection(inputs_embeds, zeros)
            shifted = projection(inputs_embeds, delta)

        torch.testing.assert_close(
            shifted - base,
            delta * qtext.per_layer_input_scale,
            atol=1e-6,
            rtol=1e-6,
        )

    # ---------------------------------------------------------------- torch.export

    def test_embedding_adapter_exports_with_token_embedding_dynamic_contract(self):
        """ple_embedding follows the token_embedding ``(1, S)`` dynamic contract."""
        from tico.quantization.wrapq.wrappers.llama.export_adapters import (
            make_token_embedding_dynamic_shapes,
            register_fake_quant_meta_kernels_for_dynamic_export,
        )

        max_seq_len = 16
        for quantized in (False, True):
            with self.subTest(quantized=quantized):
                _fp_model, qtext = _make_quant_text_model(self.cfg, quantized=quantized)
                embedding, _projection = self._adapters(qtext)
                register_fake_quant_meta_kernels_for_dynamic_export()
                exported = torch.export.export(
                    embedding,
                    (self._ids(max_seq_len),),
                    dynamic_shapes=make_token_embedding_dynamic_shapes(max_seq_len),
                    strict=False,
                )
                for seq_len in (1, 5, max_seq_len):
                    ids = self._ids(seq_len)
                    with torch.no_grad():
                        actual = exported.module()(ids)
                        expected = qtext.get_per_layer_inputs(ids, None)
                    self.assertTrue(torch.equal(actual, expected))

    def test_projection_adapter_exports_static_prefill_and_decode_shapes(self):
        """One adapter instance traces for both fixed sequence lengths."""
        for quantized in (False, True):
            with self.subTest(quantized=quantized):
                fp_model, qtext = _make_quant_text_model(self.cfg, quantized=quantized)
                _embedding, projection = self._adapters(qtext)
                for seq_len in (6, 1):
                    ids = self._ids(seq_len)
                    inputs_embeds = fp_model.embed_tokens(ids)
                    with torch.no_grad():
                        token_inputs = qtext.get_per_layer_inputs(ids, None)
                        expected = projection(inputs_embeds, token_inputs)
                    exported = torch.export.export(
                        projection,
                        (inputs_embeds, token_inputs),
                        strict=False,
                    )
                    with torch.no_grad():
                        actual = exported.module()(inputs_embeds, token_inputs)
                    self.assertTrue(torch.equal(actual, expected))
                    self.assertFalse(
                        any(
                            "embed_tokens" in name
                            for name in exported.state_dict.keys()
                        )
                    )

    def test_circle_conversion_smoke(self):
        """Both adapters convert to Circle with small static shapes."""
        import tico
        from tico.quantization.wrapq.wrappers.llama.export_adapters import (
            make_token_embedding_dynamic_shapes,
            register_fake_quant_meta_kernels_for_dynamic_export,
        )

        for quantized in (False, True):
            with self.subTest(quantized=quantized):
                fp_model, qtext = _make_quant_text_model(self.cfg, quantized=quantized)
                embedding, projection = self._adapters(qtext)
                register_fake_quant_meta_kernels_for_dynamic_export()

                max_seq_len = 4
                ids = self._ids(max_seq_len)
                with torch.no_grad():
                    circle_embedding = tico.convert(
                        embedding,
                        (ids,),
                        dynamic_shapes=make_token_embedding_dynamic_shapes(max_seq_len),
                        strict=False,
                    )
                    inputs_embeds = fp_model.embed_tokens(ids)
                    token_inputs = qtext.get_per_layer_inputs(ids, None)
                    circle_projection = tico.convert(
                        projection,
                        (inputs_embeds, token_inputs),
                        strict=False,
                    )
                self.assertIsNotNone(circle_embedding)
                self.assertIsNotNone(circle_projection)


if __name__ == "__main__":
    unittest.main()
