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

"""Offline, real-HF tests for unified LLaMA eager and static RoPE semantics."""

import copy
import io
import unittest
from typing import cast

import torch
from tico.quantization.recipes.debug.export_regression import _state_contract
from tico.quantization.wrapq.utils.static_export import prepared_static_export
from tico.quantization.wrapq.wrappers.llama.quant_decoder_layer import (
    QuantLlamaDecoderLayer,
)
from tico.quantization.wrapq.wrappers.llama.quant_model import QuantLlamaModel
from tico.quantization.wrapq.wrappers.llama.static_inputs import (
    migrate_llama_rope_state,
    prepare_llama_static_inputs,
)

from test.quantization.quant_spec_helpers import make_affine_ptq_config


class TestLlamaRoPEConsistency(unittest.TestCase):
    """Check actual wrappers, not mocks, using tiny randomly initialized models."""

    def setUp(self):
        torch.manual_seed(37)

    @staticmethod
    def _model(kind="llama3", *, head_dim=16, capacity=131072):
        """Build a small model without downloading weights or configuration."""
        from transformers.models.llama.configuration_llama import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaModel

        scaling = {
            "default": None,
            "linear": {"rope_type": "linear", "factor": 4.0},
            "dynamic": {"rope_type": "dynamic", "factor": 4.0},
            "llama3": {
                "rope_type": "llama3",
                "factor": 32.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192,
            },
        }[kind]
        config = LlamaConfig.from_dict(
            {
                "vocab_size": 64,
                "hidden_size": 2 * head_dim,
                "intermediate_size": 4 * head_dim,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": head_dim,
                "max_position_embeddings": capacity,
                "rope_theta": 500000.0,
                "rope_scaling": scaling,
                "attention_dropout": 0.0,
            }
        )
        config._attn_implementation = "eager"
        return LlamaModel(config).eval()

    @staticmethod
    def _wrapper(model, profile="reference_eval"):
        """Avoid mutating the reference FP model through attention fusion."""
        return QuantLlamaModel(
            copy.deepcopy(model),
            qcfg=make_affine_ptq_config(model_args={"profile": profile}),
        ).eval()

    @staticmethod
    def _expected(model, length, *, negate=False):
        """Use the original model-level implementation as the numerical oracle."""
        positions = torch.arange(length).unsqueeze(0)
        with torch.no_grad():
            cos, sin = model.rotary_emb(torch.empty((), dtype=torch.float32), positions)
        if negate:
            half = sin.shape[-1] // 2
            sin = torch.cat((-sin[..., :half], sin[..., half:]), dim=-1)
        return cos, sin

    def test_default_linear_llama3_match_in_both_conventions(self):
        """Independent layer wrapping must retain all configured scaling fields."""
        for kind in ("default", "linear", "llama3"):
            for profile in ("reference_eval", "npu_export"):
                with self.subTest(kind=kind, profile=profile):
                    model = self._model(kind)
                    full = self._wrapper(model, profile)
                    standalone = QuantLlamaDecoderLayer(
                        copy.deepcopy(model.layers[0]),
                        qcfg=make_affine_ptq_config(model_args={"profile": profile}),
                        layer_idx=0,
                        return_type="tuple",
                    ).eval()
                    expected = self._expected(model, 32, negate=profile == "npu_export")
                    for owner in (
                        full,
                        standalone,
                        *(layer.wrapped for layer in full.layers),
                    ):
                        inputs = prepare_llama_static_inputs(owner, 32)
                        for actual, target in zip((inputs.cos, inputs.sin), expected):
                            torch.testing.assert_close(actual, target, atol=0, rtol=0)

    def test_llama3_is_not_the_old_unscaled_theta_formula(self):
        """Catch a regression that could otherwise make both wrappers wrong together."""
        model = self._model("llama3")
        layer = QuantLlamaDecoderLayer(
            copy.deepcopy(model.layers[0]),
            qcfg=make_affine_ptq_config(model_args={"profile": "reference_eval"}),
        ).eval()
        actual = prepare_llama_static_inputs(layer, 32)
        raw = 1.0 / (500000.0 ** (torch.arange(0, 16, 2).float() / 16))
        phases = torch.outer(torch.arange(32).float(), raw)
        old_sin = torch.cat((phases, phases), dim=-1).sin().unsqueeze(0)
        # The original unscaled implementation must fail this test.
        self.assertGreater(float((actual.sin - old_sin).abs().max()), 1e-4)
        for value, target in zip((actual.cos, actual.sin), self._expected(model, 32)):
            torch.testing.assert_close(value, target, atol=0, rtol=0)

    def test_head_dim_128_preserves_gqa_rope_and_decoder_parity(self):
        """Exercise a realistic rotary width with short CPU-only sequences."""
        model = self._model("linear", head_dim=128, capacity=128)
        wrapper = self._wrapper(model)
        with torch.no_grad():
            tokens = torch.arange(16).unsqueeze(0)
            expected = model(tokens, use_cache=False).last_hidden_state
            actual = wrapper(tokens, use_cache=False).last_hidden_state
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        for length in (16, 32):
            tables = prepare_llama_static_inputs(wrapper, length)
            self.assertEqual(tables.cos.shape, (1, length, 128))
            for actual, expected in zip(
                (tables.cos, tables.sin), self._expected(model, length)
            ):
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_actual_dynamic_rope_threshold_does_not_mutate_static_source(self):
        """Cross the HF dynamic scaling threshold and recheck the static profile."""
        model = self._model("dynamic", capacity=16)
        wrapper = self._wrapper(model)
        before = prepare_llama_static_inputs(wrapper, 16)
        initial_frequency = wrapper.rotary_emb.inv_freq.clone()
        with torch.no_grad():
            wrapper(torch.arange(48).unsqueeze(0), use_cache=False)
        self.assertFalse(torch.equal(initial_frequency, wrapper.rotary_emb.inv_freq))
        after = prepare_llama_static_inputs(wrapper, 16)
        torch.testing.assert_close(before.cos, after.cos, atol=0, rtol=0)
        torch.testing.assert_close(before.sin, after.sin, atol=0, rtol=0)
        for actual, expected in zip((after.cos, after.sin), self._expected(model, 16)):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        with self.assertRaisesRegex(ValueError, "\\[1, 16\\]"):
            prepare_llama_static_inputs(wrapper, 17)

    def test_model_source_overrides_layer_fallback_and_can_be_passed_explicitly(self):
        """Custom pretrained frequency/scaling values propagate to every layer."""
        model = self._model()
        model.rotary_emb.inv_freq.mul_(0.75)
        model.rotary_emb.attention_scaling = 1.25
        full = self._wrapper(model)
        independent = QuantLlamaDecoderLayer(
            copy.deepcopy(model.layers[0]),
            qcfg=make_affine_ptq_config(model_args={"profile": "reference_eval"}),
            rotary_emb=model.rotary_emb,
        ).eval()
        expected = self._expected(model, 16)
        for owner in (full, independent, *(layer.wrapped for layer in full.layers)):
            prepared = prepare_llama_static_inputs(owner, 16)
            for actual, target in zip((prepared.cos, prepared.sin), expected):
                torch.testing.assert_close(actual, target, atol=0, rtol=0)
            self.assertIsNot(owner._export_rotary_emb, model.rotary_emb)
            self.assertNotEqual(
                owner._export_rotary_emb.inv_freq.data_ptr(),
                model.rotary_emb.inv_freq.data_ptr(),
            )

    def test_eager_mutations_cannot_change_static_tables_or_serialized_snapshot(self):
        """The snapshot survives an eager call, live mutation, and pickle roundtrip."""
        model = self._wrapper(self._model())
        before = prepare_llama_static_inputs(model.layers[0].wrapped, 16)
        with torch.no_grad():
            model(torch.randint(0, 64, (1, 48)), use_cache=False)
        for owner in (model, *(layer.wrapped for layer in model.layers)):
            owner.rotary_emb.inv_freq.mul_(100)
            owner.rotary_emb.attention_scaling = 9.0
        buffer = io.BytesIO()
        torch.save(model, buffer)
        buffer.seek(0)
        restored = torch.load(buffer, weights_only=False)
        for owner in (model, restored, *(layer.wrapped for layer in restored.layers)):
            after = prepare_llama_static_inputs(owner, 16)
            torch.testing.assert_close(before.cos, after.cos, atol=0, rtol=0)
            torch.testing.assert_close(before.sin, after.sin, atol=0, rtol=0)

    def test_staged_prefill_and_delta_decode_match_full_model(self):
        """Use identical weights with padded static caches for two decode steps."""
        for profile in ("reference_eval", "npu_export"):
            with self.subTest(profile=profile), torch.no_grad():
                model = self._wrapper(self._model(), profile)
                length, valid = 16, 5
                tokens = torch.randint(0, 64, (1, valid + 2))
                hidden = model.embed_tokens(
                    torch.nn.functional.pad(tokens[:, :valid], (0, length - valid))
                )
                inputs = [
                    prepare_llama_static_inputs(layer.wrapped, length)
                    for layer in model.layers
                ]
                caches = []
                for layer, prepared in zip(model.layers, inputs):
                    mask = prepared.mask(length, length, 0, device="cpu").clone()
                    mask[:, :, valid:] = float(
                        layer.wrapped.qcfg.attention_mask_fill_value
                    )
                    hidden, key, value = layer.wrapped.as_export_module(
                        "prefill", return_kv=True
                    )(
                        hidden,
                        attention_mask=mask,
                        position_embeddings=(prepared.cos, prepared.sin),
                    )
                    caches.append(
                        (
                            key[:, :, : length - 1].clone(),
                            value[:, :, : length - 1].clone(),
                        )
                    )
                expected = model(tokens[:, :valid], use_cache=False).last_hidden_state
                torch.testing.assert_close(
                    model.norm(hidden)[:, :valid], expected, atol=2e-6, rtol=2e-5
                )
                for position in range(valid, valid + 2):
                    hidden = model.embed_tokens(tokens[:, position : position + 1])
                    for index, (layer, prepared) in enumerate(
                        zip(model.layers, inputs)
                    ):
                        mask = torch.full(
                            (1, 1, length),
                            float(layer.wrapped.qcfg.attention_mask_fill_value),
                        )
                        mask[..., :position] = 0
                        mask[..., -1] = 0
                        rope = prepared.rope(
                            position, 1, device="cpu", dtype=torch.float32
                        )
                        hidden, delta_k, delta_v = layer.wrapped.as_export_module(
                            "decode"
                        )(
                            hidden,
                            attention_mask=mask,
                            position_embeddings=rope,
                            past_key_value=caches[index],
                        )
                        self.assertEqual(tuple(delta_k.shape), (1, 1, 1, 16))
                        self.assertEqual(tuple(delta_v.shape), (1, 1, 1, 16))
                        caches[index][0][:, :, position : position + 1] = delta_k
                        caches[index][1][:, :, position : position + 1] = delta_v
                    expected = model(
                        tokens[:, : position + 1], use_cache=False
                    ).last_hidden_state[:, -1:]
                    torch.testing.assert_close(
                        model.norm(hidden), expected, atol=2e-6, rtol=2e-5
                    )

    def test_explicit_migration_preserves_frozen_weights_and_qparams(self):
        """Changing positional provenance must not recalibrate frozen observers."""
        model = self._wrapper(self._model())
        model.enable_calibration()
        with torch.no_grad():
            model(torch.randint(0, 64, (1, 8)), use_cache=False)
        model.freeze_qparams()
        original = _state_contract(model)
        identities = {name: id(value) for name, value in model.named_parameters()}
        for owner in (model, *(layer.wrapped for layer in model.layers)):
            del owner._export_rotary_emb
            del owner._export_rope_contract_version
            owner.register_buffer(
                "rope_cos_template", torch.ones(1, 16, 16), persistent=False
            )
            owner.register_buffer(
                "rope_sin_template", torch.zeros(1, 16, 16), persistent=False
            )
        with self.assertRaisesRegex(ValueError, "migrate_llama_rope_state"):
            prepare_llama_static_inputs(model, 16)
        report = migrate_llama_rope_state(model)
        self.assertEqual(report["roots"][0]["source"], "recorded_config")
        self.assertEqual(original, _state_contract(model))
        self.assertEqual(
            identities, {name: id(value) for name, value in model.named_parameters()}
        )
        full = prepare_llama_static_inputs(model, 16)
        for layer in model.layers:
            result = prepare_llama_static_inputs(layer.wrapped, 16)
            torch.testing.assert_close(full.cos, result.cos, atol=0, rtol=0)
            torch.testing.assert_close(full.sin, result.sin, atol=0, rtol=0)
            self.assertFalse(hasattr(layer.wrapped, "rope_cos_template"))
        migrate_llama_rope_state(model)
        self.assertEqual(original, _state_contract(model))

    def test_migration_rejects_calibration_and_prepared_export(self):
        """Reject lifecycle transitions that could invalidate active work."""
        model = self._wrapper(self._model())
        model.enable_calibration()
        with self.assertRaisesRegex(ValueError, "calibration"):
            migrate_llama_rope_state(model)
        model = self._wrapper(self._model())
        with prepared_static_export(model, 8):
            with self.assertRaisesRegex(ValueError, "static export"):
                migrate_llama_rope_state(model)

    def test_layer_export_graph_has_no_dynamic_rope_generation(self):
        """The new source must not insert rotary calculations into the layer IR."""
        model = self._wrapper(self._model(), "npu_export")
        layer = cast(QuantLlamaDecoderLayer, model.layers[0].wrapped)
        prepared = prepare_llama_static_inputs(layer, 8)
        hidden = torch.randn(1, 8, 32)
        kwargs = {
            "attention_mask": prepared.mask(8, 8, 0, device="cpu"),
            "position_embeddings": (prepared.cos, prepared.sin),
        }
        adapter = layer.as_export_module("prefill", return_kv=True)
        program = torch.export.export(adapter, (hidden,), kwargs=kwargs, strict=False)
        targets = [
            str(node.target)
            for node in program.graph.nodes
            if node.op == "call_function"
        ]
        for target in targets:
            self.assertFalse(
                any(op in target for op in ("aten.sin.", "aten.cos.", "aten.arange.")),
                target,
            )
        with torch.no_grad():
            for actual, expected in zip(
                program.module()(hidden, **kwargs), adapter(hidden, **kwargs)
            ):
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
