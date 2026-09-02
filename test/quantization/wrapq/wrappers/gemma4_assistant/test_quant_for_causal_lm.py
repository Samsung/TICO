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

"""Tests for QuantGemma4AssistantForCausalLM on a tiny synthetic assistant."""

import copy
import tempfile
import unittest
from pathlib import Path

import torch

from tico.quantization import convert, prepare
from tico.quantization.config.gemma4_assistant_builders import (
    build_gemma4_assistant_ptq_config,
)


_SKIP_MSG = "required transformers Gemma4 assistant modules are not installed"


def _has_gemma4_assistant() -> bool:
    """Return whether transformers provides the Gemma4 assistant."""
    try:
        from transformers.models.gemma4_assistant.modeling_gemma4_assistant import (  # noqa: F401
            Gemma4AssistantForCausalLM,
        )
    except Exception:
        return False
    return True


if _has_gemma4_assistant():
    from tico.quantization.recipes.debug.wrapper_smoke.cases.gemma4_assistant import (
        make_tiny_gemma4_assistant_model,
    )
    from tico.quantization.wrapq.wrappers.gemma4_assistant.quant_for_causal_lm import (
        QuantGemma4AssistantForCausalLM,
    )
    from tico.quantization.wrapq.wrappers.gemma4_assistant.utils import (
        Gemma4AssistantGenerationAdapter,
    )


def _make_sample(model: torch.nn.Module, kv_len: int = 10) -> dict:
    """Create one shape-valid draft-one assistant sample."""
    text_cfg = model.config.get_text_config()
    kv_heads = int(text_cfg.num_key_value_heads)
    return {
        "inputs_embeds": torch.randn(1, 1, 2 * int(model.config.backbone_hidden_size)),
        "position_ids": torch.tensor([[kv_len - 1]]),
        "attention_mask": torch.ones(1, kv_len, dtype=torch.long),
        "shared_kv_states": {
            "full_attention": (
                torch.randn(1, kv_heads, kv_len, int(text_cfg.global_head_dim)),
                torch.randn(1, kv_heads, kv_len, int(text_cfg.global_head_dim)),
            ),
            "sliding_attention": (
                torch.randn(1, kv_heads, kv_len, int(text_cfg.head_dim)),
                torch.randn(1, kv_heads, kv_len, int(text_cfg.head_dim)),
            ),
        },
        "use_cache": False,
    }


@unittest.skipUnless(_has_gemma4_assistant(), _SKIP_MSG)
class TestQuantGemma4AssistantForCausalLM(unittest.TestCase):
    """Wrapper registry, FP parity, PTQ flow, and tied-weight contracts."""

    def setUp(self):
        torch.manual_seed(2026)
        self.fp_model = make_tiny_gemma4_assistant_model()
        self.fp_ref = copy.deepcopy(self.fp_model).eval()
        self.sample = _make_sample(self.fp_model)

    def _prepare(self, model=None):
        qcfg = build_gemma4_assistant_ptq_config(num_hidden_layers=2)
        return prepare(model or self.fp_model, qcfg)

    def test_registry_maps_hf_assistant_class(self):
        """The exact HF assistant class must resolve to the new wrapper."""
        from tico.quantization.wrapq.wrappers.registry import lookup
        from transformers.models.gemma4_assistant.modeling_gemma4_assistant import (
            Gemma4AssistantForCausalLM,
        )

        self.assertIs(
            lookup(Gemma4AssistantForCausalLM), QuantGemma4AssistantForCausalLM
        )

    def test_fp_parity_matches_hf(self):
        """The prepared (FP-numerics) wrapper must match the HF assistant."""
        prepared = self._prepare()
        with torch.no_grad():
            ref = self.fp_ref(**self.sample)
            out = prepared(**self.sample)

        torch.testing.assert_close(
            out.last_hidden_state, ref.last_hidden_state, atol=1e-6, rtol=1e-6
        )
        torch.testing.assert_close(out.logits, ref.logits, atol=1e-6, rtol=1e-6)
        self.assertEqual(
            int(out.logits.argmax(-1).item()), int(ref.logits.argmax(-1).item())
        )

    def test_shared_kv_layers_own_no_kv_projection(self):
        """Every assistant attention layer must consume target shared KV."""
        prepared = self._prepare()
        wrapper = prepared.wrapped
        for layer in wrapper.model.layers:
            attention = layer.wrapped.self_attn.wrapped
            self.assertTrue(attention.is_kv_shared_layer)
            self.assertFalse(hasattr(attention, "k_proj"))
            self.assertFalse(hasattr(attention, "v_proj"))

    def test_layer_types_consume_their_own_shared_kv(self):
        """Full layers must read full KV, sliding layers sliding KV."""
        prepared = self._prepare()
        with torch.no_grad():
            base = prepared(**self.sample).last_hidden_state

        # Perturbing the sliding shared KV must change the output; the full
        # shared KV entry stays intact, so the change can only come from the
        # sliding-attention consumer layer.
        perturbed = copy.deepcopy(self.sample)
        sliding_key, sliding_value = perturbed["shared_kv_states"]["sliding_attention"]
        perturbed["shared_kv_states"]["sliding_attention"] = (
            sliding_key + 1.0,
            sliding_value,
        )
        with torch.no_grad():
            changed = prepared(**perturbed).last_hidden_state
        self.assertGreater((base - changed).abs().max().item(), 0.0)

    def test_missing_shared_kv_entry_raises(self):
        """A missing per-layer-type shared KV entry must fail loudly."""
        prepared = self._prepare()
        broken = dict(self.sample)
        broken["shared_kv_states"] = {
            "full_attention": self.sample["shared_kv_states"]["full_attention"]
        }
        with self.assertRaisesRegex(ValueError, "sliding_attention"):
            prepared(**broken)

    def test_forward_requires_embeds_and_shared_kv(self):
        """The HF contract requires inputs_embeds and shared_kv_states."""
        prepared = self._prepare()
        with self.assertRaisesRegex(ValueError, "cannot be None"):
            prepared(inputs_embeds=None, shared_kv_states=None)

    def test_output_collection_kwargs_rejected(self):
        """Unsupported per-layer output collection must not be silent."""
        prepared = self._prepare()
        with self.assertRaises(NotImplementedError):
            prepared(**self.sample, output_hidden_states=True)

    def test_ptq_flow_produces_qparams_and_draft_token(self):
        """prepare → calibrate → convert must leave no missing qparams."""
        prepared = self._prepare()
        with torch.no_grad():
            prepared(**self.sample)
            prepared(**_make_sample(self.fp_ref, kv_len=6))
        converted = convert(prepared)

        missing = [
            name
            for name, obs in converted.wrapped.named_observers()
            if hasattr(obs, "has_qparams") and not obs.has_qparams
        ]
        self.assertEqual(missing, [])

        with torch.no_grad():
            out = converted(**self.sample)
        self.assertTrue(torch.isfinite(out.logits).all())
        self.assertTrue(torch.isfinite(out.last_hidden_state).all())
        draft_token = int(out.logits.argmax(-1).item())
        self.assertTrue(0 <= draft_token < converted.wrapped.vocab_size)

    def test_tied_weight_single_source(self):
        """lm_head must be the single quantized tied-weight source."""
        prepared = self._prepare()
        wrapper = prepared.wrapped
        self.assertEqual(
            wrapper.lm_head.wrapped.module.weight.data_ptr(),
            wrapper.module.model.embed_tokens.weight.data_ptr(),
        )
        # No second wrapper quantizes the embedding copy.
        self.assertFalse(hasattr(wrapper.model, "embed_tokens"))

    def test_untied_weights_with_tied_config_rejected(self):
        """A tied config with untied tensors must be rejected."""
        broken = copy.deepcopy(self.fp_ref)
        broken.lm_head.weight = torch.nn.Parameter(
            broken.lm_head.weight.detach().clone()
        )
        with self.assertRaisesRegex(ValueError, "tie_word_embeddings"):
            self._prepare(broken)

    def test_checkpoint_round_trip_preserves_tie_and_outputs(self):
        """Save/load must keep the tied-weight invariant and outputs."""
        prepared = self._prepare()
        with torch.no_grad():
            prepared(**self.sample)
        converted = convert(prepared)
        with torch.no_grad():
            expected = converted(**self.sample)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "quantized_model.pt"
            torch.save(converted, path)
            loaded = torch.load(path, weights_only=False)

        wrapper = loaded.wrapped
        self.assertEqual(
            wrapper.lm_head.wrapped.module.weight.data_ptr(),
            wrapper.module.model.embed_tokens.weight.data_ptr(),
        )
        with torch.no_grad():
            actual = loaded(**self.sample)
        torch.testing.assert_close(actual.logits, expected.logits)
        torch.testing.assert_close(actual.last_hidden_state, expected.last_hidden_state)

    def test_generation_adapter_matches_hf_dispatch_contract(self):
        """The generation adapter must satisfy the HF MTP dispatch rules."""
        prepared = self._prepare()
        adapter = Gemma4AssistantGenerationAdapter(prepared)
        self.assertTrue(type(adapter).__name__.startswith("Gemma4Assistant"))
        self.assertIs(adapter.config, prepared.wrapped.config)

        with torch.no_grad():
            expected = prepared(**self.sample)
            actual = adapter(**self.sample)
        torch.testing.assert_close(actual.logits, expected.logits)

    def test_generation_adapter_casts_target_dtype_inputs(self):
        """bfloat16 target-side tensors must be bridged to the assistant."""
        prepared = self._prepare()
        adapter = Gemma4AssistantGenerationAdapter(prepared)
        bf16_sample = copy.deepcopy(self.sample)
        bf16_sample["inputs_embeds"] = bf16_sample["inputs_embeds"].to(torch.bfloat16)
        bf16_sample["shared_kv_states"] = {
            layer_type: (key.to(torch.bfloat16), value.to(torch.bfloat16))
            for layer_type, (key, value) in bf16_sample["shared_kv_states"].items()
        }
        with torch.no_grad():
            out = adapter(**bf16_sample)
        self.assertEqual(out.logits.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
