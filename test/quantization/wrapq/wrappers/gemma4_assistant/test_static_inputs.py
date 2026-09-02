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

"""Tests for the Gemma4 assistant static-shape input canonicalization."""

import copy
import unittest

import torch

from tico.quantization import prepare
from tico.quantization.config.gemma4_assistant_builders import (
    build_gemma4_assistant_ptq_config,
)


_SKIP_MSG = "required transformers Gemma4 assistant modules are not installed"


def _has_gemma4_assistant() -> bool:
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
    from tico.quantization.wrapq.wrappers.gemma4_assistant.static_inputs import (
        canonicalize_gemma4_assistant_static_inputs,
        GEMMA4_ASSISTANT_CORE_INPUT_NAMES,
        Gemma4AssistantStaticShapeConfig,
    )
else:
    make_tiny_gemma4_assistant_model = None  # type: ignore[assignment]
    canonicalize_gemma4_assistant_static_inputs = None  # type: ignore[assignment]
    GEMMA4_ASSISTANT_CORE_INPUT_NAMES = None  # type: ignore[assignment]
    Gemma4AssistantStaticShapeConfig = None  # type: ignore[assignment, misc]


def _make_dynamic_inputs(
    model: torch.nn.Module,
    *,
    kv_len: int,
    attention_mask: torch.Tensor | None = None,
) -> dict:
    """Create dynamic HF assistant inputs for one draft-one step."""
    text_cfg = model.config.get_text_config()
    kv_heads = int(text_cfg.num_key_value_heads)
    if attention_mask is None:
        attention_mask = torch.ones(1, kv_len, dtype=torch.long)
    return {
        "inputs_embeds": torch.randn(1, 1, 2 * int(model.config.backbone_hidden_size)),
        "position_ids": torch.tensor([[kv_len - 1]]),
        "attention_mask": attention_mask,
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
class TestGemma4AssistantStaticCanonicalization(unittest.TestCase):
    """Dynamic HF execution vs canonicalized fixed-shape execution."""

    def setUp(self):
        torch.manual_seed(2026)
        self.fp_model = make_tiny_gemma4_assistant_model()
        self.window = int(self.fp_model.config.get_text_config().sliding_window)

    def _canonicalize(self, inputs, shape):
        return canonicalize_gemma4_assistant_static_inputs(
            inputs_embeds=inputs["inputs_embeds"],
            position_ids=inputs["position_ids"],
            attention_mask=inputs["attention_mask"],
            shared_kv_states=inputs["shared_kv_states"],
            shape=shape,
            model_or_config=self.fp_model.config,
            rotary_emb=self.fp_model.model.rotary_emb,
        )

    def _assert_dynamic_static_parity(self, inputs, shape):
        """Padded fixed-shape execution must match the dynamic HF output."""
        with torch.no_grad():
            ref = self.fp_model(**inputs)

        static = self._canonicalize(inputs, shape)
        prepared = prepare(
            copy.deepcopy(self.fp_model),
            build_gemma4_assistant_ptq_config(num_hidden_layers=2),
        )
        with torch.no_grad():
            out = prepared(
                inputs_embeds=static.assistant_input,
                attention_mask=static.attention_mask_mapping(),
                shared_kv_states=static.shared_kv_mapping(),
                position_embeddings=static.position_embeddings_mapping(),
            )

        torch.testing.assert_close(
            out.last_hidden_state, ref.last_hidden_state, atol=1e-5, rtol=1e-5
        )
        # The HF scatter fill value is data dependent, so compare the draft
        # decision and the logits at valid vocabulary entries instead of the
        # padded fill.
        self.assertEqual(
            int(out.logits.argmax(-1).item()), int(ref.logits.argmax(-1).item())
        )
        torch.testing.assert_close(
            out.logits.max(), ref.logits.max(), atol=1e-5, rtol=1e-5
        )
        return static

    def test_parity_without_padding(self):
        inputs = _make_dynamic_inputs(self.fp_model, kv_len=8)
        shape = Gemma4AssistantStaticShapeConfig(
            full_kv_length=16, sliding_kv_length=16
        )
        static = self._assert_dynamic_static_parity(inputs, shape)
        self.assertEqual(static.full_key.shape[2], 16)
        self.assertEqual(static.sliding_key.shape[2], 16)

    def test_parity_with_left_padding(self):
        mask = torch.ones(1, 8, dtype=torch.long)
        mask[:, :3] = 0
        inputs = _make_dynamic_inputs(self.fp_model, kv_len=8, attention_mask=mask)
        shape = Gemma4AssistantStaticShapeConfig(
            full_kv_length=16, sliding_kv_length=16
        )
        self._assert_dynamic_static_parity(inputs, shape)

    def test_sliding_overcapacity_raises_error(self):
        """Inputs that exceed sliding capacity must fail, not silently crop."""
        inputs = _make_dynamic_inputs(self.fp_model, kv_len=12)
        # window=4 → visible sliding span is the last 5 positions; capacity 5
        # Input has 12 positions > capacity, must raise.
        shape = Gemma4AssistantStaticShapeConfig(
            full_kv_length=16, sliding_kv_length=self.window + 1
        )
        with self.assertRaisesRegex(ValueError, "Sliding shared KV exceeds"):
            self._canonicalize(inputs, shape)

    def test_padded_kv_does_not_change_output(self):
        """Values in padded KV slots must be fully masked out.

        Canonicalization zero-fills the padded slots; the additive mask must
        also suppress any calibration-scale values that could appear there,
        so the padded region cannot contribute to attention.
        """
        inputs = _make_dynamic_inputs(self.fp_model, kv_len=10)
        shape = Gemma4AssistantStaticShapeConfig(
            full_kv_length=16, sliding_kv_length=16
        )
        static = self._canonicalize(inputs, shape)
        self.assertEqual(static.full_key[:, :, 10:, :].abs().max().item(), 0.0)
        prepared = prepare(
            copy.deepcopy(self.fp_model),
            build_gemma4_assistant_ptq_config(num_hidden_layers=2),
        )

        poisoned_kv = static.shared_kv_mapping()
        full_key, full_value = poisoned_kv["full_attention"]
        full_key = full_key.clone()
        full_value = full_value.clone()
        full_key[:, :, 10:, :] = torch.randn_like(full_key[:, :, 10:, :])
        full_value[:, :, 10:, :] = torch.randn_like(full_value[:, :, 10:, :])
        poisoned_kv["full_attention"] = (full_key, full_value)

        with torch.no_grad():
            base = prepared(
                inputs_embeds=static.assistant_input,
                attention_mask=static.attention_mask_mapping(),
                shared_kv_states=static.shared_kv_mapping(),
                position_embeddings=static.position_embeddings_mapping(),
            )
            poisoned = prepared(
                inputs_embeds=static.assistant_input,
                attention_mask=static.attention_mask_mapping(),
                shared_kv_states=poisoned_kv,
                position_embeddings=static.position_embeddings_mapping(),
            )
        torch.testing.assert_close(base.last_hidden_state, poisoned.last_hidden_state)

    def test_as_tuple_matches_abi_order(self):
        inputs = _make_dynamic_inputs(self.fp_model, kv_len=10)
        shape = Gemma4AssistantStaticShapeConfig(
            full_kv_length=16, sliding_kv_length=16
        )
        static = self._canonicalize(inputs, shape)
        flattened = static.as_tuple()
        self.assertEqual(len(flattened), len(GEMMA4_ASSISTANT_CORE_INPUT_NAMES))
        for name, tensor in zip(GEMMA4_ASSISTANT_CORE_INPUT_NAMES, flattened):
            self.assertIs(getattr(static, name), tensor)

    def test_invalid_batch_is_rejected(self):
        inputs = _make_dynamic_inputs(self.fp_model, kv_len=10)
        inputs["inputs_embeds"] = inputs["inputs_embeds"].repeat(2, 1, 1)
        shape = Gemma4AssistantStaticShapeConfig(
            full_kv_length=16, sliding_kv_length=16
        )
        with self.assertRaisesRegex(ValueError, "batch"):
            self._canonicalize(inputs, shape)

    def test_invalid_query_length_is_rejected(self):
        inputs = _make_dynamic_inputs(self.fp_model, kv_len=10)
        inputs["inputs_embeds"] = inputs["inputs_embeds"].repeat(1, 2, 1)
        shape = Gemma4AssistantStaticShapeConfig(
            full_kv_length=16, sliding_kv_length=16
        )
        with self.assertRaisesRegex(ValueError, "query length"):
            self._canonicalize(inputs, shape)

    def test_over_capacity_full_kv_is_rejected(self):
        inputs = _make_dynamic_inputs(self.fp_model, kv_len=20)
        shape = Gemma4AssistantStaticShapeConfig(
            full_kv_length=16, sliding_kv_length=16
        )
        with self.assertRaisesRegex(ValueError, "static capacity"):
            self._canonicalize(inputs, shape)

    def test_sliding_capacity_below_visible_window_is_rejected(self):
        """A capacity that could drop visible positions must not validate."""
        shape = Gemma4AssistantStaticShapeConfig(
            full_kv_length=16, sliding_kv_length=self.window
        )
        with self.assertRaisesRegex(ValueError, "sliding_kv_length"):
            shape.validate(self.fp_model.config.get_text_config())

    def test_missing_shared_kv_entry_is_rejected(self):
        inputs = _make_dynamic_inputs(self.fp_model, kv_len=10)
        del inputs["shared_kv_states"]["sliding_attention"]
        shape = Gemma4AssistantStaticShapeConfig(
            full_kv_length=16, sliding_kv_length=16
        )
        with self.assertRaisesRegex(ValueError, "sliding_attention"):
            self._canonicalize(inputs, shape)


if __name__ == "__main__":
    unittest.main()
