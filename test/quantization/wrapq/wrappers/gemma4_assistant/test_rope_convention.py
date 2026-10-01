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

"""Assistant eager/static sine conventions and shared-KV invariants."""

import copy
import dataclasses
import unittest
from typing import Any, cast

import torch

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.recipes.debug.wrapper_smoke.cases.gemma4_assistant import (
    make_tiny_gemma4_assistant_model,
)
from tico.quantization.wrapq.wrappers.gemma4.rope import prepare_gemma4_rope_sin
from tico.quantization.wrapq.wrappers.gemma4_assistant.quant_backbone import (
    QuantGemma4AssistantBackbone,
)
from tico.quantization.wrapq.wrappers.gemma4_assistant.static_inputs import (
    canonicalize_gemma4_assistant_static_inputs,
    Gemma4AssistantStaticShapeConfig,
)


def _available():
    # Use the same optional-dependency probe as the existing smoke case.
    try:
        from transformers.models.gemma4_assistant.modeling_gemma4_assistant import (
            Gemma4AssistantForCausalLM,
        )
    except ImportError:
        return False
    return Gemma4AssistantForCausalLM is not None


@unittest.skipUnless(_available(), "transformers Gemma4 assistant is not installed")
class TestGemma4AssistantRoPEConvention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(915)
        self.fp = make_tiny_gemma4_assistant_model().eval()
        self.shape = Gemma4AssistantStaticShapeConfig(
            full_kv_length=8, sliding_kv_length=5
        )
        self.positions = torch.tensor([[4]])
        # ``nn.Module.__getattr__`` types submodules as ``Tensor | Module``.
        self.config = cast(Any, self.fp.config)
        self.backbone = cast(torch.nn.Module, self.fp.model)
        self.rotary_emb = cast(Any, self.backbone).rotary_emb
        self.inputs = torch.randn(1, 1, 2 * self.config.backbone_hidden_size)
        self.shared = {
            layer_type: (
                torch.randn(1, 1, 5, width),
                torch.randn(1, 1, 5, width),
            )
            for layer_type, width in (("full_attention", 32), ("sliding_attention", 16))
        }

    def _canonical(self, rope=None):
        return canonicalize_gemma4_assistant_static_inputs(
            inputs_embeds=self.inputs,
            position_ids=self.positions,
            shared_kv_states=self.shared,
            shape=self.shape,
            model_or_config=self.config,
            rotary_emb=self.rotary_emb,
            **({} if rope is None else {"rope": rope}),
        )

    def test_npu_profile_and_static_default_match_without_rope_override(self):
        qcfg = PTQConfig(model_args={"profile": "npu_export"})
        backbone = QuantGemma4AssistantBackbone(
            copy.deepcopy(self.backbone), qcfg=qcfg
        ).eval()
        self.assertEqual(backbone.rope_convention, "pre_negated_sin")
        for layer in backbone.layers:
            decoder = cast(Any, layer.wrapped)
            self.assertEqual(
                decoder.self_attn.wrapped.attn_options.rope, "pre_negated_sin"
            )
        default_static = self._canonical()
        explicit_static = self._canonical("pre_negated_sin")
        for actual, expected in zip(
            default_static.as_tuple(), explicit_static.as_tuple()
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        hidden = torch.randn(1, 1, self.config.get_text_config().hidden_size)
        eager_tables = backbone.build_position_embeddings(hidden, self.positions)
        for layer_type, pair in eager_tables.items():
            for actual, expected in zip(
                pair, default_static.position_embeddings_mapping()[layer_type]
            ):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_static_conversion_preserves_keys_values_masks_and_cosines(self):
        original_shared = copy.deepcopy(self.shared)
        hf = self._canonical("hf")
        pre = self._canonical("pre_negated_sin")
        changed = {"full_sin", "sliding_sin"}
        for field in dataclasses.fields(hf):
            name = field.name
            expected = getattr(hf, name)
            if name in changed:
                expected = prepare_gemma4_rope_sin(expected, "pre_negated_sin")
            torch.testing.assert_close(getattr(pre, name), expected, rtol=0, atol=0)
        for layer_type in self.shared:
            for actual, expected in zip(
                self.shared[layer_type], original_shared[layer_type]
            ):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_backbone_eager_and_explicit_static_tables_match(self):
        hidden = torch.randn(1, 1, self.config.get_text_config().hidden_size)
        for rope in ("hf", "pre_negated_sin"):
            with self.subTest(rope=rope), torch.no_grad():
                qcfg = PTQConfig(model_args={"attention": {"rope": rope}})
                backbone = QuantGemma4AssistantBackbone(
                    copy.deepcopy(self.backbone), qcfg=qcfg
                ).eval()
                static = self._canonical(rope)
                tables = backbone.build_position_embeddings(hidden, self.positions)
                static_tables = static.position_embeddings_mapping()
                for layer_type, pair in tables.items():
                    for actual, expected in zip(pair, static_tables[layer_type]):
                        torch.testing.assert_close(actual, expected)
                kwargs = {
                    "attention_masks": static.attention_mask_mapping(),
                    "shared_kv_states": static.shared_kv_mapping(),
                }
                eager = backbone(hidden, position_ids=self.positions, **kwargs)
                explicit = backbone(hidden, position_embeddings=static_tables, **kwargs)
                torch.testing.assert_close(eager, explicit, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
