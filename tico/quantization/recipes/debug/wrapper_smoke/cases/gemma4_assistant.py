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

"""Smoke cases for the Gemma4 assistant (MTP draft) wrappers."""

import copy
from typing import Any, Mapping

import torch

from tico.quantization.recipes.debug.wrapper_smoke.case import (
    CaseAvailability,
    ForwardInput,
    WrapperSmokeCase,
)


def _has_gemma4_assistant() -> CaseAvailability:
    """Return whether the installed transformers provides the assistant."""
    try:
        from transformers.models.gemma4_assistant.modeling_gemma4_assistant import (  # noqa: F401
            Gemma4AssistantForCausalLM,
        )
    except Exception:
        return CaseAvailability(
            False, "transformers Gemma4 assistant modules are not installed"
        )
    return CaseAvailability(True)


def make_tiny_gemma4_assistant_config() -> Any:
    """Create a tiny synthetic Gemma4 assistant config (no download)."""
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4_assistant.configuration_gemma4_assistant import (
        Gemma4AssistantConfig,
    )

    text_cfg = Gemma4TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        global_head_dim=32,
        max_position_embeddings=128,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=4,
        rope_parameters={
            "full_attention": {
                "rope_type": "proportional",
                "partial_rotary_factor": 0.25,
                "rope_theta": 1_000_000.0,
            },
            "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
        },
        attention_bias=False,
        attention_dropout=0.0,
        use_cache=False,
        enable_moe_block=False,
        num_kv_shared_layers=2,
        hidden_size_per_layer_input=0,
        vocab_size_per_layer_input=0,
    )
    config = Gemma4AssistantConfig(
        text_config=text_cfg,
        backbone_hidden_size=24,
        use_ordered_embeddings=True,
        num_centroids=8,
        centroid_intermediate_top_k=2,
        tie_word_embeddings=True,
    )
    config._attn_implementation = "eager"
    config.text_config._attn_implementation = "eager"
    return config


def make_tiny_gemma4_assistant_model() -> torch.nn.Module:
    """Create a tiny synthetic Gemma4 assistant model (no download)."""
    from transformers.models.gemma4_assistant.modeling_gemma4_assistant import (
        Gemma4AssistantForCausalLM,
    )

    model = Gemma4AssistantForCausalLM(make_tiny_gemma4_assistant_config()).eval()
    with torch.no_grad():
        model.masked_embedding.token_ordering.copy_(
            torch.randperm(model.config.get_text_config().vocab_size)
        )
    return model


def _make_assistant_sample(model: torch.nn.Module, kv_len: int) -> ForwardInput:
    """Create one synthetic draft-one assistant sample."""
    text_cfg = model.config.get_text_config()
    kv_heads = int(text_cfg.num_key_value_heads)
    shared_kv_states = {
        "full_attention": (
            torch.randn(1, kv_heads, kv_len, int(text_cfg.global_head_dim)),
            torch.randn(1, kv_heads, kv_len, int(text_cfg.global_head_dim)),
        ),
        "sliding_attention": (
            torch.randn(1, kv_heads, kv_len, int(text_cfg.head_dim)),
            torch.randn(1, kv_heads, kv_len, int(text_cfg.head_dim)),
        ),
    }
    return ForwardInput(
        (),
        {
            "inputs_embeds": torch.randn(
                1, 1, 2 * int(model.config.backbone_hidden_size)
            ),
            "position_ids": torch.tensor([[kv_len - 1]]),
            "attention_mask": torch.ones(1, kv_len, dtype=torch.long),
            "shared_kv_states": shared_kv_states,
            "use_cache": False,
        },
    )


class Gemma4AssistantCoreCase(WrapperSmokeCase):
    """Quantize the tiny assistant draft-one core end to end."""

    name = "gemma4_assistant_core"
    description = (
        "Quantize a tiny synthetic Gemma4 assistant and export the static "
        "draft-one core (tensor-only ABI, host sparse head excluded)."
    )
    tags = ("gemma4_assistant", "assistant", "core", "shared_kv")
    kv_len = 10
    full_kv_length = 16
    sliding_kv_length = 8

    def availability(self) -> CaseAvailability:
        return _has_gemma4_assistant()

    def ptq_config(self, cfg: Mapping[str, Any]):
        from tico.quantization.config.gemma4_assistant_builders import (
            build_gemma4_assistant_ptq_config,
        )

        return build_gemma4_assistant_ptq_config(num_hidden_layers=2)

    def build(self, cfg: Mapping[str, Any]):
        torch.manual_seed(2026)
        model = make_tiny_gemma4_assistant_model()
        return model, copy.deepcopy(model)

    def calibration_inputs(self, prepared, cfg: Mapping[str, Any]):
        torch.manual_seed(2027)
        return [self._reference_model_sample(prepared) for _ in range(4)]

    def _reference_model_sample(self, prepared) -> ForwardInput:
        module = getattr(prepared, "wrapped", prepared)
        return _make_assistant_sample(module, self.kv_len)

    def output_tensor(self, output: Any) -> torch.Tensor:
        return output.last_hidden_state

    def export_module(self, quantized, cfg: Mapping[str, Any]):
        from tico.quantization.wrapq.wrappers.gemma4_assistant.export_adapters import (
            Gemma4AssistantCoreExportAdapter,
        )

        return Gemma4AssistantCoreExportAdapter(quantized)

    def export_input(self, eval_sample: ForwardInput, cfg: Mapping[str, Any]):
        from tico.quantization.wrapq.wrappers.gemma4_assistant.static_inputs import (
            canonicalize_gemma4_assistant_static_inputs,
            Gemma4AssistantStaticShapeConfig,
        )

        model = make_tiny_gemma4_assistant_model()
        kwargs = dict(eval_sample.kwargs)
        static = canonicalize_gemma4_assistant_static_inputs(
            inputs_embeds=kwargs["inputs_embeds"],
            position_ids=kwargs["position_ids"],
            attention_mask=kwargs["attention_mask"],
            shared_kv_states=kwargs["shared_kv_states"],
            shape=Gemma4AssistantStaticShapeConfig(
                full_kv_length=self.full_kv_length,
                sliding_kv_length=self.sliding_kv_length,
            ),
            model_or_config=model.config,
            rotary_emb=model.model.rotary_emb,
        )
        return ForwardInput(static.as_tuple(), {})


class Gemma4AssistantSparseHeadCase(WrapperSmokeCase):
    """Quantize the ordered sparse LM head and check HF parity."""

    name = "gemma4_assistant_sparse_head"
    description = (
        "Quantize the Gemma4 assistant ordered sparse LM head and compare "
        "against the HF masked embedder."
    )
    tags = ("gemma4_assistant", "assistant", "sparse_head")
    supports_circle_export = False
    circle_export_unsupported_reason = (
        "The ordered sparse head runs on the host: dynamic top-k, "
        "token-ordering gathers, and the full-vocabulary scatter are "
        "intentionally excluded from the NPU core graph."
    )

    def availability(self) -> CaseAvailability:
        return _has_gemma4_assistant()

    def build(self, cfg: Mapping[str, Any]):
        from transformers.models.gemma4_assistant.modeling_gemma4_assistant import (
            Gemma4AssistantMaskedEmbedder,
        )

        torch.manual_seed(2028)
        config = make_tiny_gemma4_assistant_config()
        embedder = Gemma4AssistantMaskedEmbedder(config).eval()
        with torch.no_grad():
            embedder.token_ordering.copy_(
                torch.randperm(config.get_text_config().vocab_size)
            )
        self._lm_head_weight = torch.randn(
            config.get_text_config().vocab_size,
            config.get_text_config().hidden_size,
        )
        return embedder, copy.deepcopy(embedder)

    def calibration_inputs(self, prepared, cfg: Mapping[str, Any]):
        torch.manual_seed(2029)
        hidden_size = self._lm_head_weight.shape[1]
        return [
            ForwardInput((torch.randn(1, 1, hidden_size), self._lm_head_weight), {})
            for _ in range(4)
        ]


GEMMA4_ASSISTANT_CASES = (
    Gemma4AssistantCoreCase(),
    Gemma4AssistantSparseHeadCase(),
)
