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

"""Shared helpers for the Gemma4 assistant (MTP draft) wrapper family."""

from typing import Any, Mapping

import torch
import torch.nn as nn

from tico.quantization.wrapq.wrappers.gemma4.utils import assert_gemma4_e2b_no_moe


HF_GEMMA4_ASSISTANT_CLASS_PATH = (
    "transformers.models.gemma4_assistant.modeling_gemma4_assistant"
    ".Gemma4AssistantForCausalLM"
)
HF_GEMMA4_ASSISTANT_MASKED_EMBEDDER_CLASS_PATH = (
    "transformers.models.gemma4_assistant.modeling_gemma4_assistant"
    ".Gemma4AssistantMaskedEmbedder"
)

SUPPORTED_ASSISTANT_LAYER_TYPES = frozenset(("full_attention", "sliding_attention"))


def extract_assistant_text_config(model_or_config: Any) -> Any:
    """Return the Gemma4 text config owned by an assistant model or config."""
    config = getattr(model_or_config, "config", model_or_config)
    if hasattr(config, "get_text_config"):
        return config.get_text_config()
    return getattr(config, "text_config", config)


def assistant_layer_type_head_dim(text_config: Any, layer_type: str) -> int:
    """Return the attention head dim used by one assistant layer type.

    Mirrors ``Gemma4TextAttention.__init__``: full-attention layers use
    ``global_head_dim`` when it is set, sliding layers always use ``head_dim``.
    """
    if layer_type not in SUPPORTED_ASSISTANT_LAYER_TYPES:
        raise ValueError(f"Unsupported Gemma4 assistant layer type: {layer_type!r}.")
    global_head_dim = getattr(text_config, "global_head_dim", None)
    if layer_type == "full_attention" and global_head_dim:
        return int(global_head_dim)
    return int(text_config.head_dim)


def assistant_shared_kv_num_heads(text_config: Any) -> int:
    """Return the KV head count expected from the target shared-KV states."""
    num_heads = int(text_config.num_attention_heads)
    num_kv_heads = int(text_config.num_key_value_heads)
    if num_kv_heads <= 0 or num_heads % num_kv_heads:
        raise ValueError(
            "Invalid Gemma4 assistant GQA head configuration: "
            f"num_attention_heads={num_heads}, num_key_value_heads={num_kv_heads}."
        )
    return num_kv_heads


def validate_gemma4_assistant_architecture(model_or_config: Any) -> None:
    """Validate the Gemma4 assistant contract supported by the PTQ wrappers.

    The static assistant runtime supports the dense, fully shared-KV draft-one
    architecture only. Every violated invariant raises with an actionable
    message instead of silently degrading semantics.
    """
    assert_gemma4_e2b_no_moe(model_or_config)

    config = getattr(model_or_config, "config", model_or_config)
    text_config = extract_assistant_text_config(config)
    if text_config is None:
        raise ValueError("Gemma4 assistant config requires a text_config.")

    ple_dim = int(getattr(text_config, "hidden_size_per_layer_input", 0) or 0)
    if ple_dim != 0:
        raise ValueError(
            "Gemma4 assistant requires hidden_size_per_layer_input == 0 "
            f"(no per-layer embeddings), got {ple_dim}."
        )
    ple_vocab = int(getattr(text_config, "vocab_size_per_layer_input", 0) or 0)
    if ple_vocab != 0:
        raise ValueError(
            "Gemma4 assistant requires vocab_size_per_layer_input == 0, "
            f"got {ple_vocab}."
        )

    num_layers = int(text_config.num_hidden_layers)
    num_shared = int(getattr(text_config, "num_kv_shared_layers", 0) or 0)
    if num_shared != num_layers:
        raise ValueError(
            "Every Gemma4 assistant layer must consume target shared KV states: "
            f"num_kv_shared_layers={num_shared}, num_hidden_layers={num_layers}."
        )

    layer_types = tuple(text_config.layer_types)
    unsupported = sorted(set(layer_types) - SUPPORTED_ASSISTANT_LAYER_TYPES)
    if unsupported:
        raise ValueError(f"Unsupported Gemma4 assistant layer types: {unsupported}.")
    if layer_types and "sliding_attention" in layer_types:
        window = int(getattr(text_config, "sliding_window", 0) or 0)
        if window <= 0:
            raise ValueError(
                "Gemma4 assistant sliding_attention layers require a positive "
                f"sliding_window, got {window}."
            )

    if bool(getattr(config, "use_ordered_embeddings", False)):
        num_centroids = int(getattr(config, "num_centroids", 0) or 0)
        centroid_top_k = int(getattr(config, "centroid_intermediate_top_k", 0) or 0)
        vocab_size = int(text_config.vocab_size)
        if num_centroids <= 0:
            raise ValueError(
                "Gemma4 assistant ordered embeddings require num_centroids > 0, "
                f"got {num_centroids}."
            )
        if not 0 < centroid_top_k <= num_centroids:
            raise ValueError(
                "Gemma4 assistant centroid_intermediate_top_k must be in "
                f"(0, num_centroids]: top_k={centroid_top_k}, "
                f"num_centroids={num_centroids}."
            )
        if vocab_size % num_centroids:
            raise ValueError(
                "Gemma4 assistant ordered embeddings require vocab_size to be "
                f"divisible by num_centroids: vocab_size={vocab_size}, "
                f"num_centroids={num_centroids}."
            )

    if isinstance(model_or_config, nn.Module):
        inner_model = getattr(model_or_config, "model", None)
        layers = getattr(inner_model, "layers", None)
        if layers is not None:
            for idx, layer in enumerate(layers):
                attention = getattr(layer, "self_attn", None)
                if attention is None or not bool(
                    getattr(attention, "is_kv_shared_layer", False)
                ):
                    raise ValueError(
                        "Gemma4 assistant layers must all be shared-KV "
                        f"consumers, but layer {idx} owns its own K/V projections."
                    )
        if bool(getattr(config, "use_ordered_embeddings", False)) and (
            getattr(model_or_config, "masked_embedding", None) is None
        ):
            raise ValueError(
                "Gemma4 assistant with use_ordered_embeddings=True requires a "
                "masked_embedding module."
            )


class Gemma4AssistantGenerationAdapter(nn.Module):
    """Adapter exposing a quantized assistant to HF assisted generation.

    Hugging Face `generate()` selects the single-position MTP candidate
    generator by checking that ``assistant_model.__class__.__name__`` starts
    with ``"Gemma4Assistant"``. TICO wrapper classes follow the ``Quant*``
    naming convention, so this thin delegate carries the required class name
    while every forward call still runs through the quantized wrapper.
    """

    main_input_name = "input_ids"

    def __init__(self, quant_model: nn.Module):
        super().__init__()
        # Accept either a top-level PTQWrapper or the quant module itself.
        wrapped = getattr(quant_model, "wrapped", quant_model)
        if not type(wrapped).__name__.startswith("QuantGemma4Assistant"):
            raise TypeError(
                "Gemma4AssistantGenerationAdapter requires a "
                "QuantGemma4AssistantForCausalLM (optionally inside a "
                f"PTQWrapper), got {type(quant_model).__name__}."
            )
        self.assistant = wrapped

    def _cast_to_assistant(self, value: Any) -> Any:
        """Move floating inputs to the assistant's device and compute dtype.

        The target model may run in a lower precision (e.g. bfloat16) than
        the assistant being calibrated (float32). Integer tensors such as
        position ids and padding masks are moved but never cast.
        """
        device = self.assistant.device
        dtype = next(self.assistant.parameters()).dtype
        if isinstance(value, torch.Tensor):
            if torch.is_floating_point(value):
                return value.to(device=device, dtype=dtype)
            return value.to(device=device)
        if isinstance(value, tuple):
            return tuple(self._cast_to_assistant(item) for item in value)
        if isinstance(value, Mapping):
            return {key: self._cast_to_assistant(item) for key, item in value.items()}
        return value

    def forward(self, *args, **kwargs):
        args = tuple(self._cast_to_assistant(arg) for arg in args)
        kwargs = {key: self._cast_to_assistant(val) for key, val in kwargs.items()}
        return self.assistant(*args, **kwargs)

    @property
    def config(self):
        return self.assistant.config

    @property
    def device(self):
        return self.assistant.device

    @property
    def generation_config(self):
        return self.assistant.generation_config

    def can_generate(self) -> bool:
        return True

    def _supports_logits_to_keep(self) -> bool:
        return False

    def get_input_embeddings(self):
        return self.assistant.get_input_embeddings()

    def get_output_embeddings(self):
        return self.assistant.get_output_embeddings()
