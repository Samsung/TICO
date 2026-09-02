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

"""Static export adapters for the Gemma4 assistant draft-one core.

The core graph is tensor-only: no dictionaries, no HF output objects, no
dynamic top-k/gather/scatter, and no assistant-owned KV cache. The ordered
sparse LM head runs on the host from ``assistant_hidden`` and
``centroid_logits``.
"""

from typing import Tuple

import torch
import torch.nn as nn

from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.wrappers.gemma4_assistant.quant_for_causal_lm import (
    QuantGemma4AssistantForCausalLM,
)
from tico.quantization.wrapq.wrappers.gemma4_assistant.static_inputs import (
    GEMMA4_ASSISTANT_CORE_INPUT_NAMES,
    GEMMA4_ASSISTANT_CORE_OUTPUT_NAMES,
)


def _resolve_quant_assistant(model: nn.Module) -> QuantGemma4AssistantForCausalLM:
    """Return the assistant quant wrapper hidden behind an optional PTQWrapper."""
    wrapped = getattr(model, "wrapped", model)
    if not isinstance(wrapped, QuantGemma4AssistantForCausalLM):
        raise TypeError(
            "Gemma4 assistant core export requires a "
            "QuantGemma4AssistantForCausalLM (optionally inside a PTQWrapper), "
            f"got {type(model).__name__}."
        )
    return wrapped


class Gemma4AssistantCoreExportAdapter(nn.Module):
    """Export adapter for the batch=1, q_len=1 assistant draft-one core.

    Input contract (see ``GEMMA4_ASSISTANT_CORE_INPUT_NAMES``):
        ``assistant_input``          (1, 1, 2 * backbone_hidden_size)
        ``full_key``/``full_value``  (1, kv_heads, FULL_KV_LEN, full_head_dim)
        ``sliding_key``/``sliding_value``
                                     (1, kv_heads, SLIDING_KV_LEN, head_dim)
        ``full_attention_mask``      (1, 1, 1, FULL_KV_LEN) additive
        ``sliding_attention_mask``   (1, 1, 1, SLIDING_KV_LEN) additive
        ``full_cos``/``full_sin``    (1, 1, full_head_dim)
        ``sliding_cos``/``sliding_sin``
                                     (1, 1, head_dim)

    Output contract (see ``GEMMA4_ASSISTANT_CORE_OUTPUT_NAMES``):
        ``projected_state``  (1, 1, backbone_hidden_size)
        ``assistant_hidden`` (1, 1, hidden_size)
        ``centroid_logits``  (1, 1, num_centroids)
    """

    input_names = GEMMA4_ASSISTANT_CORE_INPUT_NAMES
    output_names = GEMMA4_ASSISTANT_CORE_OUTPUT_NAMES

    def __init__(self, quant_model: nn.Module, *, allow_float: bool = False):
        super().__init__()
        assistant = _resolve_quant_assistant(quant_model)
        if assistant._mode is not Mode.QUANT and not (
            allow_float and assistant._mode is Mode.NO_QUANT
        ):
            raise RuntimeError(
                "Gemma4 assistant core export requires a converted (QUANT) "
                f"assistant, got mode {assistant._mode}. Run prepare → "
                "calibrate → convert first, or pass allow_float=True for a "
                "floating-point reference export."
            )
        if assistant.masked_embedding is None:
            raise ValueError(
                "Gemma4 assistant core export requires "
                "use_ordered_embeddings=True; the ordered sparse head defines "
                "the centroid_logits output of the core graph."
            )
        if set(assistant.model.unique_layer_types) != {
            "full_attention",
            "sliding_attention",
        }:
            raise ValueError(
                "Gemma4 assistant core export requires both full_attention "
                "and sliding_attention layer types, got "
                f"{assistant.model.unique_layer_types}."
            )

        self.pre_projection = assistant.pre_projection
        self.backbone = assistant.model
        self.post_projection = assistant.post_projection
        self.centroids = assistant.masked_embedding.centroids

    def forward(
        self,
        assistant_input: torch.Tensor,
        full_key: torch.Tensor,
        full_value: torch.Tensor,
        sliding_key: torch.Tensor,
        sliding_value: torch.Tensor,
        full_attention_mask: torch.Tensor,
        sliding_attention_mask: torch.Tensor,
        full_cos: torch.Tensor,
        full_sin: torch.Tensor,
        sliding_cos: torch.Tensor,
        sliding_sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the fixed-shape draft-one core graph."""
        hidden_states = self.pre_projection(assistant_input)
        hidden_states = self.backbone(
            hidden_states,
            attention_masks={
                "full_attention": full_attention_mask,
                "sliding_attention": sliding_attention_mask,
            },
            shared_kv_states={
                "full_attention": (full_key, full_value),
                "sliding_attention": (sliding_key, sliding_value),
            },
            position_embeddings={
                "full_attention": (full_cos, full_sin),
                "sliding_attention": (sliding_cos, sliding_sin),
            },
        )
        projected_state = self.post_projection(hidden_states)
        centroid_logits = self.centroids(hidden_states)
        return projected_state, hidden_states, centroid_logits
