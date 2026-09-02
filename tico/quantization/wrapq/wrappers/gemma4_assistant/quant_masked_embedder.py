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

"""PTQ wrapper for the Gemma4 assistant ordered sparse LM head."""

from typing import Iterable, Optional

import torch
import torch.nn as nn

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.utils.utils import join_name
from tico.quantization.wrapq.wrappers.gemma4_assistant.sparse_head import (
    scatter_full_vocab_logits,
    select_sparse_candidates,
    SparseCandidateResult,
)
from tico.quantization.wrapq.wrappers.gemma4_assistant.utils import (
    HF_GEMMA4_ASSISTANT_MASKED_EMBEDDER_CLASS_PATH,
)
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import try_register


@try_register(HF_GEMMA4_ASSISTANT_MASKED_EMBEDDER_CLASS_PATH)
class QuantGemma4AssistantMaskedEmbedder(QuantModuleBase):
    """PTQ wrapper for ``Gemma4AssistantMaskedEmbedder``.

    Only the centroid projection is quantized tensor compute; it belongs to
    the NPU core graph via :meth:`forward_centroid_logits`. Top-k selection,
    token-ordering gathers, the selected-row matmul, and the full-vocabulary
    scatter are dynamic host operations kept for eager Hugging Face parity
    only and must never be exported. ``token_ordering`` is integer metadata
    and is intentionally not quantized.
    """

    def __init__(
        self,
        fp_embedder: nn.Module,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
    ):
        super().__init__(qcfg, fp_name=fp_name)
        self.module = fp_embedder
        self.num_centroids = int(fp_embedder.num_centroids)
        self.centroid_top_k = int(fp_embedder.centroid_intermediate_top_k)
        self.vocab_size = int(fp_embedder.vocab_size)
        self.hidden_size = int(fp_embedder.hidden_size)
        self.vocab_per_centroid = int(fp_embedder.vocab_size_per_centroid)

        self.centroids = PTQWrapper(
            fp_embedder.centroids,
            qcfg=qcfg.child("centroids") if qcfg else None,
            fp_name=join_name(fp_name, "centroids"),
        )

    @property
    def token_ordering(self) -> torch.Tensor:
        """Return the integer token-ordering metadata (never quantized)."""
        return self.module.token_ordering

    def forward_centroid_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the static centroid projection (NPU-exportable path)."""
        return self.centroids(hidden_states)

    def select_candidates(
        self,
        hidden_states: torch.Tensor,
        lm_head_weight: torch.Tensor,
    ) -> SparseCandidateResult:
        """Select sparse candidate tokens and logits (host-side path)."""
        centroid_logits = self.forward_centroid_logits(hidden_states)
        return select_sparse_candidates(
            hidden_states,
            lm_head_weight,
            centroid_logits=centroid_logits,
            token_ordering=self.token_ordering,
            num_centroids=self.num_centroids,
            centroid_top_k=self.centroid_top_k,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        lm_head_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Return full-vocabulary logits with HF eager semantics.

        The full scatter exists only so the eager wrapper can plug into
        Hugging Face assisted generation. Static export uses
        :meth:`forward_centroid_logits` plus the host sparse head instead.
        """
        return scatter_full_vocab_logits(
            self.select_candidates(hidden_states, lm_head_weight),
            vocab_size=self.vocab_size,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

    def _all_observers(self) -> Iterable:
        """Return observers owned directly by this wrapper."""
        return ()
