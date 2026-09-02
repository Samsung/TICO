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

"""Host-side ordered sparse LM head for the Gemma4 assistant.

The NPU core graph ends at ``centroid_logits``. Dynamic top-k centroid
selection, token-ordering gathers, the selected-row matmul, and the candidate
argmax stay on the host. These helpers mirror
``Gemma4AssistantMaskedEmbedder`` exactly for the selected token set, without
materializing the full-vocabulary scatter tensor.
"""

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn


SPARSE_HEAD_ARTIFACT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SparseCandidateResult:
    """Sparse candidate logits selected by the ordered LM head.

    Attributes:
        selected_token_ids: Canonical token ids with shape ``(B, L, N)`` where
            ``N = centroid_top_k * vocab_per_centroid``.
        selected_logits: Logits for the selected tokens, shape ``(B, L, N)``.
        top_k_centroid_indices: Selected centroid indices, shape ``(B, L, K)``.
    """

    selected_token_ids: torch.Tensor
    selected_logits: torch.Tensor
    top_k_centroid_indices: torch.Tensor


def select_sparse_candidates(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    *,
    centroid_logits: torch.Tensor,
    token_ordering: torch.Tensor,
    num_centroids: int,
    centroid_top_k: int,
) -> SparseCandidateResult:
    """Select candidate tokens and logits from the ordered sparse head.

    Mirrors ``Gemma4AssistantMaskedEmbedder.forward`` up to (but excluding)
    the full-vocabulary scatter.
    """
    if hidden_states.dim() != 3:
        raise ValueError(
            "Gemma4 assistant sparse head expects hidden states shaped "
            f"(B, L, hidden), got {tuple(hidden_states.shape)}."
        )
    vocab_size = int(token_ordering.numel())
    if vocab_size % int(num_centroids):
        raise ValueError(
            "token_ordering length must be divisible by num_centroids: "
            f"vocab={vocab_size}, num_centroids={num_centroids}."
        )
    vocab_per_centroid = vocab_size // int(num_centroids)

    batch, seq_len = hidden_states.shape[:2]
    _, top_k_indices = torch.topk(centroid_logits, k=int(centroid_top_k), dim=-1)
    canonical_positions_per_cluster = token_ordering.long().view(
        int(num_centroids), vocab_per_centroid
    )

    selected_canonical = canonical_positions_per_cluster[top_k_indices]
    selected_flat = selected_canonical.reshape(-1)
    selected_embeddings = lm_head_weight[selected_flat].view(
        batch,
        seq_len,
        int(centroid_top_k) * vocab_per_centroid,
        hidden_states.shape[-1],
    )
    selected_logits = (
        hidden_states.unsqueeze(-2) @ selected_embeddings.transpose(-1, -2)
    ).squeeze(-2)

    return SparseCandidateResult(
        selected_token_ids=selected_canonical.view(batch, seq_len, -1),
        selected_logits=selected_logits,
        top_k_centroid_indices=top_k_indices,
    )


def sparse_top1_token(result: SparseCandidateResult) -> torch.Tensor:
    """Return the canonical top-1 token id per position, shape ``(B, L)``.

    Matches HF behavior: when logits tie, the selected token with the smallest
    canonical id is chosen (not the first occurrence in the selected array).
    """
    max_logits = result.selected_logits.amax(dim=-1, keepdim=True)
    sentinel = torch.iinfo(result.selected_token_ids.dtype).max
    tied_token_ids = torch.where(
        result.selected_logits == max_logits,
        result.selected_token_ids,
        torch.full_like(result.selected_token_ids, sentinel),
    )
    return tied_token_ids.amin(dim=-1)


def scatter_full_vocab_logits(
    result: SparseCandidateResult,
    *,
    vocab_size: int,
    dtype: torch.dtype,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Reconstruct full-vocabulary logits for debugging and HF parity only.

    This mirrors the masked fill value used by
    ``Gemma4AssistantMaskedEmbedder`` (``selected_logits.min() - 1``). It must
    never be part of the NPU core graph.
    """
    batch, seq_len = result.selected_logits.shape[:2]
    mask_value = result.selected_logits.min().item() - 1.0
    output = torch.full(
        (batch, seq_len, int(vocab_size)),
        fill_value=mask_value,
        dtype=dtype,
        device=device,
    )
    return output.scatter_(
        dim=-1,
        index=result.selected_token_ids.to(output.device),
        src=result.selected_logits.to(device=output.device, dtype=dtype),
    )


def _dequantize_affine(
    int_weight: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    *,
    channel_axis: int | None,
) -> torch.Tensor:
    """Dequantize an integer weight stored by the sparse-head artifact."""
    weight = int_weight.to(torch.float32)
    if channel_axis is None:
        return (weight - zero_point.to(torch.float32)) * scale.to(torch.float32)

    view_shape = [1] * weight.dim()
    view_shape[int(channel_axis)] = -1
    scale = scale.to(torch.float32).reshape(view_shape)
    zero_point = zero_point.to(torch.float32).reshape(view_shape)
    return (weight - zero_point) * scale


class Gemma4AssistantSparseHead(nn.Module):
    """Host (CPU) ordered sparse LM head fed by the NPU core outputs.

    Inputs are the two NPU core outputs ``assistant_hidden`` and
    ``centroid_logits``. The head owns the (dequantized) tied LM-head weight
    and the integer ``token_ordering`` metadata.
    """

    def __init__(
        self,
        *,
        lm_head_weight: torch.Tensor | None = None,
        token_ordering: torch.Tensor,
        num_centroids: int,
        centroid_top_k: int,
        lm_head_weight_int: torch.Tensor | None = None,
        lm_head_weight_scale: torch.Tensor | None = None,
        lm_head_weight_zero_point: torch.Tensor | None = None,
        lm_head_weight_channel_axis: int | None = None,
    ):
        super().__init__()
        is_quantized = lm_head_weight_int is not None
        if is_quantized and lm_head_weight is not None:
            raise ValueError(
                "Cannot specify both lm_head_weight (dequantized) and "
                "lm_head_weight_int; choose one path."
            )
        if not is_quantized and lm_head_weight is None:
            raise ValueError(
                "Must specify either lm_head_weight (dequantized) or "
                "lm_head_weight_int (quantized); exactly one is required."
            )

        if lm_head_weight_int is not None:
            if lm_head_weight_scale is None or lm_head_weight_zero_point is None:
                raise ValueError(
                    "lm_head_weight_int requires both lm_head_weight_scale and "
                    "lm_head_weight_zero_point."
                )
            vocab_size = int(lm_head_weight_int.shape[0])
            hidden_size = None
        else:
            assert lm_head_weight is not None
            if lm_head_weight.dim() != 2:
                raise ValueError(
                    "Sparse-head lm_head_weight must be rank 2 (vocab, hidden), "
                    f"got {tuple(lm_head_weight.shape)}."
                )
            vocab_size = int(lm_head_weight.shape[0])
            hidden_size = int(lm_head_weight.shape[1])
            self.register_buffer("lm_head_weight", lm_head_weight.detach().clone())

        if token_ordering.numel() != vocab_size:
            raise ValueError(
                "token_ordering length must match the LM-head vocabulary: "
                f"ordering={token_ordering.numel()}, "
                f"vocab={vocab_size}."
            )

        self.register_buffer("token_ordering", token_ordering.detach().clone().long())
        self.num_centroids = int(num_centroids)
        self.centroid_top_k = int(centroid_top_k)
        self.vocab_size = vocab_size
        self.vocab_per_centroid = self.vocab_size // self.num_centroids

        if lm_head_weight_int is not None:
            assert lm_head_weight_scale is not None
            assert lm_head_weight_zero_point is not None
            self.register_buffer(
                "_lm_head_weight_int", lm_head_weight_int.detach().clone()
            )
            self.register_buffer(
                "_lm_head_weight_scale", lm_head_weight_scale.detach().clone()
            )
            self.register_buffer(
                "_lm_head_weight_zero_point", lm_head_weight_zero_point.detach().clone()
            )
            self._lm_head_weight_channel_axis = lm_head_weight_channel_axis
            self.hidden_size = None
        else:
            self.hidden_size = hidden_size

    @classmethod
    def from_artifact(cls, artifact: Mapping[str, Any]) -> "Gemma4AssistantSparseHead":
        """Build a host sparse head from a saved sparse-head artifact."""
        schema = int(artifact.get("schema_version", -1))
        if schema != SPARSE_HEAD_ARTIFACT_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported Gemma4 assistant sparse-head artifact schema: "
                f"got {schema}, expected {SPARSE_HEAD_ARTIFACT_SCHEMA_VERSION}."
            )
        return cls(
            token_ordering=artifact["token_ordering"],
            num_centroids=int(artifact["num_centroids"]),
            centroid_top_k=int(artifact["centroid_top_k"]),
            lm_head_weight_int=artifact["lm_head_weight_int"],
            lm_head_weight_scale=artifact["lm_head_weight_scale"],
            lm_head_weight_zero_point=artifact["lm_head_weight_zero_point"],
            lm_head_weight_channel_axis=artifact.get("lm_head_weight_channel_axis"),
        )

    def select_candidates(
        self,
        assistant_hidden: torch.Tensor,
        centroid_logits: torch.Tensor,
    ) -> SparseCandidateResult:
        """Select sparse candidates from the NPU core outputs."""
        lm_head_weight = self._get_dequantized_weight()
        return select_sparse_candidates(
            assistant_hidden,
            lm_head_weight,
            centroid_logits=centroid_logits,
            token_ordering=self.token_ordering,
            num_centroids=self.num_centroids,
            centroid_top_k=self.centroid_top_k,
        )

    def _get_dequantized_weight(self) -> torch.Tensor:
        """Get the dequantized weight, materializing only selected rows if quantized."""
        if hasattr(self, "lm_head_weight"):
            return self.lm_head_weight
        return _dequantize_affine(
            self._lm_head_weight_int,
            self._lm_head_weight_scale,
            self._lm_head_weight_zero_point,
            channel_axis=self._lm_head_weight_channel_axis,
        )

    def forward(
        self,
        assistant_hidden: torch.Tensor,
        centroid_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Return the canonical next-token id per position, shape ``(B, L)``."""
        return sparse_top1_token(
            self.select_candidates(assistant_hidden, centroid_logits)
        )

    def full_logits(
        self,
        assistant_hidden: torch.Tensor,
        centroid_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Debug/reference full-vocabulary logits reconstruction."""
        return scatter_full_vocab_logits(
            self.select_candidates(assistant_hidden, centroid_logits),
            vocab_size=self.vocab_size,
            dtype=assistant_hidden.dtype,
            device=assistant_hidden.device,
        )
