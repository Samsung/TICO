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

"""Tests for the Gemma4 assistant ordered sparse LM head."""

import unittest

import torch

from tico.quantization.wrapq.wrappers.gemma4_assistant.sparse_head import (
    Gemma4AssistantSparseHead,
    scatter_full_vocab_logits,
    select_sparse_candidates,
    SPARSE_HEAD_ARTIFACT_SCHEMA_VERSION,
    sparse_top1_token,
)


_SKIP_MSG = "required transformers Gemma4 assistant modules are not installed"


def _has_gemma4_assistant() -> bool:
    try:
        from transformers.models.gemma4_assistant.modeling_gemma4_assistant import (  # noqa: F401
            Gemma4AssistantMaskedEmbedder,
        )
    except Exception:
        return False
    return True


def _make_hf_masked_embedder():
    """Create a tiny HF masked embedder with a random token ordering."""
    from tico.quantization.recipes.debug.wrapper_smoke.cases.gemma4_assistant import (
        make_tiny_gemma4_assistant_config,
    )
    from transformers.models.gemma4_assistant.modeling_gemma4_assistant import (
        Gemma4AssistantMaskedEmbedder,
    )

    config = make_tiny_gemma4_assistant_config()
    embedder = Gemma4AssistantMaskedEmbedder(config).eval()
    with torch.no_grad():
        embedder.token_ordering.copy_(
            torch.randperm(config.get_text_config().vocab_size)
        )
    return embedder


@unittest.skipUnless(_has_gemma4_assistant(), _SKIP_MSG)
class TestGemma4AssistantSparseHead(unittest.TestCase):
    """Parity between the HF masked embedder and the host sparse head."""

    def setUp(self):
        torch.manual_seed(2030)
        self.embedder = _make_hf_masked_embedder()
        self.hidden_size = int(self.embedder.hidden_size)
        self.vocab_size = int(self.embedder.vocab_size)
        self.lm_head_weight = torch.randn(self.vocab_size, self.hidden_size)
        self.hidden = torch.randn(1, 1, self.hidden_size)

    def _select(self, hidden=None):
        hidden = self.hidden if hidden is None else hidden
        centroid_logits = self.embedder.centroids(hidden)
        return select_sparse_candidates(
            hidden,
            self.lm_head_weight,
            centroid_logits=centroid_logits,
            token_ordering=self.embedder.token_ordering,
            num_centroids=int(self.embedder.num_centroids),
            centroid_top_k=int(self.embedder.centroid_intermediate_top_k),
        )

    def test_top1_matches_hf_without_full_scatter(self):
        """The sparse top-1 must equal the HF full-vocabulary argmax."""
        with torch.no_grad():
            hf_logits = self.embedder(self.hidden, self.lm_head_weight)
        result = self._select()
        self.assertEqual(
            int(sparse_top1_token(result).item()),
            int(hf_logits.argmax(dim=-1).item()),
        )

    def test_selected_ids_and_logits_match_hf(self):
        """Selected ids and logits must match the HF scatter exactly."""
        with torch.no_grad():
            hf_logits = self.embedder(self.hidden, self.lm_head_weight)
        result = self._select()

        selected = int(self.embedder.centroid_intermediate_top_k) * int(
            self.embedder.vocab_size_per_centroid
        )
        self.assertEqual(result.selected_token_ids.shape, (1, 1, selected))

        gathered = hf_logits.gather(-1, result.selected_token_ids)
        torch.testing.assert_close(gathered, result.selected_logits)

    def test_full_scatter_reconstruction_matches_hf(self):
        """The debug full-logits reconstruction must equal HF exactly."""
        with torch.no_grad():
            hf_logits = self.embedder(self.hidden, self.lm_head_weight)
        result = self._select()
        reconstructed = scatter_full_vocab_logits(
            result, vocab_size=self.vocab_size, dtype=self.hidden.dtype
        )
        torch.testing.assert_close(reconstructed, hf_logits)

    def test_host_head_module_matches_hf(self):
        """The host sparse-head module must reproduce the HF top-1 token."""
        head = Gemma4AssistantSparseHead(
            lm_head_weight=self.lm_head_weight,
            token_ordering=self.embedder.token_ordering,
            num_centroids=int(self.embedder.num_centroids),
            centroid_top_k=int(self.embedder.centroid_intermediate_top_k),
        )
        centroid_logits = self.embedder.centroids(self.hidden)
        with torch.no_grad():
            hf_logits = self.embedder(self.hidden, self.lm_head_weight)
        self.assertEqual(
            int(head(self.hidden, centroid_logits).item()),
            int(hf_logits.argmax(dim=-1).item()),
        )

    def test_artifact_round_trip_preserves_top1(self):
        """Integer artifact save/load must reproduce the same top-1 token."""
        scale = self.lm_head_weight.abs().amax(dim=1) / 127.0
        int_weight = torch.clamp(
            torch.round(self.lm_head_weight / scale[:, None]), -128, 127
        ).to(torch.uint8)
        zp = torch.zeros_like(scale, dtype=torch.uint8)
        artifact = {
            "schema_version": SPARSE_HEAD_ARTIFACT_SCHEMA_VERSION,
            "lm_head_weight_int": int_weight,
            "lm_head_weight_scale": scale,
            "lm_head_weight_zero_point": zp,
            "lm_head_weight_channel_axis": 0,
            "token_ordering": self.embedder.token_ordering,
            "num_centroids": int(self.embedder.num_centroids),
            "centroid_top_k": int(self.embedder.centroid_intermediate_top_k),
        }
        head = Gemma4AssistantSparseHead.from_artifact(artifact)

        dequantized = int_weight.to(torch.float32) * scale[:, None]
        reference = Gemma4AssistantSparseHead(
            lm_head_weight=dequantized,
            token_ordering=self.embedder.token_ordering,
            num_centroids=int(self.embedder.num_centroids),
            centroid_top_k=int(self.embedder.centroid_intermediate_top_k),
        )
        torch.testing.assert_close(
            head._get_dequantized_weight(), reference.lm_head_weight
        )

        centroid_logits = self.embedder.centroids(self.hidden)
        self.assertEqual(
            int(head(self.hidden, centroid_logits).item()),
            int(reference(self.hidden, centroid_logits).item()),
        )

    def test_unknown_artifact_schema_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "schema"):
            Gemma4AssistantSparseHead.from_artifact({"schema_version": -42})


if __name__ == "__main__":
    unittest.main()
