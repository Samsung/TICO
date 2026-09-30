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

"""Synthetic CPU coverage for pre-quantization Gemma4 PLE scale folding."""

import copy
import io
import unittest
from typing import Any
from unittest import mock

import torch
import torch.nn as nn

from tico.quantization.wrapq.wrappers.gemma4.embedding_scale_fusion import (
    fuse_gemma4_ple_embedding_scale,
    FusedGemma4PLEEmbedding,
    gemma4_ple_scale_fusion_enabled,
)


class Gemma4TextScaledWordEmbedding(nn.Embedding):
    """Tiny stand-in for the HF module's exact embedding-times-scalar contract."""

    def __init__(self, scale=16.0, dtype=torch.float32, **kwargs):
        super().__init__(19, 12, padding_idx=0, dtype=dtype, **kwargs)
        self.register_buffer("embed_scale", torch.tensor(scale), persistent=False)

    def forward(self, input_ids):
        values = super().forward(input_ids)
        return values * self.embed_scale.to(values)


def make_model(scale=16.0, dtype=torch.float32):
    """Build a PLE plus unrelated tied token embedding/LM-head pair."""
    model = nn.Module()
    model.embed_tokens_per_layer = Gemma4TextScaledWordEmbedding(scale, dtype)
    model.embed_tokens = nn.Embedding(19, 8, dtype=dtype)
    model.lm_head = nn.Linear(8, 19, bias=False, dtype=dtype)
    model.lm_head.weight = model.embed_tokens.weight
    return model.eval()


class TestGemma4PLEScaleFusion(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_fp_parity_and_parameter_identity(self):
        """The offline multiply uses the same dtype as runtime multiplication."""
        for dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
            for scale in (16.0, 0.125, 1.0, 3.7):
                with self.subTest(dtype=dtype, scale=scale):
                    model = make_model(scale, dtype)
                    ids = torch.tensor([[0, 3, 18, 3], [4, 2, 9, 0]])
                    reference = model.embed_tokens_per_layer(ids).detach()
                    original = model.embed_tokens_per_layer.weight
                    pointer = original.data_ptr()
                    self.assertEqual(
                        fuse_gemma4_ple_embedding_scale(model),
                        ("embed_tokens_per_layer",),
                    )
                    folded = model.embed_tokens_per_layer
                    self.assertIsInstance(folded, FusedGemma4PLEEmbedding)
                    self.assertIs(folded.weight, original)
                    self.assertEqual(folded.weight.data_ptr(), pointer)
                    torch.testing.assert_close(folded(ids), reference, atol=0, rtol=0)
                    self.assertEqual(float(folded.embed_scale), 1.0)

    def test_non_ple_tied_weights_unchanged(self):
        model = make_model()
        token_weight = model.embed_tokens.weight.detach().clone()
        head = model.lm_head.weight
        fuse_gemma4_ple_embedding_scale(model)
        self.assertIs(model.lm_head.weight, head)
        self.assertIs(model.embed_tokens.weight, head)
        torch.testing.assert_close(head, token_weight, atol=0, rtol=0)

    def test_idempotent(self):
        model = make_model()
        fuse_gemma4_ple_embedding_scale(model)
        weight = model.embed_tokens_per_layer.weight.detach().clone()
        module = model.embed_tokens_per_layer
        self.assertEqual(fuse_gemma4_ple_embedding_scale(model), ())
        self.assertIs(model.embed_tokens_per_layer, module)
        torch.testing.assert_close(module.weight, weight, atol=0, rtol=0)

    def test_shared_parameter_rejected_before_mutation(self):
        model = make_model()
        model.other_weight = model.embed_tokens_per_layer.weight
        original = model.other_weight.detach().clone()
        with self.assertRaisesRegex(ValueError, "shared storage"):
            fuse_gemma4_ple_embedding_scale(model)
        torch.testing.assert_close(model.other_weight, original, atol=0, rtol=0)

    def test_buffer_view_alias_with_offset_rejected(self):
        model = make_model()
        model.register_buffer("alias", model.embed_tokens_per_layer.weight.detach()[1:])
        with self.assertRaisesRegex(ValueError, "shared storage"):
            fuse_gemma4_ple_embedding_scale(model)

    def test_aliased_module_rejected(self):
        model = make_model()
        model.other_embedding = model.embed_tokens_per_layer
        with self.assertRaisesRegex(ValueError, "shared storage"):
            fuse_gemma4_ple_embedding_scale(model)

    def test_invalid_scalar_rejected(self):
        for scale in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(scale=scale):
                model = make_model(scale)
                weight = model.embed_tokens_per_layer.weight.detach().clone()
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    fuse_gemma4_ple_embedding_scale(model)
                torch.testing.assert_close(
                    model.embed_tokens_per_layer.weight, weight, atol=0, rtol=0
                )

    def test_nonscalar_and_trainable_scales_rejected(self):
        for scale in (torch.ones(2), torch.ones((), requires_grad=True)):
            model = make_model()
            model.embed_tokens_per_layer.embed_scale = scale
            with self.assertRaisesRegex(ValueError, "scalar|constant"):
                fuse_gemma4_ple_embedding_scale(model)

    def test_weight_overflow_and_nonfinite_rejected(self):
        for bad_value in (60000.0, float("inf"), float("nan")):
            model = make_model(dtype=torch.float16)
            with torch.no_grad():
                model.embed_tokens_per_layer.weight[2, 1] = bad_value
            original = model.embed_tokens_per_layer
            with self.assertRaisesRegex(ValueError, "NaN/Inf"):
                fuse_gemma4_ple_embedding_scale(model)
            self.assertIs(model.embed_tokens_per_layer, original)

    def test_all_candidates_checked_before_mutation(self):
        model = nn.Module()
        model.first = make_model()
        model.second = make_model(float("inf"))
        original = model.first.embed_tokens_per_layer.weight.detach().clone()
        with self.assertRaises(ValueError):
            fuse_gemma4_ple_embedding_scale(model)
        torch.testing.assert_close(
            model.first.embed_tokens_per_layer.weight, original, atol=0, rtol=0
        )

    def test_chunk_validation(self):
        model = make_model()
        ids = torch.tensor([[1, 2, 18]])
        reference = model.embed_tokens_per_layer(ids).detach()
        with mock.patch(
            "tico.quantization.wrapq.wrappers.gemma4.embedding_scale_fusion."
            "_MAX_CHUNK_ELEMENTS",
            24,
        ):
            fuse_gemma4_ple_embedding_scale(model)
        torch.testing.assert_close(
            model.embed_tokens_per_layer(ids), reference, atol=0, rtol=0
        )

    def test_max_norm_rejected(self):
        model = make_model()
        model.embed_tokens_per_layer.max_norm = 1.0
        with self.assertRaisesRegex(ValueError, "max_norm"):
            fuse_gemma4_ple_embedding_scale(model)

    def test_hooks_rejected(self):
        model = make_model()
        handle = model.embed_tokens_per_layer.register_forward_hook(lambda *args: None)
        try:
            with self.assertRaisesRegex(ValueError, "hooks"):
                fuse_gemma4_ple_embedding_scale(model)
        finally:
            handle.remove()

    def test_meta_and_noncontiguous_weights_rejected(self):
        model = make_model()
        model.embed_tokens_per_layer.to("meta")
        with self.assertRaisesRegex(ValueError, "materialized"):
            fuse_gemma4_ple_embedding_scale(model)
        model = make_model()
        model.embed_tokens_per_layer.weight = nn.Parameter(torch.rand(12, 19).t())
        with self.assertRaisesRegex(ValueError, "contiguous"):
            fuse_gemma4_ple_embedding_scale(model)

    def test_prepared_and_quantized_model_rejected(self):
        for mode in ("NO_QUANT", "CALIB", "QUANT"):
            model = make_model()
            model.prepared = nn.Module()
            model.prepared.qcfg = object()
            model.prepared._mode = mode
            with self.assertRaisesRegex(RuntimeError, "fresh FP model"):
                fuse_gemma4_ple_embedding_scale(model)
        model = make_model()
        model.is_quantized = True
        with self.assertRaisesRegex(RuntimeError, "fresh FP model"):
            fuse_gemma4_ple_embedding_scale(model)

    def test_wrong_or_missing_target_rejected(self):
        with self.assertRaisesRegex(ValueError, "No embed_tokens_per_layer"):
            fuse_gemma4_ple_embedding_scale(nn.Module())
        model = make_model()
        model.embed_tokens_per_layer = nn.Embedding(19, 12)
        with self.assertRaisesRegex(TypeError, "Gemma4TextScaledWordEmbedding"):
            fuse_gemma4_ple_embedding_scale(model)

    def test_state_dict_and_whole_module_roundtrip(self):
        model = make_model()
        fuse_gemma4_ple_embedding_scale(model)
        ids = torch.tensor([[0, 3, 18, 2]])
        expected = model.embed_tokens_per_layer(ids).detach()
        # State dictionaries require the same transformed architecture.
        restored = make_model()
        fuse_gemma4_ple_embedding_scale(restored)
        restored.load_state_dict(model.state_dict(), strict=True)
        torch.testing.assert_close(
            restored.embed_tokens_per_layer(ids), expected, atol=0, rtol=0
        )
        buffer = io.BytesIO()
        torch.save(model.embed_tokens_per_layer, buffer)
        buffer.seek(0)
        restored_module = torch.load(buffer, weights_only=False)
        torch.testing.assert_close(restored_module(ids), expected, atol=0, rtol=0)
        self.assertIsInstance(copy.deepcopy(restored_module), FusedGemma4PLEEmbedding)

    def test_dynamic_prefill_decode_export_has_no_mul(self):
        model = make_model()
        ids = torch.tensor([[1, 2, 3, 4]])
        original = torch.export.export(model.embed_tokens_per_layer, (ids,))
        self.assertTrue(
            any(n.target == torch.ops.aten.mul.Tensor for n in original.graph.nodes)
        )
        reference = copy.deepcopy(model.embed_tokens_per_layer)
        fuse_gemma4_ple_embedding_scale(model)
        ep = torch.export.export(
            model.embed_tokens_per_layer,
            (ids,),
            dynamic_shapes={"input": {1: torch.export.Dim("seq", min=1, max=16)}},
        )
        targets = [n.target for n in ep.graph.nodes if n.op == "call_function"]
        self.assertIn(torch.ops.aten.embedding.default, targets)
        self.assertFalse(any(str(t).startswith("aten.mul.") for t in targets))
        for seq in (1, 4, 16):
            tokens = (torch.arange(seq) % 19).reshape(1, seq)
            torch.testing.assert_close(
                ep.module()(tokens), reference(tokens), atol=0, rtol=0
            )


class TestGemma4PLEScaleFusionConfig(unittest.TestCase):
    def test_default_and_explicit_override(self):
        self.assertTrue(gemma4_ple_scale_fusion_enabled({}))
        self.assertTrue(gemma4_ple_scale_fusion_enabled({"text": {}}))
        for enabled in (True, False):
            self.assertIs(
                gemma4_ple_scale_fusion_enabled(
                    {"text": {"ple_embedding_scale_fusion": enabled}}
                ),
                enabled,
            )

    def test_invalid_config_rejected(self):
        for value in ("true", "false", 0, 1, None, [], {}):
            with self.subTest(value=value), self.assertRaisesRegex(TypeError, "bool"):
                gemma4_ple_scale_fusion_enabled(
                    {"text": {"ple_embedding_scale_fusion": value}}
                )
        invalid_configs: list[Any] = [None, [], {"text": None}, {"text": "true"}]
        for config in invalid_configs:
            with self.subTest(config=config), self.assertRaisesRegex(
                TypeError, "mapping"
            ):
                gemma4_ple_scale_fusion_enabled(config)


if __name__ == "__main__":
    unittest.main()
