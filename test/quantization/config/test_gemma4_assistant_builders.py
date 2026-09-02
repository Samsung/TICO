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

"""Tests for the Gemma4 assistant PTQConfig builder."""

import unittest

from tico.quantization.config.gemma4_assistant_builders import (
    build_gemma4_assistant_ptq_config,
)
from tico.quantization.config.specs import affine
from tico.quantization.wrapq.dtypes import DType


class TestBuildGemma4AssistantPtqConfig(unittest.TestCase):
    def test_default_profile_is_w8a16(self):
        """The default profile must be int16 activations, uint8 weights, int16 norm."""
        cfg = build_gemma4_assistant_ptq_config(num_hidden_layers=4)

        self.assertEqual(cfg.activation.dtype, DType.int(16))
        self.assertEqual(cfg.weight.dtype, DType.uint(8))
        self.assertTrue(cfg.strict_wrap)
        self.assertEqual(
            cfg.overrides["pre_projection"]["weight"]["dtype"],  # type: ignore[index]
            DType.uint(8),
        )
        self.assertEqual(
            cfg.overrides["post_projection"]["weight"]["dtype"],  # type: ignore[index]
            DType.uint(8),
        )
        self.assertEqual(
            cfg.overrides["lm_head"]["weight"]["dtype"],  # type: ignore[index]
            DType.uint(8),
        )
        self.assertEqual(
            cfg.overrides["masked_embedding"]["centroids"]["weight"]["dtype"],  # type: ignore[index]
            DType.uint(8),
        )
        # Norm weights must default to int16, not inherit root uint8.
        self.assertEqual(
            cfg.overrides["model"]["norm"]["weight"]["dtype"],  # type: ignore[index]
            DType.int(16),
        )
        for idx in range(4):
            layer = cfg.overrides["model"]["layers"][str(idx)]  # type: ignore[index]
            self.assertEqual(layer["input_layernorm"]["weight"]["dtype"], DType.int(16))

    def test_layer_paths_cover_every_layer(self):
        """model.layers.{i} overrides must exist for every hidden layer."""
        num_layers = 4
        cfg = build_gemma4_assistant_ptq_config(
            num_hidden_layers=num_layers,
            norm_weight=affine(DType.int(16)),
        )
        layers = cfg.overrides["model"]["layers"]  # type: ignore[index]
        self.assertEqual(sorted(layers), [str(i) for i in range(num_layers)])
        for idx in range(num_layers):
            layer = layers[str(idx)]
            self.assertEqual(
                layer["self_attn"]["q_proj"]["weight"]["dtype"], DType.uint(8)
            )
            self.assertEqual(
                layer["mlp"]["down_proj"]["weight"]["dtype"], DType.uint(8)
            )
            self.assertEqual(layer["input_layernorm"]["weight"]["dtype"], DType.int(16))
        self.assertEqual(
            cfg.overrides["model"]["norm"]["weight"]["dtype"],  # type: ignore[index]
            DType.int(16),
        )

    def test_shared_kv_layers_do_not_require_kv_projection_specs(self):
        """Extra k/v projection keys must stay optional for shared-KV layers.

        The override tree may carry k_proj/v_proj entries (mirroring the HF
        layer structure), but the assistant wrapper never constructs those
        children, so strict wrapping must not depend on them. This is covered
        end to end by the wrapper tests; here we only pin that the overrides
        stay plain optional mappings instead of required specs.
        """
        cfg = build_gemma4_assistant_ptq_config(num_hidden_layers=1)
        attention = cfg.overrides["model"]["layers"]["0"]["self_attn"]  # type: ignore[index]
        self.assertIn("q_proj", attention)
        # No exception path: consuming a child scope for a missing module is
        # a no-op by PTQConfig design.
        self.assertIsNotNone(cfg.child("model").child("layers").child("0"))

    def test_compact_w4_override_keeps_sensitive_boundaries_at_w8(self):
        """linear_weight=uint4 must not lower the projection/head weights."""
        cfg = build_gemma4_assistant_ptq_config(
            num_hidden_layers=2,
            linear_weight=affine(DType.uint(4)),
            projection_weight=affine(DType.uint(8)),
            centroid_weight=affine(DType.uint(8)),
            lm_head_weight=affine(DType.uint(8)),
        )
        self.assertEqual(
            cfg.overrides["model"]["layers"]["0"]["self_attn"]["q_proj"]["weight"]["dtype"],  # type: ignore[index]
            DType.uint(4),
        )
        self.assertEqual(
            cfg.overrides["pre_projection"]["weight"]["dtype"],  # type: ignore[index]
            DType.uint(8),
        )
        self.assertEqual(
            cfg.overrides["post_projection"]["weight"]["dtype"],  # type: ignore[index]
            DType.uint(8),
        )
        self.assertEqual(
            cfg.overrides["masked_embedding"]["centroids"]["weight"]["dtype"],  # type: ignore[index]
            DType.uint(8),
        )
        self.assertEqual(
            cfg.overrides["lm_head"]["weight"]["dtype"],  # type: ignore[index]
            DType.uint(8),
        )

    def test_invalid_layer_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "num_hidden_layers"):
            build_gemma4_assistant_ptq_config(num_hidden_layers=0)

    def test_full_kv_length_sets_attention_capacity(self):
        """model_args.assistant.full_kv_length must bound the mask capacity."""
        cfg = build_gemma4_assistant_ptq_config(
            num_hidden_layers=1,
            model_args={"assistant": {"full_kv_length": 128}},
        )
        self.assertEqual(cfg.get_model_arg("max_seq"), 128)

        explicit = build_gemma4_assistant_ptq_config(
            num_hidden_layers=1,
            model_args={"assistant": {"full_kv_length": 128}, "max_seq": 256},
        )
        self.assertEqual(explicit.get_model_arg("max_seq"), 256)


if __name__ == "__main__":
    unittest.main()
