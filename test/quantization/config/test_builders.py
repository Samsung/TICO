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

import unittest

from tico.quantization.config.builders import (
    _build_llama_layer_overrides,
    _build_llama_overrides,
    _build_norm_override,
    _build_weight_override,
    build_llm_ptq_config,
    build_qwen3_vl_ptq_config,
)
from tico.quantization.config.gemma4_builders import build_gemma4_e2b_ptq_config
from tico.quantization.config.llama_attention import DEFAULT_EXECUTION_PROFILE
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.config.specs import affine, mx
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.observers.mx import MXObserver
from tico.quantization.wrapq.qscheme import QScheme


class TestBuilderHelpers(unittest.TestCase):
    def test_build_weight_override_from_quant_spec(self):
        override = _build_weight_override(affine(DType.uint(8)))
        self.assertEqual(override["weight"]["dtype"], DType.uint(8))
        self.assertEqual(override["weight"]["qscheme"], QScheme.PER_CHANNEL_ASYMM)
        self.assertTrue(override["weight"]["__quant_spec_replace_role__"])
        self.assertEqual(_build_weight_override(None), {})

    def test_build_norm_override_from_quant_specs(self):
        override = _build_norm_override(
            norm=affine(DType.uint(8)),
            norm_weight=affine(DType.uint(4)),
        )

        self.assertEqual(override["act_in"]["dtype"], DType.uint(8))
        self.assertEqual(override["act_out"]["dtype"], DType.uint(8))
        self.assertEqual(override["weight"]["dtype"], DType.uint(4))
        self.assertEqual(override["weight"]["qscheme"], QScheme.PER_CHANNEL_ASYMM)

    def test_build_norm_override_empty_when_no_specs(self):
        self.assertEqual(_build_norm_override(norm=None, norm_weight=None), {})


class TestLlamaOverrideBuilders(unittest.TestCase):
    def test_build_llama_layer_overrides(self):
        overrides = _build_llama_layer_overrides(
            linear_weight=affine(DType.uint(8)),
            norm=affine(DType.uint(8)),
            norm_weight=affine(DType.uint(4)),
        )

        self.assertEqual(
            overrides["self_attn"]["q_proj"]["weight"]["qscheme"],
            QScheme.PER_CHANNEL_ASYMM,
        )
        self.assertEqual(overrides["input_layernorm"]["act_in"]["dtype"], DType.uint(8))
        self.assertEqual(
            overrides["input_layernorm"]["weight"]["qscheme"],
            QScheme.PER_CHANNEL_ASYMM,
        )

    def test_build_llama_overrides(self):
        overrides = _build_llama_overrides(
            num_hidden_layers=2,
            linear_weight=affine(DType.uint(8)),
            embedding_weight=affine(DType.uint(4)),
            lm_head_weight=affine(DType.uint(8)),
            spin_rotation_weight=affine(DType.int(16)),
            norm=affine(DType.int(16)),
            norm_weight=affine(DType.uint(4)),
        )

        self.assertEqual(len(overrides["model"]["layers"]), 2)
        self.assertEqual(
            overrides["model"]["embed_tokens"]["weight"]["dtype"], DType.uint(4)
        )
        self.assertEqual(overrides["lm_head"]["weight"]["dtype"], DType.uint(8))
        self.assertEqual(
            overrides["model"]["rotate_embedding"]["weight"]["dtype"], DType.int(16)
        )
        self.assertEqual(
            overrides["model"]["layers"]["1"]["mlp"]["up_proj"]["weight"]["qscheme"],
            QScheme.PER_CHANNEL_ASYMM,
        )


class TestBuildLlmPtqConfig(unittest.TestCase):
    def test_build_llm_ptq_config_llama(self):
        cfg = build_llm_ptq_config(
            model_type="llama",
            num_hidden_layers=2,
            activation=affine(DType.uint(8)),
            weight=affine(DType.int(16)),
            linear_weight=affine(DType.uint(8)),
            embedding_weight=affine(DType.uint(4)),
            lm_head_weight=affine(DType.uint(8)),
            spin_rotation_weight=affine(DType.int(16)),
            norm=affine(DType.int(16)),
            norm_weight=affine(DType.uint(4)),
            strict_wrap=False,
        )

        self.assertIsInstance(cfg, PTQConfig)
        self.assertEqual(cfg.activation.dtype, DType.uint(8))
        self.assertEqual(cfg.weight.dtype, DType.int(16))
        self.assertFalse(cfg.strict_wrap)
        self.assertEqual(cfg.model_args["profile"], DEFAULT_EXECUTION_PROFILE)
        self.assertEqual(
            cfg.overrides["model"]["embed_tokens"]["weight"]["dtype"], DType.uint(4)  # type: ignore[index]
        )
        self.assertEqual(
            cfg.overrides["lm_head"]["weight"]["qscheme"], QScheme.PER_CHANNEL_ASYMM  # type: ignore[index]
        )

    def test_build_llm_ptq_config_supports_mx_activation(self):
        cfg = build_llm_ptq_config(
            model_type="llama",
            num_hidden_layers=1,
            activation=mx("fp8_e4m3", axis=-1),
            linear_weight=affine(DType.uint(4)),
        )

        self.assertIs(cfg.activation.observer, MXObserver)
        self.assertEqual(
            cfg.overrides["model"]["layers"]["0"]["self_attn"]["q_proj"]["weight"][  # type: ignore[index]
                "dtype"
            ],
            DType.uint(4),
        )

    def test_build_llm_ptq_config_sets_reference_eval_profile(self):
        cfg = build_llm_ptq_config(
            model_type="llama",
            num_hidden_layers=1,
            profile="reference_eval",
        )
        self.assertEqual(cfg.model_args, {"profile": "reference_eval"})

    def test_build_llm_ptq_config_invalid_profile_raises(self):
        with self.assertRaises(ValueError):
            build_llm_ptq_config(
                model_type="llama",
                num_hidden_layers=1,
                profile="invalid_profile",  # type: ignore[arg-type]
            )

    def test_build_llm_ptq_config_unsupported_model_type_raises(self):
        with self.assertRaises(NotImplementedError):
            build_llm_ptq_config(model_type="mistral", num_hidden_layers=1)


class TestBuildQwen3VlPtqConfig(unittest.TestCase):
    def test_build_qwen3_vl_ptq_config(self):
        cfg = build_qwen3_vl_ptq_config(
            num_vision_blocks=2,
            num_text_layers=3,
            num_deepstack_mergers=1,
            model_args={"vision": {"grid_thw": (1, 2, 3)}},
            activation=affine(DType.int(16)),
            linear_weight=affine(DType.uint(4)),
            vision_patch_embed_weight=affine(DType.uint(8)),
            embedding_weight=affine(DType.uint(8)),
            lm_head_weight=affine(DType.uint(8)),
            spin_rotation_weight=affine(DType.int(16)),
            norm=affine(DType.int(16)),
            norm_weight=affine(DType.int(16)),
            strict_wrap=False,
        )

        self.assertFalse(cfg.strict_wrap)
        self.assertEqual(cfg.model_args["vision"]["grid_thw"], (1, 2, 3))
        self.assertEqual(
            cfg.overrides["model"]["visual"]["patch_embed"]["proj"]["weight"]["dtype"],  # type: ignore[index]
            DType.uint(8),
        )
        self.assertEqual(
            cfg.overrides["model"]["language_model"]["layers"]["2"]["self_attn"][  # type: ignore[index]
                "q_proj"
            ][
                "weight"
            ][
                "dtype"
            ],
            DType.uint(4),
        )
        self.assertEqual(
            cfg.overrides["model"]["language_model"]["rotate_embedding"]["weight"][  # type: ignore[index]
                "dtype"
            ],
            DType.int(16),
        )
        self.assertEqual(
            cfg.overrides["rotate_lm_head"]["weight"]["dtype"],  # type: ignore[index]
            DType.int(16),
        )
        self.assertEqual(
            cfg.overrides["lm_head"]["weight"]["qscheme"], QScheme.PER_CHANNEL_ASYMM  # type: ignore[index]
        )

    def test_build_qwen3_vl_ptq_config_omits_spin_rotation_when_not_requested(self):
        cfg = build_qwen3_vl_ptq_config(
            num_vision_blocks=1,
            num_text_layers=1,
            num_deepstack_mergers=0,
            model_args={"vision": {"grid_thw": (1, 1, 1)}},
            linear_weight=affine(DType.uint(4)),
        )

        self.assertNotIn(
            "rotate_embedding",
            cfg.overrides["model"]["language_model"],  # type: ignore[index]
        )
        self.assertNotIn("rotate_lm_head", cfg.overrides)  # type: ignore[operator]


class TestBuildGemma4E2BPtqConfig(unittest.TestCase):
    def test_build_gemma4_e2b_ptq_config_follows_hf_text_and_model_scopes(self):
        cfg = build_gemma4_e2b_ptq_config(
            num_text_layers=2,
            num_vision_layers=1,
            linear_weight=affine(DType.uint(4)),
            embedding_weight=affine(DType.uint(8)),
            lm_head_weight=affine(DType.uint(8)),
            vision_patch_embed_weight=affine(DType.uint(8)),
            norm_weight=affine(DType.uint(4)),
            strict_wrap=False,
        )

        self.assertFalse(cfg.strict_wrap)

        # Direct Gemma4TextModel path.
        self.assertEqual(
            cfg.overrides["embed_tokens"]["weight"]["dtype"],  # type: ignore[index]
            DType.uint(8),
        )
        self.assertEqual(
            cfg.overrides["layers"]["1"]["self_attn"]["q_proj"]["weight"]["dtype"],  # type: ignore[index]
            DType.uint(4),
        )
        self.assertEqual(
            cfg.overrides["layers"]["1"]["self_attn"]["q_norm"]["weight"]["dtype"],  # type: ignore[index]
            DType.uint(4),
        )
        self.assertEqual(
            cfg.overrides["norm"]["weight"]["dtype"],  # type: ignore[index]
            DType.uint(4),
        )

        # Gemma4ForCausalLM path: model is Gemma4TextModel.
        self.assertEqual(
            cfg.overrides["model"]["embed_tokens"]["weight"]["dtype"],  # type: ignore[index]
            DType.uint(8),
        )
        self.assertEqual(
            cfg.overrides["model"]["layers"]["1"]["mlp"]["up_proj"]["weight"]["dtype"],  # type: ignore[index]
            DType.uint(4),
        )

        # Direct Gemma4Model path: language_model is Gemma4TextModel.
        self.assertEqual(
            cfg.overrides["language_model"]["layers"]["0"]["self_attn"]["o_proj"]["weight"]["dtype"],  # type: ignore[index]
            DType.uint(4),
        )
        self.assertEqual(
            cfg.overrides["vision_tower"]["patch_embedder"]["input_proj"]["weight"]["dtype"],  # type: ignore[index]
            DType.uint(8),
        )
        self.assertEqual(
            cfg.overrides["vision_tower"]["encoder"]["layers"]["0"]["self_attn"]["q_norm"]["weight"]["dtype"],  # type: ignore[index]
            DType.uint(4),
        )
        self.assertEqual(
            cfg.overrides["vision_tower"]["encoder"]["layers"]["0"]["input_layernorm"]["weight"]["dtype"],  # type: ignore[index]
            DType.uint(4),
        )
        self.assertEqual(
            cfg.overrides["embed_vision"]["embedding_pre_projection_norm"]["weight"]["dtype"],  # type: ignore[index]
            DType.uint(4),
        )

        # Gemma4ForConditionalGeneration path: model is Gemma4Model.
        self.assertEqual(
            cfg.overrides["model"]["language_model"]["layers"]["0"]["self_attn"]["k_proj"]["weight"]["dtype"],  # type: ignore[index]
            DType.uint(4),
        )
        self.assertEqual(
            cfg.overrides["model"]["vision_tower"]["encoder"]["layers"]["0"]["mlp"]["down_proj"]["weight"]["dtype"],  # type: ignore[index]
            DType.uint(4),
        )
        self.assertEqual(
            cfg.overrides["model"]["embed_vision"]["embedding_projection"]["weight"]["dtype"],  # type: ignore[index]
            DType.uint(4),
        )
        self.assertEqual(
            cfg.overrides["model"]["vision_tower"]["encoder"]["layers"]["0"]["self_attn"]["v_norm"]["weight"]["dtype"],  # type: ignore[index]
            DType.uint(4),
        )
        self.assertEqual(
            cfg.overrides["model"]["vision_tower"]["encoder"]["layers"]["0"]["post_feedforward_layernorm"]["weight"]["dtype"],  # type: ignore[index]
            DType.uint(4),
        )
        self.assertEqual(
            cfg.overrides["model"]["embed_vision"]["embedding_pre_projection_norm"]["weight"]["dtype"],  # type: ignore[index]
            DType.uint(4),
        )
        self.assertEqual(
            cfg.overrides["lm_head"]["weight"]["dtype"],  # type: ignore[index]
            DType.uint(8),
        )


if __name__ == "__main__":
    unittest.main()
