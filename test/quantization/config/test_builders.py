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
from tico.quantization.config.llama_attention import DEFAULT_EXECUTION_PROFILE
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.config.specs import affine, mx
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.dtypes import MXDtype
from tico.quantization.wrapq.observers.mx import MXObserver
from tico.quantization.wrapq.observers.minmax import MinMaxObserver
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
        self.assertEqual(
            override["weight"]["observer"],
            MinMaxObserver,
        )

    def test_build_norm_override_empty_when_no_specs(self):
        self.assertEqual(_build_norm_override(norm=None, norm_weight=None), {})

    def test_build_norm_override_weight_observer_not_overridden_by_io_observer(self):
        """Weight observer must always be derived from weight dtype, never from io_observer."""
        mx8 = MXDtype(elem_format="int8")
        override = _build_norm_override(
            norm_dtype=None,
            norm_weight_dtype=DType.int(16),
            norm_io_dtype=mx8,
            norm_io_observer=MXObserver,
        )

        # Weight observer must be MinMaxObserver (from DType.int(16)), NOT MXObserver
        self.assertEqual(
            override["weight"]["observer"],
            MinMaxObserver,
        )
        # I/O observers must be MXObserver
        self.assertEqual(
            override["act_in"]["observer"],
            MXObserver,
        )
        self.assertEqual(
            override["act_out"]["observer"],
            MXObserver,
        )

    def test_build_norm_override_weight_observer_not_overridden_by_io_observer(self):
        """Weight observer must always be derived from weight dtype, never from io_observer."""
        mx8 = MXDtype(elem_format="int8")
        override = _build_norm_override(
            norm_dtype=None,
            norm_weight_dtype=DType.int(16),
            norm_io_dtype=mx8,
            norm_io_observer=MXObserver,
        )

        # Weight observer must be MinMaxObserver (from DType.int(16)), NOT MXObserver
        self.assertEqual(
            override["weight"]["observer"],
            MinMaxObserver,
        )
        # I/O observers must be MXObserver
        self.assertEqual(
            override["act_in"]["observer"],
            MXObserver,
        )
        self.assertEqual(
            override["act_out"]["observer"],
            MXObserver,
        )


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

    def test_build_llama_layer_overrides_with_linear_io_dtype(self):
        """linear_io_dtype produces act_in/act_out on linear projections and fine-grained activations."""
        mx8 = MXDtype(elem_format="int8")
        overrides = _build_llama_layer_overrides(
            linear_weight_dtype=DType.uint(4),
            linear_io_dtype=mx8,
        )

        # Linear projections get act_in/act_out with MX observer
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            self.assertEqual(
                overrides["self_attn"][proj]["act_in"]["dtype"], mx8
            )
            self.assertEqual(
                overrides["self_attn"][proj]["act_in"]["observer"], MXObserver
            )
            self.assertEqual(
                overrides["self_attn"][proj]["act_out"]["dtype"], mx8
            )

        # Fine-grained activations (driven by linear_io_dtype)
        self.assertEqual(
            overrides["self_attn"]["hidden"]["dtype"], mx8
        )
        self.assertEqual(
            overrides["self_attn"]["attn_mask"]["dtype"], mx8
        )
        self.assertEqual(
            overrides["self_attn"]["logits"]["dtype"], mx8
        )
        self.assertEqual(
            overrides["mlp"]["mul"]["dtype"], mx8
        )
        self.assertEqual(
            overrides["attn_mask"]["dtype"], mx8
        )
        self.assertEqual(
            overrides["mlp_residual_out"]["dtype"], mx8
        )
        self.assertEqual(
            overrides["self_attn_residual_out"]["dtype"], mx8
        )

    def test_build_llama_layer_overrides_with_rms_norm_io(self):
        """rms_norm_io_dtype produces act_in/act_out on norms and mlp.act_in."""
        mx8 = MXDtype(elem_format="int8")
        overrides = _build_llama_layer_overrides(
            linear_weight_dtype=DType.uint(4),
            norm_weight_dtype=DType.int(16),
            rms_norm_io_dtype=mx8,
        )

        # Norm act_in/act_out
        for norm in ["input_layernorm", "post_attention_layernorm"]:
            self.assertEqual(overrides[norm]["act_in"]["dtype"], mx8)
            self.assertEqual(overrides[norm]["act_in"]["observer"], MXObserver)
            self.assertEqual(overrides[norm]["act_out"]["dtype"], mx8)

        # mlp.act_in (driven by rms_norm_io_dtype)
        self.assertEqual(overrides["mlp"]["act_in"]["dtype"], mx8)

        # self_attn.hidden is now driven by linear_io_dtype, not rms_norm_io_dtype
        self.assertNotIn("hidden", overrides["self_attn"])

    def test_build_llama_layer_overrides_with_softmax_override(self):
        """softmax_dtype produces override on self_attn.softmax and mask_add."""
        mx8 = MXDtype(elem_format="int8")
        overrides = _build_llama_layer_overrides(
            linear_weight_dtype=DType.uint(4),
            softmax_dtype=mx8,
        )

        self.assertEqual(overrides["self_attn"]["softmax"]["dtype"], mx8)
        self.assertEqual(overrides["self_attn"]["softmax"]["observer"], MXObserver)
        self.assertEqual(overrides["self_attn"]["mask_add"]["dtype"], mx8)
        self.assertEqual(overrides["self_attn"]["mask_add"]["observer"], MXObserver)

    def test_build_llama_overrides_with_linear_io_produces_causal_mask(self):
        """linear_io_dtype produces model-level causal_mask override."""
        mx8 = MXDtype(elem_format="int8")
        overrides = _build_llama_overrides(
            num_hidden_layers=1,
            linear_weight_dtype=DType.uint(4),
            linear_io_dtype=mx8,
        )

        self.assertEqual(overrides["model"]["causal_mask"]["dtype"], mx8)
        self.assertEqual(
            overrides["model"]["causal_mask"]["observer"], MXObserver
        )

    def test_build_llama_overrides_lm_head_gets_act_in_act_out(self):
        """lm_head gets full linear desc (weight + act_in + act_out) when io is specified."""
        mx8 = MXDtype(elem_format="int8")
        overrides = _build_llama_overrides(
            num_hidden_layers=1,
            linear_weight_dtype=DType.uint(4),
            lm_head_weight_dtype=DType.uint(8),
            linear_io_dtype=mx8,
        )

        self.assertEqual(overrides["lm_head"]["act_in"]["dtype"], mx8)
        self.assertEqual(overrides["lm_head"]["act_out"]["dtype"], mx8)
        self.assertEqual(overrides["lm_head"]["weight"]["dtype"], DType.uint(8))

    def test_no_fine_grained_overrides_when_no_io_specified(self):
        """No fine-grained activation overrides when no io dtype/observer is given."""
        overrides = _build_llama_layer_overrides(
            linear_weight_dtype=DType.uint(4),
        )

        # No act_in/act_out on linear projections
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            self.assertNotIn("act_in", overrides["self_attn"][proj])
            self.assertNotIn("act_out", overrides["self_attn"][proj])

        # No fine-grained activations
        self.assertNotIn("attn_mask", overrides["self_attn"])
        self.assertNotIn("softmax", overrides["self_attn"])
        self.assertNotIn("hidden", overrides["self_attn"])
        self.assertNotIn("mul", overrides.get("mlp", {}))
        self.assertNotIn("attn_mask", overrides)
        self.assertNotIn("self_attn_residual_out", overrides)
        self.assertNotIn("mlp_residual_out", overrides)


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
            cfg.overrides["lm_head"]["weight"]["qscheme"], QScheme.PER_CHANNEL_ASYMM  # type: ignore[index]
        )


if __name__ == "__main__":
    unittest.main()
