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
    _resolve_weight_dtype,
    _weight_dtype_from_bits,
    build_llm_ptq_config,
)
from tico.quantization.config.llama_attention import DEFAULT_EXECUTION_PROFILE
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.config.utils import auto_qscheme_for
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.dtypes import MXDtype
from tico.quantization.wrapq.observers.ema import EMAObserver
from tico.quantization.wrapq.observers.minmax import MinMaxObserver
from tico.quantization.wrapq.observers.mx import MXObserver
from tico.quantization.wrapq.qscheme import QScheme


class TestBuilderHelpers(unittest.TestCase):
    def test_auto_qscheme_for_unsigned_activation(self):
        self.assertEqual(
            auto_qscheme_for(DType.uint(8), "act_in"),
            QScheme.PER_TENSOR_ASYMM,
        )

    def test_auto_qscheme_for_unsigned_weight(self):
        self.assertEqual(
            auto_qscheme_for(DType.uint(8), "weight"),
            QScheme.PER_CHANNEL_ASYMM,
        )

    def test_auto_qscheme_for_signed_dtype(self):
        self.assertEqual(
            auto_qscheme_for(DType.int(8), "weight"),
            QScheme.PER_TENSOR_SYMM,
        )

    def test_weight_dtype_from_bits(self):
        self.assertEqual(_weight_dtype_from_bits(16), DType.int(16))
        self.assertEqual(_weight_dtype_from_bits(8), DType.uint(8))
        self.assertEqual(_weight_dtype_from_bits(4), DType.uint(4))

    def test_weight_dtype_from_bits_invalid_raises(self):
        with self.assertRaises(ValueError):
            _weight_dtype_from_bits(3)

    def test_resolve_weight_dtype_prefers_explicit_dtype(self):
        self.assertEqual(
            _resolve_weight_dtype(dtype=DType.int(8), bits=4),
            DType.int(8),
        )

    def test_resolve_weight_dtype_falls_back_to_bits(self):
        self.assertEqual(
            _resolve_weight_dtype(dtype=None, bits=4),
            DType.uint(4),
        )
        self.assertIsNone(_resolve_weight_dtype(dtype=None, bits=None))

    def test_build_weight_override_includes_qscheme_and_observer(self):
        override = _build_weight_override(DType.uint(8))
        self.assertEqual(
            override,
            {
                "weight": {
                    "dtype": DType.uint(8),
                    "qscheme": QScheme.PER_CHANNEL_ASYMM,
                    "observer": MinMaxObserver,
                }
            },
        )
        self.assertEqual(_build_weight_override(None), {})

    def test_build_weight_override_signed_dtype_uses_symmetric_qscheme(self):
        override = _build_weight_override(DType.int(16))
        self.assertEqual(
            override,
            {
                "weight": {
                    "dtype": DType.int(16),
                    "qscheme": QScheme.PER_TENSOR_SYMM,
                    "observer": MinMaxObserver,
                }
            },
        )

    def test_build_norm_override_includes_module_and_weight_qscheme(self):
        override = _build_norm_override(
            norm_dtype=DType.uint(8),
            norm_weight_dtype=DType.uint(4),
        )

        self.assertEqual(override["dtype"], DType.uint(8))
        self.assertEqual(override["qscheme"], QScheme.PER_TENSOR_ASYMM)
        self.assertEqual(override["weight"]["dtype"], DType.uint(4))
        self.assertEqual(
            override["weight"]["qscheme"],
            QScheme.PER_CHANNEL_ASYMM,
        )
        self.assertEqual(
            override["weight"]["observer"],
            MinMaxObserver,
        )

    def test_build_norm_override_empty_when_no_overrides_requested(self):
        self.assertEqual(
            _build_norm_override(norm_dtype=None, norm_weight_dtype=None),
            {},
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
            linear_weight_dtype=DType.uint(8),
            norm_dtype=DType.uint(8),
            norm_weight_dtype=DType.uint(4),
        )

        self.assertEqual(
            overrides["self_attn"]["q_proj"]["weight"]["qscheme"],
            QScheme.PER_CHANNEL_ASYMM,
        )
        self.assertEqual(
            overrides["mlp"]["down_proj"]["weight"]["qscheme"],
            QScheme.PER_CHANNEL_ASYMM,
        )
        self.assertEqual(
            overrides["input_layernorm"]["qscheme"],
            QScheme.PER_TENSOR_ASYMM,
        )
        self.assertEqual(
            overrides["input_layernorm"]["weight"]["qscheme"],
            QScheme.PER_CHANNEL_ASYMM,
        )

    def test_build_llama_overrides(self):
        overrides = _build_llama_overrides(
            num_hidden_layers=2,
            linear_weight_dtype=DType.uint(8),
            embedding_weight_dtype=DType.uint(4),
            lm_head_weight_dtype=DType.uint(8),
            spin_rotation_weight_dtype=None,
            norm_dtype=DType.int(16),
            norm_weight_dtype=DType.uint(4),
        )

        self.assertIn("model", overrides)
        self.assertIn("layers", overrides["model"])
        self.assertEqual(len(overrides["model"]["layers"]), 2)
        self.assertEqual(
            overrides["model"]["embed_tokens"]["weight"]["qscheme"],
            QScheme.PER_CHANNEL_ASYMM,
        )
        self.assertEqual(
            overrides["lm_head"]["weight"]["qscheme"],
            QScheme.PER_CHANNEL_ASYMM,
        )
        self.assertNotIn("rotate_embedding", overrides["model"])
        self.assertNotIn("rotate_lm_head", overrides)
        self.assertEqual(
            overrides["model"]["norm"]["qscheme"],
            QScheme.PER_TENSOR_SYMM,
        )
        self.assertEqual(
            overrides["model"]["layers"]["0"]["self_attn"]["o_proj"]["weight"][
                "qscheme"
            ],
            QScheme.PER_CHANNEL_ASYMM,
        )

    def test_build_llama_overrides_with_spin_rotation_weights(self):
        overrides = _build_llama_overrides(
            num_hidden_layers=1,
            linear_weight_dtype=None,
            embedding_weight_dtype=None,
            lm_head_weight_dtype=None,
            spin_rotation_weight_dtype=DType.int(16),
            norm_dtype=None,
            norm_weight_dtype=None,
        )

        self.assertEqual(
            overrides["model"]["rotate_embedding"]["weight"]["dtype"],
            DType.int(16),
        )
        self.assertEqual(
            overrides["model"]["rotate_embedding"]["weight"]["qscheme"],
            QScheme.PER_TENSOR_SYMM,
        )
        self.assertEqual(
            overrides["rotate_lm_head"]["weight"]["dtype"],
            DType.int(16),
        )
        self.assertEqual(
            overrides["rotate_lm_head"]["weight"]["qscheme"],
            QScheme.PER_TENSOR_SYMM,
        )

    def test_build_llama_overrides_without_optional_weights(self):
        overrides = _build_llama_overrides(
            num_hidden_layers=1,
            linear_weight_dtype=None,
            embedding_weight_dtype=None,
            lm_head_weight_dtype=None,
            spin_rotation_weight_dtype=None,
            norm_dtype=None,
            norm_weight_dtype=None,
        )

        self.assertIn("model", overrides)
        self.assertIn("layers", overrides["model"])
        self.assertEqual(len(overrides["model"]["layers"]), 1)
        self.assertNotIn("embed_tokens", overrides["model"])
        self.assertNotIn("rotate_embedding", overrides["model"])
        self.assertNotIn("norm", overrides["model"])
        self.assertNotIn("lm_head", overrides)
        self.assertNotIn("rotate_lm_head", overrides)
        self.assertEqual(overrides["model"]["layers"]["0"], {})

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
            activation_dtype=DType.uint(8),
            default_qscheme=QScheme.PER_TENSOR_ASYMM,
            linear_weight_dtype=DType.uint(8),
            embedding_weight_dtype=DType.uint(4),
            lm_head_weight_dtype=DType.uint(8),
            spin_rotation_weight_dtype=DType.int(16),
            norm_dtype=DType.int(16),
            norm_weight_dtype=DType.uint(4),
            strict_wrap=False,
        )

        self.assertIsInstance(cfg, PTQConfig)
        self.assertEqual(cfg.default_dtype, DType.uint(8))
        self.assertEqual(cfg.default_qscheme, QScheme.PER_TENSOR_ASYMM)
        self.assertFalse(cfg.strict_wrap)
        self.assertEqual(cfg.model_args["profile"], DEFAULT_EXECUTION_PROFILE)

        self.assertEqual(
            cfg.overrides["model"]["embed_tokens"]["weight"]["qscheme"],
            QScheme.PER_CHANNEL_ASYMM,
        )
        self.assertEqual(
            cfg.overrides["lm_head"]["weight"]["qscheme"],
            QScheme.PER_CHANNEL_ASYMM,
        )
        self.assertEqual(
            cfg.overrides["model"]["rotate_embedding"]["weight"]["dtype"],
            DType.int(16),
        )
        self.assertEqual(
            cfg.overrides["model"]["rotate_embedding"]["weight"]["qscheme"],
            QScheme.PER_TENSOR_SYMM,
        )
        self.assertEqual(
            cfg.overrides["rotate_lm_head"]["weight"]["dtype"],
            DType.int(16),
        )
        self.assertEqual(
            cfg.overrides["rotate_lm_head"]["weight"]["qscheme"],
            QScheme.PER_TENSOR_SYMM,
        )
        self.assertEqual(
            cfg.overrides["model"]["layers"]["1"]["mlp"]["up_proj"]["weight"][
                "qscheme"
            ],
            QScheme.PER_CHANNEL_ASYMM,
        )
        self.assertEqual(
            cfg.overrides["model"]["norm"]["qscheme"],
            QScheme.PER_TENSOR_SYMM,
        )

    def test_build_llm_ptq_config_sets_reference_eval_profile(self):
        cfg = build_llm_ptq_config(
            model_type="llama",
            num_hidden_layers=1,
            profile="reference_eval",
        )

        self.assertEqual(cfg.model_args, {"profile": "reference_eval"})

    def test_build_llm_ptq_config_sets_npu_export_profile(self):
        cfg = build_llm_ptq_config(
            model_type="llama",
            num_hidden_layers=1,
            profile="npu_export",
        )

        self.assertEqual(cfg.model_args, {"profile": "npu_export"})

    def test_build_llm_ptq_config_invalid_profile_raises(self):
        with self.assertRaises(ValueError):
            build_llm_ptq_config(
                model_type="llama",
                num_hidden_layers=1,
                profile="invalid_profile",  # type: ignore[arg-type]
            )

    def test_explicit_dtype_takes_precedence_over_bits(self):
        cfg = build_llm_ptq_config(
            model_type="llama",
            num_hidden_layers=1,
            linear_weight_bits=4,
            linear_weight_dtype=DType.uint(8),
        )

        self.assertEqual(
            cfg.overrides["model"]["layers"]["0"]["self_attn"]["q_proj"]["weight"][
                "dtype"
            ],
            DType.uint(8),
        )
        self.assertEqual(
            cfg.overrides["model"]["layers"]["0"]["self_attn"]["q_proj"]["weight"][
                "qscheme"
            ],
            QScheme.PER_CHANNEL_ASYMM,
        )

    def test_spin_rotation_explicit_dtype_takes_precedence_over_bits(self):
        cfg = build_llm_ptq_config(
            model_type="llama",
            num_hidden_layers=1,
            spin_rotation_weight_bits=16,
            spin_rotation_weight_dtype=DType.uint(8),
        )

        self.assertEqual(
            cfg.overrides["model"]["rotate_embedding"]["weight"]["dtype"],
            DType.uint(8),
        )
        self.assertEqual(
            cfg.overrides["model"]["rotate_embedding"]["weight"]["qscheme"],
            QScheme.PER_CHANNEL_ASYMM,
        )
        self.assertEqual(
            cfg.overrides["rotate_lm_head"]["weight"]["dtype"],
            DType.uint(8),
        )
        self.assertEqual(
            cfg.overrides["rotate_lm_head"]["weight"]["qscheme"],
            QScheme.PER_CHANNEL_ASYMM,
        )

    def test_build_llm_ptq_config_unsupported_model_type_raises(self):
        with self.assertRaises(NotImplementedError):
            build_llm_ptq_config(
                model_type="mistral",
                num_hidden_layers=1,
            )

    def test_build_llm_ptq_config_accepts_default_observer(self):
        cfg = build_llm_ptq_config(
            model_type="llama",
            num_hidden_layers=1,
            default_observer=EMAObserver,
        )

        self.assertIs(cfg.default_observer, EMAObserver)

    def test_build_llm_ptq_config_bits_fallbacks(self):
        cfg = build_llm_ptq_config(
            model_type="llama",
            num_hidden_layers=1,
            linear_weight_bits=8,
            embedding_weight_bits=4,
            lm_head_weight_bits=8,
            spin_rotation_weight_bits=16,
            norm_weight_bits=4,
            norm_dtype=DType.uint(8),
        )

        self.assertEqual(
            cfg.overrides["model"]["layers"]["0"]["self_attn"]["k_proj"]["weight"][
                "dtype"
            ],
            DType.uint(8),
        )
        self.assertEqual(
            cfg.overrides["model"]["embed_tokens"]["weight"]["dtype"],
            DType.uint(4),
        )
        self.assertEqual(
            cfg.overrides["lm_head"]["weight"]["dtype"],
            DType.uint(8),
        )
        self.assertEqual(
            cfg.overrides["model"]["rotate_embedding"]["weight"]["dtype"],
            DType.int(16),
        )
        self.assertEqual(
            cfg.overrides["model"]["rotate_embedding"]["weight"]["qscheme"],
            QScheme.PER_TENSOR_SYMM,
        )
        self.assertEqual(
            cfg.overrides["rotate_lm_head"]["weight"]["dtype"],
            DType.int(16),
        )
        self.assertEqual(
            cfg.overrides["rotate_lm_head"]["weight"]["qscheme"],
            QScheme.PER_TENSOR_SYMM,
        )
        self.assertEqual(
            cfg.overrides["model"]["norm"]["weight"]["dtype"],
            DType.uint(4),
        )

    def test_build_llm_ptq_config_without_optional_weight_overrides(self):
        cfg = build_llm_ptq_config(
            model_type="llama",
            num_hidden_layers=1,
        )

        self.assertIsInstance(cfg, PTQConfig)
        self.assertEqual(cfg.default_dtype, DType.int(16))
        self.assertEqual(cfg.default_qscheme, QScheme.PER_TENSOR_SYMM)
        self.assertTrue(cfg.strict_wrap)
        self.assertEqual(cfg.model_args, {"profile": DEFAULT_EXECUTION_PROFILE})
        self.assertIn("model", cfg.overrides)
        self.assertIn("layers", cfg.overrides["model"])
        self.assertNotIn("rotate_embedding", cfg.overrides["model"])
        self.assertNotIn("rotate_lm_head", cfg.overrides)
        self.assertEqual(cfg.overrides["model"]["layers"]["0"], {})
