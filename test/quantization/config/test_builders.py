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
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.config.utils import auto_qscheme_for
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.observers.ema import EMAObserver
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

    def test_build_weight_override_includes_qscheme(self):
        override = _build_weight_override(DType.uint(8))
        self.assertEqual(
            override,
            {
                "weight": {
                    "dtype": DType.uint(8),
                    "qscheme": QScheme.PER_CHANNEL_ASYMM,
                }
            },
        )
        self.assertEqual(_build_weight_override(None), {})

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

    def test_build_norm_override_empty_when_no_overrides_requested(self):
        self.assertEqual(
            _build_norm_override(norm_dtype=None, norm_weight_dtype=None),
            {},
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

    def test_build_llama_overrides_without_optional_weights(self):
        overrides = _build_llama_overrides(
            num_hidden_layers=1,
            linear_weight_dtype=None,
            embedding_weight_dtype=None,
            lm_head_weight_dtype=None,
            norm_dtype=None,
            norm_weight_dtype=None,
        )

        self.assertIn("model", overrides)
        self.assertIn("layers", overrides["model"])
        self.assertEqual(len(overrides["model"]["layers"]), 1)
        self.assertNotIn("embed_tokens", overrides["model"])
        self.assertNotIn("norm", overrides["model"])
        self.assertNotIn("lm_head", overrides)
        self.assertEqual(overrides["model"]["layers"]["0"], {})


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
            norm_dtype=DType.int(16),
            norm_weight_dtype=DType.uint(4),
            strict_wrap=False,
        )

        self.assertIsInstance(cfg, PTQConfig)
        self.assertEqual(cfg.default_dtype, DType.uint(8))
        self.assertEqual(cfg.default_qscheme, QScheme.PER_TENSOR_ASYMM)
        self.assertFalse(cfg.strict_wrap)

        self.assertEqual(
            cfg.overrides["model"]["embed_tokens"]["weight"]["qscheme"],
            QScheme.PER_CHANNEL_ASYMM,
        )
        self.assertEqual(
            cfg.overrides["lm_head"]["weight"]["qscheme"],
            QScheme.PER_CHANNEL_ASYMM,
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
        self.assertIn("model", cfg.overrides)
        self.assertIn("layers", cfg.overrides["model"])
        self.assertEqual(cfg.overrides["model"]["layers"]["0"], {})
