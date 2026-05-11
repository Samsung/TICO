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

from tico.quantization.config.llama_attention import (
    DEFAULT_EXECUTION_PROFILE,
    get_llama_attention_options,
    is_npu_export_attention_options,
    LlamaAttentionOptions,
    normalize_execution_profile,
)
from tico.quantization.config.ptq import PTQConfig


class TestExecutionProfileValidation(unittest.TestCase):
    def test_normalize_execution_profile_accepts_supported_profiles(self):
        self.assertEqual(
            normalize_execution_profile("reference_eval"),
            "reference_eval",
        )
        self.assertEqual(normalize_execution_profile("npu_export"), "npu_export")

    def test_normalize_execution_profile_rejects_unknown_profile(self):
        with self.assertRaises(ValueError):
            normalize_execution_profile("debug")

    def test_normalize_execution_profile_rejects_non_string_profile(self):
        with self.assertRaises(TypeError):
            normalize_execution_profile(123)


class TestLlamaAttentionOptionsResolver(unittest.TestCase):
    def test_default_options_preserve_npu_export_profile(self):
        options = get_llama_attention_options(None)

        self.assertEqual(DEFAULT_EXECUTION_PROFILE, "npu_export")
        self.assertEqual(options.scale_fusion, "k_proj")
        self.assertEqual(options.rope, "pre_negated_sin")
        self.assertEqual(options.layout, "unrolled")
        self.assertTrue(is_npu_export_attention_options(options))

    def test_root_reference_eval_profile(self):
        qcfg = PTQConfig(model_args={"profile": "reference_eval"})

        options = get_llama_attention_options(qcfg)

        self.assertEqual(options.scale_fusion, "none")
        self.assertEqual(options.rope, "hf")
        self.assertEqual(options.layout, "batched")
        self.assertFalse(is_npu_export_attention_options(options))

    def test_root_npu_export_profile(self):
        qcfg = PTQConfig(model_args={"profile": "npu_export"})

        options = get_llama_attention_options(qcfg)

        self.assertEqual(options.scale_fusion, "k_proj")
        self.assertEqual(options.rope, "pre_negated_sin")
        self.assertEqual(options.layout, "unrolled")
        self.assertTrue(is_npu_export_attention_options(options))

    def test_attention_string_overrides_root_profile(self):
        qcfg = PTQConfig(
            model_args={
                "profile": "reference_eval",
                "attention": "npu_export",
            }
        )

        options = get_llama_attention_options(qcfg)

        self.assertEqual(options.scale_fusion, "k_proj")
        self.assertEqual(options.rope, "pre_negated_sin")
        self.assertEqual(options.layout, "unrolled")

    def test_attention_mapping_profile_overrides_root_profile(self):
        qcfg = PTQConfig(
            model_args={
                "profile": "reference_eval",
                "attention": {
                    "profile": "npu_export",
                },
            }
        )

        options = get_llama_attention_options(qcfg)

        self.assertEqual(options.scale_fusion, "k_proj")
        self.assertEqual(options.rope, "pre_negated_sin")
        self.assertEqual(options.layout, "unrolled")

    def test_attention_mapping_can_override_individual_fields(self):
        qcfg = PTQConfig(
            model_args={
                "profile": "reference_eval",
                "attention": {
                    "layout": "unrolled",
                    "scale_fusion": "q_proj",
                },
            }
        )

        options = get_llama_attention_options(qcfg)

        self.assertEqual(options.scale_fusion, "q_proj")
        self.assertEqual(options.rope, "hf")
        self.assertEqual(options.layout, "unrolled")

    def test_attention_none_uses_root_profile(self):
        qcfg = PTQConfig(
            model_args={
                "profile": "reference_eval",
                "attention": None,
            }
        )

        options = get_llama_attention_options(qcfg)

        self.assertEqual(options.scale_fusion, "none")
        self.assertEqual(options.rope, "hf")
        self.assertEqual(options.layout, "batched")

    def test_unknown_attention_option_raises(self):
        qcfg = PTQConfig(
            model_args={
                "attention": {
                    "unknown": True,
                }
            }
        )

        with self.assertRaises(ValueError):
            get_llama_attention_options(qcfg)

    def test_invalid_attention_option_value_raises(self):
        qcfg = PTQConfig(
            model_args={
                "attention": {
                    "layout": "flat",
                }
            }
        )

        with self.assertRaises(ValueError):
            get_llama_attention_options(qcfg)

    def test_invalid_attention_payload_type_raises(self):
        qcfg = PTQConfig(model_args={"attention": 1})

        with self.assertRaises(TypeError):
            get_llama_attention_options(qcfg)

    def test_invalid_root_profile_raises(self):
        qcfg = PTQConfig(model_args={"profile": "invalid"})

        with self.assertRaises(ValueError):
            get_llama_attention_options(qcfg)

    def test_is_npu_export_attention_options_checks_exact_graph_contract(self):
        self.assertTrue(
            is_npu_export_attention_options(
                LlamaAttentionOptions(
                    scale_fusion="k_proj",
                    rope="pre_negated_sin",
                    layout="unrolled",
                )
            )
        )
        self.assertFalse(
            is_npu_export_attention_options(
                LlamaAttentionOptions(
                    scale_fusion="none",
                    rope="pre_negated_sin",
                    layout="unrolled",
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
