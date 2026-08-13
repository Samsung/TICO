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

from tico.quantization.config.gemma4_attention import (
    DEFAULT_EXECUTION_PROFILE,
    Gemma4TextAttentionOptions,
    get_gemma4_text_attention_options,
    is_npu_export_text_attention_options,
    normalize_execution_profile,
)
from tico.quantization.config.ptq import PTQConfig


class TestGemma4ExecutionProfileValidation(unittest.TestCase):
    """Validate Gemma4 execution-profile parsing."""

    def test_accepts_supported_profiles(self):
        """Supported execution profiles should be returned unchanged."""
        self.assertEqual(
            normalize_execution_profile("reference_eval"),
            "reference_eval",
        )
        self.assertEqual(normalize_execution_profile("npu_export"), "npu_export")

    def test_rejects_unknown_profile(self):
        """Unknown execution profiles should fail fast."""
        with self.assertRaises(ValueError):
            normalize_execution_profile("debug")

    def test_rejects_non_string_profile(self):
        """Non-string execution profiles should fail with a type error."""
        with self.assertRaises(TypeError):
            normalize_execution_profile(123)


class TestGemma4TextAttentionOptionsResolver(unittest.TestCase):
    """Validate Gemma4 text-attention option resolution."""

    def test_default_options_preserve_npu_export_graph(self):
        """The default should keep the existing NPU-oriented behavior."""
        options = get_gemma4_text_attention_options(None)

        self.assertEqual(DEFAULT_EXECUTION_PROFILE, "npu_export")
        self.assertEqual(options.layout, "unrolled")
        self.assertTrue(is_npu_export_text_attention_options(options))

    def test_root_reference_eval_profile_selects_batched_layout(self):
        """The reference profile should select the batched attention graph."""
        qcfg = PTQConfig(model_args={"profile": "reference_eval"})

        options = get_gemma4_text_attention_options(qcfg)

        self.assertEqual(options.layout, "batched")
        self.assertFalse(is_npu_export_text_attention_options(options))

    def test_root_npu_export_profile_selects_unrolled_layout(self):
        """The NPU profile should select the fully unrolled graph."""
        qcfg = PTQConfig(model_args={"profile": "npu_export"})

        options = get_gemma4_text_attention_options(qcfg)

        self.assertEqual(options.layout, "unrolled")
        self.assertTrue(is_npu_export_text_attention_options(options))

    def test_attention_string_overrides_root_profile(self):
        """A string attention profile should override the root profile."""
        qcfg = PTQConfig(
            model_args={
                "profile": "reference_eval",
                "attention": "npu_export",
            }
        )

        options = get_gemma4_text_attention_options(qcfg)

        self.assertEqual(options.layout, "unrolled")

    def test_attention_mapping_profile_overrides_root_profile(self):
        """A nested attention profile should override the root profile."""
        qcfg = PTQConfig(
            model_args={
                "profile": "reference_eval",
                "attention": {"profile": "npu_export"},
            }
        )

        options = get_gemma4_text_attention_options(qcfg)

        self.assertEqual(options.layout, "unrolled")

    def test_attention_mapping_can_override_layout(self):
        """Individual graph options should override the selected preset."""
        qcfg = PTQConfig(
            model_args={
                "profile": "reference_eval",
                "attention": {"layout": "unrolled"},
            }
        )

        options = get_gemma4_text_attention_options(qcfg)

        self.assertEqual(options.layout, "unrolled")

    def test_attention_none_uses_root_profile(self):
        """A null attention override should retain the root profile."""
        qcfg = PTQConfig(
            model_args={
                "profile": "reference_eval",
                "attention": None,
            }
        )

        options = get_gemma4_text_attention_options(qcfg)

        self.assertEqual(options.layout, "batched")

    def test_unknown_attention_option_raises(self):
        """Unknown graph-option fields should be rejected."""
        qcfg = PTQConfig(model_args={"attention": {"unknown": True}})

        with self.assertRaises(ValueError):
            get_gemma4_text_attention_options(qcfg)

    def test_invalid_attention_layout_raises(self):
        """Unsupported layouts should be rejected after preset resolution."""
        qcfg = PTQConfig(model_args={"attention": {"layout": "grouped"}})

        with self.assertRaises(ValueError):
            get_gemma4_text_attention_options(qcfg)

    def test_invalid_attention_payload_type_raises(self):
        """Attention overrides must be a profile string, mapping, or null."""
        qcfg = PTQConfig(model_args={"attention": 1})

        with self.assertRaises(TypeError):
            get_gemma4_text_attention_options(qcfg)

    def test_invalid_root_profile_raises(self):
        """Invalid root profiles should not silently fall back."""
        qcfg = PTQConfig(model_args={"profile": "invalid"})

        with self.assertRaises(ValueError):
            get_gemma4_text_attention_options(qcfg)

    def test_npu_contract_requires_exact_unrolled_layout(self):
        """Only the fully unrolled layout should satisfy the NPU contract."""
        self.assertTrue(
            is_npu_export_text_attention_options(
                Gemma4TextAttentionOptions(layout="unrolled")
            )
        )
        self.assertFalse(
            is_npu_export_text_attention_options(
                Gemma4TextAttentionOptions(layout="batched")
            )
        )


if __name__ == "__main__":
    unittest.main()
