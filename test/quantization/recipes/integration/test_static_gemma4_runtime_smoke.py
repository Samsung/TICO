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

"""Smoke tests for StaticGemma4Runtime.

These tests require transformers with Gemma4 support and are gated behind
the RUN_INTERNAL_TESTS environment variable.
"""

import importlib.util
import os
import unittest

import torch

HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None
RUN_INTERNAL_TESTS = os.environ.get("RUN_INTERNAL_TESTS", "0") == "1"


@unittest.skipUnless(
    HAS_TRANSFORMERS and RUN_INTERNAL_TESTS,
    "transformers and RUN_INTERNAL_TESTS=1 required for static Gemma4 runtime smoke tests",
)
class TestStaticGemma4RuntimeSmoke(unittest.TestCase):
    """Smoke tests for StaticGemma4Runtime end-to-end flow."""

    def test_gemma4_runtime_import(self):
        """StaticGemma4Runtime should be importable."""
        from tico.quantization.recipes.debug.static_gemma4_runtime import (
            StaticGemma4Runtime,
            StaticGemma4RuntimeConfig,
        )

        self.assertIsNotNone(StaticGemma4Runtime)
        self.assertIsNotNone(StaticGemma4RuntimeConfig)

    def test_gemma4_runtime_config_defaults(self):
        """StaticGemma4RuntimeConfig should have sensible defaults."""
        from tico.quantization.recipes.debug.static_gemma4_runtime import (
            StaticGemma4RuntimeConfig,
        )

        cfg = StaticGemma4RuntimeConfig()

        self.assertEqual(cfg.model, "google/gemma-4-e2b-it")
        self.assertEqual(cfg.max_seq, 2048)
        self.assertEqual(cfg.image_height, 896)
        self.assertEqual(cfg.image_width, 896)
        self.assertEqual(cfg.visual_start_idx, 0)
        self.assertEqual(cfg.num_visual_tokens, 256)
        self.assertEqual(cfg.padding_side, "right")
        self.assertEqual(cfg.device, "cpu")
        self.assertIsInstance(cfg.prompt, str)
        self.assertEqual(cfg.verify_steps, 4)
        self.assertEqual(cfg.gen_steps, 16)

    def test_gemma4_layout_validation(self):
        """StaticGemma4Layout should validate correctly."""
        from tico.quantization.recipes.debug.static_gemma4_runtime import (
            StaticGemma4Layout,
        )

        # Valid layout
        layout = StaticGemma4Layout(
            max_seq=2048,
            visual_start_idx=0,
            num_visual_tokens=256,
            batch_size=1,
        )
        # Should not raise
        layout.validate()

        # Invalid: visual tokens exceed max_seq
        invalid_layout = StaticGemma4Layout(
            max_seq=100,
            visual_start_idx=0,
            num_visual_tokens=256,
            batch_size=1,
        )
        with self.assertRaises(ValueError):
            invalid_layout.validate()


if __name__ == "__main__":
    unittest.main()
