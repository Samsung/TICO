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

try:
    from quantization.recipes.optional_dependency_stubs import (
        install_optional_dependency_stubs,
    )
except ModuleNotFoundError:
    from optional_dependency_stubs import install_optional_dependency_stubs

install_optional_dependency_stubs()

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import tico.quantization.recipes.adapters.gemma4 as gemma4_mod
from tico.quantization.recipes.adapters.gemma4 import Gemma4Adapter
from tico.quantization.recipes.context import RecipeContext


class _TestGemma4Adapter(Gemma4Adapter):
    """Concrete test double of Gemma4Adapter for testing."""

    def export(self, ctx: RecipeContext) -> None:
        pass


class TestGemma4AdapterStaticCalibration(unittest.TestCase):
    """Tests for Gemma4 static image sizing during calibration input build."""

    def test_build_calibration_inputs_forces_static_image_processor_size(self):
        """Static Gemma4 image dimensions should be applied during preprocessing."""
        adapter = _TestGemma4Adapter()
        image_processor = SimpleNamespace(
            do_resize=False,
            size={"height": 224, "width": 224},
            crop_size={"height": 224, "width": 224},
        )
        processor = SimpleNamespace(image_processor=image_processor)
        ctx = RecipeContext(
            cfg={
                "calibration": {
                    "dataset": "vqav2",
                    "split": "testdev",
                    "n_samples": 2,
                    "seq_len": 2048,
                },
                "runtime": {"seed": 123},
                "model_args": {
                    "vision": {
                        "image_height": 896,
                        "image_width": 896,
                        "num_visual_tokens": 256,
                    }
                },
            },
            adapter=adapter,
            processor=processor,
        )
        calls = []

        def fake_build_vlm_calibration_inputs(**kwargs):
            calls.append(kwargs)
            self.assertTrue(image_processor.do_resize)
            self.assertEqual(image_processor.size, {"height": 896, "width": 896})
            self.assertEqual(image_processor.crop_size, {"height": 896, "width": 896})
            return [{"input_ids": "sample"}]

        with patch.object(
            gemma4_mod,
            "build_vlm_calibration_inputs",
            fake_build_vlm_calibration_inputs,
        ):
            result = adapter.build_calibration_inputs(ctx)

        self.assertEqual(result, [{"input_ids": "sample"}])
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0]["processor"], processor)
        self.assertEqual(calls[0]["dataset"], "vqav2")
        self.assertEqual(calls[0]["split"], "testdev")
        self.assertEqual(calls[0]["n_samples"], 2)
        self.assertEqual(calls[0]["max_seq_len"], 2048)
        self.assertEqual(calls[0]["seed"], 123)
        self.assertFalse(image_processor.do_resize)
        self.assertEqual(image_processor.size, {"height": 224, "width": 224})
        self.assertEqual(image_processor.crop_size, {"height": 224, "width": 224})

    def test_build_calibration_inputs_without_static_size_keeps_processor(self):
        """Omitting static image dimensions should preserve processor settings."""
        adapter = _TestGemma4Adapter()
        image_processor = SimpleNamespace(
            do_resize=False,
            size={"height": 224, "width": 224},
        )
        processor = SimpleNamespace(image_processor=image_processor)
        ctx = RecipeContext(
            cfg={
                "calibration": {"dataset": "vqav2"},
                "runtime": {},
                "model_args": {"vision": {"num_visual_tokens": 256}},
            },
            adapter=adapter,
            processor=processor,
        )

        with patch.object(
            gemma4_mod,
            "build_vlm_calibration_inputs",
            lambda **kwargs: [],
        ):
            result = adapter.build_calibration_inputs(ctx)

        self.assertEqual(result, [])
        self.assertFalse(image_processor.do_resize)
        self.assertEqual(image_processor.size, {"height": 224, "width": 224})

    def test_static_calibration_image_size_requires_height_and_width(self):
        """Partial static image size configuration should fail clearly."""
        adapter = _TestGemma4Adapter()
        ctx = RecipeContext(
            cfg={
                "model_args": {
                    "vision": {
                        "image_height": 896,
                        "num_visual_tokens": 256,
                    }
                }
            },
            adapter=adapter,
            processor=SimpleNamespace(image_processor=SimpleNamespace()),
        )

        with self.assertRaisesRegex(ValueError, "image_height"):
            adapter.build_calibration_inputs(ctx)


if __name__ == "__main__":
    unittest.main()
