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

"""Input-contract tests for Qwen3-VL wrapper-smoke cases."""

import unittest

import torch

from tico.quantization.recipes.debug.wrapper_smoke.cases.qwen3_vl import (
    _create_image_input,
    _has_qwen3_vl,
    _make_tiny_qwen3vl_config,
    QwenVisionModelCase,
    QwenVisionPatchEmbedCase,
)


_AVAILABILITY = _has_qwen3_vl()


@unittest.skipUnless(
    _AVAILABILITY.available,
    _AVAILABILITY.reason or "Qwen3-VL modules are unavailable",
)
class TestQwen3VLWrapperSmokeInputs(unittest.TestCase):
    """Ensure synthetic Qwen3-VL vision inputs use flattened patches."""

    @staticmethod
    def _expected_shape(vision_cfg, thw):
        num_patches = thw[0] * thw[1] * thw[2]
        patch_dim = (
            vision_cfg.in_channels
            * vision_cfg.temporal_patch_size
            * vision_cfg.patch_size
            * vision_cfg.patch_size
        )
        return (1, num_patches, patch_dim)

    def test_patch_embed_case_uses_batch_one_flattened_patches(self):
        """Patch-embed calibration must not reintroduce an NCTHW input."""
        case = QwenVisionPatchEmbedCase()
        module, _ = case.build({})

        samples = case.calibration_inputs(module, {})
        self.assertGreater(len(samples), 0)
        for sample in samples:
            pixel_values = sample.args[0]
            self.assertEqual(
                tuple(pixel_values.shape),
                self._expected_shape(case.vision_cfg, case.grid_tuple),
            )
            self.assertEqual(pixel_values.ndim, 3)

    def test_vision_model_case_uses_batch_one_flattened_patches(self):
        """Vision-model calibration and export must share the rank-3 ABI."""
        case = QwenVisionModelCase()
        module, _ = case.build({})

        sample = case.calibration_inputs(module, {})[0]
        pixel_values, grid_thw = sample.args
        self.assertEqual(
            tuple(pixel_values.shape),
            self._expected_shape(case.vision_cfg, case.grid_tuple),
        )
        self.assertEqual(pixel_values.ndim, 3)
        torch.testing.assert_close(grid_thw, case.grid_thw)

    def test_multimodal_cases_share_flattened_image_helper(self):
        """Full-model smoke inputs must pass processor-shaped patches."""
        cfg = _make_tiny_qwen3vl_config()
        thw = (1, 8, 8)

        sample = _create_image_input(cfg, seq_len=50, thw=thw)
        pixel_values = sample["pixel_values"]
        self.assertEqual(
            tuple(pixel_values.shape),
            self._expected_shape(cfg.vision_config, thw),
        )
        self.assertEqual(pixel_values.ndim, 3)
        self.assertFalse(
            any(
                isinstance(value, torch.Tensor) and value.ndim == 5
                for value in sample.values()
            )
        )


if __name__ == "__main__":
    unittest.main()
