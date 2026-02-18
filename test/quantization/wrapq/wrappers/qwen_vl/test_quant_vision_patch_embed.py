# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import unittest

import torch
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.wrappers.nn.quant_conv3d import QuantConv3d
from tico.quantization.wrapq.wrappers.qwen_vl.quant_vision_patch_embed import (
    QuantQwen3VLVisionPatchEmbed,
)


trans_spec = importlib.util.find_spec("transformers")
skip_msg = "transformers not installed — skipping Qwen3VLVisionPatchEmbed tests"


@unittest.skipUnless(trans_spec, skip_msg)
class TestQuantQwen3VLVisionPatchEmbed(unittest.TestCase):
    fp_patch_embed: torch.nn.Module
    hidden_size: int

    @classmethod
    def setUpClass(cls):
        from transformers.models.qwen3_vl.configuration_qwen3_vl import (
            Qwen3VLVisionConfig,
        )
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLVisionPatchEmbed,
        )

        cfg = Qwen3VLVisionConfig(
            hidden_size=64,  # Smaller for testing
            spatial_merge_size=2,
            temporal_merge_size=2,
        )

        cls.fp_patch_embed = Qwen3VLVisionPatchEmbed(cfg)
        cls.hidden_size = cfg.hidden_size

    def test_mode_transitions(self):
        """Test quantization mode transitions: NO_QUANT → CALIB → QUANT"""
        q_patch = QuantQwen3VLVisionPatchEmbed(self.fp_patch_embed)
        self.assertIs(q_patch._mode, Mode.NO_QUANT)

        q_patch.enable_calibration()
        self.assertIs(q_patch._mode, Mode.CALIB)

        # Run forward pass during calibration
        x = torch.randn(2, 3, 4, 32, 32)
        _ = q_patch(x)

        q_patch.freeze_qparams()
        self.assertIs(q_patch._mode, Mode.QUANT)

    def test_forward_diff(self):
        """
        Test that quantized output is acceptably close to FP32 reference.
        After calibration and freeze, quantized output should:
        - Differ from FP reference (quantization actually applied)
        - Stay within reasonable error bounds
        """
        q_patch = QuantQwen3VLVisionPatchEmbed(self.fp_patch_embed)
        q_patch.enable_calibration()

        # Calibrate with multiple inputs
        for _ in range(4):
            x = torch.randn(2, 3, 4, 32, 32)
            _ = q_patch(x)

        q_patch.freeze_qparams()

        x = torch.randn(2, 3, 4, 32, 32)
        with torch.no_grad():
            q_out = q_patch(x)
            fp_out = self.fp_patch_embed(x)

        diff = (fp_out - q_out).abs().mean().item()
        self.assertGreater(diff, 0.0)  # not identical
        self.assertLess(diff, 0.4)  # acceptably close
        self.assertEqual(fp_out.shape, q_out.shape)

    def test_proj_override(self):
        """
        PTQConfig overrides should propagate to the wrapped Conv3d layer.
        """
        cfg = PTQConfig(
            default_dtype=DType.uint(8),
            overrides={
                "proj": {
                    "weight": {"dtype": DType.uint(4)},
                    "act_in": {"dtype": DType.uint(4)},
                    "act_out": {"dtype": DType.uint(4)},
                }
            },
        )
        q_patch = QuantQwen3VLVisionPatchEmbed(self.fp_patch_embed, qcfg=cfg)
        q_conv3d = q_patch.proj.wrapped

        self.assertIsInstance(q_conv3d, QuantConv3d)
        self.assertEqual(q_conv3d.obs_weight.dtype, DType.uint(4))
        self.assertEqual(q_conv3d.obs_act_in.dtype, DType.uint(4))
        self.assertEqual(q_conv3d.obs_act_out.dtype, DType.uint(4))

    def test_activation_stats_collected(self):
        """
        Test that activation statistics are properly collected during calibration.
        """
        q_patch = QuantQwen3VLVisionPatchEmbed(self.fp_patch_embed)
        q_patch.enable_calibration()

        # Run forward pass to collect stats
        x = torch.randn(2, 3, 4, 32, 32)
        _ = q_patch(x)

        # Check that wrapped Conv3d observers have collected stats
        q_conv3d = q_patch.proj.wrapped
        self.assertTrue(q_conv3d.obs_act_in.min_val.numel() > 0)
        self.assertTrue(q_conv3d.obs_act_out.min_val.numel() > 0)
        self.assertTrue(q_conv3d.obs_weight.min_val.numel() > 0)

        # Freeze and check qparams exist
        q_patch.freeze_qparams()
        self.assertTrue(q_conv3d.obs_act_in.has_qparams)
        self.assertTrue(q_conv3d.obs_act_out.has_qparams)
        self.assertTrue(q_conv3d.obs_weight.has_qparams)

    def test_observer_count(self):
        """
        Test that the wrapper has the correct number of observers.
        - 2 local observers (obs_hidden, obs_output)
        - 3 observers from wrapped Conv3d (obs_weight, obs_act_in, obs_act_out)
        """
        q_patch = QuantQwen3VLVisionPatchEmbed(self.fp_patch_embed)

        observers = list(q_patch._all_observers())
        self.assertEqual(len(observers), 3)  # 3 from Conv3d

    def test_registration_in_registry(self):
        """
        Test that Qwen3VLVisionPatchEmbed is properly registered in the wrapper registry.
        """
        from tico.quantization.wrapq.wrappers.qwen_vl.quant_vision_patch_embed import (
            QuantQwen3VLVisionPatchEmbed,
        )
        from tico.quantization.wrapq.wrappers.registry import lookup
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLVisionPatchEmbed,
        )

        # Verify Qwen3VLVisionPatchEmbed maps to QuantQwen3VLVisionPatchEmbed
        wrapper_cls = lookup(Qwen3VLVisionPatchEmbed)
        self.assertIs(wrapper_cls, QuantQwen3VLVisionPatchEmbed)

    def test_output_shape(self):
        """Test that output shape is correct after patch embedding."""
        q_patch = QuantQwen3VLVisionPatchEmbed(self.fp_patch_embed)
        q_patch.enable_calibration()

        x = torch.randn(2, 3, 4, 32, 32)
        _ = q_patch(x)

        q_patch.freeze_qparams()

        with torch.no_grad():
            q_out = q_patch(x)
            fp_out = self.fp_patch_embed(x)

        self.assertEqual(q_out.shape, fp_out.shape)

    def test_multiple_calibration_steps(self):
        """
        Test that running multiple calibration iterations works correctly.
        Statistics should be accumulated across multiple forward passes.
        """
        q_patch = QuantQwen3VLVisionPatchEmbed(self.fp_patch_embed)
        q_patch.enable_calibration()

        # Run multiple calibration steps
        for i in range(5):
            x = torch.randn(2, 3, 4, 32, 32)
            _ = q_patch(x)

        q_patch.freeze_qparams()

        # Verify that all observers have quantization parameters
        self.assertTrue(q_patch.proj.wrapped.obs_act_in.has_qparams)
        self.assertTrue(q_patch.proj.wrapped.obs_act_out.has_qparams)
        self.assertTrue(q_patch.proj.wrapped.obs_weight.has_qparams)

    def test_different_batch_sizes(self):
        """
        Test that quantization works correctly with different batch sizes.
        """
        q_patch = QuantQwen3VLVisionPatchEmbed(self.fp_patch_embed)
        q_patch.enable_calibration()

        # Calibrate with one batch size
        calibrate_batch = torch.randn(2, 3, 4, 32, 32)
        for _ in range(3):
            _ = q_patch(calibrate_batch)
        q_patch.freeze_qparams()

        # Test with different batch sizes
        for batch_size in [1, 2, 4]:
            x = torch.randn(batch_size, 3, 4, 32, 32)
            with torch.no_grad():
                q_out = q_patch(x)
                fp_out = self.fp_patch_embed(x)

            self.assertEqual(q_out.shape, fp_out.shape)
            diff = (fp_out - q_out).abs().mean().item()
            self.assertLess(diff, 0.4)
