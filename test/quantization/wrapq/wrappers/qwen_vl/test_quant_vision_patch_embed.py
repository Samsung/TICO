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

import unittest

import torch
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.utils.version import has_transformers_for
from tico.quantization.wrapq.wrappers.qwen_vl.quant_vision_patch_embed import (
    QuantQwen3VLVisionPatchEmbed,
)

from test.quantization.quant_spec_helpers import make_affine_ptq_config


skip_msg = (
    "required transformers not installed — skipping Qwen3VLVisionPatchEmbed tests"
)


@unittest.skipUnless(has_transformers_for("qwen3-vl"), skip_msg)
class TestQuantQwen3VLVisionPatchEmbed(unittest.TestCase):
    fp_patch_embed: torch.nn.Module
    hidden_size: int
    patch_dim: int

    @classmethod
    def setUpClass(cls):
        from transformers.models.qwen3_vl.configuration_qwen3_vl import (
            Qwen3VLVisionConfig,
        )
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLVisionPatchEmbed,
        )

        cfg = Qwen3VLVisionConfig(
            hidden_size=64,
            spatial_merge_size=2,
            temporal_merge_size=2,
        )

        cls.fp_patch_embed = Qwen3VLVisionPatchEmbed(cfg)
        cls.hidden_size = cfg.hidden_size
        cls.patch_dim = (
            cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size
        )

    @classmethod
    def _make_input(cls, *, batch_size: int = 1, num_patches: int = 8) -> torch.Tensor:
        """Create processor-style flattened Qwen3-VL patch input."""
        return torch.randn(batch_size, num_patches, cls.patch_dim)

    def test_mode_transitions(self):
        """Test quantization mode transitions: NO_QUANT → CALIB → QUANT."""
        q_patch = QuantQwen3VLVisionPatchEmbed(self.fp_patch_embed)
        self.assertIs(q_patch._mode, Mode.NO_QUANT)

        q_patch.enable_calibration()
        self.assertIs(q_patch._mode, Mode.CALIB)

        _ = q_patch(self._make_input())

        q_patch.freeze_qparams()
        self.assertIs(q_patch._mode, Mode.QUANT)

    def test_linearized_projection_matches_fp_reference(self):
        """The Linear projection must exactly match the original Conv3d path."""
        q_patch = QuantQwen3VLVisionPatchEmbed(self.fp_patch_embed)
        x = self._make_input(batch_size=2, num_patches=5)

        with torch.no_grad():
            q_out = q_patch(x)
            fp_out = self.fp_patch_embed(x)

        torch.testing.assert_close(q_out, fp_out)

    def test_projection_parameters_are_flattened_without_reordering(self):
        """Conv3d parameters must be copied to Linear in C-T-H-W order."""
        q_patch = QuantQwen3VLVisionPatchEmbed(self.fp_patch_embed)
        q_linear = q_patch.proj.wrapped
        linear = q_linear.module
        conv3d = self.fp_patch_embed.proj

        self.assertIsInstance(linear, torch.nn.Linear)
        self.assertEqual(
            linear.weight.shape,
            (conv3d.out_channels, self.patch_dim),
        )
        torch.testing.assert_close(
            linear.weight,
            conv3d.weight.reshape(conv3d.out_channels, self.patch_dim),
        )
        if conv3d.bias is not None:
            self.assertIsNotNone(linear.bias)
            torch.testing.assert_close(linear.bias, conv3d.bias)

    def test_forward_diff(self):
        """Quantized output should differ from and remain close to FP output."""
        q_patch = QuantQwen3VLVisionPatchEmbed(self.fp_patch_embed)
        q_patch.enable_calibration()

        for _ in range(4):
            _ = q_patch(self._make_input())

        q_patch.freeze_qparams()

        x = self._make_input()
        with torch.no_grad():
            q_out = q_patch(x)
            fp_out = self.fp_patch_embed(x)

        diff = (fp_out - q_out).abs().mean().item()
        self.assertGreater(diff, 0.0)
        self.assertLess(diff, 0.7)
        self.assertEqual(fp_out.shape, q_out.shape)

    def test_proj_override(self):
        """PTQConfig overrides should propagate to the wrapped Linear layer."""
        cfg = make_affine_ptq_config(
            dtype=DType.uint(8),
            overrides={
                "proj": {
                    "weight": {"dtype": DType.uint(4)},
                    "act_in": {"dtype": DType.uint(4)},
                    "act_out": {"dtype": DType.uint(4)},
                }
            },
        )
        q_patch = QuantQwen3VLVisionPatchEmbed(self.fp_patch_embed, qcfg=cfg)
        q_linear = q_patch.proj.wrapped

        self.assertEqual(type(q_linear).__name__, "QuantLinear")
        self.assertEqual(q_linear.obs_weight.dtype, DType.uint(4))
        self.assertEqual(q_linear.obs_act_in.dtype, DType.uint(4))
        self.assertEqual(q_linear.obs_act_out.dtype, DType.uint(4))

    def test_activation_stats_collected(self):
        """Activation and weight statistics should be collected by QuantLinear."""
        q_patch = QuantQwen3VLVisionPatchEmbed(self.fp_patch_embed)
        q_patch.enable_calibration()

        _ = q_patch(self._make_input())

        q_linear = q_patch.proj.wrapped
        self.assertTrue(q_linear.obs_act_in.min_val.numel() > 0)
        self.assertTrue(q_linear.obs_act_out.min_val.numel() > 0)
        self.assertTrue(q_linear.obs_weight.min_val.numel() > 0)

        q_patch.freeze_qparams()
        self.assertTrue(q_linear.obs_act_in.has_qparams)
        self.assertTrue(q_linear.obs_act_out.has_qparams)
        self.assertTrue(q_linear.obs_weight.has_qparams)

    def test_observer_count(self):
        """The wrapper should expose three observers through ``proj``."""
        q_patch = QuantQwen3VLVisionPatchEmbed(self.fp_patch_embed)

        self.assertEqual(len(list(q_patch._all_observers())), 0)

        observers = list(q_patch.named_observers())
        self.assertEqual(len(observers), 3)
        self.assertSetEqual(
            {name for name, _ in observers},
            {"proj.weight", "proj.act_in", "proj.act_out"},
        )

    def test_registration_in_registry(self):
        """Qwen3VLVisionPatchEmbed should map to this wrapper."""
        from tico.quantization.wrapq.wrappers.registry import lookup
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLVisionPatchEmbed,
        )

        wrapper_cls = lookup(Qwen3VLVisionPatchEmbed)
        self.assertIs(wrapper_cls, QuantQwen3VLVisionPatchEmbed)

    def test_output_shape(self):
        """The wrapper should preserve the original flattened output shape."""
        q_patch = QuantQwen3VLVisionPatchEmbed(self.fp_patch_embed)
        q_patch.enable_calibration()

        x = self._make_input(num_patches=7)
        _ = q_patch(x)
        q_patch.freeze_qparams()

        with torch.no_grad():
            q_out = q_patch(x)
            fp_out = self.fp_patch_embed(x)

        self.assertEqual(q_out.shape, fp_out.shape)
        self.assertEqual(q_out.shape, (7, self.hidden_size))

    def test_multiple_calibration_steps(self):
        """Statistics should accumulate across multiple calibration steps."""
        q_patch = QuantQwen3VLVisionPatchEmbed(self.fp_patch_embed)
        q_patch.enable_calibration()

        for _ in range(5):
            _ = q_patch(self._make_input())

        q_patch.freeze_qparams()
        q_linear = q_patch.proj.wrapped
        self.assertTrue(q_linear.obs_act_in.has_qparams)
        self.assertTrue(q_linear.obs_act_out.has_qparams)
        self.assertTrue(q_linear.obs_weight.has_qparams)

    def test_flattens_leading_dimensions_into_batch_one_sequence(self):
        """Leading input dimensions should be folded without changing semantics."""
        q_patch = QuantQwen3VLVisionPatchEmbed(self.fp_patch_embed)

        for batch_size in [1, 2, 4]:
            with self.subTest(batch_size=batch_size):
                x = self._make_input(batch_size=batch_size, num_patches=3)
                with torch.no_grad():
                    q_out = q_patch(x)
                    fp_out = self.fp_patch_embed(x)

                torch.testing.assert_close(q_out, fp_out)
                self.assertEqual(
                    q_out.shape,
                    (batch_size * 3, self.hidden_size),
                )

    def test_export_graph_contains_linear_without_conv3d(self):
        """Export should expose rank-3 Linear input and no Conv3d operation."""
        q_patch = QuantQwen3VLVisionPatchEmbed(self.fp_patch_embed).eval()
        x = self._make_input(num_patches=4)

        exported = torch.export.export(q_patch, (x,))
        call_nodes = [
            node for node in exported.graph.nodes if node.op == "call_function"
        ]
        targets = {node.target for node in call_nodes}

        self.assertIn(torch.ops.aten.linear.default, targets)
        self.assertNotIn(torch.ops.aten.conv3d.default, targets)
        self.assertNotIn(torch.ops.aten.conv3d.padding, targets)

        linear_node = next(
            node for node in call_nodes if node.target == torch.ops.aten.linear.default
        )
        linear_input = linear_node.args[0]
        self.assertIsInstance(linear_input, torch.fx.Node)
        self.assertEqual(
            tuple(linear_input.meta["val"].shape),
            (1, 4, self.patch_dim),
        )


if __name__ == "__main__":
    unittest.main()
