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

import copy
import pathlib
import tempfile
import unittest
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionPatchMerger

import warnings

import tico

import torch
from tico.circle import CircleDocument
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.config.specs import affine, mx
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.observers.base import ObserverBase
from tico.quantization.wrapq.utils.version import has_transformers_for
from tico.quantization.wrapq.wrappers.nn.quant_layernorm import QuantLayerNorm
from tico.quantization.wrapq.wrappers.nn.quant_linear import QuantLinear
from tico.quantization.wrapq.wrappers.qwen_vl.quant_vision_patch_merger import (
    QuantQwen3VLVisionPatchMerger,
)

from test.quantization.quant_spec_helpers import make_affine_ptq_config
from test.support.quantization.mx_layout import assert_qwen_mx_layout

skip_msg = (
    "required transformers not installed — skipping Qwen3VLVisionPatchMerger tests"
)


@unittest.skipUnless(has_transformers_for("qwen3-vl"), skip_msg)
class TestQuantQwen3VLVisionPatchMerger(unittest.TestCase):
    fp_merger: "Qwen3VLVisionPatchMerger"
    hidden_size: int
    out_hidden_size: int
    spatial_merge_size: int

    @classmethod
    def setUpClass(cls):
        from transformers.models.qwen3_vl.configuration_qwen3_vl import (
            Qwen3VLVisionConfig,
        )
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLVisionPatchMerger,
        )

        # Use smaller sizes for testing
        cfg = Qwen3VLVisionConfig(
            hidden_size=64,
            spatial_merge_size=2,
            out_hidden_size=32,
        )

        cls.fp_merger = Qwen3VLVisionPatchMerger(cfg, use_postshuffle_norm=False)
        cls.hidden_size = cfg.hidden_size
        cls.out_hidden_size = cfg.out_hidden_size
        cls.spatial_merge_size = cfg.spatial_merge_size

    def test_mode_transitions(self):
        """Test quantization mode transitions: NO_QUANT → CALIB → QUANT"""
        q_merger = QuantQwen3VLVisionPatchMerger(self.fp_merger)
        self.assertIs(q_merger._mode, Mode.NO_QUANT)

        q_merger.enable_calibration()
        self.assertIs(q_merger._mode, Mode.CALIB)

        # Run forward pass during calibration
        x = torch.randn(32, self.hidden_size)
        _ = q_merger(x)

        q_merger.freeze_qparams()
        self.assertIs(q_merger._mode, Mode.QUANT)

    def test_forward_diff(self):
        """
        Test that quantized output is acceptably close to FP32 reference.
        """
        torch.manual_seed(42)
        q_merger = QuantQwen3VLVisionPatchMerger(self.fp_merger)
        q_merger.enable_calibration()

        # Calibrate with multiple inputs
        for _ in range(4):
            x = torch.randn(32, self.hidden_size)
            _ = q_merger(x)

        q_merger.freeze_qparams()

        x = torch.randn(32, self.hidden_size)
        with torch.no_grad():
            q_out = q_merger(x)
            fp_out = self.fp_merger(x)

        self.assertEqual(fp_out.shape, q_out.shape)
        diff = (fp_out - q_out).abs().mean().item()
        self.assertGreater(diff, 0.0)  # not identical
        self.assertLess(diff, 0.7)  # acceptably close

    def test_module_override(self):
        """
        PTQConfig overrides should propagate to wrapped submodules.
        """
        cfg = make_affine_ptq_config(
            dtype=DType.uint(8),
            overrides={
                "linear_fc1": {
                    "weight": {"dtype": DType.uint(4)},
                    "act_in": {"dtype": DType.uint(4)},
                    "act_out": {"dtype": DType.uint(4)},
                },
                "linear_fc2": {
                    "weight": {"dtype": DType.uint(4)},
                },
                "act_fn": {
                    "act_in": {"dtype": DType.uint(4)},
                    "act_out": {"dtype": DType.uint(4)},
                },
            },
        )
        q_merger = QuantQwen3VLVisionPatchMerger(self.fp_merger, qcfg=cfg)

        # Check linear_fc1
        q_fc1 = q_merger.linear_fc1.wrapped
        self.assertIsInstance(q_fc1, QuantLinear)
        self.assertEqual(q_fc1.obs_weight.dtype, DType.uint(4))
        self.assertEqual(q_fc1.obs_act_in.dtype, DType.uint(4))
        self.assertEqual(q_fc1.obs_act_out.dtype, DType.uint(4))

        # Check linear_fc2
        q_fc2 = q_merger.linear_fc2.wrapped
        self.assertIsInstance(q_fc2, QuantLinear)
        self.assertEqual(q_fc2.obs_weight.dtype, DType.uint(4))

        # Check act_fn (QuantGELU via QuantElementwise)
        q_act = q_merger.act_fn.wrapped
        self.assertEqual(q_act.act_in_obs.dtype, DType.uint(4))
        self.assertEqual(q_act.act_out_obs.dtype, DType.uint(4))

    def test_registration_in_registry(self):
        """
        Test that Qwen3VLVisionPatchMerger is properly registered.
        """
        from tico.quantization.wrapq.wrappers.qwen_vl.quant_vision_patch_merger import (
            QuantQwen3VLVisionPatchMerger,
        )
        from tico.quantization.wrapq.wrappers.registry import lookup
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLVisionPatchMerger,
        )

        wrapper_cls = lookup(Qwen3VLVisionPatchMerger)
        self.assertIs(wrapper_cls, QuantQwen3VLVisionPatchMerger)

    def test_output_shape(self):
        """
        Test that output shape is correct.
        Input: (N, hidden_size)
        Output: (N // self.spatial_merge_size**2, out_hidden_size) = (N/4, 32)
        """
        q_merger = QuantQwen3VLVisionPatchMerger(self.fp_merger)
        q_merger.enable_calibration()

        num_patches = 32
        x = torch.randn(num_patches, self.hidden_size)
        _ = q_merger(x)

        q_merger.freeze_qparams()

        with torch.no_grad():
            q_out = q_merger(x)
            fp_out = self.fp_merger(x)

        expected_shape = (
            num_patches // self.spatial_merge_size**2,
            self.out_hidden_size,
        )
        self.assertEqual(q_out.shape, expected_shape)
        self.assertEqual(fp_out.shape, expected_shape)

    def test_use_postshuffle_norm(self):
        """
        Test with use_postshuffle_norm=True flag.
        """
        from transformers.models.qwen3_vl.configuration_qwen3_vl import (
            Qwen3VLVisionConfig,
        )
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLVisionPatchMerger,
        )

        cfg = Qwen3VLVisionConfig(
            hidden_size=64, spatial_merge_size=2, out_hidden_size=32
        )

        fp_merger = Qwen3VLVisionPatchMerger(cfg, use_postshuffle_norm=True)
        q_merger = QuantQwen3VLVisionPatchMerger(fp_merger)
        self.assertEqual(q_merger.hidden_size, fp_merger.hidden_size)

        num_patches = 32

        q_merger.enable_calibration()
        x = torch.randn(num_patches, fp_merger.hidden_size)
        _ = q_merger(x)
        q_merger.freeze_qparams()

        with torch.no_grad():
            q_out = q_merger(x)
            fp_out = fp_merger(x)

        expected_shape = (num_patches, cfg.out_hidden_size)
        self.assertEqual(fp_out.shape, expected_shape)
        self.assertEqual(q_out.shape, expected_shape)
        diff = (fp_out - q_out).abs().mean().item()
        self.assertLess(diff, 0.7)

    def test_different_batch_sizes(self):
        """
        Test that quantization works correctly with different batch sizes.
        """
        q_merger = QuantQwen3VLVisionPatchMerger(self.fp_merger)
        q_merger.enable_calibration()

        # Calibrate with one size
        calibrate_batch = torch.randn(32, self.hidden_size)
        for _ in range(3):
            _ = q_merger(calibrate_batch)
        q_merger.freeze_qparams()

        # Test with different sizes
        for num_patches in [16, 32, 64]:
            x = torch.randn(num_patches, self.hidden_size)
            with torch.no_grad():
                q_out = q_merger(x)
                fp_out = self.fp_merger(x)

            expected_shape = (
                num_patches // self.spatial_merge_size**2,
                self.out_hidden_size,
            )
            self.assertEqual(q_out.shape, expected_shape)
            self.assertEqual(fp_out.shape, expected_shape)
            diff = (fp_out - q_out).abs().mean().item()
            self.assertLess(diff, 0.7)

    def test_mx_fc_preserves_affine_norm_view_and_frozen_grids(self):
        """Both norm/view modes keep layout affine and preserve checkpoint grids."""
        from transformers.models.qwen3_vl.configuration_qwen3_vl import (
            Qwen3VLVisionConfig,
        )
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLVisionPatchMerger,
        )

        for postshuffle in (False, True):
            with self.subTest(postshuffle=postshuffle):
                torch.manual_seed(79)
                config = Qwen3VLVisionConfig(
                    hidden_size=32, spatial_merge_size=2, out_hidden_size=64
                )
                fp = Qwen3VLVisionPatchMerger(
                    config, use_postshuffle_norm=postshuffle
                ).eval()
                qcfg = PTQConfig(
                    activation=affine(DType.int(16)),
                    weight=affine(DType.uint(8)),
                    overrides={
                        f"{fc}.{site}": mx(axis=-1)
                        for fc in ("linear_fc1", "linear_fc2")
                        for site in ("act_in", "act_out")
                    },
                )
                merger = QuantQwen3VLVisionPatchMerger(fp, qcfg=qcfg).eval()
                x = torch.linspace(-2, 3, 8 * 32).reshape(8, 32)
                merger.enable_calibration()
                merger(x)
                merger.freeze_qparams()
                expected = merger(x)
                restored = QuantQwen3VLVisionPatchMerger(
                    copy.deepcopy(fp), qcfg=qcfg
                ).eval()
                restored.load_state_dict(copy.deepcopy(merger.state_dict()))
                self.assertTrue(torch.equal(restored(x), expected))
                before = {
                    name: observer.require_frozen()
                    for name, observer in restored.named_modules()
                    if isinstance(observer, ObserverBase)
                }
                for optimize in (False, True):
                    with self.subTest(circle_optimize=optimize):
                        converted = tico.convert(
                            restored,
                            (x,),
                            config=tico.CompileConfigV1(
                                legalize_mx_affine_domains=True,
                                move_affine_quantize_before_layout=True,
                                circle_optimize=optimize,
                            ),
                        )
                        document = CircleDocument.from_bytes(converted.circle_binary)
                        document.verify()
                        counts = assert_qwen_mx_layout(self, document)
                        self.assertGreater(counts["FULLY_CONNECTED"], 0)
                        self.assertGreater(counts["RESHAPE"], 0)
                for name, observer in restored.named_modules():
                    if not isinstance(observer, ObserverBase):
                        continue
                    old, new = before[name], observer.require_frozen()
                    self.assertEqual(old.policy, new.policy, name)
                    self.assertEqual(old.kind, new.kind, name)
                    for field in ("scale", "zero_point"):
                        a, b = getattr(old, field), getattr(new, field)
                        if a is None:
                            self.assertIsNone(b, name)
                        else:
                            self.assertTrue(torch.equal(a, b), name)
                self.assertTrue(torch.equal(restored(x), expected))

    def test_subgraph_export(self):
        """
        Test that quantized merger can be exported to Circle format.
        """
        q_merger = QuantQwen3VLVisionPatchMerger(self.fp_merger).eval()
        x = torch.randn(16, self.hidden_size)

        # Calibrate and freeze
        q_merger.enable_calibration()
        _ = q_merger(x)
        q_merger.freeze_qparams()

        self.assertIs(q_merger._mode, Mode.QUANT)

        # Export to Circle
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "patch_merger.circle"
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                exported = tico.convert(q_merger, (x,))
            exported.save(path)
            self.assertTrue(path.exists())
