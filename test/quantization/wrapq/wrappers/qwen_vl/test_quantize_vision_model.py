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

"""Smoke tests migrated from the Qwen3-VL vision model quantization example."""

import copy
import os
import unittest

import tico.quantization

import torch
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.utils.version import has_transformers_for
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper

IS_INTERNAL_TEST = os.environ.get("RUN_INTERNAL_TESTS", "0") == "1"

skip_msg = "required transformers not installed — skipping Qwen3-VL vision model tests"


@unittest.skipIf(
    not IS_INTERNAL_TEST, "Internal test — run only if --include-internal is set"
)
@unittest.skipUnless(has_transformers_for("qwen3-vl"), skip_msg)
class TestQwenVisionModelExample(unittest.TestCase):
    """Exercise the old vision-model PTQ flow with a tiny static grid."""

    def test_prepare_convert_vision_model_flow_matches_example(self):
        """Quantize Qwen3VLVisionModel and validate pooled vision output."""
        from tico.quantization.wrapq.wrappers.qwen_vl.quant_vision_model import (
            QuantQwen3VLVisionModel,
        )
        from transformers.models.qwen3_vl.configuration_qwen3_vl import (
            Qwen3VLVisionConfig,
        )
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

        torch.manual_seed(123)
        cfg = Qwen3VLVisionConfig(
            hidden_size=64,
            num_heads=4,
            depth=1,
            num_position_embeddings=64,
            temporal_patch_size=2,
            patch_size=16,
            out_hidden_size=64,
            spatial_merge_size=2,
            deepstack_visual_indexes=[0],
        )
        if not hasattr(cfg, "_attn_implementation"):
            setattr(cfg, "_attn_implementation", "eager")
        else:
            cfg._attn_implementation = "eager"

        grid_tuple = (1, 2, 2)
        grid_thw = torch.tensor([grid_tuple])
        sample_shape = (
            1,
            cfg.in_channels,
            grid_tuple[0] * cfg.temporal_patch_size,
            grid_tuple[1] * cfg.patch_size,
            grid_tuple[2] * cfg.patch_size,
        )
        model = Qwen3VLVisionModel(cfg).eval()
        fp_ref = copy.deepcopy(model).eval()
        qcfg = PTQConfig(model_args={"vision": {"grid_thw": grid_tuple}})
        prepared = tico.quantization.prepare(model, qcfg, inplace=True)
        self.assertIsInstance(prepared, PTQWrapper)

        calibration_data = [torch.randn(sample_shape) for _ in range(2)]
        with torch.no_grad():
            for sample in calibration_data:
                prepared(sample, grid_thw)

        quantized = tico.quantization.convert(prepared, inplace=True)
        self.assertIs(quantized._mode, Mode.QUANT)

        with torch.no_grad():
            quant_out = quantized(calibration_data[0], grid_thw)
            fp_out = fp_ref(calibration_data[0], grid_thw)
            if QuantQwen3VLVisionModel.has_deepstack_model_output:
                quant_tensor = quant_out.pooler_output
                fp_tensor = fp_out.pooler_output
            else:
                quant_tensor = quant_out[0]
                fp_tensor = fp_out[0]

        self.assertEqual(quant_tensor.shape, fp_tensor.shape)
        self.assertTrue(torch.isfinite(quant_tensor).all())


if __name__ == "__main__":
    unittest.main()
