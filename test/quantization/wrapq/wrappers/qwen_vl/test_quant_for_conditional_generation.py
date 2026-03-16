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

import pathlib
import tempfile
import unittest
import warnings

import tico

import torch
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.utils.version import has_transformers_for
from tico.quantization.wrapq.wrappers.nn.quant_linear import QuantLinear
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.qwen_vl.quant_for_conditional_generation import (
    QuantQwen3VLForConditionalGeneration,
)


skip_msg = "required transformers not installed — skipping Qwen3VLForConditionalGeneration tests"


@unittest.skipUnless(has_transformers_for("qwen3-vl"), skip_msg)
class TestQuantQwen3VLForConditionalGeneration(unittest.TestCase):
    model_fp: torch.nn.Module
    config: object  # Will be Qwen3VLConfig but we don't want to import it at module level

    @classmethod
    def setUpClass(cls):
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

        # Import the original model class
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLForConditionalGeneration,
        )

        # Create a small config for testing
        config = Qwen3VLConfig(
            vision_config={
                "hidden_size": 32,
                "intermediate_size": 64,
                "depth": 2,
                "num_heads": 4,
                "patch_size": 14,
                "temporal_patch_size": 1,
                "in_channels": 3,
                "num_position_embeddings": 144,  # 12*12
                "spatial_merge_size": 2,
                "deepstack_visual_indexes": [],
            },
            text_config={
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "max_position_embeddings": 128,
                "vocab_size": 1000,
                "pad_token_id": 0,
                "rope_scaling": {},
            },
            image_token_id=1,
            video_token_id=2,
            vision_start_token_id=3,
        )

        cls.model_fp = Qwen3VLForConditionalGeneration(config)
        cls.config = config

    def test_mode_transitions(self):
        qmodel = QuantQwen3VLForConditionalGeneration(self.model_fp)
        self.assertIs(qmodel._mode, Mode.NO_QUANT)

        qmodel.enable_calibration()
        self.assertIs(qmodel._mode, Mode.CALIB)

        # Create dummy inputs
        input_ids = torch.randint(0, self.config.text_config.vocab_size, (1, 10))
        # For simplicity, not providing pixel_values, so no vision processing

        _ = qmodel(input_ids=input_ids)

        # For simplicity, not providing pixel_values, so no vision processing

        qmodel.freeze_qparams()
        self.assertIs(qmodel._mode, Mode.QUANT)

    def test_forward_diff(self):
        qmodel = QuantQwen3VLForConditionalGeneration(self.model_fp)
        qmodel.enable_calibration()
        for _ in range(2):
            inp = torch.randint(0, self.config.text_config.vocab_size, (1, 10))
            _ = qmodel(input_ids=inp)
        qmodel.freeze_qparams()

        x = torch.randint(0, self.config.text_config.vocab_size, (1, 10))
        with torch.no_grad():
            q_out = qmodel(input_ids=x)
            fp_out = self.model_fp(input_ids=x)

        # Check that outputs are close but not identical (due to quantization)
        diff = (fp_out.logits - q_out.logits).abs().mean().item()
        self.assertGreater(diff, 0.0)
        # The threshold might need adjustment based on actual behavior
        self.assertLess(diff, 1.0)

    def test_lm_head_override(self):
        cfg = PTQConfig(
            default_dtype=DType.uint(8),
            overrides={
                "lm_head": {
                    "act_in": {"dtype": DType.uint(4)},
                    "act_out": {"dtype": DType.uint(4)},
                }
            },
        )
        qmodel = QuantQwen3VLForConditionalGeneration(self.model_fp, qcfg=cfg)
        # We know qmodel.lm_head is a PTQWrapper wrapping a QuantLinear
        assert isinstance(qmodel.lm_head, PTQWrapper)
        q_lin = qmodel.lm_head.wrapped

        self.assertIsInstance(q_lin, QuantLinear)
        # type: ignore below because obs_act_in and obs_act_out are not in the base class interface
        self.assertEqual(q_lin.obs_act_in.dtype, DType.uint(4))
        self.assertEqual(q_lin.obs_act_out.dtype, DType.uint(4))


class TestSubgraphExport(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLForConditionalGeneration,
        )

        config = Qwen3VLConfig(
            vision_config={
                "hidden_size": 16,
                "intermediate_size": 32,
                "depth": 1,
                "num_heads": 2,
                "patch_size": 4,
                "temporal_patch_size": 1,
                "in_channels": 3,
                "num_position_embeddings": 16,  # 4*4
                "spatial_merge_size": 2,
                "deepstack_visual_indexes": [],
            },
            text_config={
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "max_position_embeddings": 32,
                "vocab_size": 100,
                "pad_token_id": 0,
                "rope_scaling": {},
            },
            image_token_id=1,
            video_token_id=2,
            vision_start_token_id=3,
        )

        model_fp = Qwen3VLForConditionalGeneration(config)
        self.model_int8 = QuantQwen3VLForConditionalGeneration(model_fp).eval()
        self.input_ids = torch.randint(0, config.text_config.vocab_size, (1, 8))

    def test_calib_quant_export(self):
        # calib
        self.model_int8.enable_calibration()
        _ = self.model_int8(self.input_ids)
        self.model_int8.freeze_qparams()

        self.assertIs(self.model_int8._mode, Mode.QUANT)

        # export
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "model.circle"
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                exported = tico.convert(self.model_int8, (self.input_ids,))
            exported.save(path)
            self.assertTrue(path.exists())
