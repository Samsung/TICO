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

"""Smoke tests migrated from the Qwen3-VL model quantization examples."""

import copy
import os
import unittest
from typing import Tuple

import tico.quantization

import torch
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.utils.version import has_transformers_for
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper

IS_INTERNAL_TEST = os.environ.get("RUN_INTERNAL_TESTS", "0") == "1"

skip_msg = "required transformers not installed — skipping Qwen3-VL model example tests"


def _make_tiny_qwen3vl_config():
    """Create a tiny Qwen3-VL config that is large enough for image-token tests."""
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

    return Qwen3VLConfig(
        vision_config={
            "hidden_size": 64,
            "num_heads": 4,
            "depth": 2,
            "temporal_patch_size": 2,
            "patch_size": 16,
            "out_hidden_size": 64,
            "spatial_merge_size": 2,
            "deepstack_visual_indexes": [0, 1],
        },
        text_config={
            "hidden_size": 64,
            "intermediate_size": 256,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "num_hidden_layers": 2,
            "attention_bias": False,
            "attention_dropout": 0.0,
            "max_position_embeddings": 1024,
            "vocab_size": 1000,
            "use_cache": False,
            "rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 2]},
        },
        image_token_id=998,
        video_token_id=999,
    )


def _make_ptq_config(cfg, thw: Tuple[int, int, int], visual_start_idx: int = 0):
    """Create the PTQ config used by the old synthetic Qwen3-VL examples."""
    return PTQConfig(
        model_args={
            "vision": {
                "grid_thw": thw,
                "visual_start_idx": visual_start_idx,
                "spatial_merge_size": cfg.vision_config.spatial_merge_size,
            }
        }
    )


def _compute_3d_position_ids(
    input_ids: torch.Tensor,
    thw: Tuple[int, int, int],
    spatial_merge_size: int,
    image_token_id: int,
) -> torch.Tensor:
    """Compute multimodal 3D RoPE position IDs for a single visual segment."""
    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    position_ids = torch.ones(
        3,
        batch_size,
        seq_len,
        dtype=input_ids.dtype,
        device=device,
    )

    for i in range(batch_size):
        image_mask = input_ids[i] == image_token_id
        image_positions = torch.nonzero(image_mask, as_tuple=True)[0]

        llm_pos_ids_list: list[torch.Tensor] = []
        st = 0

        if len(image_positions) > 0:
            start_pos = image_positions[0].item()

            text_len = start_pos - st
            if text_len > 0:
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                llm_pos_ids_list.append(
                    torch.arange(text_len, device=device).view(1, -1).expand(3, -1)
                    + st_idx
                )

            llm_grid_t = 1
            llm_grid_h = thw[1] // spatial_merge_size
            llm_grid_w = thw[2] // spatial_merge_size

            t_index = (
                torch.arange(llm_grid_t, device=device)
                .view(-1, 1)
                .expand(-1, llm_grid_h * llm_grid_w)
                .flatten()
            )
            h_index = (
                torch.arange(llm_grid_h, device=device)
                .view(1, -1, 1)
                .expand(llm_grid_t, -1, llm_grid_w)
                .flatten()
            )
            w_index = (
                torch.arange(llm_grid_w, device=device)
                .view(1, 1, -1)
                .expand(llm_grid_t, llm_grid_h, -1)
                .flatten()
            )
            st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
            llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + st_idx)

            num_visual_tokens = (thw[1] // spatial_merge_size) * (
                thw[2] // spatial_merge_size
            )
            st = start_pos + num_visual_tokens

        if st < seq_len:
            st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
            text_len = seq_len - st
            llm_pos_ids_list.append(
                torch.arange(text_len, device=device).view(1, -1).expand(3, -1) + st_idx
            )

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        position_ids[..., i, :] = llm_positions

    return position_ids


def _create_image_input(
    cfg,
    seq_len: int,
    thw: Tuple[int, int, int],
    *,
    visual_start_idx: int = 0,
    include_generation_fields: bool = False,
):
    """Create one synthetic Qwen3-VL image prompt without using a processor."""
    spatial_merge_size = cfg.vision_config.spatial_merge_size
    num_visual_tokens = (thw[1] // spatial_merge_size) * (thw[2] // spatial_merge_size)
    assert visual_start_idx + num_visual_tokens <= seq_len

    input_ids = torch.randint(
        low=0,
        high=cfg.text_config.vocab_size - 2,
        size=(1, seq_len),
        dtype=torch.long,
    )
    input_ids[
        0, visual_start_idx : visual_start_idx + num_visual_tokens
    ] = cfg.image_token_id

    mm_token_type_ids = torch.zeros_like(input_ids)
    mm_token_type_ids[
        0, visual_start_idx : visual_start_idx + num_visual_tokens
    ] = 1  # text=0, image=1, video=2

    pixel_values = torch.randn(
        1,
        3,
        thw[0] * cfg.vision_config.temporal_patch_size,
        thw[1] * cfg.vision_config.patch_size,
        thw[2] * cfg.vision_config.patch_size,
    )
    image_grid_thw = torch.tensor([thw])
    position_ids = _compute_3d_position_ids(
        input_ids=input_ids,
        thw=thw,
        spatial_merge_size=spatial_merge_size,
        image_token_id=cfg.image_token_id,
    )

    example = {
        "input_ids": input_ids,
        "attention_mask": None,
        "position_ids": position_ids,
        "past_key_values": None,
        "inputs_embeds": None,
        "pixel_values": pixel_values,
        "pixel_values_videos": None,
        "image_grid_thw": image_grid_thw,
        "mm_token_type_ids": mm_token_type_ids,
        "video_grid_thw": None,
        "cache_position": None,
    }
    if include_generation_fields:
        example["labels"] = None
        example["logits_to_keep"] = 0
    return example


def _run_calibration(prepared: torch.nn.Module, calibration_data: list[dict]) -> None:
    """Run calibration samples through a prepared Qwen3-VL model."""
    with torch.no_grad():
        for calibration_input in calibration_data:
            prepared(**calibration_input)


@unittest.skipIf(
    not IS_INTERNAL_TEST, "Internal test — run only if --include-internal is set"
)
@unittest.skipUnless(has_transformers_for("qwen3-vl"), skip_msg)
class TestQwenForConditionalGenerationExample(unittest.TestCase):
    """Exercise the old quantize_for_conditional_generation.py flow."""

    def test_prepare_convert_qwen3vl_generation_flow_matches_example(self):
        """Quantize Qwen3VLForConditionalGeneration and verify quantized logits."""
        from tico.quantization.wrapq.wrappers.qwen_vl.quant_for_conditional_generation import (
            QuantQwen3VLForConditionalGeneration,
        )
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLForConditionalGeneration,
        )

        torch.manual_seed(123)
        cfg = _make_tiny_qwen3vl_config()
        thw = (1, 8, 8)
        ptq_config = _make_ptq_config(cfg, thw)

        model = Qwen3VLForConditionalGeneration(cfg).eval()
        fp_ref = copy.deepcopy(model).eval()

        prepared = tico.quantization.prepare(model, ptq_config, inplace=True)
        self.assertIsInstance(prepared, PTQWrapper)
        self.assertIsInstance(prepared.wrapped, QuantQwen3VLForConditionalGeneration)
        self.assertIs(prepared._mode, Mode.CALIB)

        calibration_data = [
            _create_image_input(
                cfg,
                seq_len=50,
                thw=thw,
                include_generation_fields=True,
            )
            for _ in range(2)
        ]
        _run_calibration(prepared, calibration_data)

        quantized = tico.quantization.convert(prepared, inplace=True)
        self.assertIs(quantized._mode, Mode.QUANT)

        test_input = dict(calibration_data[0])
        test_input["position_ids"] = None
        with torch.no_grad():
            quant_out = quantized(**test_input, return_dict=False)[0]
            fp_out = fp_ref(**test_input, return_dict=False)[0]

        diff = (quant_out - fp_out).abs().mean().item()
        self.assertEqual(quant_out.shape, fp_out.shape)
        self.assertGreaterEqual(diff, 0.0)
        self.assertLess(diff, 10.0)
        self.assertTrue(torch.isfinite(quant_out).all())


if __name__ == "__main__":
    unittest.main()
