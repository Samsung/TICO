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

"""Deterministic, download-free HF fixtures for CPU recipe integration checks."""

from typing import Any

import torch

from tico.quantization.recipes.context import RecipeContext


def qwen_model(*, seed: int = 1073, head_dim: int = 32, layers: int = 2):
    """Construct a bounded GQA/mRoPE model with a real, one-block vision tower."""
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    torch.manual_seed(seed)
    if head_dim not in (32, 128):
        raise ValueError("The CPU fixture supports head_dim 32 or 128.")
    sections = [4, 6, 6] if head_dim == 32 else [16, 24, 24]
    hidden = head_dim * 2
    config = Qwen3VLConfig.from_dict(
        dict(
            text_config={
                "vocab_size": 128,
                "hidden_size": hidden,
                "intermediate_size": hidden * 2,
                "num_hidden_layers": layers,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": head_dim,
                "max_position_embeddings": 128,
                "rope_scaling": {"rope_type": "default", "mrope_section": sections},
                "attention_dropout": 0.0,
                "use_cache": True,
            },
            vision_config={
                "depth": 1,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_heads": 2,
                "out_hidden_size": hidden,
                "patch_size": 2,
                "temporal_patch_size": 2,
                "spatial_merge_size": 2,
                "deepstack_visual_indexes": [0],
                "num_position_embeddings": 16,
            },
            image_token_id=121,
            video_token_id=122,
            vision_start_token_id=120,
            vision_end_token_id=124,
            pad_token_id=0,
            eos_token_id=127,
        )
    )
    config._attn_implementation = "eager"
    return Qwen3VLForConditionalGeneration(config).eval()


def qwen_batch(
    *, image: bool = True, length: int = 8, grid_thw: tuple[int, int, int] = (1, 2, 2)
) -> dict[str, Any]:
    """Pack a deterministic local RGB image into the flattened-patch contract.

    Spatial 2x2 patches hold two identical temporal frames in merge-block order.
    The merged grid starts at placeholder index two. Remaining token IDs are
    fixed teacher-forcing data; no generation is calibrated.
    """
    temporal, grid_h, grid_w = grid_thw
    if temporal != 1 or grid_h % 2 or grid_w % 2 or min(grid_h, grid_w) < 2:
        raise ValueError(
            "The image fixture requires t=1 and positive even spatial grids."
        )
    visual_tokens = grid_h * grid_w // 4
    if length < visual_tokens + 4:
        raise ValueError(
            "The image fixture requires room for its complete visual span."
        )
    ids = torch.arange(5, length + 5, dtype=torch.long).unsqueeze(0)
    batch: dict[str, Any] = {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
    }
    if image:
        ids[0, 1] = 120
        ids[0, 2 : 2 + visual_tokens] = 121
        ids[0, 2 + visual_tokens] = 124
        height, width = grid_h * 2, grid_w * 2
        rgb = torch.arange(height * width * 3, dtype=torch.float32).reshape(
            height, width, 3
        )
        rgb = rgb / (rgb.numel() - 1)
        pixels = rgb.permute(2, 0, 1).unsqueeze(1).expand(3, 2, height, width)
        batch["pixel_values"] = (
            pixels.reshape(3, 2, grid_h // 2, 2, 2, grid_w // 2, 2, 2)
            .permute(2, 5, 3, 6, 0, 1, 4, 7)
            .reshape(grid_h * grid_w, 24)
            .contiguous()
        )
        batch["image_grid_thw"] = torch.tensor([grid_thw], dtype=torch.long)
        batch["mm_token_type_ids"] = (ids == 121).long()
    return batch


def llama_model(*, seed: int = 1073, head_dim: int = 32, layers: int = 2):
    """Construct a bounded two-query-head GQA Llama with scaled canonical RoPE."""
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    config = LlamaConfig.from_dict(
        dict(
            vocab_size=128,
            hidden_size=head_dim * 2,
            intermediate_size=head_dim * 4,
            num_hidden_layers=layers,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=head_dim,
            max_position_embeddings=128,
            rope_scaling={"rope_type": "linear", "factor": 2.0},
            attention_dropout=0.0,
            pad_token_id=0,
            eos_token_id=127,
            use_cache=True,
        )
    )
    config._attn_implementation = "eager"
    return LlamaForCausalLM(config).eval()


def fixture_context(family: str, *, seed: int = 1073) -> RecipeContext:
    """Build standard adapter/PTQ inputs with calibration 8 and export 16."""
    from tico.quantization.recipes.adapters.llama import LlamaAdapter
    from tico.quantization.recipes.adapters.qwen3_vl import Qwen3VLAdapter

    stage = {
        "name": "ptq",
        "activation": "int16",
        "linear_weight": "uint4",
        "embedding_weight": "uint8",
        "lm_head_weight": "uint8",
        "decode_calibration_steps": 2,
        "reuse_gptq_qparams": False,
    }
    config: dict[str, Any] = {
        "model": {"family": family, "name_or_path": "local-random-config"},
        "runtime": {
            "device": "cpu",
            "dtype": "float32",
            "seed": seed,
            "show_progress": False,
        },
        "model_args": {"profile": "npu_export"},
        "calibration": {"seq_len": 8, "n_samples": 2},
        "pipeline": [stage],
        "evaluation": {"enabled": False, "max_seq_len": 64},
        "export": {"enabled": True, "max_seq_len": 16, "prefill_decode": True},
    }
    adapter: LlamaAdapter | Qwen3VLAdapter
    batches: list[Any]
    if family == "qwen3_vl":
        config["model_args"]["vision"] = {
            "grid_thw": [1, 2, 2],
            "visual_start_idx": 2,
            "spatial_merge_size": 2,
        }
        model = qwen_model(seed=seed)
        adapter = Qwen3VLAdapter()
        batches = [qwen_batch(image=False), qwen_batch(image=True)]
    elif family == "llama":
        model = llama_model(seed=seed)
        adapter = LlamaAdapter()
        batches = [torch.arange(5, 13).unsqueeze(0), torch.arange(21, 29).unsqueeze(0)]
    else:
        raise ValueError(f"Unknown CPU fixture family: {family}")
    return RecipeContext(
        cfg=config, adapter=adapter, model=model, calibration_inputs=batches
    )


def gemma_model(*, seed: int = 1073):
    """Build a dense text smoke model with bounded PLE defaults and sliding RoPE.

    PLE and shared KV are disabled explicitly; this fixture does not validate
    either feature, the vision tower, or a multimodal Gemma export.
    """
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig

    torch.manual_seed(seed)
    config = Gemma4TextConfig.from_dict(
        dict(
            vocab_size=128,
            vocab_size_per_layer_input=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_global_key_value_heads=1,
            head_dim=32,
            global_head_dim=32,
            max_position_embeddings=128,
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=8,
            rope_parameters={
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                "full_attention": {"rope_type": "default", "rope_theta": 10000.0},
            },
            hidden_size_per_layer_input=0,
            attention_k_eq_v=False,
            num_kv_shared_layers=0,
            enable_moe_block=False,
            final_logit_softcapping=30.0,
            attention_dropout=0.0,
            use_cache=True,
            pad_token_id=0,
            eos_token_id=127,
        )
    )
    config._attn_implementation = "eager"
    return Gemma4ForCausalLM(config).eval()


def gemma4_multimodal_model(*, seed: int = 1073):
    """Build a tiny multimodal Gemma4 (vision tower + text) for CPU checks.

    Unlike :func:`gemma_model`, this fixture includes the vision tower with
    the ``patch_embedder.input_proj`` Linear, so GPTQ/PTQ policies that
    special-case the vision patch embedder can be exercised end to end.
    """
    from transformers import Gemma4Config, Gemma4ForConditionalGeneration

    torch.manual_seed(seed)
    config = Gemma4Config(
        text_config={
            "vocab_size": 128,
            "vocab_size_per_layer_input": 128,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "num_global_key_value_heads": 1,
            "head_dim": 32,
            "global_head_dim": 32,
            "max_position_embeddings": 128,
            "layer_types": ["sliding_attention", "full_attention"],
            "sliding_window": 8,
            "rope_parameters": {
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                "full_attention": {"rope_type": "default", "rope_theta": 10000.0},
            },
            "hidden_size_per_layer_input": 0,
            "attention_k_eq_v": False,
            "num_kv_shared_layers": 0,
            "enable_moe_block": False,
            "final_logit_softcapping": 30.0,
            "attention_dropout": 0.0,
            "use_cache": True,
            "pad_token_id": 0,
            "eos_token_id": 127,
        },
        vision_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "patch_size": 2,
            "position_embedding_size": 64,
            "pooling_kernel_size": 2,
        },
        # The Gemma4 GPTQ quantizer does not process the audio tower; keep a
        # small one so checks can verify it is excluded from GPTQ planning.
        audio_config={
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "subsampling_conv_channels": (8, 4),
            "conv_kernel_size": 3,
            "output_proj_dims": 16,
        },
        image_token_id=126,
    )
    config._attn_implementation = "eager"
    return Gemma4ForConditionalGeneration(config).eval()
