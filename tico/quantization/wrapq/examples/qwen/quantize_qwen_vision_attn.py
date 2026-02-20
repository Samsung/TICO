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

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor  # since 4.5

from tico.quantization import convert, prepare
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.evaluation.metric import compute_peir
from tico.quantization.evaluation.utils import plot_two_outputs
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.wrappers.qwen_vl.quant_vision_attn import (
    QuantQwen3VLVisionAttention,
)
from tico.utils.utils import SuppressWarning


def get_position_embeddings(model, grid_thw: torch.Tensor):
    pos_embeds = model.fast_pos_embed_interpolate(grid_thw)

    rotary_pos_emb = model.rot_pos_emb(grid_thw)

    seq_len, _ = pos_embeds.size()
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())
    return position_embeddings


def get_cu_seqlens(grid_thw: torch.Tensor):
    cu_seqlens = torch.repeat_interleave(
        grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
    ).cumsum(
        dim=0,
        # Select dtype based on the following factors:
        #  - FA2 requires that cu_seqlens_q must have dtype int32
        #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
        # See https://github.com/huggingface/transformers/pull/34852 for more information
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)
    return cu_seqlens


# -------------------------------------------------------------------------
# 0. Load a Qwen3-VL model (text tower) + tokenizer
# -------------------------------------------------------------------------
name = "Qwen/Qwen3-VL-2B-Instruct"
model = AutoModelForImageTextToText.from_pretrained(
    name,
    device_map="cpu",
    trust_remote_code=True,
    dtype=torch.float32,
)
model.eval()

processor = AutoProcessor.from_pretrained(name, trust_remote_code=True)
# 1) Build chat-style multimodal messages (image token + text)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {
                "type": "text",
                "text": (
                    f"Describe the picture\n"
                    "Return ONLY the final answer with no extra words."
                ),
            },
        ],
    }
]

# 2) Render prompt that includes image tokens
prompt = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)


# -------------------------------------------------------------------------
# 1. Replace layer-0’s attn with QuantQwen3VLVisionAttention
# -------------------------------------------------------------------------
orig_attn = model.model.visual.blocks[0].attn
attn_q = prepare(orig_attn, PTQConfig())
attn_q.eval()
assert isinstance(attn_q.wrapped, QuantQwen3VLVisionAttention)

# -------------------------------------------------------------------------
# 2. calibration
# -------------------------------------------------------------------------
image_size = (3, 512, 640)
examples = [
    torch.randint(0, 255, image_size),
    torch.randint(0, 255, image_size),
    torch.randint(0, 255, image_size),
]

attn_inputs = []
with torch.no_grad():
    for example in examples:
        inputs = processor(
            text=prompt,
            images=example,
            return_tensors="pt",
        )
        grid_thw = inputs["image_grid_thw"]
        pixel_values = inputs["pixel_values"]

        hidden_states = model.model.visual.patch_embed(pixel_values)
        position_embeddings = get_position_embeddings(model.model.visual, grid_thw)
        cu_seqlens = get_cu_seqlens(grid_thw)

        _ = attn_q(hidden_states, cu_seqlens, None, position_embeddings)
        attn_inputs.append((hidden_states, cu_seqlens, None, position_embeddings))

convert(attn_q)
assert attn_q._mode is Mode.QUANT, "Quantization mode should be active now."

# -------------------------------------------------------------------------
# 3. Quick diff check (INT-sim vs FP32)
# -------------------------------------------------------------------------
attn_input = attn_inputs[0]

with torch.no_grad():
    int8_out = attn_q(*attn_input)
    fp_out = orig_attn(*attn_input)

print("┌───────────── Quantization Error Summary ─────────────")
print(f"│ Mean |diff|: {(int8_out - fp_out).abs().mean().item():.6f}")
print(f"│ PEIR       : {compute_peir(fp_out, int8_out) * 100:.6f} %")
print("└──────────────────────────────────────────────────────")
print(plot_two_outputs(fp_out, int8_out))

# -------------------------------------------------------------------------
# 4. Export the quantized block
# -------------------------------------------------------------------------
import tico

save_path = pathlib.Path("qwen3vl_vision_attn.q.circle")

with SuppressWarning(UserWarning, ".*"):
    cm = tico.convert(attn_q, attn_input)
cm.save(save_path)

print(f"Quantized Circle model saved to {save_path.resolve()}")
