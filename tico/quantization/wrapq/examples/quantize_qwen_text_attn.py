# Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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
from transformers import AutoModelForVision2Seq, AutoTokenizer

from tico.quantization import convert, prepare
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.evaluation.metric import compute_peir
from tico.quantization.evaluation.utils import plot_two_outputs
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.wrappers.qwen_vl.quant_text_attn import (
    QuantQwen3VLTextAttention,
)
from tico.utils.utils import SuppressWarning

# -------------------------------------------------------------------------
# 0. Load a Qwen3-VL model (text tower) + tokenizer
# -------------------------------------------------------------------------
name = "Qwen/Qwen3-VL-2B-Instruct"
model = AutoModelForVision2Seq.from_pretrained(
    name,
    device_map="cpu",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)

MAX_SEQ = 512
text_cfg = model.config.text_config
text_cfg.max_position_embeddings = MAX_SEQ

# -------------------------------------------------------------------------
# 1. Replace layer-0’s self-attention with QuantQwen3VLTextAttention
# -------------------------------------------------------------------------
orig_attn = model.model.language_model.layers[0].self_attn
model.model.language_model.layers[0].self_attn = prepare(orig_attn, PTQConfig())
model.eval()

attn_q = model.model.language_model.layers[0].self_attn  # quant wrapper
assert isinstance(attn_q.wrapped, QuantQwen3VLTextAttention)
rotary = model.model.language_model.rotary_emb

# -------------------------------------------------------------------------
# 2. Single-pass calibration
# -------------------------------------------------------------------------
PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "In 2025, AI systems accelerated hardware-software co-design at scale.",
    "양자화는 왜 어려울까? 분포, 길이, 마스크가 관건이다.",
    "今日はいい天気ですね。ところでRoPE角度は長さに依存します。",
    "def quicksort(arr):\n    if len(arr) <= 1: return arr\n    ...",
    "Prices rose 3.14% — see Figure 2; emails: foo@bar.com!",
]

with torch.no_grad():
    for prompt in PROMPTS:
        ids = tokenizer(prompt, return_tensors="pt")
        input_ids = ids["input_ids"]
        embeds = model.model.language_model.embed_tokens(input_ids)
        cos_sin = rotary(embeds, input_ids)
        S = cos_sin[0].shape[1]
        float_mask = torch.zeros(1, 1, S, S)
        _ = attn_q(embeds, cos_sin, attention_mask=float_mask)

convert(attn_q)
assert attn_q._mode is Mode.QUANT, "Quantization mode should be active now."

# -------------------------------------------------------------------------
# 3. Quick diff check (INT-sim vs FP32)
# -------------------------------------------------------------------------
ids = tokenizer("check", return_tensors="pt")
input_ids = ids["input_ids"]
emb = model.model.language_model.embed_tokens(input_ids)
pos = rotary(emb, input_ids)
S = pos[0].shape[1]
float_mask = torch.zeros(1, 1, S, S)
with torch.no_grad():
    int8_out, _ = attn_q(emb, pos, attention_mask=float_mask)
    fp_out, _ = orig_attn(emb, position_embeddings=pos, attention_mask=float_mask)

print("┌───────────── Quantization Error Summary ─────────────")
print(f"│ Mean |diff|: {(int8_out - fp_out).abs().mean().item():.6f}")
print(f"│ PEIR       : {compute_peir(fp_out, int8_out) * 100:.6f} %")
print("└──────────────────────────────────────────────────────")
print(plot_two_outputs(fp_out, int8_out))

# -------------------------------------------------------------------------
# 4. Export the quantized block
# -------------------------------------------------------------------------
import tico

save_path = pathlib.Path("qwen3vl_text_attn.q.circle")
B, S, D = 1, 4, text_cfg.hidden_size
example = torch.randn(B, S, D)
example_ids = torch.arange(S)[None, :]
example_pos = rotary(example, example_ids)

with SuppressWarning(UserWarning, ".*"):
    cm = tico.convert(attn_q, (example, example_pos, None))
cm.save(save_path)

print(f"Quantized Circle model saved to {save_path.resolve()}")
