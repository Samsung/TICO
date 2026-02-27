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
    dtype=torch.float32,
)
tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)

# Make sure pad token exists (Llama often uses eos as pad)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

"""
As max_seq increases, the proportion of padded tokens grows, 
 which directly affects calibration statistics and tends to increase PEIR.

This is not a modeling issue but a calibration artifact: 
 observers see a distribution dominated by padding rather than real tokens.

For a more representative and realistic accuracy evaluation, 
the calibration dataset should be adjusted (e.g., longer or more diverse 
 sequence lengths, or padding-aware calibration) so that activation 
statistics better reflect actual inference workloads.
"""
MAX_SEQ = 128
text_cfg = model.config.text_config
text_cfg.max_position_embeddings = MAX_SEQ

# -------------------------------------------------------------------------
# 1. Replace layer-0’s self-attention with QuantQwen3VLTextAttention
# -------------------------------------------------------------------------
orig_attn = model.model.language_model.layers[0].self_attn
model.model.language_model.layers[0].self_attn = prepare(orig_attn, PTQConfig())
model.eval()

attn_q = model.model.language_model.layers[0].self_attn
assert isinstance(attn_q.wrapped, QuantQwen3VLTextAttention)

# -------------------------------------------------------------------------
# Helpers: fixed-length tokenize + embed
# -------------------------------------------------------------------------
def make_fixed_inputs(prompt: str):
    batch = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQ,
    )
    input_ids = batch["input_ids"]  # [1,MAX_SEQ]

    hidden = model.model.language_model.embed_tokens(input_ids)  # [1,MAX_SEQ,D]
    return hidden


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
        hidden = make_fixed_inputs(prompt)
        rotary = model.model.language_model.rotary_emb
        position_ids = torch.arange(MAX_SEQ).unsqueeze(0)
        pos = rotary(hidden, position_ids)
        _ = attn_q(hidden, pos)

convert(attn_q)
assert attn_q._mode is Mode.QUANT, "Quantization mode should be active now."

# -------------------------------------------------------------------------
# 3. Quick diff check (INT-sim vs FP32)
# -------------------------------------------------------------------------
hidden = make_fixed_inputs("check")

with torch.no_grad():
    rotary = model.model.language_model.rotary_emb
    position_ids = torch.arange(MAX_SEQ).unsqueeze(0)
    pos = rotary(hidden, position_ids)

    int8_out, _ = attn_q(hidden, pos)

    mask = torch.full((1, 1, MAX_SEQ, MAX_SEQ), float("-120"))
    mask.triu_(1)
    fp_out, _ = orig_attn(hidden, position_embeddings=pos, attention_mask=mask)

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
B, S, D = 1, MAX_SEQ, text_cfg.hidden_size
example = torch.randn(B, S, D)
example_ids = torch.arange(S)[None, :]
example_pos = rotary(example, example_ids)

with SuppressWarning(UserWarning, ".*"):
    cm = tico.convert(attn_q, (example, example_pos, None))
cm.save(save_path)

print(f"Quantized Circle model saved to {save_path.resolve()}")
