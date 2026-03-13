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
from transformers import AutoModelForImageTextToText, AutoTokenizer

from tico.quantization import convert, prepare
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.evaluation.metric import compute_peir
from tico.quantization.evaluation.utils import plot_two_outputs
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.wrappers.qwen_vl.quant_text_decoder_layer import (
    QuantQwen3VLTextDecoderLayer,
)
from tico.utils.utils import SuppressWarning

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
tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

MAX_SEQ = 128
text_cfg = model.config.text_config
text_cfg.max_position_embeddings = MAX_SEQ

# -------------------------------------------------------------------------
# 1. Wrap layer-0's decoder layer with QuantQwen3VLTextDecoderLayer
#
# QuantQwen3VLTextDecoderLayer pre-computes static causal mask and RoPE
# templates internally, so calibration only requires hidden_states input.
# -------------------------------------------------------------------------
orig_layer = model.model.language_model.layers[0]
model.model.language_model.layers[0] = prepare(orig_layer, PTQConfig())
model.eval()

layer_q = model.model.language_model.layers[0]
assert isinstance(layer_q.wrapped, QuantQwen3VLTextDecoderLayer)

# -------------------------------------------------------------------------
# Helpers: tokenize → embed to get hidden states for calibration
# -------------------------------------------------------------------------
def make_hidden(prompt: str) -> torch.Tensor:
    batch = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQ,
    )
    with torch.no_grad():
        return model.model.language_model.embed_tokens(batch["input_ids"])


# -------------------------------------------------------------------------
# 2. Calibration
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
        hidden = make_hidden(prompt)
        # position_embeddings and attention_mask are built internally
        _ = layer_q(hidden)

convert(layer_q)
assert layer_q._mode is Mode.QUANT, "Quantization mode should be active now."

# -------------------------------------------------------------------------
# 3. Quick diff check (INT-sim vs FP32)
# -------------------------------------------------------------------------
hidden = make_hidden("check")

mask = torch.full((1, 1, MAX_SEQ, MAX_SEQ), float("-120"))
mask.triu_(1)

rotary = model.model.language_model.rotary_emb
position_ids = torch.arange(MAX_SEQ).unsqueeze(0)

with torch.no_grad():
    q_out = layer_q(hidden)
    q_out = q_out[0] if isinstance(q_out, tuple) else q_out

    pos = rotary(hidden, position_ids)
    fp_out = orig_layer(hidden, attention_mask=mask, position_embeddings=pos)
    fp_out = fp_out[0] if isinstance(fp_out, tuple) else fp_out

print("┌───────────── Quantization Error Summary ─────────────")
print(f"│ Mean |diff|: {(q_out - fp_out).abs().mean().item():.6f}")
print(f"│ PEIR       : {compute_peir(fp_out, q_out) * 100:.6f} %")
print("└──────────────────────────────────────────────────────")
print(plot_two_outputs(fp_out, q_out))

# -------------------------------------------------------------------------
# 4. Export the quantized decoder layer to Circle
# -------------------------------------------------------------------------
import tico

save_path = pathlib.Path("qwen3vl_text_decoder_layer.q.circle")
B, S, D = 1, MAX_SEQ, text_cfg.hidden_size
example_hidden = torch.randn(B, S, D)

with SuppressWarning(UserWarning, ".*"):
    cm = tico.convert(layer_q, (example_hidden,))
cm.save(save_path)

print(f"Quantized Circle model saved to {save_path.resolve()}")
