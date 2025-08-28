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

# =============================================================================
# POST-TRAINING QUANTIZATION EXAMPLE — Llama Decoder Layer (Self-Attn + MLP)
# -----------------------------------------------------------------------------
# This demo shows how to:
#   1. Replace a single FP32 `LlamaDecoderLayer` with `QuantLlamaDecoderLayer`.
#   2. Collect activation statistics in one calibration sweep.
#   3. Freeze scales / zero-points and switch to INT-simulation mode.
#   4. Compare INT-8 vs FP32 outputs with a quick mean-absolute-diff check.
#   5. Export the calibrated, quantized block to a Circle model.
# -----------------------------------------------------------------------------
# Style / layout is kept identical to the `quantize_llama_attn.py` and
# `quantize_llama_mlp.py` examples for easy side-by-side reading.
# =============================================================================

import pathlib

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tico.experimental.quantization.evaluation.metric import compute_peir
from tico.experimental.quantization.evaluation.utils import plot_two_outputs
from tico.experimental.quantization.ptq.mode import Mode
from tico.experimental.quantization.ptq.wrappers.llama.quant_decoder_layer import (
    QuantLlamaDecoderLayer,
)
from tico.utils.utils import SuppressWarning

MODEL_NAME = "Maykeye/TinyLLama-v0"
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model.eval()  # disable dropout, etc.
rotary = model.model.rotary_emb  # RoPE helper

# -------------------------------------------------------------------------
# 1. Swap in the quant wrapper
# -------------------------------------------------------------------------
fp32_layer = model.model.layers[0]  # keep a reference for diff check
model.model.layers[0] = QuantLlamaDecoderLayer(
    fp32_layer
)  # PTQWrapper(fp32_layer) is also fine
model.eval()

qlayer = model.model.layers[0]  # alias for brevity

# -------------------------------------------------------------------------
# 2. Single-pass calibration (gather activation ranges)
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
    qlayer.enable_calibration()
    for prompt in PROMPTS:
        ids = tokenizer(prompt, return_tensors="pt")
        hidden = model.model.embed_tokens(ids["input_ids"])
        pos = rotary(hidden, ids["input_ids"])  # (cos, sin) tuple
        S = pos[0].shape[1]
        attn_mask = torch.zeros(1, 1, S, S)  # causal-mask placeholder
        _ = qlayer(hidden, attention_mask=attn_mask, position_embeddings=pos)
    qlayer.freeze_qparams()

assert qlayer._mode is Mode.QUANT, "Quantization mode should be active now."

# -------------------------------------------------------------------------
# 3. Quick INT-sim vs FP32 sanity check
# -------------------------------------------------------------------------
ids = tokenizer("check", return_tensors="pt")
hidden = model.model.embed_tokens(ids["input_ids"])
pos = rotary(hidden, ids["input_ids"])
S = pos[0].shape[1]
attn_mask = torch.zeros(1, 1, S, S)

with torch.no_grad():
    int8_out = qlayer(hidden, attention_mask=attn_mask, position_embeddings=pos)
    int8 = int8_out[0] if isinstance(int8_out, tuple) else int8_out
    fp32_out = fp32_layer(hidden, attention_mask=attn_mask, position_embeddings=pos)
    fp32 = fp32_out[0] if isinstance(fp32_out, tuple) else fp32_out

print("┌───────────── Quantization Error Summary ─────────────")
print(f"│ Mean |diff|: {(int8 - fp32).abs().mean().item():.6f}")
print(f"│ PEIR       : {compute_peir(fp32, int8) * 100:.6f} %")
print("└──────────────────────────────────────────────────────")
print(plot_two_outputs(fp32, int8))

# -------------------------------------------------------------------------
# 4. Export the calibrated layer to Circle
# -------------------------------------------------------------------------
import tico

save_path = pathlib.Path("decoder_layer.q.circle")
B, S, D = 1, 4, model.config.hidden_size
example_hidden = torch.randn(B, S, D)
example_pos = rotary(example_hidden, torch.arange(S)[None, :])
attn_mask = torch.zeros(1, 1, S, S)

with SuppressWarning(UserWarning, ".*"):
    cm = tico.convert(
        qlayer, (example_hidden, attn_mask), {"position_embeddings": example_pos}
    )
# Note that the model is not fully quantized.
cm.save(save_path)

print(f"Quantized Circle model saved to {save_path.resolve()}")
