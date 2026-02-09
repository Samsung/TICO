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

from tico.quantization import convert, prepare
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.evaluation.metric import compute_peir
from tico.quantization.evaluation.utils import plot_two_outputs
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.wrappers.llama.quant_decoder_layer import (
    QuantLlamaDecoderLayer,
)
from tico.utils.utils import SuppressWarning

MODEL_NAME = "Maykeye/TinyLLama-v0"
MAX_S = 256

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Make sure pad token exists (Llama often uses eos as pad)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model.config.max_position_embeddings = MAX_S
model.eval()  # disable dropout, etc.

# -------------------------------------------------------------------------
# 1. Swap in the quant wrapper
# -------------------------------------------------------------------------
fp32_layer = model.model.layers[0]  # keep a reference for diff check
model.model.layers[0] = prepare(fp32_layer, PTQConfig())
model.eval()

qlayer = model.model.layers[0]  # alias for brevity
assert isinstance(qlayer.wrapped, QuantLlamaDecoderLayer)

# -------------------------------------------------------------------------
# Helpers: fixed-length tokenize + embed
# -------------------------------------------------------------------------
def make_fixed_inputs(prompt: str):
    batch = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_S,
    )
    input_ids = batch["input_ids"]  # [1,MAX_S]

    hidden = model.model.embed_tokens(input_ids)  # [1,MAX_S,D]
    return hidden


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
    for prompt in PROMPTS:
        hidden = make_fixed_inputs(prompt)
        # IMPORTANT:
        # - Do NOT pass position_embeddings; layer will use its static templates
        _ = qlayer(hidden)

convert(qlayer)
assert qlayer._mode is Mode.QUANT, "Quantization mode should be active now."

# -------------------------------------------------------------------------
# 3. Quick INT-sim vs FP32 sanity check
# -------------------------------------------------------------------------
hidden = make_fixed_inputs("check")

with torch.no_grad():
    int8_out = qlayer(hidden)
    int8 = int8_out[0] if isinstance(int8_out, tuple) else int8_out

    rotary = model.model.rotary_emb
    position_ids = torch.arange(MAX_S).unsqueeze(0)
    pos = rotary(hidden, position_ids)

    fp32_out = fp32_layer(hidden, position_embeddings=pos)
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
B, S, D = 1, MAX_S, model.config.hidden_size
example_hidden = torch.randn(B, S, D)

with SuppressWarning(UserWarning, ".*"):
    cm = tico.convert(
        qlayer,
        (example_hidden,),
    )
# Note that the model is not fully quantized.
cm.save(save_path)

print(f"Quantized Circle model saved to {save_path.resolve()}")
