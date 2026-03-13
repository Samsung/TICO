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

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

from tico.quantization import convert, prepare
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.evaluation.metric import compute_peir
from tico.quantization.evaluation.utils import plot_two_outputs
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.wrappers.qwen_vl.quant_text_model import (
    QuantQwen3VLTextModel,
)

# -------------------------------------------------------------------------
# 0. Load a Qwen3-VL model + tokenizer
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
# 1. Wrap the language model backbone with QuantQwen3VLTextModel
#
# QuantQwen3VLTextModel replaces the text backbone (language_model) and:
#   - Pre-computes a shared static causal mask
#   - Pre-computes shared RoPE cos/sin templates (sliced per seq_len)
#   - Passes them to every decoder layer once, avoiding redundant computation
# -------------------------------------------------------------------------
orig_lm = model.model.language_model
model.model.language_model = prepare(orig_lm, PTQConfig())
model.eval()

lm_q = model.model.language_model
assert isinstance(lm_q.wrapped, QuantQwen3VLTextModel)

# -------------------------------------------------------------------------
# Helpers: fixed-length tokenize → (input_ids, attention_mask)
#
# We return the 2D attention_mask alongside input_ids.  Passing it to the
# model lets QuantQwen3VLTextModel combine the static causal mask with a
# per-sequence padding correction, so padding tokens are masked out during
# both calibration and quantized inference.
# -------------------------------------------------------------------------
def make_inputs(prompt: str):
    batch = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQ,
    )
    return batch["input_ids"], batch["attention_mask"]


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
        input_ids, attention_mask = make_inputs(prompt)
        _ = lm_q(input_ids, attention_mask=attention_mask)

convert(lm_q)
assert lm_q._mode is Mode.QUANT, "Quantization mode should be active now."

# -------------------------------------------------------------------------
# 3. Quick diff check (INT-sim vs FP32)
# -------------------------------------------------------------------------
input_ids, attention_mask = make_inputs("check")

with torch.no_grad():
    q_out = lm_q(input_ids, attention_mask=attention_mask, return_dict=False)[0]
    fp_out = orig_lm(input_ids, attention_mask=attention_mask, return_dict=False)[0]

print("┌───────────── Quantization Error Summary ─────────────")
print(f"│ Mean |diff|: {(q_out - fp_out).abs().mean().item():.6f}")
print(f"│ PEIR       : {compute_peir(fp_out, q_out) * 100:.6f} %")
print("└──────────────────────────────────────────────────────")
print(plot_two_outputs(fp_out, q_out))

# Note on PEIR at the model level
# --------------------------------
# PEIR here measures the L2 divergence of the full text backbone output, not a
# single layer.  It is expected to be significantly higher than per-layer PEIR
# (e.g. QuantQwen3VLTextAttention) for two reasons:
#
# 1. Error accumulation: quantization errors from every decoder layer compound
#    multiplicatively.  Even 1-2 % per layer becomes tens of percent over 28+
#    layers.
#
# 2. Padding in calibration data: even with the padding mask fix applied above,
#    the static causal mask baked into the quantized model does not carry
#    sequence-specific padding information.  During calibration, padding is
#    correctly masked out, which improves observer statistics.  However, the
#    PEIR comparison above also passes attention_mask to orig_lm, so both models
#    see the same mask and the measurement is fair.
#
# PEIR is not a direct proxy for task accuracy.  Use downstream metrics
# (e.g. perplexity, VQA score) to evaluate the quantized model's real accuracy.

# -------------------------------------------------------------------------
# Note on Circle export
# -------------------------------------------------------------------------
# Exporting QuantQwen3VLTextModel directly to Circle is not shown here
# because torch.export.export() requires the model to return a plain
# Tensor or a flat tuple of Tensors. The current forward() returns
# BaseModelOutputWithPast (a named tuple), which requires a thin adapter
# wrapper before calling tico.convert().
#
# For subgraph-level export, see the individual layer examples:
#   - quantize_text_attn.py
#   - quantize_text_decoder_layer.py
