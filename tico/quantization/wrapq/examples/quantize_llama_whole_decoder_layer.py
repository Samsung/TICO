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

import os
import pathlib

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tico.quantization import convert, prepare
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.evaluation.metric import compute_peir
from tico.quantization.evaluation.utils import plot_two_outputs
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.observers.minmax import MinMaxObserver
from tico.quantization.wrapq.observers.mx import MXObserver
from tico.quantization.wrapq.qscheme import QScheme
from tico.quantization.wrapq.wrappers.llama.quant_decoder_layer import (
    QuantLlamaDecoderLayer,
)
from tico.utils.utils import SuppressWarning

MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct"  # "Maykeye/TinyLLama-v0" #"unsloth/Llama-3.2-3B-Instruct"  # "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, cache_dir="/mnt/storage/transformers_cache"
)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, cache_dir="/mnt/storage/transformers_cache"
)
model.config.max_position_embeddings = 2048  # we need this to prevent RAM exhaust
model.config.use_cache = True  # False

model.eval()  # disable dropout, etc.
rotary = model.model.rotary_emb  # RoPE helper

# -------------------------------------------------------------------------
# 1. Swap in the quant wrapper
# -------------------------------------------------------------------------
fp32_layer = model.model.layers[0]  # keep a reference for diff check

cfg = PTQConfig(
    default_dtype=DType.int(16),
    default_qscheme=QScheme.PER_TENSOR_SYMM,
    default_observer=MinMaxObserver,  # type: ignore[type-abstract]
    overrides={
        "mlp": {
            "gate_proj": {
                "weight": {"dtype": DType.uint(4), "observer": MinMaxObserver},
                "act_in": {"observer": MXObserver},
                "act_out": {"observer": MXObserver},
            },
            "up_proj": {
                "weight": {"dtype": DType.uint(4), "observer": MinMaxObserver},
                "act_in": {"observer": MXObserver},
                "act_out": {"observer": MXObserver},
            },
            "down_proj": {
                "weight": {"dtype": DType.uint(4), "observer": MinMaxObserver},
                "act_in": {"observer": MXObserver},
                "act_out": {"observer": MXObserver},
            },
            "act_fn": {
                "act_in": {"observer": MinMaxObserver},
                "sigmoid": {"observer": MinMaxObserver},
                "mul": {"observer": MinMaxObserver},
            },
        },
        "self_attn": {
            "q_proj": {
                "weight": {"dtype": DType.uint(4), "observer": MinMaxObserver},
                "act_in": {"observer": MXObserver},
                "act_out": {"observer": MXObserver},
            },
            "k_proj": {
                "weight": {"dtype": DType.uint(4), "observer": MinMaxObserver},
                "act_in": {"observer": MXObserver},
                "act_out": {"observer": MXObserver},
            },
            "v_proj": {
                "weight": {"dtype": DType.uint(4), "observer": MinMaxObserver},
                "act_in": {"observer": MXObserver},
                "act_out": {"observer": MXObserver},
            },
            "o_proj": {
                "weight": {"dtype": DType.uint(4), "observer": MinMaxObserver},
                "act_in": {"observer": MXObserver},
                "act_out": {"observer": MXObserver},
            },
            "scale": {"observer": MinMaxObserver},
            "mask_add": {"observer": MinMaxObserver},
            "softmax": {"observer": MinMaxObserver},
        },
        "self_attn_residual_act_out": {"observer": MinMaxObserver},
        "input_layernorm": {
            "dtype": DType.int(16),
            "weight": {"dtype": DType.int(16), "observer": MinMaxObserver},
            "act_in": {"observer": MinMaxObserver},
            "act_out": {"observer": MinMaxObserver},
        },
        "post_attention_layernorm": {
            "dtype": DType.int(16),
            "weight": {"dtype": DType.int(16), "observer": MinMaxObserver},
            "act_in": {"observer": MinMaxObserver},
            "act_out": {"observer": MinMaxObserver},
        },
    },
)

model.model.layers[0] = prepare(fp32_layer, cfg, kwargs={"return_kv_cache": True})
model.eval()

qlayer = model.model.layers[0]  # alias for brevity
assert isinstance(qlayer.wrapped, QuantLlamaDecoderLayer)

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
        ids = tokenizer(prompt, return_tensors="pt")
        hidden = model.model.embed_tokens(ids["input_ids"])
        pos = rotary(hidden, ids["input_ids"])  # (cos, sin) tuple
        S = pos[0].shape[1]
        attn_mask = torch.zeros(1, 1, S, S)  # causal-mask placeholder
        _ = qlayer(
            hidden,
            attention_mask=attn_mask,
            position_embeddings=pos,
            use_cache=model.config.use_cache,
        )

convert(qlayer)

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

save_path = pathlib.Path(
    "decoder_layer.q.circle"
)  # "decoder_layer_unsloth_LLama_3_2_1B_RMS_NORM_A16W4.q.circle"
B, S, D = 1, 4, model.config.hidden_size
example_hidden = torch.randn(B, S, D)
example_pos = rotary(example_hidden, torch.arange(S)[None, :])
attn_mask = torch.zeros(1, 1, S, S)

with SuppressWarning(UserWarning, ".*"):
    cm = tico.convert(
        qlayer,
        (example_hidden, attn_mask),
        {"position_embeddings": example_pos},
        strict=False,
    )
# os.environ["CCEX_RUNTIME"]="onert"
# args = (example_hidden, attn_mask, example_pos),
# cm_out = torch.tensor(cm(*args)[0])

# Note that the model is not fully quantized.
cm.save(save_path)

print(f"Quantized Circle model saved to {save_path.resolve()}")
