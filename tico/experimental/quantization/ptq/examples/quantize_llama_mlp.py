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
from transformers import AutoModelForCausalLM, AutoTokenizer

import tico
from tico.experimental.quantization.evaluation.metric import compute_peir
from tico.experimental.quantization.evaluation.utils import plot_two_outputs
from tico.experimental.quantization.ptq.dtypes import INT16
from tico.experimental.quantization.ptq.mode import Mode
from tico.experimental.quantization.ptq.qscheme import QScheme
from tico.experimental.quantization.ptq.quant_config import QuantConfig
from tico.experimental.quantization.ptq.wrappers.llama.quant_mlp import QuantLlamaMLP
from tico.utils.utils import SuppressWarning

name = "Maykeye/TinyLLama-v0"
model = AutoModelForCausalLM.from_pretrained(name)
tokenizer = AutoTokenizer.from_pretrained(name)
model.eval()

# -------------------------------------------------------------------------
# 1. Replace layer-0’s MLP with QuantLlamaMLP
# -------------------------------------------------------------------------
fp32_mlp = model.model.layers[0].mlp
model.model.layers[0].mlp = QuantLlamaMLP(
    fp32_mlp,
    qcfg=QuantConfig(default_dtype=INT16, default_qscheme=QScheme.PER_TENSOR_SYMM),
)  # PTQWrapper(fp32_mlp) is also fine
model.eval()

mlp_q = model.model.layers[0].mlp

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
    mlp_q.enable_calibration()
    for prompt in PROMPTS:
        enc = tokenizer(prompt, return_tensors="pt")
        emb = model.model.embed_tokens(enc["input_ids"])
        _ = mlp_q(emb)

    mlp_q.freeze_qparams()

assert mlp_q._mode is Mode.QUANT, "Quantization mode should be active now."

# -------------------------------------------------------------------------
# 3. Quick diff check (INT-sim vs FP32)
# -------------------------------------------------------------------------
with torch.no_grad():
    ids = tokenizer("quant all tensors!", return_tensors="pt")
    emb = model.model.embed_tokens(ids["input_ids"])
    int16 = mlp_q(emb)  # INT-sim
    fp32 = fp32_mlp(emb)  # baseline reference

print("┌───────────── Quantization Error Summary ─────────────")
print(f"│ Mean |diff|: {(int16 - fp32).abs().mean().item():.6f}")
print(f"│ PEIR       : {compute_peir(fp32, int16) * 100:.6f} %")
print("└──────────────────────────────────────────────────────")
print(plot_two_outputs(fp32, int16))

# -------------------------------------------------------------------------
# 4. Export the quantized block
# -------------------------------------------------------------------------
save_path = pathlib.Path("mlp.q.circle")
example_in = (torch.randn(1, 1, model.config.hidden_size),)

with SuppressWarning(UserWarning, ".*"):
    cm = tico.convert(mlp_q, example_in)
cm.save(save_path)

print(f"Quantized Circle model saved to {save_path.resolve()}")
