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

import torch
import tqdm
from datasets import load_dataset

# ─────────────────────── quant modules ────────────────────────
from tico.experimental.quantization.custom.quant_config import QuantConfig
from tico.experimental.quantization.custom.utils import perplexity
from tico.experimental.quantization.custom.wrappers.llama.quant_llama_attn import (
    QuantLlamaAttention,
)
from tico.experimental.quantization.custom.wrappers.llama.quant_llama_decoder_layer import (
    QuantLlamaDecoderLayer,
)
from tico.experimental.quantization.custom.wrappers.llama.quant_llama_mlp import (
    QuantLlamaMLP,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

# -------------------------------------------------------------------
MODEL_NAME = "meta-llama/Meta-Llama-3-1B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STRIDE = 512
RUN_FP = True

"""
Calib-token presets for PTQ
---------------------------
Numbers are empirical; tweak per model size & observer type.
"""
TOKENS = {
    # Fast sanity-check: does the graph run end-to-end?
    # Small slice keeps turnaround < 1 min on commodity GPU/CPU.
    "debug": 2_000,  # 2k tokens ≈ 16×128-seq batches
    # Recommended starting point for 1-7 B parameter LLMs.
    # Good trade-off — usually <3 % perplexity increase w/ percentile observers.
    "baseline": 50_000,  # 50k tokens
    # Use when shipping to production or using aggressive 4-bit activations.
    # Larger sample smooths out rare outliers → tighter scales.
    "production": 200_000,  # 200k tokens
}
CALIB_TOKENS = TOKENS["baseline"]
print(f"Calibrating with {CALIB_TOKENS:,} tokens.")

# ---------------- 1. FP32 baseline ---------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if RUN_FP:
    print("Loading FP32 model …")
    fp_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    fp_model.config.use_cache = False

# ---------------- 2. Quantised clone -------------------------------
else:
    print("Creating INT-8 clone …")
    int8_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    int8_model.config.use_cache = False

    qcfg = QuantConfig()  # all-uint8 defaults

    # Replace self-attention & MLP
    # for layer in int8_model.model.layers:  # replace in-place
    #     layer.mlp = QuantLlamaMLP(layer.mlp, qcfg=qcfg.child("mlp"))
    #     layer.self_attn = QuantLlamaAttention(layer.self_attn, qcfg=qcfg.child("attn"))

    # Replace DecoderLayer
    new_layers = torch.nn.ModuleList()
    for idx, fp_layer in enumerate(int8_model.model.layers):
        layer_cfg = qcfg.child(f"layer{idx}")
        q_layer = QuantLlamaDecoderLayer(fp_layer, qcfg=layer_cfg)
        new_layers.append(q_layer)
    int8_model.model.layers = new_layers

    # ---------------- 3. Calibration (statistics collection) -----------
    print("Calibrating INT-8 observers …")
    calib_txt = " ".join(
        load_dataset("wikitext", "wikitext-2-raw-v1", split="train")["text"]
    )[:CALIB_TOKENS]
    ids = tokenizer(calib_txt, return_tensors="pt").input_ids.to(DEVICE)

    # enable CALIB mode for every QuantModuleBase
    int8_model.model.layers.apply(
        lambda m: getattr(m, "enable_calibration", lambda: None)()
    )
    with torch.no_grad():
        for i in tqdm.trange(0, ids.size(1) - 1, STRIDE):
            int8_model(ids[:, i : i + STRIDE])
    # freeze → compute (scale, zp)
    int8_model.model.layers.apply(
        lambda m: getattr(m, "freeze_qparams", lambda: None)()
    )

# ---------------- 4. Perplexity helper -----------------------------

# ---------------- 5. Run and report -------------------------------
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
enc = tokenizer("\n\n".join(ds["text"]), return_tensors="pt")

print("\nCalculating perplexities …")
if RUN_FP:
    ppl_fp = perplexity(fp_model, enc, DEVICE, stride=512)
else:
    ppl_int8 = perplexity(int8_model, enc, DEVICE, stride=512)

print("\n┌── Wikitext-2 test perplexity ─────────────")
if RUN_FP:
    print(f"│ FP32  : {ppl_fp:8.2f}")
else:
    print(f"│ INT-8 : {ppl_int8:8.2f}")
print("└───────────────────────────────────────────")
