import pathlib

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import tico
from tico.experimental.quantization.custom.wrappers.llama.quant_llama_mlp import QuantLlamaMLP
from tico.experimental.quantization.custom.wrappers.mode import Mode

name = "Maykeye/TinyLLama-v0"
model = AutoModelForCausalLM.from_pretrained(name)
tokenizer = AutoTokenizer.from_pretrained(name)
model.eval()

# -------------------------------------------------------------------------
# 1. Replace layer-0’s MLP with QuantLlamaMLP
# -------------------------------------------------------------------------
fp32_mlp                  = model.model.layers[0].mlp
model.model.layers[0].mlp = QuantLlamaMLP(fp32_mlp)

mlp_q = model.model.layers[0].mlp

# -------------------------------------------------------------------------
# 2. Single-pass calibration 
# -------------------------------------------------------------------------
with torch.no_grad():
    mlp_q.enable_calibration()
    for _ in range(16):                      
        prompts = ["hello tinyllama "] * 8
        enc = tokenizer(prompts, return_tensors="pt")
        emb = model.model.embed_tokens(enc["input_ids"])
        _ = mlp_q(emb)

    mlp_q.freeze_qparams()

assert mlp_q._mode is Mode.QUANT

# -------------------------------------------------------------------------
# 3. Quick diff check (INT-sim vs FP32)
# -------------------------------------------------------------------------
with torch.no_grad():
    ids  = tokenizer("quant all tensors!", return_tensors="pt")
    emb  = model.model.embed_tokens(ids["input_ids"])
    int8 = mlp_q(emb)                    # INT-sim
    fp32 = fp32_mlp(emb)                 # baseline reference

print("mean|diff|  INT8 vs FP32 :", (int8 - fp32).abs().mean().item())

# -------------------------------------------------------------------------
# 4. Export the quantized block
# -------------------------------------------------------------------------
export_path = pathlib.Path("mlp.q.circle")
example_in  = (torch.randn(1, 1, model.config.hidden_size),)

cm = tico.convert(mlp_q, example_in)
cm.save(export_path)
