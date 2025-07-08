from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, pathlib

from tico.experimental.quantization.custom.wrappers.llama.quant_llama_attn import QuantLlamaAttention
from tico.experimental.quantization.custom.wrappers.mode import Mode

name = "Maykeye/TinyLLama-v0"
model = AutoModelForCausalLM.from_pretrained(name)
tokenizer = AutoTokenizer.from_pretrained(name)

# -------------------------------------------------------------------------
# 1. Replace layer-0’s MLP with QuantLlamaMLP
# -------------------------------------------------------------------------
orig_attn = model.model.layers[0].self_attn
model.model.layers[0].self_attn = QuantLlamaAttention(orig_attn)
model.eval()

attn_q = model.model.layers[0].self_attn      # quant wrapper
rotary = model.model.rotary_emb

# -------------------------------------------------------------------------
# 2. Single-pass calibration 
# -------------------------------------------------------------------------
with torch.no_grad():
    attn_q.enable_calibration()
    for _ in range(16):
        ids = tokenizer(["hello"]*8, return_tensors="pt")
        embeds = model.model.embed_tokens(ids["input_ids"])
        cos_sin = rotary(embeds, ids["input_ids"])
        _ = attn_q(embeds, cos_sin)           # observers collect
    attn_q.freeze_qparams()

assert attn_q._mode is Mode.QUANT

# -------------------------------------------------------------------------
# 3. Quick diff check (INT-sim vs FP32)
# -------------------------------------------------------------------------
ids = tokenizer("check", return_tensors="pt")
emb = model.model.embed_tokens(ids["input_ids"])
pos = rotary(emb, ids["input_ids"])
with torch.no_grad():
    int8 = attn_q(emb, pos)
    fp32 = orig_attn(emb, position_embeddings=pos, attention_mask=None)[0]
print("mean|diff| =", (int8 - fp32).abs().mean().item())

# -------------------------------------------------------------------------
# 4. Export the quantized block
# -------------------------------------------------------------------------
import tico
B, S, D = 1, 4, model.config.hidden_size
example = torch.randn(B, S, D)
example_pos = rotary(example, torch.arange(S)[None, :])
cm = tico.convert(attn_q, (example, example_pos))
cm.save(pathlib.Path("attn.q.circle"))
