# User input
prompt = "Lily picked up a flower."
model_name = "Maykeye/TinyLLama-v0"

# Tokenizer
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
inputs = tokenizer(
    prompt,
    return_tensors="pt",
    padding="max_length",
    max_length=32,
    truncation=True,
)

# Generator
import torch

from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

from tico.utils.record_input import RecordingInput

target_model = model.model.layers[0]
condition_fn = lambda args_dict: args_dict["past_key_value"].get_seq_length() != 0

with torch.no_grad(), RecordingInput(target_model, condition_fn) as rec:
    outputs = model.generate(
        **inputs,
        max_new_tokens=32,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    captured_input = rec.captured_input

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)


# Convert

import tico
from tico.serialize.operators.op_circle_attention import llama_attention_forward_adapter
from transformers.models.llama.modeling_llama import LlamaAttention

LlamaAttention.forward = llama_attention_forward_adapter

model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()
circle_model = tico.convert(model.model.layers[0], captured_input)
circle_model.save(f"tinyllama.layer.attn.circle")
