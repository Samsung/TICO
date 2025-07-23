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

# past_key_values
# ---------------
# During prefill, "past_key_values" not None, but an empty Cache instance.
# Passing None makes torch.export happy.

# attention_mask, cache_position
# ------------------------------
# For npu, ignore captured values generated from example prompt.

input_to_remove = ["past_key_values", "attention_mask", "cache_position"]

with torch.no_grad(), RecordingInput(model, input_to_remove=input_to_remove) as rec:
    outputs = model.generate(
        **inputs,
        max_new_tokens=32,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    captured_input = rec.captured_input

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)

# Tico
import tico

model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()
circle_model = tico.convert(model, captured_input)
circle_model.save(f"tinyllama.prefill.circle")
