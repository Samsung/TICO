# User input
prompt = "Lily picked up a flower."
model_name = "Maykeye/TinyLLama-v0"

import torch

# Capturer
captured_inputs = {}  # type: ignore[var-annotated]

import copy, inspect, types

from transformers import LlamaForCausalLM

original_forward = LlamaForCausalLM.forward


def patched_forward(self, *args, **kwargs):
    global captured_inputs

    # Prepare args tuple for TICO.convert()
    # Get arg_names in positional args order using inspect
    sig = inspect.signature(original_forward)
    args_names = [
        # signature includes `self`` and `kwargs``.
        # Just retrieve the ordinary positional inputs only
        name
        for name in sig.parameters.keys()
        if name not in ("self", "kwargs")
    ]

    args_dict = dict(zip(args_names, args))
    args_dict.update(kwargs)

    def populate_args(args_dict, filter):
        for key in filter:
            args_dict.pop(key, None)
        args_tuple = tuple(args_dict.get(name, None) for name in args_names)
        return copy.deepcopy(args_tuple)

    if captured_inputs.get("prefill", None) == None:
        input_to_remove = [
            "past_key_values",
            "use_cache",
            "attention_mask",
            "cache_position",
        ]
        captured_inputs["prefill"] = populate_args(args_dict, input_to_remove)
    elif captured_inputs.get("decode", None) == None:
        input_to_remove = ["use_cache"]
        captured_inputs["decode"] = populate_args(args_dict, input_to_remove)

    return original_forward(self, *args, **kwargs)


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
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()
model.forward = types.MethodType(patched_forward, model)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=32,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)

# Tico
import tico

for key in captured_inputs.keys():
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    circle_model = tico.convert(model, captured_inputs[key])
    circle_model.save(f"{key}.circle")
