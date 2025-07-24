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

from typing import Any, Optional, Tuple

# Define DecoderLayers

from torch import nn
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaModel


# DecoderLayers is not nn.Module. Not torch.export-able.
# Let's define decoder layers as nn.Module.


class LlamaDecoderLayers(nn.Module):
    def __init__(self, model: LlamaModel):
        super().__init__()
        self.config = model.config
        self.layers = model.layers

    # Make sure signature is same to capturing input.
    # Just copy and Paste from LlamaDecoderLayer::forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
        **kwargs: Any,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]

        return hidden_states


# Convert

import tico

# NOTE:
# If you want to restore forward, it may be implemented as context manager.
# However, it is just a simple script to export. No one uses forward after tico conversion.
from tico.serialize.operators.op_circle_attention import llama_attention_forward_adapter

LlamaAttention.forward = llama_attention_forward_adapter

model = AutoModelForCausalLM.from_pretrained(model_name)
layers = LlamaDecoderLayers(model.model)
layers.eval()
circle_model = tico.convert(layers, captured_input)
circle_model.save(f"tinyllama.layers.attn.circle")
