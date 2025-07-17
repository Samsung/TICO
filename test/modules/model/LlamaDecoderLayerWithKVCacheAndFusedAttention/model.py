# User input
prompt = "Lily picked up a flower."
model_name = "Maykeye/TinyLLama-v0"

captured_input = ()

import copy, inspect, types

from transformers.models.llama.modeling_llama import LlamaDecoderLayer

forward_org = LlamaDecoderLayer.forward


def capture_and_forward(self, *args, **kwargs):
    global captured_input

    # Prepare args tuple for TICO.convert()
    # Get arg_names in positional args order using inspect
    sig = inspect.signature(forward_org)
    args_names = [
        # signature includes `self`` and `kwargs``.
        # Just retrieve the ordinary positional inputs only
        name
        for name in sig.parameters.keys()
        if name
        not in ("self", "kwargs", "use_cache", "position_ids", "output_attentions")
    ]

    args_dict = dict(zip(args_names, args))
    args_dict.update(kwargs)

    def populate_args(args_dict, filter):
        for key in filter:
            args_dict.pop(key, None)
        args_tuple = tuple(args_dict.get(name, None) for name in args_names)
        return copy.deepcopy(args_tuple)

    if args_dict["past_key_value"].get_seq_length() != 0 and captured_input == ():
        input_to_remove = [
            "use_cache",
        ]
        captured_input = populate_args(args_dict, input_to_remove)

    return forward_org(self, *args, **kwargs)


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
model.model.layers[0].forward = types.MethodType(
    capture_and_forward, model.model.layers[0]
)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=32,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)

# ATTENTION FUSER

from typing import List, Optional, Tuple


@torch.library.impl("circle::attention.llama", "CPU")
def attention_llama_cpu(
    hidden_states,
    position_cos,
    position_sin,
    attention_mask,
    past_key,
    past_value,
    layer_idx,
    cache_position,
):
    return hidden_states


@torch.library.register_fake("circle::attention.llama")
def attention_llama(*args, **kwargs):
    (
        hidden_states,
        position_cos,
        position_sin,
        attention_mask,
        past_key,
        past_value,
        layer_idx,
        cache_position,
    ) = args
    return hidden_states


from transformers.cache_utils import Cache, DynamicCache
from transformers.models.llama.modeling_llama import LlamaAttention


def forward_adapter(
    self: LlamaAttention,
    hidden_states: torch.Tensor,
    position_embeddings: List[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[DynamicCache],
    cache_position: torch.Tensor,
    **kwargs,
):
    # past_key_value is a dict with key_cache and value_cache.
    # It needs to be decomposed for tico and circle which does not know dict.
    key_cache = past_key_value.key_cache
    value_cache = past_key_value.value_cache
    return (
        torch.ops.circle.attention.llama(
            hidden_states,
            position_embeddings[0],  # cos
            position_embeddings[1],  # sin
            attention_mask,
            # key_cache is a list of cache for each decoder layer.
            # Assumtion: key cache is continuous
            #
            #    k_cache[0] | k_cache[1] | ...  | k_cache[n]
            key_cache[0],
            value_cache[0],  # Same to value_cache
            self.layer_idx,
            cache_position,
        ),
        None,
    )


# Tico

import tico

from torch import nn
from transformers.models.llama.modeling_llama import LlamaModel

model = AutoModelForCausalLM.from_pretrained(model_name)


class LlamaDecoderLayers(nn.Module):
    def __init__(self, model: LlamaModel):
        super().__init__()
        self.config = model.config
        self.layers = model.layers

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]

        return hidden_states


layers = LlamaDecoderLayers(model.model)
LlamaAttention.forward = forward_adapter
layers.eval()
circle_model = tico.convert(layers, captured_input)
circle_model.save(f"tinyllama.model.attn.circle")
