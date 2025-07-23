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


# ATTENTION FUSER

from typing import List, Optional


@torch.library.impl("circle::attention.llama", "CPU")
def attention_llama_cpu(
    hidden_states,
    q_proj,
    k_proj,
    v_proj,
    o_proj,
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
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        position_cos,
        position_sin,
        attention_mask,
        past_key,
        past_value,
        layer_idx,
        cache_position,
    ) = args
    return hidden_states


from transformers.cache_utils import DynamicCache
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
    key_cache = past_key_value.key_cache  # type: ignore[union-attr]
    value_cache = past_key_value.value_cache  # type: ignore[union-attr]
    return (
        torch.ops.circle.attention.llama(
            hidden_states,
            self.q_proj.weight,
            self.k_proj.weight,
            self.v_proj.weight,
            self.o_proj.weight,
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

model = AutoModelForCausalLM.from_pretrained(model_name)

LlamaAttention.forward = forward_adapter

model.eval()
circle_model = tico.convert(model.model.layers[0], captured_input)
circle_model.save(f"tinyllama.layer.attn.circle")
