from transformers.models.llama.modeling_llama import LlamaConfig, eager_attention_forward, apply_rotary_pos_emb
import torch
from torch import nn
from typing import Callable, Optional, Tuple
from transformers.cache_utils import Cache, DynamicCache


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # attention_interface: Callable = eager_attention_forward
        # if self.config._attn_implementation != "eager":
        #     if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
        #         logger.warning_once(
        #             "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
        #             'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
        #         )
        #     else:
        #         attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        
        attention_interface: Callable = eager_attention_forward #sdpa_attention_forward
        
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            # **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        # Without this, the matcher cannot find multiple attention layers.
        #
        # return attn_output, attn_weights
        return attn_output, attn_weights, key_states, value_states
    
    def get_example_inputs(self):
        seq_len = 5# 1  # Assume token generation
        hidden_size = self.config.hidden_size
        head_dim = self.config.head_dim
        num_heads = self.config.num_attention_heads

        hidden_states = torch.randn(1, seq_len, hidden_size)
        position_embeddings = (
            torch.randn(1, seq_len, head_dim),
            torch.randn(1, seq_len, head_dim),
        )
        attention_mask = torch.Tensor([[[[0.0]] * seq_len]])  # shape: 1, 1, seq_len, 1
        # This attention_mask will become a causal_mask of shape: (batch_size, 1, query_length, key_value_length)
        prev_seq_len = 4
        past_key_values = DynamicCache()

        past_key_values.update(
            torch.randn(1, num_heads, prev_seq_len, head_dim),
            torch.randn(1, num_heads, prev_seq_len, head_dim),
            0,
        )
        cache_position = torch.tensor([[prev_seq_len]])
        return (hidden_states, position_embeddings, attention_mask, past_key_values,)
    

class LlamaModelAttentionProjection(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        
        return query_states, key_states, value_states
    
    def get_example_inputs(self):
        hidden_size = self.config.hidden_size
        head_dim = self.config.head_dim
        num_heads = self.config.num_attention_heads
        seq_len = 1
        hidden_states = torch.randn(1, seq_len, hidden_size)
        return (hidden_states,)