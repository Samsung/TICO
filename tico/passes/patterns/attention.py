from transformers.models.llama.modeling_llama import LlamaConfig, LlamaAttention
import torch
from transformers.cache_utils import DynamicCache

class LlamaAttentionWithOutputs(LlamaAttention):
    def __init__(self):
        super().__init__(config = LlamaConfig(
            hidden_size=512,
            num_hidden_layers=8,
            num_attention_heads=8,
            use_cache=True,
            attn_implementation = "eager"
        ), layer_idx = 0)
        
    def forward(self,
        hidden_states: torch.Tensor,
        position_embeddings,
        attention_mask= None,
        past_key_value= None,
        cache_position= None,
        **kwargs,
    ):
        attn_output, _attn_weights = super().forward(hidden_states, position_embeddings, attention_mask, past_key_value, cache_position, **kwargs)
        
        return attn_output, past_key_value.key_cache, past_key_value.value_cache
    
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
    