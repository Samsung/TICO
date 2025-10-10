import os
import tempfile
import time
from typing import Any, List, Tuple

import torch

from transformers import LlamaConfig
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRotaryEmbedding,
)


class Llama32LayerBase(torch.nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.config = self.get_config()
        self.model = LlamaDecoderLayer(config=self.config, layer_idx=0)

        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)
        self.cache_position = torch.arange(self.seq_len)
        position_ids = self.cache_position.unsqueeze(0)
        self.register_buffer("position_ids", position_ids)

    def get_config(self):
        raise NotImplementedError("Must implement get_config")

    def num_hidden_layers(self):
        return self.config.num_hidden_layers

    def _create_causal_mask(self):
        min_val = torch.finfo(torch.float32).min
        causal_mask = torch.full(
            (self.seq_len, self.seq_len), fill_value=min_val, dtype=torch.float32
        )
        causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask = torch.reshape(causal_mask, [1, 1, self.seq_len, self.seq_len])
        return causal_mask

    def forward(self, *args, **kwargs):
        hidden_states = args[0]
        # A new, empty cache is created for each forward pass for stateless testing.
        past_key_values = DynamicCache()
        position_embeddings = self.rotary_emb(hidden_states, self.position_ids)
        causal_mask = self._create_causal_mask()
        layer_outputs = self.model.forward(
            *args,
            **{
                "position_ids": self.position_ids,
                "cache_position": self.cache_position,
                "attention_mask": causal_mask,
                "past_key_value": past_key_values,
                "use_cache": self.config.use_cache,
                "position_embeddings": position_embeddings,
            },
        )
        hidden_states = layer_outputs[0]
        return (
            hidden_states,
            past_key_values.to_legacy_cache(),
        )

    def get_example_inputs(self):
        hidden_states = torch.rand([1, self.seq_len, self.config.hidden_size])
        return (hidden_states,), {}


class Llama32_1B(Llama32LayerBase):
    def __init__(self, seq_len):
        super().__init__(seq_len=seq_len)

    def get_config(self):
        return LlamaConfig(
            _attn_implementation_autoset=True,
            architectures=["LlamaForCausalLM"],
            attention_bias=False,
            attention_dropout=0.0,
            bos_token_id=128000,
            eos_token_id=128001,
            head_dim=64,
            hidden_act="silu",
            hidden_size=2048,
            initializer_range=0.02,
            intermediate_size=8192,
            max_position_embeddings=131072,
            mlp_bias=False,
            model_type="llama",
            num_attention_heads=32,
            num_hidden_layers=16,
            num_key_value_heads=8,
            pretraining_tp=1,
            rms_norm_eps=1e-05,
            rope_scaling={
                "factor": 32.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3",
            },
            rope_theta=500000.0,
            tie_word_embeddings=True,
            torch_dtype=torch.float16,
            use_cache=True,
            vocab_size=128256,
        )


class Llama32_3B(Llama32LayerBase):
    def __init__(self, seq_len):
        super().__init__(seq_len=seq_len)

    def get_config(self):
        return LlamaConfig(
            _attn_implementation_autoset=True,
            architectures=["LlamaForCausalLM"],
            attention_bias=False,
            attention_dropout=0.0,
            bos_token_id=128000,
            eos_token_id=128001,
            head_dim=128,
            hidden_act="silu",
            hidden_size=3072,
            initializer_range=0.02,
            intermediate_size=8192,
            max_position_embeddings=131072,
            mlp_bias=False,
            model_type="llama",
            num_attention_heads=24,
            num_hidden_layers=28,
            num_key_value_heads=8,
            pretraining_tp=1,
            rms_norm_eps=1e-05,
            rope_scaling={
                "factor": 32.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3",
            },
            rope_theta=500000.0,
            tie_word_embeddings=True,
            torch_dtype=torch.float16,
            use_cache=True,
            vocab_size=128256,
        )


def load_model(module_name: str) -> Tuple[torch.nn.Module, Tuple[Any, ...]]:
    """
    Dynamically import a test model module and return the instantiated model
    together with its example inputs (as returned by get_example_inputs()).
    """
    if module_name == "Llama-3.2-1B":
        model = Llama32_1B(seq_len=256)
    elif module_name == "Llama-3.2-3B":
        model = Llama32_3B(seq_len=256)  # type: ignore[assignment]
    else:
        raise RuntimeError(f"Not yet implemented module: {module_name}")

    model.eval()

    inputs, _ = model.get_example_inputs()
    return model, inputs


def measure_time(func, *args, repeat: int = 3) -> List[float]:
    """Run ``func`` ``repeat`` times and return a list of elapsed seconds."""
    timings = []
    for _ in range(repeat):
        start = time.perf_counter()
        func(*args)
        timings.append(time.perf_counter() - start)
    return timings


def size_of_bytes(data: bytes) -> int:
    """Return the size of a bytes object."""
    return len(data)


def temp_state_dict_size(model: torch.nn.Module) -> int:
    """Serialize the model's state_dict to a temporary file and return its size."""
    with tempfile.NamedTemporaryFile() as tmp:
        torch.save(model.state_dict(), tmp.name)
        size = os.path.getsize(tmp.name)
    return size
