# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from typing import Any

import torch

import tico
from tico.quantization.wrapq.wrappers.llama.export_adapters import (
    LlamaLMHeadExportAdapter,
    LlamaTokenEmbeddingExportAdapter,
    make_token_embedding_dynamic_shapes,
    make_token_embedding_example_input,
    register_fake_quant_meta_kernels_for_dynamic_export,
)
from tico.utils.utils import SuppressWarning


def _convert_and_save(
    module: torch.nn.Module,
    example_inputs: tuple[Any, ...],
    save_path: Path,
    *,
    kwargs: dict[str, Any] | None = None,
    dynamic_shapes: Any | None = None,
    strict: bool = False,
) -> None:
    print(f"Saving {save_path.name} to {save_path.resolve()}")
    with torch.no_grad(), SuppressWarning(UserWarning, ".*"):
        cm = tico.convert(
            module.eval(),
            example_inputs,
            kwargs=kwargs,
            dynamic_shapes=dynamic_shapes,
            strict=strict,
        )
    cm.save(save_path)


def _make_random_position_embeddings(batch: int, head_dim: int, device: str):
    cos = torch.randn(batch, 1, head_dim, device=device)
    sin = torch.randn(batch, 1, head_dim, device=device)
    return cos, sin


def _make_random_decode_attn_mask(batch: int, max_seq: int, device: str):
    effective_len = torch.randint(low=1, high=max_seq + 1, size=(1,)).item()
    mask = torch.zeros(batch, 1, max_seq, device=device, dtype=torch.float32)
    if effective_len < max_seq:
        mask[:, :, effective_len:] = float("-120")
    return mask


def _make_random_decode_batch(model, batch: int, device: str, max_seq: int):
    hidden_size = model.config.hidden_size
    head_dim = getattr(
        model.config, "head_dim", hidden_size // model.config.num_attention_heads
    )
    n_kv = model.config.num_key_value_heads

    hidden = torch.randn(batch, 1, hidden_size, device=device)
    pos = _make_random_position_embeddings(batch, head_dim, device)
    mask = _make_random_decode_attn_mask(batch, max_seq, device)
    past_k = torch.randn(batch, n_kv, max_seq - 1, head_dim, device=device)
    past_v = torch.randn(batch, n_kv, max_seq - 1, head_dim, device=device)
    return hidden, pos, mask, (past_k, past_v)


def export_token_embedding(
    qmodel: torch.nn.Module, max_seq_len: int, output_dir: Path
) -> None:
    register_fake_quant_meta_kernels_for_dynamic_export()
    example = make_token_embedding_example_input(qmodel=qmodel, max_seq_len=max_seq_len)
    dynamic_shapes = make_token_embedding_dynamic_shapes(max_seq_len)
    _convert_and_save(
        LlamaTokenEmbeddingExportAdapter(qmodel),
        (example,),
        output_dir / "token_embedding.q.circle",
        dynamic_shapes=dynamic_shapes,
    )


def export_lm_head(qmodel: torch.nn.Module, output_dir: Path) -> None:
    example_hidden = torch.randn(1, 1, int(qmodel.config.hidden_size), device="cpu")
    _convert_and_save(
        LlamaLMHeadExportAdapter(qmodel),
        (example_hidden,),
        output_dir / "lm_head.q.circle",
    )


def export_llama_per_layer(
    *,
    q_model: torch.nn.Module,
    max_seq_len: int,
    output_dir: str | Path,
    prefill_decode: bool = False,
) -> None:
    """Export token embedding, decoder layers, and LM head as Circle artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    q_model.eval().cpu()
    if not hasattr(q_model, "wrapped"):
        raise TypeError("Per-layer LLaMA export requires a PTQ-wrapped model.")

    qmodel = q_model.wrapped
    layers = qmodel.model.wrapped.layers
    config = qmodel.config

    export_token_embedding(qmodel, max_seq_len, output_dir)

    for i, qlayer in enumerate(layers):
        suffix = "prefill_" if prefill_decode else ""
        layer_name = f"decoder_layer_{suffix}{i}"
        save_path = output_dir / f"{layer_name}.q.circle"

        batch, seq, hidden_size = 1, max_seq_len, config.hidden_size
        example_hidden = torch.randn(batch, seq, hidden_size, device="cpu")
        attention_mask = (
            qlayer.wrapped.causal_mask_template[..., :seq, :seq].squeeze(0).to("cpu")
        )
        position_embeddings = qlayer.wrapped._slice_rope(
            start=0,
            seq_len=seq,
            device="cpu",
            dtype=example_hidden.dtype,
        )

        _convert_and_save(
            qlayer.wrapped.as_export_module("prefill", return_kv=prefill_decode),
            (example_hidden,),
            save_path,
            kwargs={
                "attention_mask": attention_mask,
                "position_embeddings": position_embeddings,
            },
        )

        if prefill_decode:
            layer_name = f"decoder_layer_decode_{i}"
            hidden, pos, mask, past = _make_random_decode_batch(
                qmodel, 1, "cpu", max_seq_len
            )
            _convert_and_save(
                qlayer.wrapped.as_export_module("decode"),
                (hidden,),
                output_dir / f"{layer_name}.q.circle",
                kwargs={
                    "attention_mask": mask,
                    "past_key_value": past,
                    "position_embeddings": pos,
                },
            )

    export_lm_head(qmodel, output_dir)
