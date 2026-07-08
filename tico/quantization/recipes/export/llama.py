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
from typing import Any, Sequence

import torch

import tico
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.wrap_helper import PTQWrapHelper
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


def _make_random_position_embeddings(
    batch: int, seq_len: int, head_dim: int, device: str
):
    """Create example RoPE position embeddings with static export shapes."""
    cos = torch.randn(batch, seq_len, head_dim, device=device)
    sin = torch.randn(batch, seq_len, head_dim, device=device)
    return cos, sin


def _make_random_decode_attn_mask(batch: int, max_seq: int, device: str):
    """Create an example static additive attention mask for decode export."""
    effective_len = torch.randint(low=1, high=max_seq + 1, size=(1,)).item()
    mask = torch.zeros(batch, 1, max_seq, device=device, dtype=torch.float32)
    if effective_len < max_seq:
        mask[:, :, effective_len:] = -120.0
    return mask


def _make_random_append_prefill_attn_mask(
    batch: int,
    max_seq: int,
    append_seq: int,
    device: str,
):
    """Create an example static additive attention mask for append-prefill export."""
    if append_seq < 1:
        raise ValueError(f"append_seq must be positive, got {append_seq}.")
    if append_seq > max_seq:
        raise ValueError(
            f"append_seq must be less than or equal to max_seq. "
            f"Got append_seq={append_seq}, max_seq={max_seq}."
        )

    max_past_len = max_seq - append_seq
    if max_past_len > 0:
        effective_past_len = torch.randint(
            low=0, high=max_past_len + 1, size=(1,)
        ).item()
    else:
        effective_past_len = 0

    mask = torch.full(
        (batch, append_seq, max_seq), -120.0, device=device, dtype=torch.float32
    )
    if effective_past_len > 0:
        mask[:, :, :effective_past_len] = 0.0

    tail_start = max_seq - append_seq
    causal = torch.tril(
        torch.ones(append_seq, append_seq, device=device, dtype=torch.bool)
    )
    tail_mask = torch.zeros(append_seq, append_seq, device=device, dtype=torch.float32)
    tail_mask = tail_mask.masked_fill(~causal, -120.0)
    mask[:, :, tail_start:max_seq] = tail_mask.unsqueeze(0).expand(batch, -1, -1)
    return mask


def _make_random_decode_batch(model, batch: int, device: str, max_seq: int):
    """Create example inputs for a static single-token decode graph."""
    hidden_size = model.config.hidden_size
    head_dim = getattr(
        model.config, "head_dim", hidden_size // model.config.num_attention_heads
    )
    n_kv = model.config.num_key_value_heads

    hidden = torch.randn(batch, 1, hidden_size, device=device)
    pos = _make_random_position_embeddings(batch, 1, head_dim, device)
    mask = _make_random_decode_attn_mask(batch, max_seq, device)
    past_k = torch.randn(batch, n_kv, max_seq - 1, head_dim, device=device)
    past_v = torch.randn(batch, n_kv, max_seq - 1, head_dim, device=device)
    return hidden, pos, mask, (past_k, past_v)


def _make_random_append_prefill_batch(
    model,
    batch: int,
    device: str,
    max_seq: int,
    append_seq: int,
):
    """Create example inputs for a static append-prefill graph."""
    hidden_size = model.config.hidden_size
    head_dim = getattr(
        model.config, "head_dim", hidden_size // model.config.num_attention_heads
    )
    n_kv = model.config.num_key_value_heads

    if append_seq < 1:
        raise ValueError(f"append_seq must be positive, got {append_seq}.")
    if append_seq > max_seq:
        raise ValueError(
            f"append_seq must be less than or equal to max_seq. "
            f"Got append_seq={append_seq}, max_seq={max_seq}."
        )

    hidden = torch.randn(batch, append_seq, hidden_size, device=device)
    pos = _make_random_position_embeddings(batch, append_seq, head_dim, device)
    mask = _make_random_append_prefill_attn_mask(batch, max_seq, append_seq, device)
    past_len = max_seq - append_seq
    past_k = torch.randn(batch, n_kv, past_len, head_dim, device=device)
    past_v = torch.randn(batch, n_kv, past_len, head_dim, device=device)
    return hidden, pos, mask, (past_k, past_v)


def _normalize_append_prefill_buckets(
    append_prefill_buckets: Sequence[int] | None,
    max_seq_len: int,
) -> tuple[int, ...]:
    """Validate and normalize append-prefill bucket sizes."""
    if append_prefill_buckets is None:
        return ()

    buckets = tuple(sorted({int(bucket) for bucket in append_prefill_buckets}))
    for bucket in buckets:
        if bucket < 1:
            raise ValueError(f"append-prefill bucket size must be positive: {bucket}")
        if bucket > max_seq_len:
            raise ValueError(
                f"append-prefill bucket size {bucket} exceeds max_seq_len "
                f"{max_seq_len}."
            )
    return buckets


def _is_wrapped_export_model(model: torch.nn.Module) -> bool:
    """Return whether the model already exposes the PTQ wrapper export layout."""
    wrapped = getattr(model, "wrapped", None)
    return wrapped is not None and hasattr(wrapped, "model")


def _float_artifact_tag(model: torch.nn.Module) -> str:
    """Validate a float32 model and return its artifact tag."""
    try:
        dtype = next(model.parameters()).dtype
    except StopIteration:
        dtype = torch.float32

    if dtype is not torch.float32:
        raise TypeError(
            "Floating-point LLaMA export currently supports float32 only. "
            f"Got parameter dtype {dtype}."
        )
    return "f32"


def _prepare_llama_export_model(
    model: torch.nn.Module,
) -> tuple[torch.nn.Module, str, bool]:
    """Normalize a checkpoint or floating-point model for per-layer export.

    Floating-point models are wrapped structurally with the NPU export profile.
    The wrapper remains in its default NO_QUANT mode, so no calibration or
    fake-quantization is introduced. Already wrapped checkpoints are preserved.

    Returns:
        A tuple containing the export model, artifact tag, and whether fake-quant
        meta kernels may be required during dynamic export.
    """
    model = model.eval().cpu()
    if _is_wrapped_export_model(model):
        return model, "q", True

    artifact_tag = _float_artifact_tag(model)
    wrapper_config = PTQConfig(
        model_args={"profile": "npu_export"},
        strict_wrap=True,
    )
    export_model = PTQWrapHelper(strict_wrap=wrapper_config.strict_wrap).wrap_supported(
        model, wrapper_config
    )
    if not _is_wrapped_export_model(export_model):
        raise TypeError(
            "Floating-point LLaMA export requires a top-level wrapper with a "
            "wrapped model."
        )
    return export_model, artifact_tag, False


def _circle_name(stem: str, artifact_tag: str) -> str:
    """Build a Circle artifact name with an explicit precision tag."""
    return f"{stem}.{artifact_tag}.circle"


def export_token_embedding(
    qmodel: torch.nn.Module,
    max_seq_len: int,
    output_dir: Path,
    *,
    artifact_tag: str = "q",
    register_fake_quant_kernels: bool = True,
) -> None:
    """Export the token embedding stage with the configured precision tag."""
    if register_fake_quant_kernels:
        register_fake_quant_meta_kernels_for_dynamic_export()
    example = make_token_embedding_example_input(qmodel=qmodel, max_seq_len=max_seq_len)
    dynamic_shapes = make_token_embedding_dynamic_shapes(max_seq_len)
    _convert_and_save(
        LlamaTokenEmbeddingExportAdapter(qmodel),
        (example,),
        output_dir / _circle_name("token_embedding", artifact_tag),
        dynamic_shapes=dynamic_shapes,
    )


def export_lm_head(
    qmodel: torch.nn.Module,
    output_dir: Path,
    *,
    artifact_tag: str = "q",
) -> None:
    """Export the final normalization and LM head stage."""
    example_hidden = torch.randn(1, 1, int(qmodel.config.hidden_size), device="cpu")
    _convert_and_save(
        LlamaLMHeadExportAdapter(qmodel),
        (example_hidden,),
        output_dir / _circle_name("lm_head", artifact_tag),
    )


def export_llama_per_layer(
    *,
    q_model: torch.nn.Module,
    max_seq_len: int,
    output_dir: str | Path,
    prefill_decode: bool = False,
    append_prefill_buckets: Sequence[int] | None = None,
) -> None:
    """Export a floating-point or PTQ-wrapped LLaMA model by stage and layer.

    Args:
        q_model: Floating-point or PTQ-wrapped LLaMA model.
        max_seq_len: Static attention key length used for exported decoder graphs.
        output_dir: Directory where Circle artifacts are saved.
        prefill_decode: If True, export both prefill and single-token decode graphs.
        append_prefill_buckets: Optional fixed block sizes `Q` for multi-turn
            append-prefill graphs. Each graph consumes `(B, Q, D)` hidden states,
            `(B, Q, max_seq_len)` additive masks, `(B, Q, head_dim)` RoPE tensors,
            and `(B, num_kv_heads, max_seq_len - Q, head_dim)` past KV tensors.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (
        export_model,
        artifact_tag,
        register_fake_quant_kernels,
    ) = _prepare_llama_export_model(q_model)
    qmodel = export_model.wrapped
    layers = qmodel.model.wrapped.layers
    config = qmodel.config
    append_buckets = _normalize_append_prefill_buckets(
        append_prefill_buckets, max_seq_len
    )

    export_token_embedding(
        qmodel,
        max_seq_len,
        output_dir,
        artifact_tag=artifact_tag,
        register_fake_quant_kernels=register_fake_quant_kernels,
    )

    for i, qlayer in enumerate(layers):
        suffix = "prefill_" if prefill_decode else ""
        layer_name = f"decoder_layer_{suffix}{i}"
        save_path = output_dir / _circle_name(layer_name, artifact_tag)

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
                output_dir / _circle_name(layer_name, artifact_tag),
                kwargs={
                    "attention_mask": mask,
                    "past_key_value": past,
                    "position_embeddings": pos,
                },
            )

        for append_seq in append_buckets:
            layer_name = f"decoder_layer_append_prefill_q{append_seq}_{i}"
            hidden, pos, mask, past = _make_random_append_prefill_batch(
                qmodel, 1, "cpu", max_seq_len, append_seq
            )
            _convert_and_save(
                qlayer.wrapped.as_export_module("append_prefill"),
                (hidden,),
                output_dir / _circle_name(layer_name, artifact_tag),
                kwargs={
                    "attention_mask": mask,
                    "past_key_value": past,
                    "position_embeddings": pos,
                },
            )

    export_lm_head(qmodel, output_dir, artifact_tag=artifact_tag)
