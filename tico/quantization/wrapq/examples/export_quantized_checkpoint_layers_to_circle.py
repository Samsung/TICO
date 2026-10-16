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

# =============================================================================
# EXPORT AN ALREADY-QUANTIZED CHECKPOINT TO CIRCLE, LAYER BY LAYER
# -----------------------------------------------------------------------------
# This script loads a PTQ checkpoint previously saved with
#     torch.save(q_m, "PTQ_<model>_....pt")
# by quantize_full_qmodel_with_gptq.py, and exports it to Circle format:
#
#   Artifacts (prefill + decode):
#     - token_embedding.q.circle           (shared, dynamic seq dim)
#     - decoder_layer_prefill_{i}.q.circle (one per decoder layer)
#     - decoder_layer_decode_{i}.q.circle  (one per decoder layer)
#     - lm_head.q.circle                   (shared, single token)
#
# Example:
#   python tico/quantization/wrapq/examples/export_quantized_checkpoint_layers_to_circle.py \
#       --checkpoint output/PTQ_meta-llama_Llama-3.2-1B_GPTQ_..._npu_export_42.pt \
#       --output-dir output/circles --max-seq-len 512
#
# NOTES:
#   - The checkpoint must come from a PTQ-wrapped model (has 'wrapped').
#   - Loading uses torch.load(..., weights_only=False) because the whole module
#     is pickled; torch >= 2.6 defaults to weights_only=True which would fail.
#   - The installed tico package must match the one used at quantization time.
#   - The model must have been prepared with the 'npu_export' execution profile
#     for Circle export (same restriction as quantize_full_qmodel_with_gptq.py).
# =============================================================================

import argparse
import pathlib
from typing import Any, Optional

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export an already-quantized (torch.save'd) PTQ model to Circle "
            "format, layer by layer (prefill + decode)."
        )
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the quantized checkpoint saved with torch.save(q_m, path).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for .circle artifacts (default: checkpoint's directory).",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help=(
            "Max sequence length used for export example inputs "
            "(default: model config max_position_embeddings)."
        ),
    )
    parser.add_argument(
        "--skip-decode",
        action="store_true",
        help="Export prefill layers only (skip decoder_layer_decode_* artifacts).",
    )
    parser.add_argument(
        "--skip-token-embedding",
        action="store_true",
        help="Skip exporting token_embedding.q.circle.",
    )
    parser.add_argument(
        "--skip-lm-head",
        action="store_true",
        help="Skip exporting lm_head.q.circle.",
    )
    parser.add_argument(
        "--full-model",
        action="store_true",
        help="Also export whole-model prefill/decode circles.",
    )
    return parser


def load_quantized_checkpoint(path: pathlib.Path) -> torch.nn.Module:
    """Load a torch.save'd quantized model (q_m with PTQ wrappers)."""
    print(f"Loading quantized checkpoint from {path.resolve()}")
    q_m = torch.load(path, map_location="cpu", weights_only=False)
    if not hasattr(q_m, "wrapped"):
        raise ValueError(
            "Checkpoint is not a PTQ-wrapped model (missing 'wrapped'). "
            "Per-layer Circle export is supported only for PTQ quantized models."
        )
    q_m.eval()
    q_m.cpu()
    return q_m


# -----------------------------------------------------------------------------
# Export helpers (mirroring quantize_full_qmodel_with_gptq.py)
# -----------------------------------------------------------------------------
def save_export_module_to(
    module: torch.nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    save_path: pathlib.Path,
    artifact_name: str,
    *,
    kwargs: Optional[dict[str, Any]] = None,
    dynamic_shapes: Optional[Any] = None,
    strict: bool = False,
) -> None:
    """Convert an export module to Circle and save it."""
    print(f"Saving {artifact_name} to {save_path.resolve()}")

    with torch.no_grad():
        with SuppressWarning(UserWarning, ".*"):
            cm = tico.convert(
                module.eval(),
                example_inputs,
                kwargs=kwargs,
                dynamic_shapes=dynamic_shapes,
                strict=strict,
            )

    cm.save(save_path)


def make_random_position_embeddings(B, head_dim, DEVICE):
    """Create random RoPE tables for one decode step."""
    cos = torch.randn(B, 1, head_dim, device=DEVICE)
    sin = torch.randn(B, 1, head_dim, device=DEVICE)
    return (cos, sin)


def make_random_decode_attn_mask(B, MAX_SEQ, DEVICE):
    # Additive mask of final static width: (B, 1, MAX_SEQ)
    # Simulate that only the first L_eff positions are valid and the rest are padding.
    L_eff = torch.randint(low=1, high=MAX_SEQ + 1, size=(1,)).item()
    mask = torch.zeros(B, 1, MAX_SEQ, device=DEVICE, dtype=torch.float32)
    if L_eff < MAX_SEQ:
        mask[:, :, L_eff:] = float("-120")
    return mask


def make_random_decode_batch(model, B, DEVICE, MAX_SEQ):
    """Create a synthetic decode batch for per-layer export."""
    D = model.config.hidden_size
    head_dim = getattr(model.config, "head_dim", D // model.config.num_attention_heads)
    n_kv = model.config.num_key_value_heads

    # Single-token hidden state.
    x = torch.randn(B, 1, D, device=DEVICE)
    pos = make_random_position_embeddings(B, head_dim, DEVICE)
    mask = make_random_decode_attn_mask(B, MAX_SEQ, DEVICE)

    # Static-sized past KV (already RoPE-applied for past tokens).
    past_k = torch.randn(B, n_kv, MAX_SEQ - 1, head_dim, device=DEVICE)
    past_v = torch.randn(B, n_kv, MAX_SEQ - 1, head_dim, device=DEVICE)
    past = (past_k, past_v)

    return x, pos, mask, past


def save_token_embedding_to(
    qmodel: torch.nn.Module,
    max_seq_len: int,
    save_layers_to_folder: str | pathlib.Path,
) -> None:
    """
    Export and save the token embedding stage with a dynamic sequence dimension.

    The generated Circle model is shared by prefill and decode runtime paths.

    Circle contract:
        input_ids:     `(1, S)`
        hidden_states: `(1, S, hidden_size)`

    The sequence dimension `S` is dynamic and bounded by
    `1 <= S <= max_seq_len`.
    """
    register_fake_quant_meta_kernels_for_dynamic_export()

    artifact_name = "token_embedding"
    save_path = pathlib.Path(save_layers_to_folder, f"{artifact_name}.q.circle")

    example_input_ids = make_token_embedding_example_input(
        qmodel=qmodel,
        max_seq_len=max_seq_len,
    )
    dynamic_shapes = make_token_embedding_dynamic_shapes(max_seq_len)

    save_export_module_to(
        LlamaTokenEmbeddingExportAdapter(qmodel),
        (example_input_ids,),
        save_path,
        artifact_name,
        dynamic_shapes=dynamic_shapes,
    )


def save_lm_head_to(
    qmodel: torch.nn.Module,
    save_layers_to_folder: str | pathlib.Path,
) -> None:
    """
    Export and save the shared single-token LM head stage.

    This artifact is used for both:
        - the last real token after prefill
        - every decode token

    Circle contract:
        hidden_states: `(1, 1, hidden_size)`
        logits:        `(1, 1, vocab_size)`

    The runtime should slice or gather the last real prefill hidden state before
    calling this artifact.
    """
    artifact_name = "lm_head"
    save_path = pathlib.Path(save_layers_to_folder, f"{artifact_name}.q.circle")
    example_hidden = torch.randn(
        1,
        1,
        int(qmodel.config.hidden_size),
        device="cpu",
    )

    save_export_module_to(
        LlamaLMHeadExportAdapter(qmodel),
        (example_hidden,),
        save_path,
        artifact_name,
    )


def save_layers_to(
    q_m,
    max_seq_len: int,
    save_layers_to_folder: str | pathlib.Path,
    prefill_decode: bool = True,
    *,
    skip_token_embedding: bool = False,
    skip_lm_head: bool = False,
) -> None:
    """
    Export and save quantized token embedding, decoder layers, and LM head.

    Artifacts:
        - `token_embedding.q.circle`
            Shared by prefill and decode. Its sequence dimension is dynamic.
        - `decoder_layer_prefill_{i}.q.circle` (one per decoder layer)
            Static `max_seq_len` sequence dimension.
        - `decoder_layer_decode_{i}.q.circle` (one per decoder layer, optional)
            Single-token decode step with a static past KV cache.
        - `lm_head.q.circle`
            Shared by prefill and decode. Single-token logits.
    """
    if max_seq_len is None:
        raise ValueError("max_seq_len must be set for per-layer Circle export.")

    max_seq_len = int(max_seq_len)
    if max_seq_len < 1:
        raise ValueError(f"max_seq_len must be positive, got {max_seq_len}.")

    qmodel = q_m.wrapped
    layers = qmodel.model.wrapped.layers
    config = qmodel.config

    # Token embedding runs on CPU in the target runtime, so export it once with
    # dynamic sequence length. This one artifact covers both prefill and decode.
    if not skip_token_embedding:
        save_token_embedding_to(
            qmodel=qmodel,
            max_seq_len=max_seq_len,
            save_layers_to_folder=save_layers_to_folder,
        )

    for i, qlayer in enumerate(layers):
        layer_name = f"decoder_layer_prefill_{i}"
        save_path = pathlib.Path(save_layers_to_folder, f"{layer_name}.q.circle")
        B, S, D = 1, max_seq_len, config.hidden_size
        example_hidden = torch.randn(B, S, D, device="cpu")

        attention_mask = (
            qlayer.wrapped.causal_mask_template[..., :S, :S].squeeze(0).to("cpu")
        )
        dtype = example_hidden.dtype
        pos_embeds = qlayer.wrapped._slice_rope(
            start=0, seq_len=S, device="cpu", dtype=dtype
        )

        print(f"Saving {layer_name} to {save_path.resolve()}")
        with torch.no_grad():
            with SuppressWarning(UserWarning, ".*"):
                # Pass attention_mask and position_embeddings as inputs to avoid
                # storing them per layer and increasing model size.
                cm = tico.convert(
                    qlayer.wrapped.as_export_module(
                        "prefill", return_kv=prefill_decode
                    ).eval(),
                    (example_hidden,),
                    kwargs={
                        "attention_mask": attention_mask,
                        "position_embeddings": pos_embeds,
                    },
                )
        cm.save(save_path)

        if prefill_decode:
            layer_name = f"decoder_layer_decode_{i}"
            save_path = pathlib.Path(save_layers_to_folder, f"{layer_name}.q.circle")
            print(f"Saving {layer_name} to {save_path.resolve()}")

            with torch.no_grad():
                with SuppressWarning(UserWarning, ".*"):
                    ex_hid, pos_embeds, attn_mask, past = make_random_decode_batch(
                        q_m.wrapped, B=1, DEVICE="cpu", MAX_SEQ=max_seq_len
                    )
                    cm = tico.convert(
                        qlayer.wrapped.as_export_module("decode").eval(),
                        (ex_hid,),  # hidden_states
                        kwargs={
                            "attention_mask": attn_mask,
                            "past_key_value": past,
                            "position_embeddings": pos_embeds,
                        },
                    )
            cm.save(save_path)

    # The runtime only needs logits for one token:
    #   - the last real token after prefill
    #   - the current token during decode
    # Therefore one shared single-token LM head artifact is enough.
    if not skip_lm_head:
        save_lm_head_to(
            qmodel=qmodel,
            save_layers_to_folder=save_layers_to_folder,
        )


def save_full_model_to(
    q_m,
    max_seq_len: int,
    save_circle_to_folder: str | pathlib.Path,
    prefill_decode: bool = True,
) -> None:
    """
    Export and save the whole quantized model in circle format.

    Uses synthetic padded token ids as example input, since only shapes/dtypes
    are traced by torch.export.
    """
    qmodel = q_m.wrapped
    config = qmodel.config
    S = int(max_seq_len)

    token = torch.zeros(1, S, dtype=torch.long)

    model_name = "model_prefill"
    save_path = pathlib.Path(save_circle_to_folder, f"{model_name}.q.circle")
    print(f"saving the whole {model_name} to {save_path.resolve()}")
    attention_mask = qmodel.causal_mask_template[..., :S, :S].squeeze(0).to("cpu")
    pos_embeds = (
        qmodel.rope_cos_template[:, :S, :].to("cpu"),
        qmodel.rope_sin_template[::S, :].to("cpu"),
    )
    with torch.no_grad():
        with SuppressWarning(UserWarning, ".*"):
            cm = tico.convert(
                q_m.wrapped.as_export_module(
                    "prefill", return_kv=prefill_decode
                ).eval(),
                (token,),
                kwargs={
                    "attention_mask": attention_mask,
                    "position_embeddings": pos_embeds,
                },
                strict=False,
            )
    cm.save(save_path)

    if prefill_decode:
        model_name = "model_decode"
        save_path = pathlib.Path(save_circle_to_folder, f"{model_name}.q.circle")
        print(f"saving the whole {model_name} to {save_path.resolve()}")

        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        n_kv = config.num_key_value_heads
        past_kv = [
            (
                torch.randn(1, n_kv, S - 1, head_dim, device="cpu"),
                torch.randn(1, n_kv, S - 1, head_dim, device="cpu"),
            )
            for _ in range(config.num_hidden_layers)
        ]
        with torch.no_grad():
            with SuppressWarning(UserWarning, ".*"):
                cm = tico.convert(
                    q_m.wrapped.as_export_module("decode").eval(),
                    (token[:, :1], past_kv),
                    kwargs={
                        "attention_mask": make_random_decode_attn_mask(1, S, "cpu"),
                        "position_embeddings": make_random_position_embeddings(
                            1, head_dim, "cpu"
                        ),
                    },
                    strict=False,
                )
        cm.save(save_path)


def main():
    args = build_parser().parse_args()
    print(args)

    checkpoint = pathlib.Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    output_dir = pathlib.Path(args.output_dir) if args.output_dir else checkpoint.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    q_m = load_quantized_checkpoint(checkpoint)

    config = q_m.wrapped.config
    max_seq_len = args.max_seq_len or int(config.max_position_embeddings)
    print(f"Model: {config.name_or_path}")
    print(f"Hidden layers: {config.num_hidden_layers}")
    print(f"Max seq len: {max_seq_len}")
    print(f"Output directory: {output_dir.resolve()}")

    save_layers_to(
        q_m,
        max_seq_len,
        output_dir,
        prefill_decode=not args.skip_decode,
        skip_token_embedding=args.skip_token_embedding,
        skip_lm_head=args.skip_lm_head,
    )

    if args.full_model:
        save_full_model_to(
            q_m,
            max_seq_len,
            output_dir,
            prefill_decode=not args.skip_decode,
        )

    print("Done.")


if __name__ == "__main__":
    main()
