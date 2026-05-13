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

"""
PTQ export smoke test for tied embedding weights.

This script builds a tiny language-model-like module whose token embedding
weight and LM-head weight are tied. It then quantizes both modules with TICO
WrapQ, exports the quantized model through tico.convert(), and checks that the
serialized Circle model has only one data-backed tensor for the tied weight
shape.
"""

import argparse
import copy
import os
import pathlib
from typing import Any, Sequence

import torch
import torch.nn as nn
from circle_schema import circle

import tico
from tico.quantization import convert as qconvert, prepare
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.wrappers.nn.quant_embedding import QuantEmbedding
from tico.quantization.wrapq.wrappers.nn.quant_linear import QuantLinear
from tico.utils.utils import SuppressWarning


class TiedEmbeddingLM(nn.Module):
    """A tiny model with tied token embedding and LM-head weights."""

    def __init__(self, vocab_size: int = 16, hidden_size: int = 8) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # This is the same pattern used by many language models.
        self.lm_head.weight = self.embed.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Run embedding lookup followed by the tied LM head."""
        hidden = self.embed(token_ids)
        return self.lm_head(hidden)


def _unwrap_weight(module: nn.Module) -> torch.Tensor:
    """Return the wrapped floating-point module weight when WrapQ is applied."""
    if hasattr(module, "wrapped") and hasattr(module.wrapped, "module"):
        return module.wrapped.module.weight
    return module.weight  # type: ignore[attr-defined]


def _assert_tied_weight(model: TiedEmbeddingLM, stage: str) -> None:
    """Assert that embedding and LM-head weights share the same tensor storage."""
    embed_weight = _unwrap_weight(model.embed)
    lm_head_weight = _unwrap_weight(model.lm_head)

    assert (
        embed_weight is lm_head_weight
    ), f"{stage}: embedding and LM-head do not reference the same Parameter object."
    assert (
        embed_weight.data_ptr() == lm_head_weight.data_ptr()
    ), f"{stage}: embedding and LM-head do not share the same data_ptr."

    print(
        f"[OK] {stage}: tied weight data_ptr={embed_weight.data_ptr()}, "
        f"shape={tuple(embed_weight.shape)}"
    )


def _storage_key(tensor: torch.Tensor) -> tuple[Any, ...]:
    """Return a storage-identity key for debug printing."""
    return (
        str(tensor.device),
        tensor.data_ptr(),
        tensor.storage_offset(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.layout,
    )


def _print_exported_weight_placeholders(
    exported_program: torch.export.ExportedProgram,
) -> None:
    """Print exported placeholders that point to module weights."""
    print("\n[ExportedProgram parameter placeholders]")

    found = False
    for (
        placeholder_name,
        param_name,
    ) in exported_program.graph_signature.inputs_to_parameters.items():
        if not param_name.endswith("module.weight"):
            continue

        tensor = exported_program.state_dict[param_name]
        if not isinstance(tensor, torch.Tensor):
            continue

        found = True
        print(
            f"  {placeholder_name:<32s} -> {param_name:<40s} "
            f"data_ptr={tensor.data_ptr()} shape={tuple(tensor.shape)}"
        )

    if not found:
        print("  No module.weight placeholders were found.")


def _circle_tensor_shape(tensor) -> list[int]:
    """Return a Circle tensor shape as a Python list."""
    return [tensor.Shape(i) for i in range(tensor.ShapeLength())]


def _circle_tensor_name(tensor) -> str:
    """Return a Circle tensor name as a Python string."""
    raw_name = tensor.Name()
    if raw_name is None:
        return ""
    return raw_name.decode("utf-8")


def _circle_data_tensors_with_shape(
    circle_binary: bytes,
    shape: Sequence[int],
) -> list[tuple[int, str, int, int, int]]:
    """Return data-backed Circle tensors with the requested shape.

    Each tuple is:
        tensor_id, tensor_name, buffer_id, buffer_data_size, tensor_type
    """
    model = circle.Model.Model.GetRootAsModel(bytearray(circle_binary), 0)
    subgraph = model.Subgraphs(0)

    matched = []
    expected_shape = list(shape)

    for tensor_id in range(subgraph.TensorsLength()):
        tensor = subgraph.Tensors(tensor_id)
        tensor_shape = _circle_tensor_shape(tensor)
        if tensor_shape != expected_shape:
            continue

        buffer_id = tensor.Buffer()
        buffer = model.Buffers(buffer_id)
        buffer_data_size = buffer.DataLength()

        if buffer_data_size == 0:
            continue

        matched.append(
            (
                tensor_id,
                _circle_tensor_name(tensor),
                buffer_id,
                buffer_data_size,
                tensor.Type(),
            )
        )

    return matched


def _quantize_model(model: TiedEmbeddingLM) -> TiedEmbeddingLM:
    """Apply WrapQ PTQ to the embedding and LM-head modules."""
    qcfg = PTQConfig()

    model.embed = prepare(model.embed, qcfg)  # type: ignore[assignment]
    model.lm_head = prepare(model.lm_head, qcfg)  # type: ignore[assignment]

    assert isinstance(model.embed.wrapped, QuantEmbedding)  # type: ignore[attr-defined]
    assert isinstance(model.lm_head.wrapped, QuantLinear)  # type: ignore[attr-defined]

    _assert_tied_weight(model, "after prepare()")

    return model


def _calibrate(
    model: TiedEmbeddingLM, vocab_size: int, batch: int, seq: int, iters: int
):
    """Run a small calibration loop."""
    model.eval()

    with torch.no_grad():
        for _ in range(iters):
            token_ids = torch.randint(0, vocab_size, (batch, seq), dtype=torch.long)
            _ = model(token_ids)


def _freeze_quantized_model(model: TiedEmbeddingLM) -> TiedEmbeddingLM:
    """Freeze WrapQ observers and switch modules to quantized simulation mode."""
    model.embed = qconvert(model.embed)  # type: ignore[assignment]
    model.lm_head = qconvert(model.lm_head)  # type: ignore[assignment]

    assert model.embed._mode is Mode.QUANT  # type: ignore[attr-defined]
    assert model.lm_head._mode is Mode.QUANT  # type: ignore[attr-defined]

    _assert_tied_weight(model, "after quantization convert()")

    return model


def main() -> None:
    """Run the tied-weight quantization and Circle export smoke test."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=5)
    parser.add_argument("--calib-iters", type=int, default=16)
    parser.add_argument(
        "--save-path",
        type=pathlib.Path,
        default=pathlib.Path("tied_embedding_lm.q.circle"),
    )
    parser.add_argument(
        "--dump-graphs",
        action="store_true",
        help="Set TICO_GRAPH_DUMP=1 before running tico.convert().",
    )
    parser.add_argument(
        "--skip-circle-sharing-check",
        action="store_true",
        help="Skip the final Circle tensor sharing assertion.",
    )
    args = parser.parse_args()

    if args.dump_graphs:
        os.environ["TICO_GRAPH_DUMP"] = "1"

    torch.manual_seed(2026)

    model = TiedEmbeddingLM(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
    ).eval()
    fp32_ref = copy.deepcopy(model).eval()

    _assert_tied_weight(model, "initial model")
    _assert_tied_weight(fp32_ref, "FP32 reference")

    model = _quantize_model(model)
    _calibrate(
        model,
        vocab_size=args.vocab_size,
        batch=args.batch_size,
        seq=args.seq_len,
        iters=args.calib_iters,
    )
    model = _freeze_quantized_model(model)

    check_tokens = torch.randint(
        0,
        args.vocab_size,
        (args.batch_size, args.seq_len),
        dtype=torch.long,
    )

    with torch.no_grad():
        fp32_out = fp32_ref(check_tokens)
        quant_out = model(check_tokens)

    print("\n[Quantization sanity check]")
    print(f"  Mean |diff|: {(quant_out - fp32_out).abs().mean().item():.6f}")

    # Export once directly with torch.export so the placeholder sharing state is
    # easy to inspect before the full TICO conversion pipeline runs.
    with torch.no_grad():
        exported_program = torch.export.export(model, (check_tokens,))

    _print_exported_weight_placeholders(exported_program)

    embed_weight = _unwrap_weight(model.embed)
    lm_head_weight = _unwrap_weight(model.lm_head)
    print("\n[Tied weight storage keys]")
    print(f"  embedding: {_storage_key(embed_weight)}")
    print(f"  lm_head  : {_storage_key(lm_head_weight)}")

    # Run the full TICO conversion. This is the path that exercises:
    #   ConstPropPass -> FoldQuantOps -> RemoveWeightDequantOp -> serializer
    with SuppressWarning(UserWarning, ".*"):
        circle_model = tico.convert(model, (check_tokens,))

    circle_model.save(args.save_path)
    print(f"\n[OK] Quantized Circle model saved to {args.save_path.resolve()}")

    tied_weight_shape = [args.vocab_size, args.hidden_size]
    matched_tensors = _circle_data_tensors_with_shape(
        circle_model.circle_binary,
        tied_weight_shape,
    )

    print("\n[Circle data-backed tensors with tied weight shape]")
    print(f"  expected shape: {tied_weight_shape}")
    for tensor_id, name, buffer_id, buffer_data_size, tensor_type in matched_tensors:
        print(
            f"  tensor_id={tensor_id:<4d} buffer_id={buffer_id:<4d} "
            f"bytes={buffer_data_size:<6d} type={tensor_type:<3d} name={name}"
        )

    if not args.skip_circle_sharing_check:
        assert len(matched_tensors) == 1, (
            "Expected exactly one data-backed Circle tensor for the tied weight "
            f"shape {tied_weight_shape}, but found {len(matched_tensors)}. "
            "This usually means tied Q propagation or serializer tensor sharing "
            "is still broken."
        )
        print("\n[OK] Circle export has one shared tied-weight tensor.")


if __name__ == "__main__":
    main()
