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

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

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
        self.lm_head.weight = self.embed.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embed(token_ids)
        return self.lm_head(hidden)


@dataclass
class TiedEmbeddingSmokeConfig:
    vocab_size: int = 16
    hidden_size: int = 8
    batch_size: int = 1
    seq_len: int = 5
    calib_iters: int = 16
    save_path: str = "tied_embedding_lm.q.circle"
    skip_circle_sharing_check: bool = False


def run_tied_embedding_smoke(cfg: TiedEmbeddingSmokeConfig) -> None:
    torch.manual_seed(2026)

    model = TiedEmbeddingLM(cfg.vocab_size, cfg.hidden_size).eval()
    fp32_ref = copy.deepcopy(model).eval()

    _assert_tied_weight(model, "initial model")
    _assert_tied_weight(fp32_ref, "FP32 reference")

    model = _quantize_model(model)
    _calibrate(model, cfg.vocab_size, cfg.batch_size, cfg.seq_len, cfg.calib_iters)
    model = _freeze_quantized_model(model)

    check_tokens = torch.randint(
        0,
        cfg.vocab_size,
        (cfg.batch_size, cfg.seq_len),
        dtype=torch.long,
    )

    with torch.no_grad():
        fp32_out = fp32_ref(check_tokens)
        quant_out = model(check_tokens)

    print("\n[Quantization sanity check]")
    print(f"  Mean |diff|: {(quant_out - fp32_out).abs().mean().item():.6f}")

    with torch.no_grad():
        exported_program = torch.export.export(model, (check_tokens,))
    _print_exported_weight_placeholders(exported_program)

    save_path = Path(cfg.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with SuppressWarning(UserWarning, ".*"):
        circle_model = tico.convert(model, (check_tokens,))

    circle_model.save(save_path)
    print(f"\n[OK] Quantized Circle model saved to {save_path.resolve()}")

    tied_weight_shape = [cfg.vocab_size, cfg.hidden_size]
    matched_tensors = _circle_data_tensors_with_shape(
        circle_model.circle_binary, tied_weight_shape
    )

    print("\n[Circle data-backed tensors with tied weight shape]")
    print(f"  expected shape: {tied_weight_shape}")
    for tensor_id, name, buffer_id, buffer_data_size, tensor_type in matched_tensors:
        print(
            f"  tensor_id={tensor_id:<4d} buffer_id={buffer_id:<4d} "
            f"bytes={buffer_data_size:<6d} type={tensor_type:<3d} name={name}"
        )

    if not cfg.skip_circle_sharing_check:
        assert len(matched_tensors) == 1, (
            "Expected exactly one data-backed Circle tensor for the tied weight "
            f"shape {tied_weight_shape}, but found {len(matched_tensors)}."
        )
        print("\n[OK] Circle export has one shared tied-weight tensor.")


def _unwrap_weight(module: nn.Module) -> torch.Tensor:
    if hasattr(module, "wrapped") and hasattr(module.wrapped, "module"):
        return module.wrapped.module.weight
    return module.weight  # type: ignore[attr-defined]


def _assert_tied_weight(model: TiedEmbeddingLM, stage: str) -> None:
    embed_weight = _unwrap_weight(model.embed)
    lm_head_weight = _unwrap_weight(model.lm_head)
    assert (
        embed_weight is lm_head_weight
    ), f"{stage}: weights are not the same Parameter."
    assert (
        embed_weight.data_ptr() == lm_head_weight.data_ptr()
    ), f"{stage}: weights do not share storage."
    print(
        f"[OK] {stage}: tied weight data_ptr={embed_weight.data_ptr()}, shape={tuple(embed_weight.shape)}"
    )


def _quantize_model(model: TiedEmbeddingLM) -> TiedEmbeddingLM:
    qcfg = PTQConfig()
    model.embed = prepare(model.embed, qcfg)  # type: ignore[assignment]
    model.lm_head = prepare(model.lm_head, qcfg)  # type: ignore[assignment]

    assert isinstance(model.embed.wrapped, QuantEmbedding)  # type: ignore[attr-defined]
    assert isinstance(model.lm_head.wrapped, QuantLinear)  # type: ignore[attr-defined]
    _assert_tied_weight(model, "after prepare()")
    return model


def _calibrate(
    model: TiedEmbeddingLM, vocab_size: int, batch: int, seq: int, iters: int
) -> None:
    model.eval()
    with torch.no_grad():
        for _ in range(iters):
            token_ids = torch.randint(0, vocab_size, (batch, seq), dtype=torch.long)
            model(token_ids)


def _freeze_quantized_model(model: TiedEmbeddingLM) -> TiedEmbeddingLM:
    model.embed = qconvert(model.embed)  # type: ignore[assignment]
    model.lm_head = qconvert(model.lm_head)  # type: ignore[assignment]
    assert model.embed._mode is Mode.QUANT  # type: ignore[attr-defined]
    assert model.lm_head._mode is Mode.QUANT  # type: ignore[attr-defined]
    _assert_tied_weight(model, "after quantization convert()")
    return model


def _print_exported_weight_placeholders(
    exported_program: torch.export.ExportedProgram,
) -> None:
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
    return [tensor.Shape(i) for i in range(tensor.ShapeLength())]


def _circle_tensor_name(tensor) -> str:
    raw_name = tensor.Name()
    return "" if raw_name is None else raw_name.decode("utf-8")


def _circle_data_tensors_with_shape(
    circle_binary: bytes, shape: Sequence[int]
) -> list[tuple[int, str, int, int, int]]:
    model = circle.Model.Model.GetRootAsModel(bytearray(circle_binary), 0)
    subgraph = model.Subgraphs(0)
    matched = []
    expected_shape = list(shape)

    for tensor_id in range(subgraph.TensorsLength()):
        tensor = subgraph.Tensors(tensor_id)
        if _circle_tensor_shape(tensor) != expected_shape:
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
