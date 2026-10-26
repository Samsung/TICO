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

"""Locate the preexisting natural-grid RoPE eager/Circle discrepancy.

Run with the same fixture and natural, unoptimized artifacts as the companion
comparison script. Counterfactuals align the eager RoPE half-observer grids and
softmax tie rounding with the serialized equations. They do not change or
re-export any artifact.

Example from the repository root::

    python docs/archive/llama_kv_cache_rope_diagnostic.py \
        --repo . --fixture /tmp/tico-kv-validation-fixture \
        --circles /tmp/tico-kv-validation-candidate/natural/unoptimized \
        --output /tmp/llama-rope-diagnostic.json
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch

from llama_kv_cache_branch_validation import bind_inputs, metrics, write_json


def diagnose(args: argparse.Namespace) -> None:
    from tico.circle.runtime import CircleReferenceRuntime, ExecutionMode
    from tico.quantization import convert, prepare
    from tico.quantization.config.ptq import PTQConfig
    from tico.quantization.wrapq.wrappers.llama.static_inputs import (
        prepare_llama_static_inputs,
    )
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.set_num_threads(1)
    torch.manual_seed(1083)
    config = LlamaConfig(**json.loads((args.fixture / "config.json").read_text()))
    model = LlamaForCausalLM(config).eval()
    model.load_state_dict(
        torch.load(args.fixture / "float_state.pt", weights_only=True)
    )
    prepared = prepare(model, PTQConfig(model_args={"profile": "npu_export"}))
    with torch.no_grad():
        prepared(torch.tensor([[1, 5, 3, 7, 9, 4, 6, 8]]), use_cache=True)
        output = prepared(torch.tensor([[1, 5, 3]]), use_cache=True, return_dict=True)
        for token in (7, 9):
            output = prepared(
                torch.tensor([[token]]),
                past_key_values=output.past_key_values,
                use_cache=True,
                return_dict=True,
            )
    quantized = convert(prepared).eval()
    layer = quantized.wrapped.model.wrapped.layers[args.layer].wrapped
    attention = layer.self_attn.wrapped
    static = prepare_llama_static_inputs(layer, 8)
    runtime = CircleReferenceRuntime(
        (args.circles / f"decoder_layer_prefill_{args.layer}.q.circle").read_bytes()
    )
    values = {
        "hidden": torch.linspace(-0.08, 0.09, 128).reshape(1, 8, 16),
        "mask": static.mask(8, 8, 0, device="cpu"),
    }
    values["cos"], values["sin"] = static.rope(0, 8, device="cpu", dtype=torch.float32)
    inputs = bind_inputs(runtime, values)
    observers = {
        "norm": layer.input_layernorm.wrapped.obs_act_out,
        "k_projection": attention.k_proj.wrapped.obs_act_out,
        "k_x1": attention.obs_k_x1,
        "k_x2": attention.obs_k_x2,
        "k_cat": attention.obs_k_cat,
        "k_cos": attention.obs_k_cos,
        "k_sin": attention.obs_k_sin,
        "k_rot": attention.obs_k_rot,
        "softmax": attention.obs_softmax,
    }
    captured: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def capture(name: str, original: Any):
        def wrapped(value: torch.Tensor) -> torch.Tensor:
            result = original(value)
            captured[name] = (value.detach().clone(), result.detach().clone())
            return result

        return wrapped

    def eager():
        with torch.no_grad():
            return layer.as_export_module("prefill")(
                values["hidden"],
                attention_mask=values["mask"],
                position_embeddings=(values["cos"], values["sin"]),
            )

    with ExitStack() as stack:
        for name, observer in observers.items():
            stack.enter_context(
                patch.object(observer, "fake_quant", capture(name, observer.fake_quant))
            )
        reference = eager()
    traced = runtime.run(
        [value.numpy() for value in inputs],
        mode=ExecutionMode.FAKE_QUANTIZE,
        trace=True,
    )
    assert traced.tensor_values is not None
    suffixes = {
        "norm": "QuantRMSNorm::rms_norm",
        "k_projection": "QuantLinear::linear_1",
        "k_cat": "QuantLlamaAttention::cat",
        "k_cos": "QuantLlamaAttention::mul",
        "k_sin": "QuantLlamaAttention::mul_1",
        "k_rot": "QuantLlamaAttention::add",
        "softmax": "QuantLlamaAttention::_softmax",
    }
    report: dict[str, Any] = {
        "layer": args.layer,
        "observers": {},
        "serialized_vs_eager": {},
    }
    for name, observer in observers.items():
        scale, zero = observer.compute_qparams()
        before, after = captured[name]
        report["observers"][name] = {
            "scale": float(scale),
            "zero_point": int(zero),
            "input_min": float(before.min()),
            "input_max": float(before.max()),
            "output_min": float(after.min()),
            "output_max": float(after.max()),
            "fake_quant_change": metrics(after.numpy(), before.numpy()),
        }
        if name in suffixes:
            matches = [
                t for t in runtime.program.tensors if t.name.endswith(suffixes[name])
            ]
            assert len(matches) == 1, (name, matches)
            actual = traced.tensor_values[matches[0].index].reshape(after.shape)
            report["serialized_vs_eager"][name] = metrics(actual, after.numpy())
    report["original_k_delta"] = metrics(traced.outputs[1], reference[1].numpy())
    raw_k = captured["k_projection"][1].reshape(1, 8, 8)
    first, second = raw_k.chunk(2, dim=-1)
    circle_cat = next(
        t for t in runtime.program.tensors if t.name.endswith(suffixes["k_cat"])
    )
    report["serialized_cat_vs_unobserved_rotation"] = metrics(
        traced.tensor_values[circle_cat.index],
        torch.cat((second, first), dim=-1).numpy(),
    )
    # Diagnostic only: this is not a calibration change to either paired export.
    attention.obs_k_x1.load_qparams(
        *attention.k_proj.wrapped.obs_act_out.compute_qparams(), lock=True
    )
    counterfactual = eager()
    report["counterfactual_k_delta"] = metrics(
        traced.outputs[1], counterfactual[1].numpy()
    )
    for name in ("obs_q_x1", "obs_q_x2"):
        getattr(attention, name).load_qparams(
            *attention.q_proj.wrapped.obs_act_out.compute_qparams(), lock=True
        )
    attention.obs_k_x2.load_qparams(
        *attention.k_proj.wrapped.obs_act_out.compute_qparams(), lock=True
    )
    all_halves = eager()
    report["counterfactual_all_rope_halves"] = {
        name: metrics(actual, expected.numpy())
        for name, actual, expected in zip(
            ("hidden", "k", "v"), traced.outputs, all_halves
        )
    }
    softmax_scale, softmax_zero = attention.obs_softmax.compute_qparams()
    probabilities = captured["softmax"][0]
    scaled = probabilities / softmax_scale
    ties = scaled[scaled.frac().abs() == 0.5]
    report["softmax_half_ties"] = {
        "count": ties.numel(),
        "scaled_values": ties.tolist(),
    }

    def half_away_softmax(value: torch.Tensor) -> torch.Tensor:
        units = value / softmax_scale
        integral = torch.trunc(units)
        rounded = integral + units.sign() * ((units - integral).abs() >= 0.5)
        encoded = (rounded + softmax_zero).clamp(
            attention.obs_softmax.dtype.qmin, attention.obs_softmax.dtype.qmax
        )
        return (encoded - softmax_zero) * softmax_scale

    with patch.object(attention.obs_softmax, "fake_quant", half_away_softmax):
        same_rounding = eager()
    report["counterfactual_rope_and_softmax_rounding"] = {
        name: metrics(actual, expected.numpy())
        for name, actual, expected in zip(
            ("hidden", "k", "v"), traced.outputs, same_rounding
        )
    }
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--circles", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    diagnose(args)


if __name__ == "__main__":
    main()
