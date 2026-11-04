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

"""Compare selected-row execution with an explicit full-table DQ CPU oracle.

Run ``python -m test.performance.benchmark_rowwise_embedding`` from the repository.
No checkpoint, network, accelerator, or external runtime is required. Times exclude
model construction/loading; tracemalloc peaks are measured separately from timing.
They include tracked Python/NumPy allocations, not process RSS or allocator caches.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
import tracemalloc
from typing import Any
from unittest import mock

import numpy as np
import torch
from circle_schema import circle

from tico.circle.runtime import CircleReferenceRuntime, ExecutionMode
from tico.circle.runtime.kernels import quantize as quantize_kernels
from tico.circle.runtime.rowwise_gather import RowwiseGatherPlan

from test.support.circle.builder import CircleModelBuilder


def _options(name: str, **fields: Any) -> Any:
    result = getattr(getattr(circle, name), f"{name}T")()
    for key, value in fields.items():
        setattr(result, key, value)
    return result


def _circle(vocab: int, width: int, tokens: int) -> bytes:
    """Build a deterministic explicit lookup/Q/DQ/MUL/Q graph and serialize it."""

    builder = CircleModelBuilder(description="rowwise-embedding-cpu-benchmark")
    codes = (np.arange(vocab * width).reshape(vocab, width) * 37 % 256).astype(np.uint8)
    row = np.arange(vocab)
    weight = builder.quantized_constant(
        "weight",
        codes,
        tensor_type=circle.TensorType.TensorType.UINT8,
        scale=((row % 23 + 1) / 64).tolist(),
        zero_point=(row * 17 % 256).tolist(),
    )
    ids = builder.input("ids", [1, tokens], dtype=np.int64)
    full = builder.activation("table_fp32", [vocab, width])
    builder.operator(
        "DEQUANTIZE", [weight], [full], options=_options("DequantizeOptions")
    )
    selected = builder.activation("selected", [1, tokens, width])
    builder.operator(
        "GATHER", [full, ids], [selected], options=_options("GatherOptions", axis=0)
    )
    first_q = builder.activation(
        "embedding_q",
        [1, tokens, width],
        dtype=np.uint8,
        quantization=builder.quantization([0.125], [128]),
    )
    builder.operator(
        "QUANTIZE", [selected], [first_q], options=_options("QuantizeOptions")
    )
    first_dq = builder.activation("embedding_dq", [1, tokens, width])
    builder.operator(
        "DEQUANTIZE", [first_q], [first_dq], options=_options("DequantizeOptions")
    )
    # The scale operand itself has a quantization boundary: code 13 at scale
    # 0.1 restores to FP32 1.3, rather than an unquantized source value 1.34.
    scale_q = builder.quantized_constant(
        "scale_q",
        np.array(13, dtype=np.uint8),
        tensor_type=circle.TensorType.TensorType.UINT8,
        scale=[0.1],
        zero_point=[0],
    )
    scale = builder.activation("scale_dq", [])
    builder.operator(
        "DEQUANTIZE", [scale_q], [scale], options=_options("DequantizeOptions")
    )
    scaled = builder.mul(first_dq, scale, name="scaled")
    output = builder.activation(
        "npu_input",
        [1, tokens, width],
        dtype=np.int16,
        quantization=builder.quantization([0.002], [0]),
    )
    builder.operator(
        "QUANTIZE", [scaled], [output], options=_options("QuantizeOptions")
    )
    builder.set_outputs(output)
    return builder.build().to_bytes()


def _restored_shapes(runtime: CircleReferenceRuntime, ids: np.ndarray) -> list:
    """Observe actual restored FP32 shapes, separately from measured runs."""

    shapes = []
    original_dq = quantize_kernels.dequantize_array
    original_selected = RowwiseGatherPlan.execute

    def dequantize(value, tensor):
        result = original_dq(value, tensor)
        shapes.append(list(result.shape))
        return result

    def selected(plan, context):
        result = original_selected(plan, context)
        shapes.append(list(result[0].shape))
        return result

    with mock.patch.object(quantize_kernels, "dequantize_array", dequantize):
        with mock.patch.object(RowwiseGatherPlan, "execute", selected):
            runtime.run((ids,), mode=ExecutionMode.NATIVE)
    return shapes


def _measure(
    binary: bytes, ids: np.ndarray, *, selected: bool, warmup: int, repeat: int
) -> tuple[dict[str, Any], np.ndarray]:
    tracemalloc.start()
    runtime = CircleReferenceRuntime(binary, enable_rowwise_gather=selected)
    _, load_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    for _ in range(warmup):
        runtime.run((ids,), mode=ExecutionMode.NATIVE)
    elapsed = []
    for _ in range(repeat):
        started = time.perf_counter()
        runtime.run((ids,), mode=ExecutionMode.NATIVE)
        elapsed.append(time.perf_counter() - started)
    tracemalloc.start()
    result = runtime.run((ids,), mode=ExecutionMode.NATIVE).outputs[0]
    _, run_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    shapes = _restored_shapes(runtime, ids)
    if selected:
        assert len(runtime.rowwise_gather_plans) == 1
        assert list(runtime.program.tensor(0).shape) not in shapes
    return {
        "plan": "selected_rows" if selected else "full_table_oracle",
        "median_ms": statistics.median(elapsed) * 1000,
        "load_tracemalloc_peak_bytes": load_peak,
        "run_tracemalloc_peak_bytes": run_peak,
        "restored_fp32_shapes": shapes,
    }, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab", type=int, default=4096)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--prefill", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=30)
    args = parser.parse_args()
    if min(args.vocab, args.width, args.prefill, args.repeat) <= 0 or args.warmup < 0:
        parser.error("sizes/repeat must be positive and warmup nonnegative")
    torch.set_num_threads(1)
    report: dict[str, Any] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "torch_threads": torch.get_num_threads(),
        "table_shape": [args.vocab, args.width],
        "full_table_fp32_bytes": args.vocab * args.width * 4,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "mode": "NATIVE",
        "memory_scope": "tracemalloc Python/NumPy allocations; excludes untracked native allocations and RSS",
        "cases": [],
    }
    for label, count in (("decode", 1), ("prefill", args.prefill)):
        binary = _circle(args.vocab, args.width, count)
        ids = ((np.arange(count, dtype=np.int64) * 67 + args.vocab - 1) % args.vocab)[
            None, :
        ]
        samples = []
        expected = None
        for selected in (False, True):
            sample, result = _measure(
                binary, ids, selected=selected, warmup=args.warmup, repeat=args.repeat
            )
            if expected is None:
                expected = result
            else:
                np.testing.assert_array_equal(result, expected)
            assert result.dtype == np.int16
            samples.append(sample)
        report["cases"].append(
            {
                "case": label,
                "ids_shape": list(ids.shape),
                "selected_fp32_bytes": count * args.width * 4,
                "native_int16_exact_equal": True,
                "samples": samples,
            }
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
