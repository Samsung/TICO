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

"""Measure constant-pool and fold-preflight allocations without a real model.

Run each mode in a separate process on the base and patched source trees. The
same script works before and after the patch; no schema tables are generated.
Imports require the normal TICO development environment. Reported tracemalloc
numbers exclude input allocation and are not whole-export peak RSS estimates.
"""

from __future__ import annotations

import argparse
import functools
import gc
import json
import sys
import time
import tracemalloc
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

from tico.circle.builder import ConstantPool
from tico.circle.graph import CircleGraph
from tico.circle.value import TensorTypeRegistry, TensorTypeSpec, TensorValueCodec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload-mib", type=int, default=64)
    parser.add_argument("--aliases", type=int, default=1)
    parser.add_argument("--mode", choices=("pool", "preflight"), default="pool")
    args = parser.parse_args()
    if args.payload_mib <= 0 or args.aliases <= 0:
        parser.error("--payload-mib and --aliases must be positive")

    codec = TensorValueCodec(
        TensorTypeRegistry([TensorTypeSpec("UINT8", 3, np.uint8, np.uint8, 8, False)])
    )
    # Import session services before tracing; constructor work is measured below.
    from tico.circle.session import existing_optimization_session

    payload = np.full(args.payload_mib * 1024 * 1024, 7, dtype=np.uint8)
    tensors = [
        SimpleNamespace(
            name=f"weight_{index}",
            buffer=1,
            type=3,
            shape=[payload.size],
            shapeSignature=None,
            isVariable=False,
            quantization=None,
        )
        for index in range(args.aliases)
    ]
    model = SimpleNamespace(
        buffers=[
            SimpleNamespace(data=None, offset=0, size=0),
            SimpleNamespace(data=payload, offset=0, size=0),
        ],
        subgraphs=[
            SimpleNamespace(tensors=tensors, inputs=[], outputs=[], operators=[])
        ],
    )
    graph = CircleGraph(model, 0)
    existing_optimization_session(model)
    measured: Callable[[], Any]
    if args.mode == "pool":
        measured = functools.partial(ConstantPool, model, codec=codec)
    else:
        from tico.circle.passes.optimization.fold.constant_subgraph import (
            _required_input_payloads,
        )

        measured = functools.partial(_required_input_payloads, model, graph, (0,), (0,))

    gc.collect()
    tracemalloc.start()
    started = time.perf_counter()
    result = measured()
    elapsed = time.perf_counter() - started
    retained, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    report: dict[str, Any] = {
        "mode": args.mode,
        "payload_bytes": payload.nbytes,
        "tensor_aliases": args.aliases,
        "retained_allocated_bytes": retained,
        "peak_allocated_bytes": peak,
        "elapsed_seconds": elapsed,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scope": "incremental traced allocations; excludes input storage",
    }
    if args.mode == "pool":
        report["pool_statistics"] = result.statistics
    else:
        report["input_bytes"] = sum(len(value) for value in result.values())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
