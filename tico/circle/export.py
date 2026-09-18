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

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from tico.circle._schema import decode_text
from tico.circle.analysis import TensorContract
from tico.circle.document import CircleDocument
from tico.circle.errors import CircleRewriteError
from tico.circle.graph import as_indices, as_list
from tico.circle.passes import create_o1_pipeline
from tico.circle.passes.base import CirclePassContext


def _interface_snapshot(document: CircleDocument) -> tuple[Any, ...]:
    """Capture ordered public contracts and aliases, independently of indices.

    Compaction may renumber tensors. A port's name, metadata and alias relation
    to other ports must not change. Signature tensor references are expressed as
    port positions rather than subgraph-local tensor indices for the same reason.
    Constant buffers are deliberately excluded: folding may materialize outputs.
    """

    subgraphs = as_list(document.model.subgraphs)
    interfaces = []
    for subgraph in subgraphs:
        tensors = as_list(subgraph.tensors)
        inputs = as_indices(subgraph.inputs)
        outputs = as_indices(subgraph.outputs)
        aliases: dict[int, int] = {}
        ports = []
        for indices in (inputs, outputs):
            ports.append(
                tuple(
                    (
                        decode_text(getattr(tensors[index], "name", None)),
                        TensorContract.from_tensor(tensors[index]),
                        aliases.setdefault(index, len(aliases)),
                    )
                    for index in indices
                )
            )
        interfaces.append((decode_text(getattr(subgraph, "name", None)), *ports))

    signatures = []
    for signature in as_list(getattr(document.model, "signatureDefs", None)):
        subgraph_index = int(signature.subgraphIndex)
        subgraph = subgraphs[subgraph_index]
        mappings = []
        for field_name in ("inputs", "outputs"):
            boundary = as_indices(getattr(subgraph, field_name, None))
            mappings.append(
                tuple(
                    (
                        decode_text(getattr(tensor_map, "name", None)),
                        tuple(
                            position
                            for position, index in enumerate(boundary)
                            if index == int(tensor_map.tensorIndex)
                        ),
                    )
                    for tensor_map in as_list(getattr(signature, field_name, None))
                )
            )
        signatures.append(
            (
                decode_text(getattr(signature, "signatureKey", None)),
                subgraph_index,
                *mappings,
            )
        )
    return tuple(interfaces), tuple(signatures)


def _operator_count(document: CircleDocument) -> int:
    """Count operators across all subgraphs without building graph indexes."""

    return sum(
        len(as_list(getattr(subgraph, "operators", None)))
        for subgraph in as_list(document.model.subgraphs)
    )


def optimize_for_export(
    circle_binary: bytes,
    *,
    logger: logging.Logger | None = None,
    dump_directory: str | os.PathLike[str] | None = None,
) -> bytes:
    """Run the default Circle O1 pipeline while preserving exported I/O contracts.

    This is post-serialization optimization, not an FX pass or an NPU-specific
    legalization stage. Numerical reassociation follows the existing O1 policy;
    structural verification does not establish bit-exactness or backend support.
    All failures propagate. Callers must explicitly disable optimization to obtain
    an unoptimized artifact; there is no silent fallback to the input bytes.

    When requested, a unique debug directory holds the original and optimized
    artifacts and per-pass results, including repeated fixed-point invocations.
    The original remains available if a pass fails. No whole-model clone is made.
    """

    logger = logger or logging.getLogger(__name__)
    start = time.perf_counter()
    dump_path = None
    if dump_directory is not None:
        parent = Path(dump_directory)
        parent.mkdir(parents=True, exist_ok=True)
        dump_path = Path(tempfile.mkdtemp(prefix="tico_circle_o1_", dir=parent))
        (dump_path / "before.circle").write_bytes(circle_binary)
        logger.info("Circle O1 debug artifacts: %s", dump_path)

    document = CircleDocument.from_bytes(circle_binary)
    document.verify()
    interface = _interface_snapshot(document)
    operators_before = _operator_count(document)
    context = CirclePassContext(logger=logger, preserve_io=True)
    result = create_o1_pipeline().run(document, context)
    document.verify()
    if _interface_snapshot(document) != interface:
        raise CircleRewriteError(
            "Circle O1 changed an exported I/O or signature contract. "
            "Disable circle_optimize to inspect the unoptimized artifact."
        )
    optimized = document.to_bytes() if result.modified else circle_binary
    elapsed = time.perf_counter() - start
    operators_after = _operator_count(document)
    logger.info(
        "Circle O1: %d -> %d operators; %d -> %d bytes; %d changes in %.3fs.",
        operators_before,
        operators_after,
        len(circle_binary),
        len(optimized),
        result.changes,
        elapsed,
    )
    if dump_path is not None:
        (dump_path / "after.circle").write_bytes(optimized)
        report = {
            "preset": "o1",
            "preserve_io": True,
            "modified": result.modified,
            "changes": result.changes,
            "elapsed_seconds": elapsed,
            "operators_before": operators_before,
            "operators_after": operators_after,
            "bytes_before": len(circle_binary),
            "bytes_after": len(optimized),
            "phases": [
                {
                    "name": phase.name,
                    "executions": [
                        {
                            "pass_name": execution.pass_name,
                            "modified": execution.result.modified,
                            "changes": execution.result.changes,
                            "diagnostics": list(execution.result.diagnostics),
                        }
                        for execution in phase.result.executions
                    ],
                }
                for phase in result.phases
            ],
        }
        (dump_path / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    return optimized
