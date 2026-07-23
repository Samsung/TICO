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

import re
from collections.abc import Iterable, Sequence

from tico.circle.errors import CircleSelectionError
from tico.circle.graph import CircleGraph


def parse_operator_spec(specification: str) -> tuple[int, ...]:
    """Parse comma-separated operator indices and inclusive ranges."""

    if not specification or not specification.strip():
        raise CircleSelectionError("Operator selection is empty.")

    selected: set[int] = set()
    for raw_part in specification.split(","):
        part = raw_part.strip()
        if not part:
            continue
        separator = "-" if "-" in part else ":" if ":" in part else None
        if separator is None:
            try:
                index = int(part)
            except ValueError as error:
                raise CircleSelectionError(
                    f"Invalid operator index {part!r}."
                ) from error
            if index < 0:
                raise CircleSelectionError("Operator indices must be non-negative.")
            selected.add(index)
            continue

        pieces = part.split(separator)
        if len(pieces) != 2 or not all(piece.strip() for piece in pieces):
            raise CircleSelectionError(f"Invalid operator range {part!r}.")
        try:
            start, end = (int(piece.strip()) for piece in pieces)
        except ValueError as error:
            raise CircleSelectionError(f"Invalid operator range {part!r}.") from error
        if start < 0 or end < 0:
            raise CircleSelectionError("Operator indices must be non-negative.")
        if start > end:
            raise CircleSelectionError(
                f"Operator range start {start} is greater than end {end}."
            )
        selected.update(range(start, end + 1))

    if not selected:
        raise CircleSelectionError("Operator selection is empty.")
    return tuple(sorted(selected))


def resolve_tensor_patterns(
    graph: CircleGraph,
    patterns: Sequence[str],
    *,
    full_match: bool = False,
) -> tuple[int, ...]:
    """Resolve regular expression patterns to tensor indices."""

    if not patterns:
        return ()

    compiled: list[re.Pattern[str]] = []
    for raw_pattern in patterns:
        try:
            compiled.append(re.compile(raw_pattern))
        except re.error as error:
            raise CircleSelectionError(
                f"Invalid tensor regular expression {raw_pattern!r}: {error}."
            ) from error

    result: list[int] = []
    unmatched = set(range(len(compiled)))
    for tensor_index, tensor_name in graph.iter_tensor_names():
        for pattern_index, compiled_pattern in enumerate(compiled):
            matched = (
                compiled_pattern.fullmatch(tensor_name)
                if full_match
                else compiled_pattern.search(tensor_name)
            )
            if matched:
                unmatched.discard(pattern_index)
                if tensor_index not in result:
                    result.append(tensor_index)

    if unmatched:
        missing = [patterns[index] for index in sorted(unmatched)]
        raise CircleSelectionError(
            f"Tensor patterns did not match any tensors: {missing}."
        )
    return tuple(result)


def select_operators_by_tensor_boundaries(
    graph: CircleGraph,
    *,
    from_tensors: Iterable[int] = (),
    to_tensors: Iterable[int] = (),
) -> tuple[int, ...]:
    """Select operators on directed paths between tensor boundaries."""

    starts = tuple(dict.fromkeys(int(index) for index in from_tensors))
    ends = tuple(dict.fromkeys(int(index) for index in to_tensors))
    if not starts and not ends:
        raise CircleSelectionError(
            "At least one input or output tensor boundary must be provided."
        )

    invalid = sorted(
        index for index in (*starts, *ends) if index < 0 or index >= graph.tensor_count
    )
    if invalid:
        raise CircleSelectionError(
            f"Tensor indices {invalid} are outside subgraph {graph.subgraph_index}."
        )

    forward = graph.forward_operators(starts) if starts else None
    backward = graph.backward_operators(ends) if ends else None
    if forward is not None and backward is not None:
        selected = forward & backward
    elif forward is not None:
        selected = forward
    else:
        selected = backward or set()

    if not selected:
        raise CircleSelectionError(
            "No operators lie on the requested tensor boundary paths."
        )
    return tuple(sorted(selected))
