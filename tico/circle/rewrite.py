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

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Iterable, Iterator

from tico.circle.errors import CircleRewriteError
from tico.circle.graph import as_indices, as_list, OPTIONAL_TENSOR_INDEX


@dataclass(frozen=True)
class RewriteStats:
    """Count objects removed or references updated by a rewrite."""

    removed_tensors: int = 0
    removed_buffers: int = 0
    removed_operator_codes: int = 0
    removed_subgraphs: int = 0
    removed_signatures: int = 0
    remapped_references: int = 0

    @property
    def modified(self) -> bool:
        """Return whether the rewrite changed the model."""

        return any(
            (
                self.removed_tensors,
                self.removed_buffers,
                self.removed_operator_codes,
                self.removed_subgraphs,
                self.removed_signatures,
                self.remapped_references,
            )
        )

    def __add__(self, other: RewriteStats) -> RewriteStats:
        """Combine statistics from consecutive rewrites."""

        return RewriteStats(
            removed_tensors=self.removed_tensors + other.removed_tensors,
            removed_buffers=self.removed_buffers + other.removed_buffers,
            removed_operator_codes=(
                self.removed_operator_codes + other.removed_operator_codes
            ),
            removed_subgraphs=self.removed_subgraphs + other.removed_subgraphs,
            removed_signatures=self.removed_signatures + other.removed_signatures,
            remapped_references=(self.remapped_references + other.remapped_references),
        )


def _valid_index(index: int, size: int, path: str) -> int:
    if index < 0 or index >= size:
        raise CircleRewriteError(
            f"{path} references index {index}, but the valid range is 0..{size - 1}."
        )
    return index


def _remap_index(
    index: int,
    mapping: dict[int, int],
    *,
    path: str,
    optional: bool = False,
) -> int:
    if optional and index == OPTIONAL_TENSOR_INDEX:
        return index
    try:
        return mapping[index]
    except KeyError as error:
        raise CircleRewriteError(
            f"{path} references removed or invalid index {index}."
        ) from error


def _remap_vector(
    value: Any,
    mapping: dict[int, int],
    *,
    path: str,
    optional: bool = False,
) -> tuple[list[int], int]:
    original = as_indices(value)
    remapped = [
        _remap_index(index, mapping, path=path, optional=optional) for index in original
    ]
    changes = sum(old != new for old, new in zip(original, remapped))
    return remapped, changes


def _signature_tensor_maps(signature: Any) -> Iterator[tuple[str, Any]]:
    for field_name in ("inputs", "outputs"):
        for tensor_map in as_list(getattr(signature, field_name, None)):
            yield field_name, tensor_map


def _metadata_buffer_indices(model: Any) -> Iterator[tuple[str, Any, str]]:
    metadata_buffer = getattr(model, "metadataBuffer", None)
    for index, _buffer_index in enumerate(as_indices(metadata_buffer)):
        yield f"model.metadataBuffer[{index}]", metadata_buffer, str(index)
    for index, metadata in enumerate(as_list(getattr(model, "metadata", None))):
        if hasattr(metadata, "buffer"):
            yield f"model.metadata[{index}].buffer", metadata, "buffer"


def _get_reference(container: Any, key: str) -> int:
    if key.isdigit():
        return int(container[int(key)])
    return int(getattr(container, key))


def _set_reference(container: Any, key: str, value: int) -> None:
    if key.isdigit():
        container[int(key)] = value
    else:
        setattr(container, key, value)


def compact_tensors(
    model: Any,
    *,
    subgraph_indices: Iterable[int] | None = None,
) -> RewriteStats:
    """Remove unused tensors and remap references in selected subgraphs."""

    subgraphs = as_list(model.subgraphs)
    selected = (
        None if subgraph_indices is None else {int(index) for index in subgraph_indices}
    )
    if selected is not None:
        for index in selected:
            _valid_index(index, len(subgraphs), "tensor compaction subgraph selection")

    total_removed = 0
    total_remapped = 0
    signatures = as_list(getattr(model, "signatureDefs", None))

    for subgraph_index, subgraph in enumerate(subgraphs):
        if selected is not None and subgraph_index not in selected:
            continue
        tensors = as_list(subgraph.tensors)
        tensor_count = len(tensors)
        used: set[int] = set()

        for field_name in ("inputs", "outputs"):
            for position, tensor_index in enumerate(
                as_indices(getattr(subgraph, field_name, None))
            ):
                used.add(
                    _valid_index(
                        tensor_index,
                        tensor_count,
                        f"subgraphs[{subgraph_index}].{field_name}[{position}]",
                    )
                )

        for operator_index, operator in enumerate(as_list(subgraph.operators)):
            for field_name, optional in (
                ("inputs", True),
                ("outputs", False),
                ("intermediates", True),
            ):
                for position, tensor_index in enumerate(
                    as_indices(getattr(operator, field_name, None))
                ):
                    if optional and tensor_index == OPTIONAL_TENSOR_INDEX:
                        continue
                    used.add(
                        _valid_index(
                            tensor_index,
                            tensor_count,
                            f"subgraphs[{subgraph_index}].operators[{operator_index}]"
                            f".{field_name}[{position}]",
                        )
                    )

        for signature_index, signature in enumerate(signatures):
            if int(getattr(signature, "subgraphIndex", -1)) != subgraph_index:
                continue
            for field_name, tensor_map in _signature_tensor_maps(signature):
                tensor_index = int(getattr(tensor_map, "tensorIndex", -1))
                used.add(
                    _valid_index(
                        tensor_index,
                        tensor_count,
                        f"signatureDefs[{signature_index}].{field_name}.tensorIndex",
                    )
                )

        kept = sorted(used)
        mapping = {old: new for new, old in enumerate(kept)}
        if len(kept) == tensor_count and all(
            old == new for old, new in mapping.items()
        ):
            continue

        subgraph.tensors = [tensors[index] for index in kept]
        total_removed += tensor_count - len(kept)

        for field_name in ("inputs", "outputs"):
            remapped, changes = _remap_vector(
                getattr(subgraph, field_name, None),
                mapping,
                path=f"subgraphs[{subgraph_index}].{field_name}",
            )
            setattr(subgraph, field_name, remapped)
            total_remapped += changes

        for operator_index, operator in enumerate(as_list(subgraph.operators)):
            for field_name, optional in (
                ("inputs", True),
                ("outputs", False),
                ("intermediates", True),
            ):
                value = getattr(operator, field_name, None)
                if value is None:
                    continue
                remapped, changes = _remap_vector(
                    value,
                    mapping,
                    path=(
                        f"subgraphs[{subgraph_index}].operators[{operator_index}]"
                        f".{field_name}"
                    ),
                    optional=optional,
                )
                setattr(operator, field_name, remapped)
                total_remapped += changes

        for signature_index, signature in enumerate(signatures):
            if int(getattr(signature, "subgraphIndex", -1)) != subgraph_index:
                continue
            for field_name, tensor_map in _signature_tensor_maps(signature):
                old = int(getattr(tensor_map, "tensorIndex", -1))
                new = _remap_index(
                    old,
                    mapping,
                    path=(f"signatureDefs[{signature_index}].{field_name}.tensorIndex"),
                )
                tensor_map.tensorIndex = new
                total_remapped += old != new

    return RewriteStats(
        removed_tensors=total_removed,
        remapped_references=total_remapped,
    )


def compact_operator_codes(model: Any) -> RewriteStats:
    """Remove unused operator codes and remap operator code indices."""

    operator_codes = as_list(model.operatorCodes)
    used: set[int] = set()
    for subgraph_index, subgraph in enumerate(as_list(model.subgraphs)):
        for operator_index, operator in enumerate(as_list(subgraph.operators)):
            opcode_index = int(getattr(operator, "opcodeIndex", -1))
            used.add(
                _valid_index(
                    opcode_index,
                    len(operator_codes),
                    f"subgraphs[{subgraph_index}].operators[{operator_index}]"
                    ".opcodeIndex",
                )
            )

    kept = sorted(used)
    mapping = {old: new for new, old in enumerate(kept)}
    if len(kept) == len(operator_codes) and all(
        old == new for old, new in mapping.items()
    ):
        return RewriteStats()

    model.operatorCodes = [operator_codes[index] for index in kept]
    remapped = 0
    for subgraph in as_list(model.subgraphs):
        for operator in as_list(subgraph.operators):
            old = int(operator.opcodeIndex)
            new = _remap_index(old, mapping, path="operator.opcodeIndex")
            operator.opcodeIndex = new
            remapped += old != new

    return RewriteStats(
        removed_operator_codes=len(operator_codes) - len(kept),
        remapped_references=remapped,
    )


def compact_buffers(model: Any) -> RewriteStats:
    """Remove unused buffers and remap tensor and metadata buffer indices."""

    buffers = as_list(model.buffers)
    if not buffers:
        raise CircleRewriteError("Circle models must contain buffer 0.")

    used: set[int] = {0}
    tensor_references: list[tuple[Any, int, str]] = []
    for subgraph_index, subgraph in enumerate(as_list(model.subgraphs)):
        for tensor_index, tensor in enumerate(as_list(subgraph.tensors)):
            buffer_index = int(getattr(tensor, "buffer", 0) or 0)
            _valid_index(
                buffer_index,
                len(buffers),
                f"subgraphs[{subgraph_index}].tensors[{tensor_index}].buffer",
            )
            used.add(buffer_index)
            tensor_references.append((tensor, buffer_index, "buffer"))

    metadata_references = list(_metadata_buffer_indices(model))
    for path, container, key in metadata_references:
        buffer_index = _get_reference(container, key)
        _valid_index(buffer_index, len(buffers), path)
        used.add(buffer_index)

    kept = sorted(used)
    mapping = {old: new for new, old in enumerate(kept)}
    if len(kept) == len(buffers) and all(old == new for old, new in mapping.items()):
        return RewriteStats()

    model.buffers = [buffers[index] for index in kept]
    remapped = 0
    for tensor, old, field_name in tensor_references:
        new = _remap_index(old, mapping, path="tensor.buffer")
        setattr(tensor, field_name, new)
        remapped += old != new
    for path, container, key in metadata_references:
        old = _get_reference(container, key)
        new = _remap_index(old, mapping, path=path)
        _set_reference(container, key, new)
        remapped += old != new

    return RewriteStats(
        removed_buffers=len(buffers) - len(kept),
        remapped_references=remapped,
    )


def iter_subgraph_references(
    model: Any,
    *,
    owner_indices: Iterable[int] | None = None,
) -> Iterator[tuple[str, Any, str, int]]:
    """Yield mutable fields that refer to subgraph indices."""

    allowed_owners = set(owner_indices) if owner_indices is not None else None
    for subgraph_index, subgraph in enumerate(as_list(model.subgraphs)):
        if allowed_owners is not None and subgraph_index not in allowed_owners:
            continue
        for operator_index, operator in enumerate(as_list(subgraph.operators)):
            for options_field in ("builtinOptions", "builtinOptions2"):
                options = getattr(operator, options_field, None)
                if options is None:
                    continue
                for field_name in dir(options):
                    if field_name.startswith("_"):
                        continue
                    normalized = field_name.replace("_", "").lower()
                    try:
                        value = getattr(options, field_name)
                    except Exception:
                        continue
                    if callable(value):
                        continue

                    field_path = (
                        f"subgraphs[{subgraph_index}].operators[{operator_index}]"
                        f".{options_field}.{field_name}"
                    )
                    if normalized.endswith("subgraphindices"):
                        for position, subgraph_reference in enumerate(
                            as_indices(value)
                        ):
                            yield (
                                f"{field_path}[{position}]",
                                value,
                                str(position),
                                subgraph_reference,
                            )
                        continue

                    if not (
                        normalized.endswith("subgraphindex") or normalized == "subgraph"
                    ):
                        continue
                    if not isinstance(value, Integral):
                        continue
                    yield field_path, options, field_name, int(value)


def keep_subgraphs(model: Any, indices: Iterable[int]) -> RewriteStats:
    """Keep selected subgraphs and remap signatures and control-flow references."""

    subgraphs = as_list(model.subgraphs)
    kept = tuple(dict.fromkeys(int(index) for index in indices))
    if not kept:
        raise CircleRewriteError("At least one subgraph must be kept.")
    for index in kept:
        _valid_index(index, len(subgraphs), "subgraph selection")
    mapping = {old: new for new, old in enumerate(kept)}

    references = list(iter_subgraph_references(model, owner_indices=kept))
    for path, _container, _key, old in references:
        if old not in mapping:
            raise CircleRewriteError(
                f"{path} references subgraph {old}, which is not part of "
                "the extraction."
            )

    remapped = 0
    for _path, container, key, old in references:
        new = mapping[old]
        _set_reference(container, key, new)
        remapped += old != new

    signatures = getattr(model, "signatureDefs", None)
    removed_signatures = 0
    if signatures is not None:
        retained_signatures = []
        for signature in signatures:
            old = int(getattr(signature, "subgraphIndex", -1))
            if old not in mapping:
                removed_signatures += 1
                continue
            new = mapping[old]
            signature.subgraphIndex = new
            remapped += old != new
            retained_signatures.append(signature)
        model.signatureDefs = retained_signatures

    model.subgraphs = [subgraphs[index] for index in kept]
    return RewriteStats(
        removed_subgraphs=len(subgraphs) - len(kept),
        removed_signatures=removed_signatures,
        remapped_references=remapped,
    )


def compact_model(
    model: Any,
    *,
    subgraph_indices: Iterable[int] | None = None,
) -> RewriteStats:
    """Compact selected tensor spaces and all model-global index spaces."""

    return (
        compact_tensors(model, subgraph_indices=subgraph_indices)
        + compact_operator_codes(model)
        + compact_buffers(model)
    )
