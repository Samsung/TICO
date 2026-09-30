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

"""Bounded-memory helpers for already encoded Circle byte vectors."""

from __future__ import annotations

import copy
import hashlib
from typing import Any, Mapping

import numpy as np

# Equality may allocate a NumPy boolean array, but never one the size of a weight.
_COMPARE_CHUNK_BYTES = 1024 * 1024
PayloadFingerprint = tuple[int, bytes]
# Original payload storage objects keyed by identity while a clone borrows them.
BorrowedPayloads = dict[int, Any]


def clone_model_borrowing_payloads(model: Any) -> tuple[Any, BorrowedPayloads]:
    """Deep-copy every table except ``Buffer.data`` storage, which stays shared.

    The buffer table entries themselves are copied, so the clone can drop, reorder,
    or replace buffers without touching the source. Only the payload storage objects
    are shared, and they are returned by identity so a caller can later detach the
    ones that survive. Callers must not modify shared payloads in place while the
    clone borrows them.
    """

    borrowed: BorrowedPayloads = {}
    memo: dict[int, Any] = {}
    for buffer in getattr(model, "buffers", None) or ():
        data = getattr(buffer, "data", None)
        if data is None:
            continue
        borrowed[id(data)] = data
        # Pre-seeding deepcopy's memo maps the storage object onto itself.
        memo[id(data)] = data
    clone = copy.deepcopy(model, memo)
    return clone, borrowed


def owned_payload_copy(data: Any) -> Any:
    """Return payload storage that shares nothing with ``data`` or its base."""

    if data is None:
        return None
    if isinstance(data, np.ndarray):
        # ``copy()`` drops a view's base, so a small retained constant no longer
        # pins a multi-gigabyte source binary or file mapping.
        return data.copy()
    if isinstance(data, bytes):
        # Immutable storage is already independent of any other object.
        return data
    if isinstance(data, memoryview):
        return bytes(data)
    if isinstance(data, bytearray):
        return bytearray(data)
    return copy.deepcopy(data)


def detach_borrowed_payloads(model: Any, borrowed: Mapping[int, Any]) -> int:
    """Replace surviving borrowed payloads with owned copies, once per storage.

    Buffers that share one storage object receive the same copy, so tensors that
    alias a buffer never duplicate its bytes. Returns the number of storage objects
    copied.
    """

    copies: dict[int, Any] = {}
    for buffer in getattr(model, "buffers", None) or ():
        data = getattr(buffer, "data", None)
        if data is None or borrowed.get(id(data)) is not data:
            continue
        replacement = copies.get(id(data))
        if replacement is None:
            replacement = owned_payload_copy(data)
            copies[id(data)] = replacement
        buffer.data = replacement
    return len(copies)


def inline_payload_view(buffer: Any) -> memoryview | None:
    """Borrow contiguous bytes; reject absent or unresolved external storage.

    Normal generated uint8 arrays and resolved external buffers are not copied.
    Legacy sequences, dtype conversions and non-contiguous storage may require a
    normalization copy. The returned view is temporary, not a dictionary key.
    """

    if int(getattr(buffer, "offset", 0) or 0) or int(getattr(buffer, "size", 0) or 0):
        return None
    data = getattr(buffer, "data", None)
    if data is None:
        return None
    if isinstance(data, (bytes, bytearray, memoryview)):
        view = memoryview(data)
        if not view.c_contiguous:
            view = memoryview(view.tobytes())
        return view.cast("B")
    try:
        array = np.asarray(data, dtype=np.uint8)
    except (TypeError, ValueError):
        return None
    return memoryview(np.ascontiguousarray(array).reshape(-1))


def payload_fingerprint(payload: memoryview) -> PayloadFingerprint:
    """Hash a borrowed byte vector without constructing a payload-sized bytes."""

    return payload.nbytes, hashlib.sha256(payload).digest()


def payloads_equal(left: memoryview, right: memoryview) -> bool:
    """Compare exact bytes with bounded scratch space, including hash collisions."""

    if left.nbytes != right.nbytes:
        return False
    for start in range(0, left.nbytes, _COMPARE_CHUNK_BYTES):
        end = start + _COMPARE_CHUNK_BYTES
        lhs = np.frombuffer(left[start:end], dtype=np.uint8)
        rhs = np.frombuffer(right[start:end], dtype=np.uint8)
        if not np.array_equal(lhs, rhs):
            return False
    return True
