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

import hashlib
from typing import Any

import numpy as np

# Equality may allocate a NumPy boolean array, but never one the size of a weight.
_COMPARE_CHUNK_BYTES = 1024 * 1024
PayloadFingerprint = tuple[int, bytes]


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
