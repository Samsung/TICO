# Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

import numpy as np


def pack_buffer(flat_data: np.ndarray, dtype: str) -> np.ndarray:
    assert flat_data.ndim == 1

    if dtype == "uint4":
        if flat_data.dtype != np.uint8:
            raise RuntimeError("uint4 data should be saved in uint8.")

        if flat_data.size == 0:
            return np.empty(0, dtype=np.uint8)

        if np.any(flat_data > 15):
            raise RuntimeError("uint4 data must be in [0, 15].")

        """
        NumPy vectorized operations are faster than Python-level loops:

          - flat_data[0::2] and flat_data[1::2] use strided views (no data copy)
          - Bitwise operations (<<, |=) are executed in optimized C
          
        As a result, packing runs in bulk over the entire array, which is significantly
        faster than iterating element-by-element in Python.
        """
        packed = np.empty((flat_data.size + 1) // 2, dtype=np.uint8)
        packed[:] = flat_data[0::2]
        # For odd-sized inputs, the last packed element has no corresponding
        # upper 4-bit value, so we restrict the operation to packed[: n//2]
        # to avoid shape mismatch.
        upper = (flat_data[1::2] << 4).astype(np.uint8, copy=False)
        packed[: flat_data.size // 2] |= upper
        return packed

    raise NotImplementedError(f"NYI dtype: {dtype}")
