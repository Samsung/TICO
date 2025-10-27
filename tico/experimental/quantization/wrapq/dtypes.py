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

from dataclasses import dataclass


@dataclass(frozen=True)
class DType:
    """
    Self-contained integer dtypes for quantization.

    A DType is just an immutable value-object with two fields:
      - bits
      - signed

    Common presets (INT8, UINT4, ..) are provided as constants for convenience.
    """

    bits: int  # pylint: disable=used-before-assignment
    signed: bool = False  # False -> unsigned

    @property
    def qmin(self) -> int:
        assert self.bits is not None
        if self.signed:
            return -(1 << (self.bits - 1))
        return 0

    @property
    def qmax(self) -> int:
        assert self.bits is not None
        if self.signed:
            return (1 << (self.bits - 1)) - 1
        return (1 << self.bits) - 1

    def __str__(self) -> str:
        prefix = "int" if self.signed else "uint"
        return f"{prefix}{self.bits}"

    # ────────────────────────────────
    #  Factory helpers
    # ────────────────────────────────
    @staticmethod
    def int(bits: int):  # type: ignore[valid-type]
        return DType(bits, signed=True)

    @staticmethod
    def uint(bits: int):  # type: ignore[valid-type]
        return DType(bits, signed=False)


# ---------------------------------------------------------------------
#  Convenient canned versions
# ---------------------------------------------------------------------
UINT4 = DType.uint(4)
INT4 = DType.int(4)
INT8 = DType.int(8)
UINT8 = DType.uint(8)
INT16 = DType.int(16)
