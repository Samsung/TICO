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

from __future__ import annotations

from dataclasses import dataclass
from typing import Union


# ─────────────────────────────────────────────────────────────────────
#  Base class – every quantization dtype in wrapq inherits this
# ─────────────────────────────────────────────────────────────────────
class QuantDtype:
    """
    Common interface for all quantization dtype descriptors.

    Concrete subclasses:
      - DType   : integer affine dtypes (INT8, UINT4, …)
      - MXDtype : microscaling dtypes  (MXINT8, MXFP4, …)

    Subclasses must provide ``bits`` (int), ``signed`` (bool), and
    ``__str__`` — either as dataclass fields or as properties.
    """

    # Convenience helpers
    @property
    def is_mx(self) -> bool:
        """True if this is a microscaling (MX) dtype."""
        return isinstance(self, MXDtype)

    @property
    def is_affine_integer(self) -> bool:
        """True if this is a plain integer dtype (DType)."""
        return isinstance(self, DType)


# ─────────────────────────────────────────────────────────────────────
#  Integer affine dtype  (original DType, now extends QuantDtype)
# ─────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class DType(QuantDtype):
    """
    Self-contained integer dtypes for quantization.

    A DType is just an immutable value-object with two fields:
      - bits
      - signed

    Common presets (INT8, UINT4, ..) are provided as constants for convenience.
    """

    bits: int  # pylint: disable=used-before-assignment
    signed: bool = False  # False -> unsigned

    # -- Affine-specific properties -------------------------------------------

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


# ─────────────────────────────────────────────────────────────────────
#  Microscaling (MX) dtype
# ─────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class MXDtype(QuantDtype):
    """
    Immutable descriptor for OCP Microscaling (MX) element formats.

    An MX dtype groups *block_size* elements that share a single
    scale factor encoded with *scale_bits* bits.  Each element is
    stored in the format given by *elem_format* (e.g. ``"int8"``,
    ``"fp4"``).

    Parameters
    ----------
    elem_format : str
        Element encoding name.  Must match one of the keys recognised by
        ``tico.utils.mx.formats.ElemFormat`` (int8, fp4).
    block_size : int
        Number of elements that share one scale factor.
        OCP MX spec mandates 32.
    scale_bits : int
        Bit-width of the shared exponent / scale.
        OCP MX spec mandates 8.

    Derived properties (ebits, mbits, emax, max_norm, min_norm) are
    computed lazily from *elem_format* via
    ``tico.utils.mx.formats._get_format_params``.
    """

    elem_format: str
    block_size: int = 32
    scale_bits: int = 8

    # -- Lazy format parameter cache ------------------------------------------

    _format_params: tuple | None = None

    def _get_format_params(self) -> tuple:
        """Return (ebits, mbits, emax, max_norm, min_norm) for elem_format."""
        if self._format_params is None:
            # Import here to avoid circular imports at module level
            from tico.utils.mx.formats import _get_format_params as _gfp

            object.__setattr__(self, "_format_params", _gfp(self.elem_format))
        return self._format_params  # type: ignore[return-value]

    # -- QuantDtype interface --------------------------------------------------

    @property
    def bits(self) -> int:
        """Total bit-width of a single element (e.g. 8 for MXINT8, 4 for MXFP4)."""
        ebits, mbits, *_ = self._get_format_params()
        if ebits == 0:
            # Integer format: mbits includes the sign bit
            return mbits
        # Float format: 1 sign + ebits exponent + (mbits - 2) mantissa bits
        # fp4  -> 1+2+1 = 4,  fp6 -> 1+3+2 = 6 or 1+2+3 = 6,
        # fp8  -> 1+5+2 = 8 or 1+4+3 = 8,  fp16 -> 1+5+10 = 16
        return 1 + ebits + (mbits - 2)

    @property
    def signed(self) -> bool:
        """All MX element formats are signed by OCP spec."""
        return True

    # -- MX-specific properties -----------------------------------------------

    @property
    def ebits(self) -> int:
        """Exponent bits of the element format (0 for integer formats)."""
        return self._get_format_params()[0]

    @property
    def mbits(self) -> int:
        """Mantissa bits of the element format (includes sign and implicit bits)."""
        return self._get_format_params()[1]

    @property
    def emax(self) -> int:
        """Maximum normal exponent of the element format."""
        return self._get_format_params()[2]

    @property
    def max_norm(self) -> float:
        """Largest representable normal number in the element format."""
        return self._get_format_params()[3]

    @property
    def min_norm(self) -> float:
        """Smallest representable normal number in the element format."""
        return self._get_format_params()[4]

    @property
    def is_float(self) -> bool:
        """True if the element format is a floating-point encoding."""
        return self.ebits > 0

    @property
    def is_integer_elem(self) -> bool:
        """True if the element format is an integer encoding."""
        return self.ebits == 0

    @property
    def qmin(self) -> int:
        """
        Minimum representable integer value (only valid for integer MX formats).

        For integer MX formats the representation is sign-magnitude, so:
          qmin = -(2^(bits-1) - 1)   (no two's-complement asymmetry)
        """
        if not self.is_integer_elem:
            raise ValueError(
                f"qmin is not defined for floating-point MX format "
                f"{self.elem_format!r}. Use min_norm / max_norm instead."
            )
        return -(1 << (self.bits - 1)) + 1  # sign-magnitude: no -2^(n-1)

    @property
    def qmax(self) -> int:
        """
        Maximum representable integer value (only valid for integer MX formats).

        For integer MX formats the representation is sign-magnitude, so:
          qmax = 2^(bits-1) - 1
        """
        if not self.is_integer_elem:
            raise ValueError(
                f"qmax is not defined for floating-point MX format "
                f"{self.elem_format!r}. Use min_norm / max_norm instead."
            )
        return (1 << (self.bits - 1)) - 1

    def __str__(self) -> str:
        return f"mx{self.elem_format}"

    # ────────────────────────────────
    #  Factory helpers
    # ────────────────────────────────
    @staticmethod
    def int8() -> "MXDtype":
        """MXINT8: 8-bit signed integer elements, block_size=32."""
        return MXDtype("int8")

    @staticmethod
    def fp4() -> "MXDtype":
        """MXFP4(E2M1): 4-bit float with 2 exp / 1 mantissa bit, block_size=32."""
        return MXDtype("fp4")


# ---------------------------------------------------------------------
#  Convenient canned versions – integer affine
# ---------------------------------------------------------------------
UINT4 = DType.uint(4)
INT4 = DType.int(4)
INT8 = DType.int(8)
UINT8 = DType.uint(8)
INT16 = DType.int(16)

# ---------------------------------------------------------------------
#  Convenient canned versions – microscaling (OCP MX v1.0)
# ---------------------------------------------------------------------
MXINT8 = MXDtype.int8()
MXFP4 = MXDtype.fp4()

# ---------------------------------------------------------------------
#  Type alias for any quantization dtype
# ---------------------------------------------------------------------
AnyDtype = Union[DType, MXDtype]
