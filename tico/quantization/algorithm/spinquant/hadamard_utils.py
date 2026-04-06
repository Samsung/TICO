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

import math
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from scipy.linalg import hadamard as scipy_hadamard
except ImportError:
    scipy_hadamard = None


# ----------------------------------------------------------------------
# Optional fast_hadamard_transform integration
# ----------------------------------------------------------------------
# - Interface shape and function selection are based on the public
#    fast_hadamard_transform package interface by Tri Dao.
# - The original package is BSD-3-Clause licensed.
#
# Original copyright notice from the referenced interface:
#   Copyright (c) 2023, Tri Dao.
try:
    from fast_hadamard_transform import (
        hadamard_transform as _fht_pow2,
        hadamard_transform_12N as _fht_12n,
        hadamard_transform_ref as _fht_ref,
    )

    _FAST_HADAMARD_AVAILABLE = True
except ImportError:
    _fht_pow2 = None
    _fht_12n = None
    _fht_ref = None
    _FAST_HADAMARD_AVAILABLE = False


# ----------------------------------------------------------------------
# Public special-order text registry
# ----------------------------------------------------------------------
# Fill these strings from a public Hadamard source such as:
#   http://www.neilsloane.com/hadamard/
#
# Each row must use:
#   '+' for +1
#   '-' for -1
#
SPECIAL_HADAMARD_TEXT: Dict[int, str] = {
    12: """
    +-----------
    ++-+---+++-+
    +++-+---+++-
    +-++-+---+++
    ++-++-+---++
    +++-++-+---+
    ++++-++-+---
    +-+++-++-+--
    +--+++-++-+-
    +---+++-++-+
    ++---+++-++-
    +-+---+++-++
    """
}


def is_pow2(n: int) -> bool:
    """
    Return True if `n` is a positive power of two.

    Parameters:
        n: Integer value.

    Returns:
        Whether `n` is a positive power of two.
    """
    return n > 0 and (n & (n - 1)) == 0


def _normalize_hadamard_rows(text: str) -> list[str]:
    """
    Normalize a textual Hadamard representation into clean row strings.

    Parameters:
        text: Multiline string containing '+' and '-' entries.

    Returns:
        A list of normalized row strings.
    """
    rows: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip().replace(" ", "")
        if not line or line.startswith("#"):
            continue
        rows.append(line)
    return rows


def _parse_sign_hadamard(text: str, order: int) -> torch.Tensor:
    """
    Parse a Hadamard matrix from a textual +/- representation.

    Parameters:
        text: Multiline Hadamard string.
        order: Expected matrix order.

    Returns:
        A float64 tensor of shape [order, order] with values in {+1, -1}.

    Raises:
        ValueError: If the text is malformed or not Hadamard-orthogonal.
    """
    rows = _normalize_hadamard_rows(text)
    if len(rows) != order:
        raise ValueError(f"Expected {order} Hadamard rows, but found {len(rows)} rows.")

    parsed: list[list[float]] = []
    for row_idx, row in enumerate(rows):
        if len(row) != order:
            raise ValueError(
                f"Hadamard row {row_idx} must have length {order}, "
                f"but got {len(row)}."
            )

        values: list[float] = []
        for ch in row:
            if ch == "+":
                values.append(1.0)
            elif ch == "-":
                values.append(-1.0)
            else:
                raise ValueError(
                    f"Unsupported Hadamard character {ch!r}. "
                    "Only '+' and '-' are allowed."
                )
        parsed.append(values)

    matrix = torch.tensor(parsed, dtype=torch.float64)
    gram = matrix @ matrix.T
    expected = order * torch.eye(order, dtype=torch.float64)

    if not torch.equal(gram, expected):
        raise ValueError(f"Parsed order-{order} matrix is not a valid Hadamard matrix.")

    return matrix


def _sylvester_hadamard(order: int) -> torch.Tensor:
    """
    Construct a Sylvester Hadamard matrix for a power-of-two order.

    Parameters:
        order: Target order.

    Returns:
        A float64 tensor of shape [order, order].

    Raises:
        ValueError: If `order` is not a power of two.
    """
    if not is_pow2(order):
        raise ValueError(
            f"Sylvester construction requires a power-of-two order, got {order}."
        )

    matrix = torch.tensor([[1.0]], dtype=torch.float64)
    while matrix.shape[0] < order:
        matrix = torch.cat(
            [
                torch.cat([matrix, matrix], dim=1),
                torch.cat([matrix, -matrix], dim=1),
            ],
            dim=0,
        )
    return matrix


def _special_hadamard(order: int) -> torch.Tensor:
    """
    Return a registered special-order Hadamard matrix.

    Parameters:
        order: Matrix order.

    Returns:
        A float64 tensor of shape [order, order].

    Raises:
        ValueError: If the order is not registered.
    """
    text = SPECIAL_HADAMARD_TEXT.get(order)
    if text is None or not text.strip():
        raise ValueError(
            f"No special Hadamard text has been registered for order {order}."
        )
    return _parse_sign_hadamard(text, order)


def _resolve_factor(size: int) -> int:
    """
    Resolve the supported non-power-of-two factor for `size`.

    This implementation intentionally supports the minimum subset needed for
    LLaMA-3.2-3B-Instruct:
        - power-of-two sizes
        - 12 * 2^k sizes

    Parameters:
        size: Matrix size.

    Returns:
        The factor value.

    Raises:
        ValueError: If `size` is unsupported.
    """
    if is_pow2(size):
        return 1

    if size % 12 == 0 and is_pow2(size // 12):
        return 12

    raise ValueError(
        "Unsupported Hadamard size. This implementation currently supports "
        f"power-of-two sizes and 12 * 2^k sizes only, but got size={size}."
    )


def _build_dense_hadamard(
    size: int,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Build a normalized dense Hadamard matrix.

    For 12 * 2^k sizes, the matrix is constructed as:
        kron(H12, H2k) / sqrt(size)

    Parameters:
        size: Target matrix size.
        device: Target device.
        dtype: Target dtype.

    Returns:
        A normalized orthogonal matrix of shape [size, size].
    """
    factor = _resolve_factor(size)

    if factor == 1:
        matrix = _sylvester_hadamard(size)
    else:
        special = _special_hadamard(factor)
        residual = size // factor
        power2 = _sylvester_hadamard(residual)
        matrix = torch.kron(special, power2)

    matrix = matrix / math.sqrt(size)
    return matrix.to(device=device, dtype=dtype)


def build_hadamard_matrix(
    size: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Return a normalized dense Hadamard matrix.

    Parameters:
        size: Matrix size.
        device: Target device.
        dtype: Target dtype.

    Returns:
        A tensor of shape [size, size].
    """
    return _build_dense_hadamard(size, device=device, dtype=dtype)


def _scipy_ref_hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """
    Reference Hadamard transform for power-of-two sizes using SciPy.

    Parameters:
        x: Input tensor with shape (..., dim).
        scale: Output scale.

    Returns:
        Tensor with the same shape as `x`.

    Raises:
        ImportError: If SciPy is not available.
    """
    if scipy_hadamard is None:
        raise ImportError(
            "SciPy is required for the Hadamard reference fallback, but it is not installed."
        )

    x_shape = x.shape
    dim = x.shape[-1]
    x2 = x.reshape(-1, dim)
    dim_padded = 1 << math.ceil(math.log2(dim))

    if dim_padded != dim:
        x2 = F.pad(x2, (0, dim_padded - dim))

    had = torch.tensor(
        scipy_hadamard(dim_padded, dtype=float),
        dtype=x.dtype,
        device=x.device,
    )
    out = F.linear(x2, had) * scale
    return out[..., :dim].reshape(*x_shape)


def _dense_transform(x: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    """
    Apply a dense transform on the last dimension.

    Parameters:
        x: Input tensor with shape (..., dim).
        transform: Transform matrix with shape [dim, dim].

    Returns:
        Tensor with the same shape as `x`.
    """
    original_shape = x.shape
    x2 = x.reshape(-1, x.shape[-1]).to(transform.dtype)
    out = x2 @ transform
    return out.reshape(original_shape).to(dtype=x.dtype)


def _fallback_hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """
    Fallback Hadamard transform for power-of-two sizes.

    This prefers the reference implementation when available and otherwise
    falls back to a dense Sylvester matrix.

    Parameters:
        x: Input tensor with shape (..., dim).
        scale: Output scale.

    Returns:
        Tensor with the same shape as `x`.
    """
    dim = x.shape[-1]

    if _fht_ref is not None:
        return _fht_ref(x, scale)

    if scipy_hadamard is not None:
        return _scipy_ref_hadamard_transform(x, scale)

    transform = build_hadamard_matrix(dim, device=x.device, dtype=torch.float64)
    transform = transform * math.sqrt(dim) * scale
    return _dense_transform(x, transform)


def _fallback_hadamard_transform_12n(
    x: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """
    Fallback Hadamard transform for dimensions of the form 12 * 2^k.

    Parameters:
        x: Input tensor with shape (..., dim).
        scale: Output scale.

    Returns:
        Tensor with the same shape as `x`.
    """
    dim = x.shape[-1]
    transform = build_hadamard_matrix(dim, device=x.device, dtype=torch.float64)
    transform = transform * math.sqrt(dim) * scale
    return _dense_transform(x, transform)


def _select_fast_transform(
    dim: int,
) -> Optional[Callable[[torch.Tensor, float], torch.Tensor]]:
    """
    Select a fast Hadamard transform function if available.

    Parameters:
        dim: Last-dimension size.

    Returns:
        Callable if supported by the installed package, otherwise None.
    """
    if not _FAST_HADAMARD_AVAILABLE:
        return None

    if is_pow2(dim):
        return _fht_pow2

    if dim % 12 == 0 and is_pow2(dim // 12):
        return _fht_12n

    return None


def _apply_hadamard_last_dim(x: torch.Tensor) -> torch.Tensor:
    """
    Apply a normalized Hadamard transform to the last dimension of `x`.

    This returns x @ H where H is orthogonal and normalized by 1 / sqrt(dim).

    Parameters:
        x: Input tensor with shape (..., dim).

    Returns:
        Tensor with the same shape as `x`.
    """
    dim = x.shape[-1]
    fast_fn = _select_fast_transform(dim)

    if fast_fn is not None:
        return fast_fn(x, scale=1.0 / math.sqrt(dim))  # type: ignore[call-arg]

    if is_pow2(dim):
        return _fallback_hadamard_transform(x, scale=1.0 / math.sqrt(dim))

    if dim % 12 == 0 and is_pow2(dim // 12):
        return _fallback_hadamard_transform_12n(x, scale=1.0 / math.sqrt(dim))

    raise ValueError(
        f"Unsupported Hadamard transform dimension: {dim}. "
        "Supported shapes are power-of-two and 12 * 2^k."
    )


def make_random_hadamard_rotation(
    size: int,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Create a randomized Hadamard-structured orthogonal matrix.

    The matrix is:
        D @ H

    where:
        - D is a random diagonal sign matrix
        - H is a normalized Hadamard matrix

    Parameters:
        size: Matrix size.
        device: Target device.
        dtype: Target dtype.

    Returns:
        Tensor of shape [size, size].
    """
    if device is None:
        device = torch.device("cpu")

    signs = torch.randint(0, 2, (size,), device=device, dtype=torch.int64)
    signs = signs.to(torch.float64).mul_(2.0).sub_(1.0)
    eye = torch.eye(size, device=device, dtype=torch.float64)
    signed_eye = signs.unsqueeze(1) * eye

    out = _apply_hadamard_last_dim(signed_eye)
    return out.to(dtype=dtype)


def _apply_blockwise_right_transform(
    x: torch.Tensor,
    block_transform: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """
    Apply a blockwise right multiplication on the last dimension.

    Parameters:
        x: Input tensor.
        block_transform: Transform matrix of shape [block_size, block_size].
        block_size: Block size on the last dimension.

    Returns:
        Transformed tensor.
    """
    if x.shape[-1] % block_size != 0:
        raise ValueError(
            f"Last dimension {x.shape[-1]} is not divisible by block_size {block_size}."
        )

    original_shape = x.shape
    reshaped = x.reshape(-1, original_shape[-1] // block_size, block_size)
    transformed = reshaped @ block_transform
    return transformed.reshape(original_shape)


@torch.no_grad()
def fuse_blockwise_transform_to_linear(
    module: nn.Linear,
    had_dim: int = -1,
    output: bool = False,
    R2: Optional[torch.Tensor] = None,
) -> None:
    """
    Fuse a blockwise transform into a linear weight.

    This function supports two modes:
      1) If `R2` is provided:
         - Apply `R2` as a blockwise transform (per-head rotation).
      2) Otherwise:
         - Construct and apply a Hadamard transform.

    Parameters:
        module: Target linear module.
        had_dim:
            -1 means apply over the full selected axis.
            Otherwise, apply blockwise with the given block size.
        output:
            False applies on the input-feature axis.
            True applies on the output-feature axis.
        R2:
            Optional custom transform for blockwise mode. When provided, it
            overrides the internally constructed Hadamard transform.

    Raises:
        TypeError: If `module` is not nn.Linear.
        ValueError: If dimensions are invalid.
    """
    if not isinstance(module, nn.Linear):
        raise TypeError(f"Expected nn.Linear, but got {type(module).__name__}.")

    weight = module.weight.data
    device = weight.device
    dtype = weight.dtype
    out_features, in_features = weight.shape

    target_dim = out_features if output else in_features
    block_size = target_dim if had_dim == -1 else had_dim

    if had_dim != -1 and not is_pow2(had_dim):
        raise ValueError(
            f"Hadamard block dimension must be a power of two, got {had_dim}."
        )

    if target_dim % block_size != 0:
        raise ValueError(
            f"Target dimension {target_dim} is not divisible by block_size {block_size}."
        )

    if R2 is not None:
        transform = R2.to(device=device, dtype=torch.float64)
        if transform.ndim != 2 or transform.shape != (block_size, block_size):
            raise ValueError(
                f"R2 must have shape ({block_size}, {block_size}), "
                f"but got {tuple(transform.shape)}."
            )
    else:
        if had_dim == -1:
            transform = build_hadamard_matrix(
                block_size,
                device=device,
                dtype=torch.float64,
            )
        else:
            transform = build_hadamard_matrix(
                block_size,
                device=device,
                dtype=torch.float64,
            )

    working = weight.to(torch.float64)

    if output:
        updated = _apply_blockwise_right_transform(
            working.T,
            transform,
            block_size,
        ).T
    else:
        updated = _apply_blockwise_right_transform(
            working,
            transform,
            block_size,
        )

    module.weight.data.copy_(updated.to(device=device, dtype=dtype))
