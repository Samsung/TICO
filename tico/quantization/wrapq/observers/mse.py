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

"""MSE grid-search observer for affine quantization."""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import torch

from tico.quantization.wrapq.dtypes import DType, UINT8
from tico.quantization.wrapq.observers.affine_base import AffineObserverBase
from tico.quantization.wrapq.qscheme import QScheme
from tico.quantization.wrapq.utils.reduce_utils import channelwise_minmax

# Guards the base-scale division for constant calibration batches. The final
# scale floor is applied later by ``AffineObserverBase._derive_grid``.
_BASE_SCALE_EPS = 1e-12


class MSEObserver(AffineObserverBase):
    """Affine observer that grid-searches the MSE-optimal quantization grid.

    For every calibration batch the observer searches affine quantization
    candidates and keeps the one that minimizes the fake-quantization error

        ``(x - Q(x; scale, zp))**2``

    summed over the batch (per-tensor) or independently per channel. Each
    batch-optimal candidate is converted to the value interval it represents,
    ``[scale * (qmin - zp), scale * (qmax - zp)]``, and merged into running
    clipping bounds by an element-wise union. Freezing derives the final grid
    from these bounds through ``AffineObserverBase``, so the final grid covers
    every batch-optimal grid and remains fully covered by the shared
    frozen-state validation.

    The scale search evaluates ``num_grid`` candidates ``alpha * base_scale``
    with ``alpha`` in ``(0, 1]``, where ``base_scale`` comes from the batch
    min/max. The zero-point of each scale candidate is derived from the batch
    minimum, so the ``alpha = 1`` candidate is exactly the MinMax grid and the
    searched grid is never worse than MinMax on the batch being searched.

    For asymmetric schemes with ``alternating_search=True``, the batch
    candidate is refined by alternating an integer zero-point search (a
    ``num_zp_grid``-wide grid around the derived zero-point) with a scale
    search at the fixed zero-point, repeated ``max_alternating_iters`` times.
    Symmetric schemes always keep ``zp = 0`` and skip the zero-point search.

    Error accumulation is chunked: at most ``search_chunk_size`` candidate-
    error elements are materialized at once, which bounds peak memory for
    large activation tensors without changing the result (exact sum
    reduction).

    Args:
        name: Observer role name reported in error messages and summaries.
        dtype: Integer quantization dtype defining the candidate grid range.
        qscheme: Affine quantization scheme (per-tensor or per-channel,
            asymmetric or symmetric).
        channel_axis: Channel axis for ``PER_CHANNEL_*`` schemes; ignored for
            per-tensor schemes.
        num_grid: Number of scale candidates in ``(0, base_scale]``.
        num_zp_grid: Number of integer zero-point candidates searched around
            the derived zero-point for asymmetric schemes.
        alternating_search: Refine asymmetric batch candidates by alternating
            scale/zero-point searches. Ignored for symmetric schemes.
        max_alternating_iters: Number of alternating refinement rounds per
            batch when ``alternating_search`` is enabled.
        search_chunk_size: Maximum number of candidate-error elements
            (rows x channels x candidates) materialized during one search
            chunk.

    Raises:
        ValueError: If ``num_grid < 2``, ``num_zp_grid < 3``,
            ``max_alternating_iters < 0`` or ``search_chunk_size < 1``.
    """

    clip_min_val: torch.Tensor
    clip_max_val: torch.Tensor

    def __init__(
        self,
        *,
        name: str,
        dtype: DType = UINT8,
        qscheme: QScheme = QScheme.PER_TENSOR_ASYMM,
        channel_axis: Optional[int] = None,
        num_grid: int = 80,
        num_zp_grid: int = 41,
        alternating_search: bool = True,
        max_alternating_iters: int = 4,
        search_chunk_size: int = 1 << 20,
    ) -> None:
        """Create an MSE grid-search observer with the given search policy."""
        if num_grid < 2:
            raise ValueError("num_grid must be at least 2.")
        if num_zp_grid < 3:
            raise ValueError("num_zp_grid must be at least 3.")
        if max_alternating_iters < 0:
            raise ValueError("max_alternating_iters must be non-negative.")
        if search_chunk_size < 1:
            raise ValueError("search_chunk_size must be positive.")

        self.num_grid = int(num_grid)
        self.num_zp_grid = int(num_zp_grid)
        self.alternating_search = bool(alternating_search)
        self.max_alternating_iters = int(max_alternating_iters)
        self.search_chunk_size = int(search_chunk_size)

        super().__init__(
            name=name, dtype=dtype, qscheme=qscheme, channel_axis=channel_axis
        )

        # Merged coverage of the batch-optimal grids. These tiny buffers are
        # persistent so an interrupted calibration can be saved and resumed.
        self.register_buffer("clip_min_val", torch.tensor(math.inf))
        self.register_buffer("clip_max_val", torch.tensor(-math.inf))
        self.reset()

    def reset(self) -> None:
        """Clear observed ranges, merged clipping bounds and cached qparams.

        Keeps the clipping buffers registered; resetting begins a fresh
        calibration cycle and deliberately clears any merged grid coverage.
        """
        super().reset()
        # The affine base calls reset() before this subclass registers its
        # buffers; the guard skips that first invocation.
        if hasattr(self, "clip_min_val"):
            self.clip_min_val = self.clip_min_val.new_full((), math.inf)
            self.clip_max_val = self.clip_max_val.new_full((), -math.inf)

    @torch.no_grad()
    def _update_stats(self, x: torch.Tensor) -> None:
        """Update ranges and merge the MSE-optimal grid of one batch.

        Args:
            x: Nonempty calibration tensor supplied by ``collect()``.
        """
        if self.channel_axis is None:
            curr_min, curr_max = x.min(), x.max()
        else:
            curr_min, curr_max = channelwise_minmax(x, self.channel_axis)
        self.min_val = torch.minimum(self.min_val, curr_min)
        self.max_val = torch.maximum(self.max_val, curr_max)

        scale, zp = self._search_batch_grid(x)
        lower, upper = self._grid_coverage(scale, zp)
        lower = lower.to(device=self.clip_min_val.device, dtype=self.clip_min_val.dtype)
        upper = upper.to(device=self.clip_max_val.device, dtype=self.clip_max_val.dtype)
        # Element-wise interval union; broadcasting grows scalar buffers to
        # the per-channel vector shape on the first batch.
        self.clip_min_val = torch.minimum(self.clip_min_val, lower)
        self.clip_max_val = torch.maximum(self.clip_max_val, upper)

    @torch.no_grad()
    def _search_batch_grid(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Search the MSE-optimal (scale, zp) candidate grid for one batch.

        Args:
            x: Nonempty calibration tensor.

        Returns:
            ``(scale, zp)`` as float32 per-channel vectors (length 1 for
            per-tensor schemes).
        """
        values = self._channelwise_flat(x)
        batch_min = values.amin(dim=0)
        batch_max = values.amax(dim=0)

        scale, _ = self._search_scale(values, batch_min, batch_max, fixed_zp=None)

        if self.qscheme.is_symmetric():
            return scale, torch.zeros_like(scale)
        if not self.alternating_search:
            return scale, self._derived_zero_point(batch_min, scale)

        for _ in range(self.max_alternating_iters):
            zp = self._search_zero_point(values, batch_min, scale)
            scale, _ = self._search_scale(values, batch_min, batch_max, fixed_zp=zp)
        zp = self._search_zero_point(values, batch_min, scale)
        return scale, zp

    def _channelwise_flat(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape an input to float32 ``[rows, channels]`` for search.

        Args:
            x: Nonempty calibration tensor.

        Returns:
            Two-dimensional float32 view with the channel axis last, or a
            single column for per-tensor schemes.
        """
        values = x.detach()
        if self.channel_axis is None:
            return values.reshape(-1, 1).to(torch.float32)
        channel_axis = self.channel_axis % values.ndim
        moved = values.movedim(channel_axis, -1)
        return moved.reshape(-1, values.shape[channel_axis]).to(torch.float32)

    def _base_scale(
        self,
        batch_min: torch.Tensor,
        batch_max: torch.Tensor,
    ) -> torch.Tensor:
        """Return the full-range scale ``[C]`` anchoring the alpha search.

        Args:
            batch_min: Per-channel batch minimum ``[C]``.
            batch_max: Per-channel batch maximum ``[C]``.

        Returns:
            Per-channel scale covering the full batch range.
        """
        qmin, qmax = self.dtype.qmin, self.dtype.qmax
        if self.qscheme.is_symmetric():
            max_abs = torch.maximum(batch_min.abs(), batch_max.abs())
            return max_abs.clamp(min=_BASE_SCALE_EPS) / qmax
        rng = (batch_max - batch_min).clamp(min=_BASE_SCALE_EPS)
        return rng / (qmax - qmin)

    def _derived_zero_point(
        self, batch_min: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        """Return the float zero-point implied by a scale and the batch min.

        Args:
            batch_min: Per-channel batch minimum ``[C]``.
            scale: Per-channel scale ``[C]`` or candidate scales ``[C, K]``.

        Returns:
            Float zero-point clamped to the integer grid range, broadcast to
            the scale shape.
        """
        qmin, qmax = self.dtype.qmin, self.dtype.qmax
        while batch_min.dim() < scale.dim():
            batch_min = batch_min.unsqueeze(-1)
        return torch.round(qmin - batch_min / scale).clamp(qmin, qmax)

    def _search_scale(
        self,
        values: torch.Tensor,
        batch_min: torch.Tensor,
        batch_max: torch.Tensor,
        fixed_zp: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Grid-search the best scale per channel over the alpha candidates.

        Args:
            values: Float32 ``[rows, C]`` batch view.
            batch_min: Per-channel batch minimum ``[C]``.
            batch_max: Per-channel batch maximum ``[C]``.
            fixed_zp: Per-channel float zero-point held constant during the
                search, or ``None`` to derive it from the batch minimum for
                each candidate scale.

        Returns:
            ``(scale, zp)`` of the best candidate as ``[C]`` float32 vectors.
        """
        base = self._base_scale(batch_min, batch_max)
        alphas = torch.linspace(
            1.0 / self.num_grid, 1.0, self.num_grid, device=values.device
        )
        scales = base.unsqueeze(1) * alphas.unsqueeze(0)  # [C, G]

        if self.qscheme.is_symmetric():
            zps = torch.zeros_like(scales)
        elif fixed_zp is not None:
            zps = (
                fixed_zp.to(values.device, torch.float32).unsqueeze(1).expand_as(scales)
            )
        else:
            zps = self._derived_zero_point(batch_min, scales)  # [C, G]

        best_idx = self._accumulated_error(values, scales, zps).argmin(dim=1)
        best = best_idx.unsqueeze(1)  # [C, 1]
        return scales.gather(1, best).squeeze(1), zps.gather(1, best).squeeze(1)

    def _search_zero_point(
        self,
        values: torch.Tensor,
        batch_min: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """Grid-search the best integer zero-point per channel at fixed scale.

        Args:
            values: Float32 ``[rows, C]`` batch view.
            batch_min: Per-channel batch minimum ``[C]``.
            scale: Per-channel fixed scale ``[C]``.

        Returns:
            Per-channel float zero-point (integer-valued) ``[C]``.
        """
        qmin, qmax = self.dtype.qmin, self.dtype.qmax
        half = self.num_zp_grid // 2
        init = self._derived_zero_point(batch_min, scale)  # [C]
        offsets = torch.arange(
            -half, half + 1, device=values.device, dtype=torch.float32
        )
        zps = (init.unsqueeze(1) + offsets.unsqueeze(0)).clamp(qmin, qmax)  # [C, Z]
        scales = scale.unsqueeze(1).expand_as(zps)

        best_idx = self._accumulated_error(values, scales, zps).argmin(dim=1)
        return zps.gather(1, best_idx.unsqueeze(1)).squeeze(1)

    def _accumulated_error(
        self,
        values: torch.Tensor,
        scales: torch.Tensor,
        zps: torch.Tensor,
    ) -> torch.Tensor:
        """Sum squared fake-quantization errors for each candidate column.

        Args:
            values: Float32 ``[rows, C]`` batch view.
            scales: Per-channel candidate scales ``[C, K]``.
            zps: Per-channel candidate float zero-points ``[C, K]``.

        Returns:
            Per-channel summed squared error ``[C, K]``.
        """
        qmin, qmax = self.dtype.qmin, self.dtype.qmax
        candidates = max(1, scales.shape[0] * scales.shape[1])
        chunk = max(1, self.search_chunk_size // candidates)
        error = torch.zeros_like(scales)
        for rows in values.split(chunk):
            rows = rows.unsqueeze(2)  # [n, C, 1]
            scales_k = scales.unsqueeze(0)  # [1, C, K]
            zps_k = zps.unsqueeze(0)  # [1, C, K]
            quantized = torch.round(rows / scales_k) + zps_k
            quantized = quantized.clamp(qmin, qmax)
            dequantized = (quantized - zps_k) * scales_k
            error += (rows - dequantized).square().sum(dim=0)
        return error

    def _grid_coverage(
        self, scale: torch.Tensor, zp: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the value interval represented by a candidate grid.

        Args:
            scale: Per-channel scale ``[C]``.
            zp: Per-channel float zero-point ``[C]``.

        Returns:
            ``(lower, upper)`` per-channel interval bounds covered by the
            grid. For symmetric schemes the interval is centered at zero so
            that re-deriving the grid reproduces the searched scale exactly.
        """
        qmin, qmax = self.dtype.qmin, self.dtype.qmax
        if self.qscheme.is_symmetric():
            upper = scale * qmax
            return -upper, upper
        lower = (qmin - zp) * scale
        upper = (qmax - zp) * scale
        return lower, upper

    def _statistics_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the merged coverage of the searched grids for freezing.

        Returns:
            ``(clip_min_val, clip_max_val)``; the affine base treats the
            initial ``(+inf, -inf)`` pair as missing calibration statistics.
        """
        return self.clip_min_val, self.clip_max_val

    def _policy(self) -> tuple:
        """Identify the search policy so incompatible state loading is rejected."""
        return super()._policy() + (
            self.num_grid,
            self.num_zp_grid,
            self.alternating_search,
            self.max_alternating_iters,
            self.search_chunk_size,
        )

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[int, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Accept persisted clipping buffers whose merged shape changed.

        Args:
            state_dict: Dictionary being loaded, keyed by the module prefix.
            prefix: Parameter-name prefix of this observer in the dictionary.
            local_metadata: Metadata supplied by ``nn.Module``.
            strict: Whether missing or unexpected keys must be reported.
            missing_keys: Accumulator for missing keys.
            unexpected_keys: Accumulator for unexpected keys.
            error_msgs: Accumulator for loading errors.
        """
        for name in ("clip_min_val", "clip_max_val"):
            key = prefix + name
            if key in state_dict:
                value = state_dict[key]
                setattr(
                    self,
                    name,
                    getattr(self, name).new_empty(value.shape, dtype=value.dtype),
                )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def extra_repr(self) -> str:
        """Return search settings for module summaries."""
        return (
            f"num_grid={self.num_grid}, num_zp_grid={self.num_zp_grid}, "
            f"alternating_search={self.alternating_search}, "
            f"max_alternating_iters={self.max_alternating_iters}, "
            f"search_chunk_size={self.search_chunk_size}"
        )
