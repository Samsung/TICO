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

"""
MSE-based activation observer.

For each calibration batch:
  1. Search for the scale that minimizes MSE between x and fake_quant(x, scale)
  2. Merge the batch-optimal scale with the running scale by taking the maximum

For asymmetric quantization, the zero-point is also optimized via alternating
search: fix zp → search scale → fix scale → search zp → repeat.
The running zero-point is the mean of batch-optimal zero-points.
"""

import math
from typing import Optional

import torch

from tico.quantization.wrapq.observers.affine_base import AffineObserverBase
from tico.quantization.wrapq.utils.reduce_utils import channelwise_minmax


class MSEObserver(AffineObserverBase):
    """
    MSE-optimal scale observer with max-merge across batches.

    For each incoming batch:
      • Compute min/max (for zero-point derivation)
      • Search for the scale that minimizes MSE(x, fake_quant(x, scale))
      • Merge: running_scale = max(running_scale, batch_optimal_scale)

    For asymmetric quantization, an alternating search can optionally be used
    (``alternating_search=True``):
      1. Fix zp → search optimal scale (grid over alpha)
      2. Fix scale → search optimal zp (grid around current zp)
      3. Repeat for max_alternating_iters (default 2)

    When ``alternating_search=True``, the running zero-point is the mean of
    batch-optimal zero-points.  When False (default), only the scale is
    searched and the zero-point is derived from the running min/max.

    Parameters
    ----------
    num_grid : int
        Number of grid points for scale search.  Default 80.
    num_zp_grid : int
        Number of grid points for zero-point search (±half around current zp).
        Default 41 (±20).
    alternating_search : bool
        If True, use alternating scale/zp search for asymmetric quantization.
        Default False (only scale is searched; zp derived from min/max).
    max_alternating_iters : int
        Number of alternating scale/zp search iterations.  Default 2.
    max_merge : bool
        If True (default), merge scales by taking the element-wise maximum
        with the running scale.
    """

    def __init__(
        self,
        *,
        num_grid: int = 80,
        num_zp_grid: int = 41,
        alternating_search: bool = True,
        max_alternating_iters: int = 4,
        max_merge: bool = True,
        search_chunk_size: int = 1 << 20,
        **kwargs,
    ):
        self.num_grid = num_grid
        self.num_zp_grid = num_zp_grid
        self.alternating_search = alternating_search
        self.max_alternating_iters = max_alternating_iters
        self.max_merge = max_merge
        # Number of tensor elements (along the flattened N dim) processed per
        # chunk during the grid searches. Bounds peak memory: intermediates are
        # [chunk, K] instead of [N, K] (critical for large tensors, e.g.
        # KV-cache of 1B+ models). Results are identical (exact sum reduction).
        self.search_chunk_size = search_chunk_size
        super().__init__(**kwargs)

        # Running optimal scale (per-tensor scalar or per-channel vector)
        self.register_buffer(
            "_running_scale", torch.tensor(math.inf), persistent=False
        )
        self._running_scale.fill_(math.inf)

        # Running zero-point (for asymmetric). Mean of batch-optimal zps.
        self.register_buffer(
            "_running_zp", torch.tensor(0.0), persistent=False
        )
        self._running_zp.fill_(0.0)
        self._zp_count = 0

    def reset(self) -> None:
        super().reset()
        if hasattr(self, "_running_scale"):
            self._running_scale.fill_(math.inf)
        if hasattr(self, "_running_zp"):
            self._running_zp.fill_(0.0)
            self._zp_count = 0

    # ------------------------------------------------------------------
    # Core: search for MSE-optimal scale on a single batch
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _search_optimal_scale(self, x: torch.Tensor, fixed_zp: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Search for the scale that minimizes MSE(x, fake_quant(x, scale)).

        If fixed_zp is provided (asymmetric), the zero-point is held constant
        during the scale search. Otherwise zp is derived from min/max.

        Returns
        -------
        torch.Tensor
            Optimal scale (scalar for per-tensor, vector for per-channel).
        """
        qmin = self.dtype.qmin
        qmax = self.dtype.qmax
        is_symmetric = self.qscheme.is_symmetric()

        if self.channel_axis is None:
            # Per-tensor
            if is_symmetric:
                max_abs = x.abs().max().clamp(min=1e-12)
                base_scale = max_abs / qmax
            else:
                x_min = x.min()
                x_max = x.max()
                rng = (x_max - x_min).clamp(min=1e-12)
                base_scale = rng / (qmax - qmin)
        else:
            # Per-channel
            curr_min, curr_max = channelwise_minmax(x, self.channel_axis)
            if is_symmetric:
                max_abs = torch.maximum(curr_min.abs(), curr_max.abs()).clamp(min=1e-12)
                base_scale = max_abs / qmax
            else:
                rng = (curr_max - curr_min).clamp(min=1e-12)
                base_scale = rng / (qmax - qmin)

        # Vectorized grid search over alpha ∈ (0, 1]
        alphas = torch.linspace(
            1.0 / self.num_grid, 1.0, self.num_grid, device=x.device, dtype=x.dtype
        )
        if self.channel_axis is None:
            scales = alphas * base_scale  # [K]
        else:
            scales = alphas[:, None] * base_scale[None, :]  # [K, C]
        norm = 2.0
        if self.channel_axis is None:
            # Per-tensor: vectorized fake-quant for all alphas at once.
            # Process x in chunks to bound peak memory: intermediates are
            # [chunk, K] instead of [N, K]. The mean is computed exactly as
            # sum over chunks / N (identical result to the unchunked version).
            x_flat = x.flatten()  # [N]
            N = x_flat.numel()
            chunk = max(int(self.search_chunk_size), 1)

            if fixed_zp is not None:
                zp_val = float(fixed_zp)
            else:
                zp_val = None

            if fixed_zp is None and not is_symmetric:
                # Derive zp from x_min for each scale
                x_min = x.min()
                zps = torch.round(qmin - x_min / scales).clamp(qmin, qmax).to(torch.int)  # [K]

            err_sum = torch.zeros_like(scales)  # [K]
            for x_chunk in x_flat.split(chunk):
                x_div = x_chunk[:, None] / scales[None, :]  # [chunk, K]
                if is_symmetric:
                    x_round = torch.round(x_div)
                    x_clamped = x_round.clamp(-qmax, qmax)
                    x_q = x_clamped * scales[None, :]
                elif zp_val is not None:
                    x_round = torch.round(x_div) + zp_val
                    x_clamped = x_round.clamp(qmin, qmax)
                    x_q = (x_clamped - zp_val) * scales[None, :]
                else:
                    x_round = torch.round(x_div) + zps[None, :].float()
                    x_clamped = x_round.clamp(qmin, qmax)
                    x_q = (x_clamped - zps[None, :].float()) * scales[None, :]
                err_sum += (torch.abs(x_chunk[:, None] - x_q) ** norm).sum(dim=0)
            mse = err_sum / N  # [K]
            assert not torch.any(torch.isnan(mse))
            
            best_idx = mse.argmin()
            best_scale = scales[best_idx]
        else:
            # Per-channel: vectorized over (channels, alphas)
            # scales is already [K, C] from above
            ca = self.channel_axis % x.ndim
            x_perm = x.movedim(ca, -1)  # [..., C]
            x_flat = x_perm.reshape(-1, x_perm.shape[-1])  # [N, C]

            if is_symmetric:
                x_div = x_flat[:, :, None] / scales.T[None, :, :]  # [N, C, K]
                x_round = torch.round(x_div)
                x_clamped = x_round.clamp(-qmax, qmax)
                x_q = x_clamped * scales.T[None, :, :]  # [N, C, K]
                mse = (torch.abs((x_flat[:, :, None] - x_q)) ** norm).mean(dim=0)  # [C, K]
            else:
                if fixed_zp is not None:
                    # fixed_zp: [C]
                    zp_b = fixed_zp.to(x.device).float()  # [C]
                    x_div = x_flat[:, :, None] / scales.T[None, :, :]  # [N, C, K]
                    x_round = torch.round(x_div) + zp_b[None, :, None]  # [N, C, K]
                    x_clamped = x_round.clamp(qmin, qmax)
                    x_q = (x_clamped - zp_b[None, :, None]) * scales.T[None, :, :]
                    mse = (torch.abs((x_flat[:, :, None] - x_q)) ** norm).mean(dim=0)  # [C, K]
                else:
                    x_min_pc = x_flat.min(dim=0).values  # [C]
                    zps = torch.round(qmin - x_min_pc[None, :] / scales).clamp(qmin, qmax).to(torch.int)
                    x_div = x_flat[:, :, None] / scales.T[None, :, :]  # [N, C, K]
                    x_round = torch.round(x_div) + zps.T[None, :, :].float()  # [N, C, K]
                    x_clamped = x_round.clamp(qmin, qmax)
                    x_q = (x_clamped - zps.T[None, :, :].float()) * scales.T[None, :, :]
                    mse = (torch.abs((x_flat[:, :, None] - x_q)) ** norm).mean(dim=0)  # [C, K]

            
            assert not torch.any(torch.isnan(mse))
            
            best_idx = mse.argmin(dim=1)  # [C]
            best_scale = scales[best_idx, torch.arange(scales.shape[1], device=x.device)]  # [C]

        return best_scale

    @torch.no_grad()
    def _search_optimal_zp(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """
        Search for the zero-point that minimizes MSE, with scale fixed.

        Grid search over zp candidates around the current zp (derived from min/max).

        Returns
        -------
        torch.Tensor
            Optimal zero-point (scalar for per-tensor, vector for per-channel).
        """
        qmin = self.dtype.qmin
        qmax = self.dtype.qmax

        if self.channel_axis is None:
            # Per-tensor
            x_flat = x.flatten()  # [N]
            # Initial zp from min/max
            x_min = x.min()
            init_zp = int(torch.round(qmin - x_min / scale).clamp(qmin, qmax).item())

            # Candidate zps: ±half around init_zp
            half = self.num_zp_grid // 2
            lo = max(qmin, init_zp - half)
            hi = min(qmax, init_zp + half)
            zp_candidates = torch.arange(lo, hi + 1, device=x.device, dtype=torch.float32)  # [Z]

            # x_q for each zp: [chunk, Z] — chunked to bound peak memory
            N = x_flat.numel()
            chunk = max(int(self.search_chunk_size), 1)
            err_sum = torch.zeros_like(zp_candidates)  # [Z]
            for x_chunk in x_flat.split(chunk):
                x_div = x_chunk[:, None] / scale  # [chunk, 1]
                x_round = torch.round(x_div) + zp_candidates[None, :]  # [chunk, Z]
                x_clamped = x_round.clamp(qmin, qmax)
                x_q = (x_clamped - zp_candidates[None, :]) * scale  # [chunk, Z]
                err_sum += ((x_chunk[:, None] - x_q) ** 2).sum(dim=0)
            mse = err_sum / N  # [Z]

            best_idx = mse.argmin()
            best_zp = zp_candidates[best_idx]
            return best_zp
        else:
            # Per-channel
            ca = self.channel_axis % x.ndim
            x_perm = x.movedim(ca, -1)
            x_flat = x_perm.reshape(-1, x_perm.shape[-1])  # [N, C]
            C = x_flat.shape[1]

            # Per-channel init zp
            x_min_pc = x_flat.min(dim=0).values  # [C]
            scale_b = scale.to(x.device)
            init_zp_pc = torch.round(qmin - x_min_pc / scale_b).clamp(qmin, qmax)  # [C]

            # For simplicity, use same range for all channels
            half = self.num_zp_grid // 2
            # Build candidate offsets: [-half, ..., +half]
            offsets = torch.arange(-half, half + 1, device=x.device, dtype=torch.float32)  # [Z]
            # zp_candidates: [C, Z]
            zp_candidates = (init_zp_pc[:, None] + offsets[None, :]).clamp(qmin, qmax)  # [C, Z]

            # x_q for each (channel, zp): [N, C, Z]
            x_div = x_flat[:, :, None] / scale_b[None, :, None]  # [N, C, 1]
            x_round = torch.round(x_div) + zp_candidates[None, :, :]  # [N, C, Z]
            x_clamped = x_round.clamp(qmin, qmax)
            x_q = (x_clamped - zp_candidates[None, :, :]) * scale_b[None, :, None]  # [N, C, Z]
            mse = ((x_flat[:, :, None] - x_q) ** 2).mean(dim=0)  # [C, Z]

            best_idx = mse.argmin(dim=1)  # [C]
            best_zp = zp_candidates[torch.arange(C, device=x.device), best_idx]  # [C]
            return best_zp

    @torch.no_grad()
    def _search_optimal_scale_and_zp(self, x: torch.Tensor) -> tuple:
        """
        Alternating search for optimal (scale, zp) for asymmetric quantization.

        1. Search scale with zp derived from min/max
        2. Search zp with scale fixed
        3. Search scale with fixed zp
        4. Repeat for max_alternating_iters

        Returns
        -------
        (scale, zp) : tuple of torch.Tensor
        """
        # Step 1: initial scale search (zp derived from min/max)
        scale = self._search_optimal_scale(x, fixed_zp=None)

        for _ in range(self.max_alternating_iters):
            # Step 2: search zp with fixed scale
            zp = self._search_optimal_zp(x, scale)

            # Step 3: search scale with fixed zp
            scale = self._search_optimal_scale(x, fixed_zp=zp)

        # Final zp search with the final scale
        zp = self._search_optimal_zp(x, scale)

        return scale, zp

    def _fake_quant_with_scale(
        self, x: torch.Tensor, scale: torch.Tensor, is_symmetric: bool
    ) -> torch.Tensor:
        """Fake-quantize x with a given scale (and zp from min/max)."""
        qmin = self.dtype.qmin
        qmax = self.dtype.qmax

        if is_symmetric:
            zp = torch.zeros_like(scale, dtype=torch.int)
        else:
            if self.channel_axis is None:
                x_min = x.min()
                x_max = x.max()
            else:
                x_min, x_max = channelwise_minmax(x, self.channel_axis)
            rng = torch.where(0 < x_min, x_max, x_max - x_min)
            rng = torch.where(0 > x_max, -x_min, rng)
            zp = torch.round(qmin - x_min / scale).clamp(qmin, qmax).to(torch.int)

        if self.channel_axis is None:
            return torch.fake_quantize_per_tensor_affine(
                x.float(), scale=scale.float(), zero_point=zp,
                quant_min=qmin, quant_max=qmax,
            )
        else:
            # channel_axis may be negative (e.g. -1 for the last dim);
            # torch requires a non-negative axis.
            ca = self.channel_axis % x.dim()
            shape = [1] * x.dim()
            shape[ca] = -1
            scale_b = scale.float().reshape(shape)
            zp_b = zp.reshape(shape)
            return torch.fake_quantize_per_channel_affine(
                x.float(), scale=scale_b, zero_point=zp_b,
                axis=ca, quant_min=qmin, quant_max=qmax,
            )

    # ------------------------------------------------------------------
    # ObserverBase interface
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _update_stats(self, x: torch.Tensor, **kwargs) -> None:
        """
        Update running min/max and search for MSE-optimal scale (and zp for asymmetric).

        Scale is merged via max-merge across batches.
        Zero-point is merged via running mean across batches.
        """
        # Update min/max
        if self.channel_axis is None:
            curr_min, curr_max = x.min(), x.max()
        else:
            curr_min, curr_max = channelwise_minmax(x, self.channel_axis)

        self.min_val = torch.minimum(self.min_val, curr_min)
        self.max_val = torch.maximum(self.max_val, curr_max)

        is_symmetric = self.qscheme.is_symmetric()

        if is_symmetric:
            # Symmetric: only search scale, zp=0
            batch_scale = self._search_optimal_scale(x, fixed_zp=None)
            batch_zp = torch.tensor(0.0, device=x.device)
        elif self.alternating_search:
            # Asymmetric with alternating scale/zp search (opt-in)
            batch_scale, batch_zp = self._search_optimal_scale_and_zp(x)
        else:
            # Asymmetric default: only search scale; zp derived from min/max
            batch_scale = self._search_optimal_scale(x, fixed_zp=None)
            batch_zp = None  # zp derived from running min/max in compute_qparams


        # Merge scale with running scale (max-merge)
        if torch.isinf(self._running_scale).any():
            self._running_scale = batch_scale
        else:
            if self.max_merge:
                self._running_scale = torch.maximum(
                    self._running_scale, batch_scale
                )
            else:
                self._running_scale = batch_scale

        # Merge zp with running zp (running mean) — only when alternating search
        # produced batch-optimal zps
        if batch_zp is not None:
            self._zp_count += 1
            if self._zp_count == 1:
                self._running_zp = batch_zp.float()
            else:
                n = self._zp_count
                self._running_zp = self._running_zp + (batch_zp.float() - self._running_zp) / n

    def compute_qparams(self):
        """
        Compute quantization parameters from the MSE-optimal running scale and zp.

        For symmetric: scale from MSE search, zp=0.
        For asymmetric: scale from MSE search (max-merged), zp from running mean.
        """
        assert isinstance(self.min_val, torch.Tensor)
        assert isinstance(self.max_val, torch.Tensor)
        qmin, qmax = self.dtype.qmin, self.dtype.qmax

        scale = self._running_scale.float()
        eps = 1e-12

        if self.qscheme.is_symmetric():
            scale = torch.clamp(scale, min=eps)
            zp = torch.zeros_like(scale, dtype=torch.int)
            self._cached_scale, self._cached_zp = scale, zp
            return scale, zp

        # Asymmetric
        scale = torch.clamp(scale, min=eps)
        if self.alternating_search:
            # Use running zp (mean of batch-optimal zps)
            zp = torch.round(self._running_zp).clamp(qmin, qmax).to(torch.int)
        else:
            # Default: keep the MSE-searched scale (max-merged) and derive
            # zp from the running min/max
            min_val = self.min_val
            zp = (
                torch.round(qmin - min_val / scale).clamp(qmin, qmax).to(torch.int)
            )

        self._cached_scale, self._cached_zp = scale, zp
        return scale, zp
