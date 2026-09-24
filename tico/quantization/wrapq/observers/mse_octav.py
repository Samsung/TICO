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
OCTAV-based MSE observer.

Uses the Newton-Raphson OCTAV algorithm (Sakr et al., ICML 2022,
arXiv:2206.06501) to find MSE-optimal clipping thresholds for each
calibration batch, then merges the thresholds across batches via
standard min/max-merge.

For symmetric quantization, the 1-D OCTAV recursion (Theorem 3.1 /
Corollary 3.2 in the paper) is used to find a single clipping scalar
``s*``.

For asymmetric quantization, a 2-D Newton-Raphson generalization
optimizes ``(x_min, x_max)`` jointly using the 2x2 Hessian of the
clipped-quantization MSE.  The same piecewise-linear indicator trick
from the paper makes all derivatives collapse to simple sufficient
statistics (counts and sums of in-range / out-of-range elements).

After merging, scale and zero-point are derived from the running
min/max by the standard affine formula in ``AffineObserverBase``.
"""

from __future__ import annotations

import torch

from tico.quantization.wrapq.observers.affine_base import AffineObserverBase
from tico.quantization.wrapq.utils.reduce_utils import channelwise_minmax


class MSEOCTAVObserver(AffineObserverBase):
    """
    OCTAV-optimal clipping observer with min/max-merge across batches.

    For each incoming batch:
      1. Run OCTAV (Newton-Raphson) to find MSE-optimal clipping thresholds
         - Symmetric: single scalar ``s*``  (1-D recursion)
         - Asymmetric: ``(x_min*, x_max*)``  (2-D recursion with 2x2 Hessian)
      2. Merge: ``min_val = min(min_val, x_min*)``,
                ``max_val = max(max_val, x_max*)``

    Scale and zero-point are derived from the running min/max by the
    standard affine formula in ``AffineObserverBase.compute_qparams``.

    Parameters
    ----------
    max_iters : int
        Maximum Newton-Raphson iterations.  Default 10 (paper's default).
    tol : float
        Convergence tolerance on the parameter update.  Default 1e-10.
    """

    def __init__(
        self,
        *,
        max_iters: int = 20,
        tol: float = 1e-10,
        max_merge: bool = True,
        num_bins: int = 255,
        **kwargs,
    ):
        self.max_iters = max_iters
        self.tol = tol
        self.max_merge = max_merge
        self.num_bins = num_bins
        super().__init__(**kwargs)

        # Two-stage (global) state — used when max_merge=False
        self._stage: int = 1
        self._hist_counts = None   # [B] or [C, B]
        self._hist_sums = None     # [B] or [C, B]
        self._hist_min = None      # scalar or [C]
        self._hist_max = None      # scalar or [C]
        self._bin_width = None     # scalar or [C]
        self._zero_count = None    # scalar or [C] (symmetric only)

    def reset(self) -> None:
        super().reset()
        self._stage = 1
        self._hist_counts = None
        self._hist_sums = None
        self._hist_min = None
        self._hist_max = None
        self._bin_width = None
        self._zero_count = None

    # ------------------------------------------------------------------
    # OCTAV core
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _octav_symmetric(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        1-D OCTAV for symmetric quantization.

        Returns (x_min, x_max) = (-s*, s*) where s* minimizes the
        clipped-quantization MSE.

        Input ``x`` is 2-D with shape ``[batch, N]`` where the reduction
        is over dim=-1 (the N dimension).  For per-tensor, batch=1;
        for per-channel, batch=C (number of channels).
        """
        qmax = self.dtype.qmax
        gamma = 1.0 / (12.0 * qmax * qmax)  # = 4^(-B) / 3 for signed B-bit

        abs_x = x.abs()
        # Exclude zeros (paper: zeros are representable, don't count them
        # toward discretization noise to avoid over-estimation for sparse tensors)
        nonzero = abs_x > 0

        # Initial guess: mean of |x| (paper's recommendation)
        s = (abs_x * nonzero).sum(dim=-1) / nonzero.sum(dim=-1).clamp(min=1)

        for _ in range(self.max_iters):
            if s.dim() > 0:
                inside = nonzero & (abs_x <= s.unsqueeze(-1))
            else:
                inside = nonzero & (abs_x <= s)
            outside = nonzero & (~inside)

            # Sufficient statistics
            num_in = inside.sum(dim=-1).float()  # N_in
            num_out = outside.sum(dim=-1).float()  # N_out
            sum_out = (abs_x * outside).sum(dim=-1).float()  # S_out

            denom = gamma * num_in + num_out
            denom = torch.clamp(denom, min=1e-30)

            s_new = sum_out / denom

            if torch.all((s_new - s).abs() < self.tol):
                s = s_new
                break
            s = s_new

        return -s, s

    @torch.no_grad()
    def _octav_asymmetric(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        2-D OCTAV for asymmetric quantization.

        Returns (x_min*, x_max*) minimizing the clipped-quantization MSE
        via 2-D Newton-Raphson with the 2x2 Hessian.

        Input ``x`` is 2-D with shape ``[batch, N]`` where the reduction
        is over dim=-1 (the N dimension).  For per-tensor, batch=1;
        for per-channel, batch=C (number of channels).

        Derivation
        -----------
        Objective (clipped-quantization MSE):

            f(a, b) = sum_{x_i < a} (x_i - a)^2
                    + sum_{x_i > b} (x_i - b)^2
                    + gamma * p * (b - a)^2

        where:
          - a      : low clipping threshold (x_min)
          - b      : high clipping threshold (x_max)
          - gamma  : 1 / (12 * (qmax - qmin)^2)  (quantization noise factor)
          - p      : count of in-range elements  (a <= x_i <= b)

        OCTAV trick: at each iteration, treat the indicator sets as fixed.
        Define:
          - L = {i : x_i < a}  (low set),   count = q,  sum = m_q
          - H = {i : x_i > b}  (high set),  count = r,  sum = m_r
          - M = {i : a <= x_i <= b}         (in-range), count = p

        With sets fixed, f(a, b) is quadratic, so the gradient is linear:

            df/da = 2*(q + gamma*p)*a - 2*gamma*p*b - 2*m_q
            df/db = -2*gamma*p*a + 2*(r + gamma*p)*b - 2*m_r

        Setting gradient = 0 gives the linear system  A * x = c:

            | q + gamma*p   -gamma*p  | | a |   | m_q |
            | -gamma*p    r + gamma*p | | b | = | m_r |

        Since f is quadratic, Newton-Raphson converges in one step:
            x_{n+1} = x_n - H^{-1} * grad(f)
        Because grad(f) = H * x - d (with constant H, d), this simplifies to:
            x_{n+1} = H^{-1} * d
        which is exactly the direct solve of grad(f) = 0 above.

        Solve by Cramer's rule:

            det = (q + gamma*p)*(r + gamma*p) - (gamma*p)^2
                = q*r + gamma*p*(q + r)

            a = | m_q        -gamma*p  | / det
                | m_r     r + gamma*p |

              = (m_q*(r + gamma*p) + gamma*p*m_r) / det
              = (gamma*p*(m_q + m_r) + m_q*r) / det

            b = | q + gamma*p   m_q | / det
                | -gamma*p      m_r |

              = ((q + gamma*p)*m_r + gamma*p*m_q) / det
              = (gamma*p*(m_q + m_r) + m_r*q) / det

        The loop iterates because the indicator sets L, H, M change when
        (a, b) change.  Each iteration:
          1. Recompute sets -> recompute p, q, r, m_q, m_r
          2. Solve the linear system exactly (one Newton step = direct solve)
          3. Repeat until (a, b) and the sets stabilize
        """
        qmin = self.dtype.qmin
        qmax = self.dtype.qmax
        gamma = 1.0 / (12.0 * (qmax - qmin) ** 2)

        # Initial guess: percentiles that ensure some data inside and outside
        if x.dim() > 0:
            a = x.quantile(0.05, dim=-1)
            b = x.quantile(0.95, dim=-1)
        else:
            a = x.quantile(0.05)
            b = x.quantile(0.95)

        for _ in range(self.max_iters):
            if a.dim() > 0:
                a_b = a.unsqueeze(-1)
                b_b = b.unsqueeze(-1)
            else:
                a_b, b_b = a, b

            mask_lo = x < a_b
            mask_hi = x > b_b
            mask_in = (~mask_lo) & (~mask_hi)

            # Sufficient statistics (see derivation in docstring above)
            p = mask_in.sum(dim=-1).float()  # N_in
            q = mask_lo.sum(dim=-1).float()  # N_lo
            r = mask_hi.sum(dim=-1).float()  # N_hi
            m_q = (x * mask_lo).sum(dim=-1).float()  # S_lo
            m_r = (x * mask_hi).sum(dim=-1).float()  # S_hi

            # det = q*r + gamma*p*(q + r)  (Cramer's rule denominator)
            denom = gamma * p * (q + r) + q * r
            denom = torch.clamp(denom, min=1e-30)

            # Cramer's rule:
            #   a = (gamma*p*(m_q + m_r) + m_q*r) / det
            #   b = (gamma*p*(m_q + m_r) + m_r*q) / det
            a_new = (gamma * p * (m_q + m_r) + m_q * r) / denom
            b_new = (gamma * p * (m_q + m_r) + m_r * q) / denom

            if torch.all((a_new - a).abs() < self.tol) and torch.all(
                (b_new - b).abs() < self.tol
            ):
                a, b = a_new, b_new
                break
            a, b = a_new, b_new

        return a, b

    # ------------------------------------------------------------------
    # Per-tensor / per-channel dispatch
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _compute_octav(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run OCTAV and return (x_min, x_max) as scalars (per-tensor)
        or vectors (per-channel).
        """
        is_symmetric = self.qscheme.is_symmetric()

        if self.channel_axis is None:
            # Per-tensor: flatten to 1-D
            x_flat = x.flatten().unsqueeze(0)  # [1, N]
            if is_symmetric:
                x_min, x_max = self._octav_symmetric(x_flat)
                return x_min.squeeze(0), x_max.squeeze(0)
            else:
                x_min, x_max = self._octav_asymmetric(x_flat)
                return x_min.squeeze(0), x_max.squeeze(0)
        else:
            # Per-channel: move channel axis to front, flatten rest
            ca = self.channel_axis % x.dim()
            x_perm = x.movedim(ca, 0)  # [C, ...]
            x_flat = x_perm.reshape(x_perm.shape[0], -1)  # [C, N]

            if is_symmetric:
                x_min, x_max = self._octav_symmetric(x_flat)
            else:
                x_min, x_max = self._octav_asymmetric(x_flat)
            return x_min, x_max

    # ------------------------------------------------------------------
    # Two-stage (global) strategy — used when max_merge=False
    # ------------------------------------------------------------------
    def prepare_stage2(self) -> None:
        if self.max_merge:
            self.enabled = False
            return

        is_symmetric = self.qscheme.is_symmetric()
        device = self.min_val.device

        if is_symmetric:
            max_abs = torch.maximum(self.max_val.abs(), self.min_val.abs()).clamp(min=1e-12)
            self._hist_min = torch.zeros_like(max_abs)
            self._hist_max = max_abs
        else:
            self._hist_min = self.min_val.clone()
            self._hist_max = self.max_val.clone()

        self._bin_width = ((self._hist_max - self._hist_min) / self.num_bins).clamp(min=1e-30)

        if self.channel_axis is None:
            self._hist_counts = torch.zeros(self.num_bins, device=device, dtype=torch.float32)
            self._hist_sums = torch.zeros(self.num_bins, device=device, dtype=torch.float32)
            self._zero_count = torch.tensor(0.0, device=device, dtype=torch.float32)
        else:
            C = self.min_val.shape[0]
            self._hist_counts = torch.zeros(C, self.num_bins, device=device, dtype=torch.float32)
            self._hist_sums = torch.zeros(C, self.num_bins, device=device, dtype=torch.float32)
            self._zero_count = torch.zeros(C, device=device, dtype=torch.float32)

        self._stage = 2

    @torch.no_grad()
    def _update_stats_global(self, x: torch.Tensor, **kwargs) -> None:
        if self.channel_axis is None:
            curr_min, curr_max = x.min(), x.max()
        else:
            curr_min, curr_max = channelwise_minmax(x, self.channel_axis)
        self.min_val = torch.minimum(self.min_val, curr_min)
        self.max_val = torch.maximum(self.max_val, curr_max)

        if self._stage == 1:
            return

        assert self._hist_counts is not None
        is_symmetric = self.qscheme.is_symmetric()
        num_bins = self.num_bins

        if self.channel_axis is None:
            x_flat = x.flatten().float()
            if is_symmetric:
                abs_x = x_flat.abs()
                nonzero = abs_x > 0
                abs_nz = abs_x[nonzero]
                if abs_nz.numel() > 0:
                    bin_idx = (abs_nz / self._bin_width).long().clamp(0, num_bins - 1)
                    self._hist_counts += torch.bincount(bin_idx, minlength=num_bins).float()
                    self._hist_sums.scatter_add_(0, bin_idx, abs_nz)
                self._zero_count += (~nonzero).sum().float()
            else:
                bin_idx = ((x_flat - self._hist_min) / self._bin_width).long().clamp(0, num_bins - 1)
                self._hist_counts += torch.bincount(bin_idx, minlength=num_bins).float()
                self._hist_sums.scatter_add_(0, bin_idx, x_flat)
        else:
            ca = self.channel_axis % x.dim()
            x_perm = x.movedim(ca, 0)
            x_flat = x_perm.reshape(x_perm.shape[0], -1).float()
            C = x_flat.shape[0]

            if is_symmetric:
                for c in range(C):
                    abs_c = x_flat[c].abs()
                    nz = abs_c > 0
                    abs_nz = abs_c[nz]
                    if abs_nz.numel() > 0:
                        bin_idx = (abs_nz / self._bin_width[c]).long().clamp(0, num_bins - 1)
                        self._hist_counts[c] += torch.bincount(bin_idx, minlength=num_bins).float()
                        self._hist_sums[c].scatter_add_(0, bin_idx, abs_nz)
                    self._zero_count[c] += (~nz).sum().float()
            else:
                for c in range(C):
                    x_c = x_flat[c]
                    bin_idx = ((x_c - self._hist_min[c]) / self._bin_width[c]).long().clamp(0, num_bins - 1)
                    self._hist_counts[c] += torch.bincount(bin_idx, minlength=num_bins).float()
                    self._hist_sums[c].scatter_add_(0, bin_idx, x_c)


    @torch.no_grad()
    def _octav_symmetric_hist(
        self, counts, sums, bin_width
    ) -> tuple[torch.Tensor, torch.Tensor]:
        qmax = self.dtype.qmax
        gamma = 1.0 / (12.0 * qmax * qmax)
        num_bins = self.num_bins

        cumcount = counts.cumsum(dim=-1)
        cumsum = sums.cumsum(dim=-1)
        total_count = cumcount[..., -1]
        total_sum = cumsum[..., -1]

        s = total_sum / total_count.clamp(min=1)

        if s.dim() == 0:
            for _ in range(self.max_iters):
                s_bin = (s / bin_width).long().clamp(0, num_bins - 2)
                num_in = cumcount[s_bin]
                sum_in = cumsum[s_bin]
                num_out = total_count - num_in
                sum_out = total_sum - sum_in
                denom = torch.clamp(gamma * num_in + num_out, min=1e-30)
                s_new = sum_out / denom
                if (s_new - s).abs() < self.tol:
                    s = s_new
                    break
                if s_new < s:
                    s = 0.5 * (s + s_new)
                else:
                    s = s_new
        else:
            active = torch.ones_like(s, dtype=torch.bool)
            for _ in range(self.max_iters):
                s_bin = (s / bin_width).long().clamp(0, num_bins - 2)
                num_in = cumcount.gather(-1, s_bin.unsqueeze(-1)).squeeze(-1)
                sum_in = cumsum.gather(-1, s_bin.unsqueeze(-1)).squeeze(-1)
                num_out = total_count - num_in
                sum_out = total_sum - sum_in
                denom = torch.clamp(gamma * num_in + num_out, min=1e-30)
                s_new = sum_out / denom
                converged = (s_new - s).abs() < self.tol
                damp = s_new < s
                s_new = torch.where(damp, 0.5 * (s + s_new), s_new)
                s = torch.where(active, s_new, s)
                active = active & ~converged
                if not active.any():
                    break

        return -s, s

    @torch.no_grad()
    def _octav_asymmetric_hist(
        self, counts, sums, hist_min, bin_width
    ) -> tuple[torch.Tensor, torch.Tensor]:
        qmin = self.dtype.qmin
        qmax = self.dtype.qmax
        gamma = 1.0 / (12.0 * (qmax - qmin) ** 2)
        num_bins = self.num_bins

        cumcount = counts.cumsum(dim=-1)
        cumsum = sums.cumsum(dim=-1)
        total_count = cumcount[..., -1]
        total_sum = cumsum[..., -1]

        a = self._hist_percentile(cumcount, total_count, 0.05, hist_min, bin_width)
        b = self._hist_percentile(cumcount, total_count, 0.95, hist_min, bin_width)

        if a.dim() == 0:
            for _ in range(self.max_iters):
                a_bin = ((a - hist_min) / bin_width).long().clamp(1, num_bins - 2)
                b_bin = ((b - hist_min) / bin_width).long().clamp(1, num_bins - 2)
                q = cumcount[a_bin]
                m_q = cumsum[a_bin]
                r = total_count - cumcount[b_bin]
                m_r = total_sum - cumsum[b_bin]
                p = total_count - q - r
                denom = torch.clamp(gamma * p * (q + r) + q * r, min=1e-30)
                a_new = (gamma * p * (m_q + m_r) + m_q * r) / denom
                b_new = (gamma * p * (m_q + m_r) + m_r * q) / denom
                if (a_new - a).abs() < self.tol and (b_new - b).abs() < self.tol:
                    a, b = a_new, b_new
                    break
                if b_new <= a_new:
                    break
                if a_new > a:
                    a_new = 0.5 * (a + a_new)
                if b_new < b:
                    b_new = 0.5 * (b + b_new)
                a, b = a_new, b_new
        else:
            active = torch.ones_like(a, dtype=torch.bool)
            for _ in range(self.max_iters):
                a_bin = ((a - hist_min) / bin_width).long().clamp(1, num_bins - 2)
                b_bin = ((b - hist_min) / bin_width).long().clamp(1, num_bins - 2)
                q = cumcount.gather(-1, a_bin.unsqueeze(-1)).squeeze(-1)
                m_q = cumsum.gather(-1, a_bin.unsqueeze(-1)).squeeze(-1)
                r = total_count - cumcount.gather(-1, b_bin.unsqueeze(-1)).squeeze(-1)
                m_r = total_sum - cumsum.gather(-1, b_bin.unsqueeze(-1)).squeeze(-1)
                p = total_count - q - r
                denom = torch.clamp(gamma * p * (q + r) + q * r, min=1e-30)
                a_new = (gamma * p * (m_q + m_r) + m_q * r) / denom
                b_new = (gamma * p * (m_q + m_r) + m_r * q) / denom
                converged = ((a_new - a).abs() < self.tol) & ((b_new - b).abs() < self.tol)
                collapse = b_new <= a_new
                damp_a = a_new > a
                damp_b = b_new < b
                a_new = torch.where(damp_a, 0.5 * (a + a_new), a_new)
                b_new = torch.where(damp_b, 0.5 * (b + b_new), b_new)
                a = torch.where(active & ~collapse, a_new, a)
                b = torch.where(active & ~collapse, b_new, b)
                active = active & ~converged & ~collapse
                if not active.any():
                    break

        return a, b

    @staticmethod
    def _hist_percentile(cumcount, total_count, pct, hist_min, bin_width):
        target = pct * total_count
        if cumcount.dim() == 1:
            mask = cumcount >= target
            idx = mask.int().argmax()
            if not mask.any():
                idx = torch.tensor(cumcount.shape[0] - 1, device=cumcount.device)
            return hist_min + (idx + 0.5) * bin_width
        else:
            mask = cumcount >= target.unsqueeze(-1)
            idx = mask.int().argmax(dim=-1)
            no_match = ~mask.any(dim=-1)
            idx = torch.where(no_match, torch.tensor(cumcount.shape[-1] - 1, device=cumcount.device, dtype=idx.dtype), idx)
            return hist_min + (idx + 0.5) * bin_width

    def compute_qparams(self):
        if not self.max_merge:
            assert self._hist_counts is not None, (
                "max_merge=False requires prepare_stage2() and a second pass"
            )
            is_symmetric = self.qscheme.is_symmetric()
            if is_symmetric:
                x_min, x_max = self._octav_symmetric_hist(
                    self._hist_counts, self._hist_sums, self._bin_width
                )
            else:
                x_min, x_max = self._octav_asymmetric_hist(
                    self._hist_counts, self._hist_sums, self._hist_min, self._bin_width
                )
            if torch.any(torch.isnan(x_min)) or torch.any(torch.isnan(x_max)):
                x_min, x_max = self.min_val, self.max_val
            if torch.any(x_min >= x_max):
                x_min = torch.minimum(x_min, self.min_val)
                x_max = torch.maximum(x_max, self.max_val)
            self.min_val = x_min
            self.max_val = x_max
            return super().compute_qparams()
        return super().compute_qparams()


    # ------------------------------------------------------------------
    # ObserverBase interface
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _update_stats(self, x: torch.Tensor, **kwargs) -> None:
        """
        Run OCTAV to find MSE-optimal clipping thresholds for this batch,
        then merge with running min/max.
        """
        if not self.max_merge:
            self._update_stats_global(x, **kwargs)
            return

        # Also update raw min/max for edge-case fallback
        if self.channel_axis is None:
            curr_min, curr_max = x.min(), x.max()
        else:
            curr_min, curr_max = channelwise_minmax(x, self.channel_axis)

        # Run OCTAV
        octav_min, octav_max = self._compute_octav(x)

        # Guard against degenerate OCTAV output (e.g. all-zero tensor)
        # by falling back to raw min/max
        if torch.any(torch.isnan(octav_min)) or torch.any(torch.isnan(octav_max)):
            octav_min, octav_max = curr_min, curr_max

        # Ensure min < max
        if torch.any(octav_min >= octav_max):
            octav_min = torch.minimum(octav_min, curr_min)
            octav_max = torch.maximum(octav_max, curr_max)

        # Merge: min-merge for min, max-merge for max
        self.min_val = torch.minimum(self.min_val, octav_min)
        self.max_val = torch.maximum(self.max_val, octav_max)
