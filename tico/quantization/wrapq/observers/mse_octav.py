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
        **kwargs,
    ):
        self.max_iters = max_iters
        self.tol = tol
        super().__init__(**kwargs)

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
    # ObserverBase interface
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _update_stats(self, x: torch.Tensor, **kwargs) -> None:
        """
        Run OCTAV to find MSE-optimal clipping thresholds for this batch,
        then merge with running min/max.
        """
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
