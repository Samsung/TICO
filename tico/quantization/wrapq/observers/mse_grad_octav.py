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
Gradient-weighted OCTAV observer.

This is a strict generalization of :class:`MSEOCTAVObserver`.  Instead of
minimising the *unweighted* clipped-quantization MSE, it minimises a
*gradient-weighted* MSE where each element ``x_i`` is weighted by
``g_i = |dL/dx_i|`` -- the magnitude of the task-loss gradient flowing back
through that element.

Motivation
----------
Standard OCTAV treats every element equally.  But in a quantized LLM, an
outlier activation that barely affects the task loss should be clipped
aggressively, while a small activation that carries important gradient
signal should be preserved.  By weighting the MSE with task-loss
gradients, the clipping thresholds are optimised for *task-relevant*
error rather than raw reconstruction error.

Mathematical formulation
------------------------
The gradient-weighted clipped-quantization MSE is:

    f(s) = sum_{|x_i| > s} g_i (|x_i| - s)^2          (symmetric)
    f(a,b) = sum_{x_i < a} g_i (x_i - a)^2
           + sum_{x_i > b} g_i (x_i - b)^2
           + gamma * G_p * (b - a)^2                    (asymmetric)

where ``g_i = |dL/dx_i|`` and ``G_p = sum_{in-range} g_i``.

Setting ``g_i = 1`` for all *i* recovers the standard (unweighted) OCTAV
objective exactly.

Sufficient statistics
~~~~~~~~~~~~~~~~~~~~~
All counts become gradient-mass sums and all element sums become
gradient-weighted sums:

================  ====================  ===================================
Statistic         Plain OCTAV           Gradient-weighted
================  ====================  ===================================
count in-range p  |M|                   G_p = sum_{i in M} g_i
count low q       |L|                   G_q = sum_{i in L} g_i
count high r       |H|                   G_r = sum_{i in H} g_i
sum low m_q        sum_{i in L} x_i      S_q = sum_{i in L} g_i x_i
sum high m_r       sum_{i in H} x_i      S_r = sum_{i in H} g_i x_i
sum outside (sym)  sum_{|x_i|>s} |x_i|   S_out = sum_{|x_i|>s} g_i |x_i|
================  ====================  ===================================

With these substitutions, the Cramer's rule formulas are *identical in
structure* to the plain OCTAV case -- only the sufficient statistics change.

Symmetric (1-D bisection)
~~~~~~~~~~~~~~~~~~~~~~~~~~

    s* = S_out / (gamma * G_in + G_out)

where ``G_in = sum_{|x_i|<=s} g_i``, ``G_out = sum_{|x_i|>s} g_i``,
``S_out = sum_{|x_i|>s} g_i |x_i|``.

Asymmetric (2-D Cramer's rule)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    det = G_q * G_r + gamma * G_p * (G_q + G_r)
    a* = (gamma * G_p * (S_q + S_r) + S_q * G_r) / det
    b* = (gamma * G_p * (S_q + S_r) + S_r * G_q) / det

Gradient delivery
-----------------
Gradients are delivered through the existing ``_fq -> collect -> _update_stats``
chain.  The calibration pipeline runs a gradient pre-pass (forward + backward
outside ``torch.no_grad()``) to capture ``|x.grad|`` per activation, then a
CALIB pass that passes the captured gradients as ``grad=`` kwarg.

A module-level registry (``_GRAD_REGISTRY``) maps ``id(observer)`` to their
captured gradient tensors.  When ``grad`` is ``None`` (e.g. no gradient
capture was run, or the observer is used in a unit test), the observer
silently falls back to plain (unweighted) OCTAV.

Limitations
-----------
- Decode-step calibration falls back to plain OCTAV (gradients are only
  computed for the prefill pass).
"""

from __future__ import annotations

import torch

from tico.quantization.wrapq.observers.mse_octav import MSEOCTAVObserver
from tico.quantization.wrapq.utils.reduce_utils import channelwise_minmax


# ---------------------------------------------------------------------------
# Module-level gradient registry
# ---------------------------------------------------------------------------
# Maps observer id -> gradient tensor (|dL/dx|, same shape as x).
# Populated by the gradient capture in the calibration pipeline.
# The observer's _update_stats looks up id(self) to retrieve the gradient.
#
# Keying by id(observer) instead of obs.name is essential because obs.name is
# only the *local* name (e.g. "value", "act_out") — in a multi-layer model
# dozens of observers share the same local name, so a name-keyed registry
# would suffer collisions and cross-talk.  id(observer) is unique per object.
_GRAD_REGISTRY: dict[tuple[int, tuple[int, ...]], torch.Tensor] = {}


def register_gradient(
    obs: "MSEGradOCTAVObserver", x: torch.Tensor, grad: torch.Tensor
) -> None:
    """Register a gradient tensor for an observer by its id() and tensor shape.

    A single observer may be called with multiple different tensor shapes in
    one forward pass (e.g. ``obs_value`` is hit per-head as ``[B, S, H]`` and
    stacked as ``[B, num_kv_heads, S, H]``).  Keying by ``(id, shape)`` ensures
    each call site retrieves its own matching gradient.
    """
    # Offload to CPU to free GPU memory between the gradient-capture pass
    # and the calibration pass.  Moved back to the activation's device in
    # _update_stats when the gradient is actually needed.
    _GRAD_REGISTRY[(id(obs), tuple(x.shape))] = grad.detach().cpu()


def clear_gradient_registry() -> None:
    """Clear all registered gradients. Called between calibration batches."""
    _GRAD_REGISTRY.clear()


def get_gradient(
    obs_id: int, shape: tuple[int, ...]
) -> torch.Tensor | None:
    """Retrieve a registered gradient, or None if not found."""
    return _GRAD_REGISTRY.get((obs_id, shape))


class MSEGradOCTAVObserver(MSEOCTAVObserver):
    """
    Gradient-weighted OCTAV observer.

    Identical to :class:`MSEOCTAVObserver` except that each element is
    weighted by ``g_i = |dL/dx_i|`` in the clipping MSE objective.

    When no gradient is available (``grad=None``), falls back to plain
    (unweighted) OCTAV -- i.e. behaves identically to the parent class.

    Parameters
    ----------
    max_iters : int
        Maximum Newton-Raphson iterations.  Default 20.
    tol : float
        Convergence tolerance on the parameter update.  Default 1e-10.
    max_merge : bool
        If True (default), use min/max-merge across batches.
        If False, use the two-stage global strategy with gradient-weighted
        histogram accumulation.
    num_bins : int
        Number of histogram bins for two-stage mode.  Default 255.
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
        super().__init__(
            max_iters=max_iters,
            tol=tol,
            max_merge=max_merge,
            num_bins=num_bins,
            **kwargs,
        )


    # ------------------------------------------------------------------
    # Gradient-weighted OCTAV core
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _octav_symmetric_grad(
        self, x: torch.Tensor, g: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        1-D gradient-weighted OCTAV for symmetric quantization.

        Returns (x_min, x_max) = (-s*, s*) where s* minimises the
        gradient-weighted clipped-quantization MSE.

        Input ``x`` is 2-D with shape ``[batch, N]`` and ``g`` has the
        same shape.  Reduction is over dim=-1.
        """
        qmax = self.dtype.qmax
        gamma = 1.0 / (12.0 * qmax * qmax)

        abs_x = x.abs()
        # Exclude zeros (zeros are representable, don't count them)
        nonzero = abs_x > 0

        # Gradient-weighted initial guess: weighted mean of |x|
        g_nz = g * nonzero
        s = (abs_x * g_nz).sum(dim=-1) / g_nz.sum(dim=-1).clamp(min=1e-30)

        for _ in range(self.max_iters):
            if s.dim() > 0:
                inside = nonzero & (abs_x <= s.unsqueeze(-1))
            else:
                inside = nonzero & (abs_x <= s)
            outside = nonzero & (~inside)

            # Gradient-weighted sufficient statistics
            g_in = (g * inside).sum(dim=-1)        # G_in
            g_out = (g * outside).sum(dim=-1)      # G_out
            sum_out = (abs_x * g * outside).sum(dim=-1)  # S_out

            denom = gamma * g_in + g_out
            denom = torch.clamp(denom, min=1e-30)

            s_new = sum_out / denom

            if torch.all((s_new - s).abs() < self.tol):
                s = s_new
                break
            s = s_new

        return -s, s


    @torch.no_grad()
    def _octav_asymmetric_grad(
        self, x: torch.Tensor, g: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        2-D gradient-weighted OCTAV for asymmetric quantization.

        Returns (x_min*, x_max*) minimising the gradient-weighted
        clipped-quantization MSE via 2-D Cramer's rule.

        Input ``x`` is 2-D with shape ``[batch, N]`` and ``g`` has the
        same shape.  Reduction is over dim=-1.

        Derivation
        -----------
        Objective (gradient-weighted clipped-quantization MSE):

            f(a, b) = sum_{x_i < a} g_i (x_i - a)^2
                    + sum_{x_i > b} g_i (x_i - b)^2
                    + gamma * G_p * (b - a)^2

        where:
          - a      : low clipping threshold (x_min)
          - b      : high clipping threshold (x_max)
          - gamma  : 1 / (12 * (qmax - qmin)^2)  (quantization noise factor)
          - G_p    : gradient-mass of in-range elements  (sum_{in-range} g_i)
          - g_i    : |dL/dx_i|  (per-element gradient weight)

        With indicator sets fixed, f(a, b) is quadratic.  Setting the
        gradient to zero gives the linear system:

            | G_q + gamma*G_p   -gamma*G_p  | | a |   | S_q |
            | -gamma*G_p    G_r + gamma*G_p | | b | = | S_r |

        where:
          - G_q = sum_{low} g_i,    S_q = sum_{low} g_i x_i
          - G_r = sum_{high} g_i,   S_r = sum_{high} g_i x_i

        Solve by Cramer's rule:

            det = G_q*G_r + gamma*G_p*(G_q + G_r)

            a* = (gamma*G_p*(S_q + S_r) + S_q*G_r) / det
            b* = (gamma*G_p*(S_q + S_r) + S_r*G_q) / det

        The loop iterates because the indicator sets change when (a, b)
        change.  Each iteration recomputes sets and solves exactly.
        """
        qmin = self.dtype.qmin
        qmax = self.dtype.qmax
        gamma = 1.0 / (12.0 * (qmax - qmin) ** 2)

        # Initial guess: percentiles
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

            # Gradient-weighted sufficient statistics
            G_p = (g * mask_in).sum(dim=-1)       # gradient-mass in-range
            G_q = (g * mask_lo).sum(dim=-1)       # gradient-mass low
            G_r = (g * mask_hi).sum(dim=-1)       # gradient-mass high
            S_q = (x * g * mask_lo).sum(dim=-1)   # gradient-weighted sum low
            S_r = (x * g * mask_hi).sum(dim=-1)   # gradient-weighted sum high
            
          #  G_p = (mask_in).sum(dim=-1)       # gradient-mass in-range
          #  G_q = (mask_lo).sum(dim=-1)       # gradient-mass low
          #  G_r = (mask_hi).sum(dim=-1)       # gradient-mass high
          #  S_q = (x * mask_lo).sum(dim=-1)   # gradient-weighted sum low
          #  S_r = (x * mask_hi).sum(dim=-1)   # gradient-weighted sum high
                        
            # det = G_q*G_r + gamma*G_p*(G_q + G_r)
            denom = gamma * G_p * (G_q + G_r) + G_q * G_r
            denom = torch.clamp(denom, min=1e-30)

            # Cramer's rule:
            a_new = (gamma * G_p * (S_q + S_r) + S_q * G_r) / denom
            b_new = (gamma * G_p * (S_q + S_r) + S_r * G_q) / denom

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
    def _compute_octav_grad(
        self, x: torch.Tensor, g: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run gradient-weighted OCTAV and return (x_min, x_max) as scalars
        (per-tensor) or vectors (per-channel).
        """
        is_symmetric = self.qscheme.is_symmetric()

        if self.channel_axis is None:
            # Per-tensor: flatten to 1-D
            x_flat = x.flatten().unsqueeze(0)  # [1, N]
            g_flat = g.flatten().unsqueeze(0)  # [1, N]
            if is_symmetric:
                x_min, x_max = self._octav_symmetric_grad(x_flat, g_flat)
                return x_min.squeeze(0), x_max.squeeze(0)
            else:
                x_min, x_max = self._octav_asymmetric_grad(x_flat, g_flat)
                return x_min.squeeze(0), x_max.squeeze(0)
        else:
            # Per-channel: move channel axis to front, flatten rest
            ca = self.channel_axis % x.dim()
            x_perm = x.movedim(ca, 0)  # [C, ...]
            x_flat = x_perm.reshape(x_perm.shape[0], -1)  # [C, N]
            g_perm = g.movedim(ca, 0)
            g_flat = g_perm.reshape(g_perm.shape[0], -1)  # [C, N]

            if is_symmetric:
                x_min, x_max = self._octav_symmetric_grad(x_flat, g_flat)
            else:
                x_min, x_max = self._octav_asymmetric_grad(x_flat, g_flat)
            return x_min, x_max

    # ------------------------------------------------------------------
    # Two-stage gradient-weighted histogram accumulation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _update_stats_global_grad(
        self, x: torch.Tensor, grad: torch.Tensor
    ) -> None:
        """
        Accumulate gradient-weighted histogram statistics on the fixed grid
        (stage 2 of two-stage mode).

        Mirrors :meth:`MSEOCTAVObserver._update_stats_global` but replaces
        plain element counts with gradient-mass sums (``g_i = |dL/dx_i|``)
        and plain value sums with gradient-weighted sums.  The parent's
        ``_octav_*_hist`` solvers are formula-identical for both plain and
        gradient-weighted statistics, so no solver override is needed.
        """
        # Update raw min/max for edge-case fallback in compute_qparams
        if self.channel_axis is None:
            curr_min, curr_max = x.min(), x.max()
        else:
            curr_min, curr_max = channelwise_minmax(x, self.channel_axis)
        self.min_val = torch.minimum(self.min_val, curr_min)
        self.max_val = torch.maximum(self.max_val, curr_max)

        assert self._hist_counts is not None
        is_symmetric = self.qscheme.is_symmetric()
        num_bins = self.num_bins

        if self.channel_axis is None:
            x_flat = x.flatten().float()
            g_flat = grad.flatten().float()
            if is_symmetric:
                abs_x = x_flat.abs()
                nonzero = abs_x > 0
                abs_nz = abs_x[nonzero]
                g_nz = g_flat[nonzero]
                if abs_nz.numel() > 0:
                    bin_idx = (abs_nz / self._bin_width).long().clamp(0, num_bins - 1)
                    self._hist_counts.scatter_add_(0, bin_idx, g_nz)
                    self._hist_sums.scatter_add_(0, bin_idx, abs_nz * g_nz)
                self._zero_count += g_flat[~nonzero].sum()
            else:
                bin_idx = ((x_flat - self._hist_min) / self._bin_width).long().clamp(0, num_bins - 1)
                self._hist_counts.scatter_add_(0, bin_idx, g_flat)
                self._hist_sums.scatter_add_(0, bin_idx, x_flat * g_flat)
        else:
            ca = self.channel_axis % x.dim()
            x_perm = x.movedim(ca, 0)
            x_flat = x_perm.reshape(x_perm.shape[0], -1).float()
            g_perm = grad.movedim(ca, 0)
            g_flat = g_perm.reshape(g_perm.shape[0], -1).float()
            C = x_flat.shape[0]

            if is_symmetric:
                for c in range(C):
                    abs_c = x_flat[c].abs()
                    nz = abs_c > 0
                    abs_nz = abs_c[nz]
                    g_nz = g_flat[c][nz]
                    if abs_nz.numel() > 0:
                        bin_idx = (abs_nz / self._bin_width[c]).long().clamp(0, num_bins - 1)
                        self._hist_counts[c].scatter_add_(0, bin_idx, g_nz)
                        self._hist_sums[c].scatter_add_(0, bin_idx, abs_nz * g_nz)
                    self._zero_count[c] += g_flat[c][~nz].sum()
            else:
                for c in range(C):
                    x_c = x_flat[c]
                    g_c = g_flat[c]
                    bin_idx = ((x_c - self._hist_min[c]) / self._bin_width[c]).long().clamp(0, num_bins - 1)
                    self._hist_counts[c].scatter_add_(0, bin_idx, g_c)
                    self._hist_sums[c].scatter_add_(0, bin_idx, x_c * g_c)

    # ------------------------------------------------------------------
    # ObserverBase interface
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _update_stats(
        self, x: torch.Tensor, grad: torch.Tensor | None = None, **kwargs
    ) -> None:
        """
        Run gradient-weighted OCTAV when ``grad`` is provided, otherwise
        fall back to plain (unweighted) OCTAV.
        """
        # If no gradient provided, try the module-level registry as fallback
        if grad is None:
            grad = get_gradient(id(self), tuple(x.shape))
            if grad is None:
                if self.max_merge is False and self._stage < 2:
                    super()._update_stats(x, **kwargs)
                return

            # Move gradient from CPU back to the activation's device
            grad = grad.to(x.device)

        # Two-stage: accumulate gradient-weighted histogram on the fixed grid.
        # The parent's compute_qparams uses _octav_*_hist on _hist_counts /
        # _hist_sums — the OCTAV formula is identical whether the statistics
        # are plain element counts or gradient-weighted masses, so we can
        # reuse the parent's solver by simply accumulating gradient-weighted
        # counts/sums instead of plain ones.
        if not self.max_merge:
            if self._stage >= 2:
                self._update_stats_global_grad(x, grad)
            else:
                super()._update_stats(x, **kwargs)
            return

        # Also update raw min/max for edge-case fallback
        if self.channel_axis is None:
            curr_min, curr_max = x.min(), x.max()
        else:
            curr_min, curr_max = channelwise_minmax(x, self.channel_axis)

        # Run gradient-weighted OCTAV
        octav_min, octav_max = self._compute_octav_grad(x, grad)

        # Guard against degenerate output
        if torch.any(torch.isnan(octav_min)) or torch.any(torch.isnan(octav_max)):
            octav_min, octav_max = curr_min, curr_max

        # Ensure min < max
        if torch.any(octav_min >= octav_max):
            octav_min = torch.minimum(octav_min, curr_min)
            octav_max = torch.maximum(octav_max, curr_max)

        # Merge: min-merge for min, max-merge for max
        self.min_val = torch.minimum(self.min_val, octav_min)
        self.max_val = torch.maximum(self.max_val, octav_max)

