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

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch


def compute_max_abs_diff(base: torch.Tensor, target: torch.Tensor) -> float:
    """
    Return the *maximum* absolute element-wise difference between two tensors.
    """
    assert base.shape == target.shape, "shape mismatch"
    return (base.detach() - target.detach()).abs().max().item()


def compute_peir(base: torch.Tensor, target: torch.Tensor) -> float:
    """
    Peak-Error-to-Interval Ratio (PEIR).

        PEIR = max(|base - target|) / (max(base) - min(base))

    The interval denominator uses the reference (*base*) tensor only — this
    makes PEIR independent of quantisation error in `target`.
    """
    assert base.shape == target.shape, "shape mismatch"
    peak_error = (base.detach() - target.detach()).abs().max().item()
    interval = (base.detach().max() - base.detach().min()).item()
    interval = 1.0 if interval == 0.0 else interval  # avoid divide-by-zero
    return peak_error / interval


def mae(base: torch.Tensor, target: torch.Tensor) -> float:
    """
    Mean Absolute Error (MAE).

    Measures the **average magnitude** of element-wise differences, treating
    all errors equally (no squaring).
    Good for a quick, scale-dependent sense of “overall drift”.

    Formula
    -------
        MAE = mean(|base - target|)

    Returns
    -------
    float
        Mean absolute error. *Lower is better*.
    """
    return (base.detach() - target.detach()).abs().mean().item()


def mse(base: torch.Tensor, target: torch.Tensor) -> float:
    """
    Mean Squared Error (MSE).

    Penalizes **larger** deviations more heavily than MAE by squaring each
    difference — helpful to expose occasional large spikes.

    Formula
    -------
        MSE = mean((base - target)²)

    Returns
    -------
    float
        Mean squared error. *Lower is better*.
    """
    return torch.mean((base.detach() - target.detach()) ** 2).item()


def rmse(base: torch.Tensor, target: torch.Tensor) -> float:
    """
    Root Mean Squared Error (RMSE).

    Square-root of MSE, restoring the units of the original data while
    retaining the strong penalty on outliers.

    Formula
    -------
        RMSE = sqrt( MSE )

    Returns
    -------
    float
        Root mean squared error. *Lower is better*.
    """
    return mse(base, target) ** 0.5


def rel_l2(base: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Relative L2 Error (scale-invariant).

    Normalizes the L2-norm of the error by the L2-norm of the reference,
    making results comparable across layers with very different magnitudes.

    Formula
    -------
        rel_L2 = ||base - target||₂ / (||base||₂ + ε)

    Returns
    -------
    float
        Relative L2 error in the range [0, ∞). *Lower is better*.
    """
    num = torch.norm(base.detach() - target.detach()).item()
    den = torch.norm(base.detach()).item() + eps
    return num / den


def cosine_sim(base: torch.Tensor, target: torch.Tensor) -> float:
    """
    Cosine Similarity.

    Captures the **directional alignment** between two tensors, ignoring
    their magnitudes — particularly insightful for attention key/query
    vectors or embedding spaces.

    Formula
    -------
        cos_sim = (base · target) / (||base||₂ · ||target||₂)

    Returns
    -------
    float
        Cosine similarity in [-1, 1]; *closer to 1 is better*.
    """
    return torch.nn.functional.cosine_similarity(
        base.flatten(), target.flatten(), dim=0
    ).item()


def sqnr_db(base: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Signal-to-Quantisation-Noise Ratio (SQNR) in decibels.

    Classic communications metric indicating how dominant the original signal
    energy is over the quantization noise.

    Formula
    -------
        SQNR = 10 · log10( Σ base² / (Σ (base - target)² + ε) )

    Returns
    -------
    float
        SQNR in dB; *higher is better*.
    """
    signal = torch.sum(base.detach() ** 2).item()
    noise = torch.sum((base.detach() - target.detach()) ** 2).item() + eps
    return 10.0 * np.log10(signal / noise)


def kl_div(base: torch.Tensor, target: torch.Tensor, *, bins: int = 2048) -> float:
    """
    Approximate KL-Divergence between the **distributions** of two tensors.

    Compares histograms built over the same range to quantify how much the
    quantised distribution diverges from the reference.

    Formula
    -------
        D_KL(p‖q) = Σ p(i) · log( p(i) / q(i) )

    Implementation details
    ----------------------
    * Histograms share bin edges [min(base), max(base)] to keep intervals equal.
    * Zero-probability bins are masked to avoid log(0).

    Returns
    -------
    float
        KL-divergence (non-negative); *closer to 0 is better*.
    """
    b_hist = torch.histc(
        base.detach(), bins=bins, min=base.min().item(), max=base.max().item()
    )
    t_hist = torch.histc(
        target.detach(), bins=bins, min=base.min().item(), max=base.max().item()
    )

    b_pdf = b_hist / b_hist.sum()
    t_pdf = t_hist / t_hist.sum()

    mask = (b_pdf > 0) & (t_pdf > 0)
    return torch.sum(b_pdf[mask] * torch.log(b_pdf[mask] / t_pdf[mask])).item()


class MetricCalculator:
    """
    Lightweight registry-and-dispatcher for **pair-wise tensor comparison metrics**.

    Purpose
    -------
    Consolidate all metrics used to assess the discrepancy between a reference
    (usually FP32) tensor and its quantized counterpart, while letting the caller
    choose *at runtime* which subset to evaluate.

    Built-in metrics
    ----------------
    Key                     Description
    --------------------    -------------------------------------------------
    "diff" / "max_abs_diff"  Maximum absolute element-wise difference
    "peir"                   Peak-Error-to-Interval Ratio

    Usage pattern
    -------------
    >>> calc = MetricCalculator(custom_metrics={'mse': mse_fn})
    >>> stats = calc.compute(fp_outs, q_outs, metrics=['diff', 'mse'])

    • **Instantiation** registers any extra user metrics
      (signature: ``fn(base: Tensor, target: Tensor) -> float``).
    • **compute(...)** takes two *equal-length* lists of tensors and an optional
      list of metric names.
        — If *metrics* is *None*, every registered metric is evaluated.
        — Returns a dict: ``{metric_name -> [value for each tensor pair]}``.

    Implementation notes
    --------------------
    * All tensors are detached before calculation to avoid autograd overhead.
    * Registrations are stored in `self.registry` (str → callable).
    * Duplicate metric names between built-ins and custom metrics raise an error
      at construction time to prevent silent shadowing.

    Practical metric bundles
    ------------------------
    • Quick sanity check ....................... ["diff", "mae", "peir"]
        — Spike, average, and range-normalized errors in one glance.

    • Layer-wise sensitivity profiling ......... ["rel_l2", "cosine"]
        — Scale-invariant magnitude plus directional alignment.

    • Whole-model quality / paper reporting .... ["sqnr"]  or  ["kl"]
        — Signal-to-noise (dB) or distribution divergence for headline numbers.
    """

    builtin_metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], float]] = {
        "diff": compute_max_abs_diff,
        "max_abs_diff": compute_max_abs_diff,
        "peir": compute_peir,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "rel_l2": rel_l2,
        "cosine": cosine_sim,
        "sqnr": sqnr_db,
        "kl": kl_div,
    }

    def __init__(
        self,
        custom_metrics: Optional[
            Dict[str, Callable[[torch.Tensor, torch.Tensor], float]]
        ] = None,
    ):
        self.registry: Dict[str, Callable] = self.builtin_metrics.copy()
        if custom_metrics:
            dup = self.registry.keys() & custom_metrics.keys()
            if dup:
                raise RuntimeError(f"Duplicate metric names: {dup}")
            assert custom_metrics is not None
            self.registry.update(custom_metrics)  # type: ignore[arg-type]

    # ----------------------------------------------------------------- #
    # Public API                                                        #
    # ----------------------------------------------------------------- #
    def compute(
        self,
        base_outputs: List[torch.Tensor],
        target_outputs: List[torch.Tensor],
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, List[Any]]:
        """
        Compute selected metrics for every (base, target) pair.

        Parameters
        ----------
        metrics
            List of metric names to evaluate **this call**.
            • None → evaluate *all* registered metrics.
        """
        sel = metrics or list(self.registry)
        unknown = set(sel) - self.registry.keys()
        if unknown:
            raise RuntimeError(f"Unknown metric(s): {unknown}")

        results: Dict[str, List[Any]] = {m: [] for m in sel}
        for base, tgt in zip(base_outputs, target_outputs):
            for m in sel:
                results[m].append(self.registry[m](base, tgt))
        return results
