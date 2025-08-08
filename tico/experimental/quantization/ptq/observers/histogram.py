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
HistogramObserver

Collects activation/weight statistics in histogram bins and determines an
optimal clipping range by *minimizing KL-divergence* between the original
float distribution and its quantized counterpart.

Why use it?
-----------
• Very robust to outliers — usually yields the **best** PTQ accuracy.  
• Works well for LLM feed-forward and embedding layers that have heavy tails.

Drawbacks
---------
• Memory- & compute-intensive (large histograms, KL search).  
• Overkill for every layer in a huge model.

TIP
------
Using a HistogramObserver on **all** layers of a large model will blow up
calibration time and memory consumption.  In practice you only attach it to
layers that are *prone to outliers* (e.g. FFN, embedding, first & last layer)
and keep cheaper observers (MinMax, EMA, …) everywhere else.  Mixing observers
is perfectly fine because fake-quant happens locally per layer.
"""

import torch
import torch.nn.functional as F

from tico.experimental.quantization.ptq.observers.affine_base import AffineObserverBase
from tico.experimental.quantization.ptq.utils.reduce_utils import channelwise_minmax


class HistogramObserver(AffineObserverBase):
    """Heavy-duty observer that finds KL-optimal clip thresholds."""

    def __init__(
        self,
        *,
        bins: int = 2048,
        search_step: int = 128,
        **kwargs,
    ):
        self.bins = bins
        self.search_step = search_step

        self.hist = torch.zeros(bins)
        self.edges = torch.linspace(-1.0, 1.0, bins + 1)

        super().__init__(**kwargs)

    # ------------------------------------------------------------------ #
    # Lifecycle overrides
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Clear running min/max and histogram."""
        super().reset()
        self.hist.zero_()

    @torch.no_grad()
    def _update_stats(self, x: torch.Tensor) -> None:
        """
        Update running min/max *and* histogram.

        For clarity only the per-tensor variant is implemented.  Per-channel
        support would histogram each channel independently (TODO).
        """
        # ---------- update global min/max first ------------------------
        if self.channel_axis is None:
            curr_min, curr_max = x.min(), x.max()
        else:
            curr_min, curr_max = channelwise_minmax(x, self.channel_axis)

        # Broadcasting handles scalar-vs-vector cases
        self.min_val = torch.minimum(self.min_val, curr_min)
        self.max_val = torch.maximum(self.max_val, curr_max)

        # ---------- update histogram -----------------------------------
        if torch.isinf(self.min_val) or torch.isinf(self.max_val):
            # Need at least one valid range before we can bin values
            return

        # (Re)allocate edges lazily once true range is known
        self.edges = torch.linspace(
            self.min_val.item(), self.max_val.item(), self.bins + 1, device=x.device
        )

        h = torch.histc(
            x.detach(),
            bins=self.bins,
            min=self.edges[0].item(),
            max=self.edges[-1].item(),
        )
        self.hist += h

    def compute_qparams(self):
        """
        Replace `min_val` / `max_val` with KL-optimal clip range
        before delegating to the base implementation.
        """
        if self.enabled:  # i.e. statistics have been collected
            self.min_val, self.max_val = self._best_clip()

        return super().compute_qparams()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _best_clip(self):
        """
        Return (clip_min, clip_max) that minimise KL divergence.

        *Always* feeds equal-length probability vectors to `kl_div` by
        resampling whichever side (p or q) is shorter.
        """
        num_qbins = self.dtype.qmax - self.dtype.qmin + 1  # e.g. 256
        best_kl = torch.tensor(torch.inf)
        best_min, best_max = self.min_val, self.max_val

        for start in range(0, self.bins // 2, self.search_step):
            for end in range(self.bins - 1, self.bins // 2, -self.search_step):
                if end <= start:
                    continue

                # ----- Original distribution in the candidate clip range
                p = self.hist[start : end + 1].clone()
                p[0] += self.hist[:start].sum()  # left tail → first bin
                p[-1] += self.hist[end + 1 :].sum()  # right tail → last bin

                n = p.numel()

                # ----- Quantised approximation (merge into num_qbins buckets)
                if n >= num_qbins:
                    # Down-sample via average pooling (stride ≥ 1 always)
                    factor = n // num_qbins or 1
                    q = F.avg_pool1d(
                        p.view(1, 1, -1),
                        kernel_size=factor,
                        stride=factor,
                        ceil_mode=False,
                    ).view(-1)[:num_qbins]
                else:
                    # Up-sample via linear interpolation
                    q = F.interpolate(
                        p.view(1, 1, -1),
                        size=num_qbins,
                        mode="linear",
                        align_corners=False,
                    ).view(-1)

                # ----- Ensure p and q have identical length ---------------
                if p.numel() != q.numel():
                    # Resample the *shorter* one to match the longer
                    longer = max(p.numel(), q.numel())
                    if p.numel() < longer:
                        p = F.interpolate(
                            p.view(1, 1, -1),
                            size=longer,
                            mode="linear",
                            align_corners=False,
                        ).view(-1)
                    else:  # q is shorter
                        q = F.interpolate(
                            q.view(1, 1, -1),
                            size=longer,
                            mode="nearest",
                        ).view(-1)

                # ----- Compute KL divergence -----------------------------
                p_prob = p / (p.sum() + 1e-12)
                q_prob = q / (q.sum() + 1e-12)

                kl = F.kl_div(q_prob.log(), p_prob, reduction="sum")
                if kl < best_kl:
                    best_kl = kl
                    best_min = self.edges[start]
                    best_max = self.edges[end]

        return best_min, best_max
