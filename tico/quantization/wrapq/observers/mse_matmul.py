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
MatMul-aware MSE observer.

Unlike :class:`MSEObserver`, which minimizes the element-wise error
``||x - Q(x)||^2``, this observer minimizes the error *at the output of the
matmul that consumes x*:

    || W @ x - W @ Q(x) ||^2

The consuming weight ``W`` is passed to :meth:`collect` as the ``weight``
kwarg by the wrapper (e.g. ``QuantLinear`` passes its layer weight).

Key trick: with ``E = x - Q(x)`` (the quantization error),

    || W @ E ||^2 = sum_n E_n^T (W^T W) E_n = sum_n <E_n, H E_n>,  H = W^T W

so the Gram matrix ``H`` (K x K, K = in_features) is computed once per layer
and each grid candidate only needs a quadratic form instead of a full matmul.
"""

from typing import Optional

import torch

from tico.quantization.wrapq.observers.mse import MSEObserver
from tico.quantization.wrapq.utils.reduce_utils import channelwise_minmax


@torch.no_grad()
def _chunked_scale_err_sum(
    x_rows: torch.Tensor,
    scales: torch.Tensor,
    gram: torch.Tensor,
    qmin: int,
    qmax: int,
    is_symmetric: bool,
    zp_val,
    zps,
    chunk: int,
) -> torch.Tensor:
    """Weighted squared error per scale candidate, chunked over rows.

    Args:
        x_rows: [N, D] reference rows.
        scales: [G] scale candidates.
        gram: [D, D] Gram matrix H = W^T W.
        zp_val: fixed zero-point (int) or None.
        zps: [G] per-scale zero-points (used when zp_val is None and
             asymmetric).
        chunk: max rows per chunk.

    Returns:
        [G] sum over rows of <e, H e>.
    """
    err_sum = torch.zeros_like(scales)  # [G]
    for x_chunk in x_rows.split(chunk):
        # [chunk, G, D]
        x_div = x_chunk[:, None, :] / scales[None, :, None]
        if is_symmetric:
            x_round = torch.round(x_div)
            x_clamped = x_round.clamp(-qmax, qmax)
            x_q = x_clamped * scales[None, :, None]
        elif zp_val is not None:
            x_round = torch.round(x_div) + zp_val
            x_clamped = x_round.clamp(qmin, qmax)
            x_q = (x_clamped - zp_val) * scales[None, :, None]
        else:
            x_round = torch.round(x_div) + zps[None, :, None].float()
            x_clamped = x_round.clamp(qmin, qmax)
            x_q = (x_clamped - zps[None, :, None].float()) * scales[None, :, None]
        e = x_chunk[:, None, :] - x_q  # [chunk, G, D]
        err_sum += ((e @ gram) * e).sum(dim=(0, 2))
    return err_sum


@torch.no_grad()
def _chunked_zp_err_sum(
    x_rows: torch.Tensor,
    scale: torch.Tensor,
    zp_candidates: torch.Tensor,
    gram: torch.Tensor,
    qmin: int,
    qmax: int,
    chunk: int,
) -> torch.Tensor:
    """Weighted squared error per zp candidate, chunked over rows.

    Args:
        x_rows: [N, D] reference rows.
        scale: scalar scale.
        zp_candidates: [Z] zero-point candidates.
        gram: [D, D] Gram matrix.

    Returns:
        [Z] sum over rows of <e, H e>.
    """
    err_sum = torch.zeros_like(zp_candidates)  # [Z]
    for x_chunk in x_rows.split(chunk):
        # [chunk, Z, D]
        x_div = x_chunk[:, None, :] / scale
        x_round = torch.round(x_div) + zp_candidates[None, :, None]
        x_clamped = x_round.clamp(qmin, qmax)
        x_q = (x_clamped - zp_candidates[None, :, None]) * scale
        e = x_chunk[:, None, :] - x_q  # [chunk, Z, D]
        err_sum += ((e @ gram) * e).sum(dim=(0, 2))
    return err_sum



class MSEMatMulObserver(MSEObserver):
    """
    MSE-optimal scale observer minimizing the matmul output error
    ``||W @ x - W @ Q(x)||^2`` instead of the element-wise error.

    The consuming weight is provided via the ``weight`` kwarg of
    :meth:`collect` (forwarded by ``QuantModuleBase._fq``). When no weight
    is provided, the observer falls back to plain element-wise MSE behavior
    (identical to :class:`MSEObserver`).

    All merge strategies (max-merge for scale, running-mean zp, alternating
    search) are inherited from :class:`MSEObserver`.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Cached Gram matrix H = W^T W for the current consumer weight.
        # Keyed by (data_ptr, shape, device) so a weight change is detected.
        self._gram: Optional[torch.Tensor] = None
        self._gram_key: Optional[tuple] = None

    def reset(self) -> None:
        super().reset()
        self._gram = None
        self._gram_key = None

    # ------------------------------------------------------------------
    # Gram matrix cache
    # ------------------------------------------------------------------
    def _get_gram(self, weight: torch.Tensor) -> torch.Tensor:
        """Return H = W^T W, cached per weight tensor."""
        key = (weight.data_ptr(), tuple(weight.shape), weight.device)
        if self._gram is None or self._gram_key != key:
            w = weight.detach().float()
            # weight: [out_features, in_features] -> H: [in_features, in_features]
            self._gram = w.t() @ w
            self._gram_key = key
        return self._gram

    # ------------------------------------------------------------------
    # Scale search (weighted)
    # ------------------------------------------------------------------
    @staticmethod
    def _is_valid_weight(weight: Optional[torch.Tensor]) -> bool:
        """Any non-None weight defines a matmul consumer.

        2D weights ``[out_features, in_features]`` use the fast Gram-matrix
        path. Higher-dimensional weights (e.g. attention weights
        ``[B, H, S, K]``) use the direct-matmul path.
        """
        return weight is not None

    @torch.no_grad()
    def _search_optimal_scale_nd(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        fixed_zp: Optional[torch.Tensor],
        qmin: int,
        qmax: int,
        is_symmetric: bool,
        scales: torch.Tensor,
        zp_val,
        zps,
    ) -> torch.Tensor:
        """Search scale via direct matmul for N-D weights.

        Here ``weight @ x`` is a batched matmul where the contraction is over
        the last dim of ``weight`` and the second-to-last dim of ``x``. The
        error ``E = x - Q(x)`` has the same shape as ``x``, and the cost is
        ``sum (weight @ E)^2``.

        Args:
            x: ``[..., K, D]`` — the tensor being quantized.
            weight: ``[..., S, K]`` — the consumer (batched over leading dims).
            scales: ``[G]`` scale candidates.
        Returns:
            Best scale (scalar tensor).
        """
        wf = weight.detach().float()
        xf = x.detach().float()

        # Flatten batch dims so x becomes [N, K, D] and weight [N, S, K]
        # where N = product of all leading dims.
        x_3d = xf.reshape(-1, xf.shape[-2], xf.shape[-1])  # [N, K, D]
        w_3d = wf.reshape(-1, wf.shape[-2], wf.shape[-1])  # [N, S, K]
        N, K, D = x_3d.shape

        chunk = max(int(self.search_chunk_size), 1)
        err_sum = torch.zeros_like(scales)  # [G]

        for x_chunk, w_chunk in zip(x_3d.split(chunk), w_3d.split(chunk)):
            # x_chunk: [n, K, D], w_chunk: [n, S, K]
            # For each scale candidate g: E_g = x_chunk - Q_g(x_chunk)
            # cost_g = sum (w_chunk @ E_chunk_g)^2
            for g_idx in range(scales.shape[0]):
                s = scales[g_idx]
                x_div = x_chunk / s
                if is_symmetric:
                    x_round = torch.round(x_div)
                    x_clamped = x_round.clamp(-qmax, qmax)
                    x_q = x_clamped * s
                elif zp_val is not None:
                    x_round = torch.round(x_div) + zp_val
                    x_clamped = x_round.clamp(qmin, qmax)
                    x_q = (x_clamped - zp_val) * s
                else:
                    x_round = torch.round(x_div) + zps[g_idx]
                    x_clamped = x_round.clamp(qmin, qmax)
                    x_q = (x_clamped - zps[g_idx]) * s

                e = x_chunk - x_q  # [n, K, D]
                # w_chunk @ e: [n, S, D]
                err = torch.bmm(w_chunk, e)
                err_sum[g_idx] += (err * err).sum()

        mse = err_sum / N
        assert not torch.isnan(mse).any()
        best_idx = mse.argmin()
        return scales[best_idx]

    @torch.no_grad()
    def _search_optimal_zp_nd(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        scale: torch.Tensor,
        zp_candidates: torch.Tensor,
        qmin: int,
        qmax: int,
    ) -> torch.Tensor:
        """Search zp via direct matmul for N-D weights (scale fixed).

        Args:
            x: ``[..., K, D]`` — the tensor being quantized.
            weight: ``[..., S, K]`` — the consumer (batched over leading dims).
            scale: scalar scale.
            zp_candidates: ``[Z]`` zero-point candidates.
        Returns:
            Best zp (scalar tensor).
        """
        wf = weight.detach().float()
        xf = x.detach().float()

        x_3d = xf.reshape(-1, xf.shape[-2], xf.shape[-1])  # [N, K, D]
        w_3d = wf.reshape(-1, wf.shape[-2], wf.shape[-1])  # [N, S, K]
        N, K, D = x_3d.shape

        chunk = max(int(self.search_chunk_size), 1)
        err_sum = torch.zeros_like(zp_candidates)  # [Z]

        for x_chunk, w_chunk in zip(x_3d.split(chunk), w_3d.split(chunk)):
            for z_idx in range(zp_candidates.shape[0]):
                zp = zp_candidates[z_idx]
                x_div = x_chunk / scale
                x_round = torch.round(x_div) + zp
                x_clamped = x_round.clamp(qmin, qmax)
                x_q = (x_clamped - zp) * scale
                e = x_chunk - x_q  # [n, K, D]
                err = torch.bmm(w_chunk, e)  # [n, S, D]
                err_sum[z_idx] += (err * err).sum()

        mse = err_sum / N
        best_idx = mse.argmin()
        return zp_candidates[best_idx]

    @torch.no_grad()
    def _search_optimal_scale(
        self, x: torch.Tensor, fixed_zp: Optional[torch.Tensor] = None,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Search scale minimizing ||W x - W Q(x)||^2 (or plain MSE if no W)."""
        if not self._is_valid_weight(weight):
            return super()._search_optimal_scale(x, fixed_zp=fixed_zp)

        qmin = self.dtype.qmin
        qmax = self.dtype.qmax
        is_symmetric = self.qscheme.is_symmetric()

        # Base scale from min/max (same grid as MSEObserver)
        if is_symmetric:
            max_abs = x.abs().max().clamp(min=1e-12)
            base_scale = max_abs / qmax
        else:
            x_min = x.min()
            x_max = x.max()
            rng = (x_max - x_min).clamp(min=1e-12)
            base_scale = rng / (qmax - qmin)

        alphas = torch.linspace(
            1.0 / self.num_grid, 1.0, self.num_grid, device=x.device, dtype=torch.float32
        )
        scales = alphas * base_scale  # [G]

        if fixed_zp is not None:
            zp_val = float(fixed_zp)
            zps = None
        else:
            zp_val = None
            if not is_symmetric:
                zps = torch.round(qmin - x_min / scales).clamp(qmin, qmax).to(torch.int)  # [G]
            else:
                zps = None

        # --- N-D weight path: direct matmul (no Gram) ---
        if weight.dim() != 2:
            return self._search_optimal_scale_nd(
                x, weight, fixed_zp, qmin, qmax, is_symmetric,
                scales, zp_val, zps,
            )

        # --- 2D weight path: Gram-matrix trick (fast) ---
        # x: [..., in_features] — rows are the vectors multiplied by W.
        # Flatten to [N, K_in].
        x_rows = x.reshape(-1, x.shape[-1]).float()
        N, K_in = x_rows.shape
        gram = self._get_gram(weight)  # [K_in, K_in]

        # Chunk over rows to bound memory: intermediates [chunk, G, K_in]
        chunk = max(int(self.search_chunk_size), 1)
        err_sum = torch.zeros_like(scales)  # [G]

        for x_chunk in x_rows.split(chunk):
            # [chunk, 1, K_in] / [1, G, 1] -> [chunk, G, K_in]
            x_div = x_chunk[:, None, :] / scales[None, :, None]
            if is_symmetric:
                x_round = torch.round(x_div)
                x_clamped = x_round.clamp(-qmax, qmax)
                x_q = x_clamped * scales[None, :, None]
            elif zp_val is not None:
                x_round = torch.round(x_div) + zp_val
                x_clamped = x_round.clamp(qmin, qmax)
                x_q = (x_clamped - zp_val) * scales[None, :, None]
            else:
                x_round = torch.round(x_div) + zps[None, :, None].float()
                x_clamped = x_round.clamp(qmin, qmax)
                x_q = (x_clamped - zps[None, :, None].float()) * scales[None, :, None]

            e = x_chunk[:, None, :] - x_q  # [chunk, G, K_in]
            # Weighted error: <e, H e> per row, summed over rows
            # (e @ gram): [chunk, G, K_in]
            err_sum += ((e @ gram) * e).sum(dim=(0, 2))

        mse = err_sum / N  # [G]
        assert not torch.isnan(mse).any()

        best_idx = mse.argmin()
        return scales[best_idx]

    # ------------------------------------------------------------------
    # Zero-point search (weighted)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _search_optimal_zp(
        self, x: torch.Tensor, scale: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Search zp minimizing ||W x - W Q(x)||^2 with scale fixed."""
        if not self._is_valid_weight(weight):
            return super()._search_optimal_zp(x, scale)

        qmin = self.dtype.qmin
        qmax = self.dtype.qmax

        x_min = x.min()
        init_zp = int(torch.round(qmin - x_min / scale).clamp(qmin, qmax).item())

        half = self.num_zp_grid // 2
        lo = max(qmin, init_zp - half)
        hi = min(qmax, init_zp + half)
        zp_candidates = torch.arange(lo, hi + 1, device=x.device, dtype=torch.float32)  # [Z]

        # --- N-D weight path: direct matmul (no Gram) ---
        if weight.dim() != 2:
            return self._search_optimal_zp_nd(
                x, weight, scale, zp_candidates, qmin, qmax,
            )

        # --- 2D weight path: Gram-matrix trick (fast) ---
        x_rows = x.reshape(-1, x.shape[-1]).float()
        N, K_in = x_rows.shape
        gram = self._get_gram(weight)

        chunk = max(int(self.search_chunk_size), 1)
        err_sum = torch.zeros_like(zp_candidates)  # [Z]

        for x_chunk in x_rows.split(chunk):
            # [chunk, Z, K_in]
            x_div = x_chunk[:, None, :] / scale
            x_round = torch.round(x_div) + zp_candidates[None, :, None]
            x_clamped = x_round.clamp(qmin, qmax)
            x_q = (x_clamped - zp_candidates[None, :, None]) * scale
            e = x_chunk[:, None, :] - x_q  # [chunk, Z, K_in]
            err_sum += ((e @ gram) * e).sum(dim=(0, 2))

        mse = err_sum / N
        best_idx = mse.argmin()
        return zp_candidates[best_idx]


    # ------------------------------------------------------------------
    # Alternating search (weighted)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _search_optimal_scale_and_zp(
        self, x: torch.Tensor, weight: Optional[torch.Tensor] = None
    ) -> tuple:
        scale = self._search_optimal_scale(x, fixed_zp=None, weight=weight)
        for _ in range(self.max_alternating_iters):
            zp = self._search_optimal_zp(x, scale, weight=weight)
            scale = self._search_optimal_scale(x, fixed_zp=zp, weight=weight)
        zp = self._search_optimal_zp(x, scale, weight=weight)
        return scale, zp

    # ----------------------------------------------------------------    # ObserverBase interface
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _update_stats(self, x: torch.Tensor, **kwargs) -> None:
        """
        Update running min/max and search for the scale (and zp) minimizing
        the matmul output error ||W x - W Q(x)||^2.

        ``weight`` kwarg: the consuming layer's weight [out_features, in_features].
        Falls back to plain MSEObserver behavior when absent.
        """
        weight = kwargs.pop("weight", None)
        if weight is None:
            return None
        
        # Update min/max (same as MSEObserver)
        if self.channel_axis is None:
            curr_min, curr_max = x.min(), x.max()
        else:
            curr_min, curr_max = channelwise_minmax(x, self.channel_axis)

        self.min_val = torch.minimum(self.min_val, curr_min)
        self.max_val = torch.maximum(self.max_val, curr_max)

        is_symmetric = self.qscheme.is_symmetric()

        if is_symmetric:
            batch_scale = self._search_optimal_scale(x, fixed_zp=None, weight=weight)
            batch_zp = torch.tensor(0.0, device=x.device)
        elif self.alternating_search:
            batch_scale, batch_zp = self._search_optimal_scale_and_zp(x, weight=weight)
        else:
            batch_scale = self._search_optimal_scale(x, fixed_zp=None, weight=weight)
            batch_zp = None

        # Merge scale with running scale (max-merge) — inherited strategy
        if torch.isinf(self._running_scale).any():
            self._running_scale = batch_scale
        else:
            if self.max_merge:
                self._running_scale = torch.maximum(self._running_scale, batch_scale)
            else:
                self._running_scale = batch_scale

        if batch_zp is not None:
            self._zp_count += 1
            if self._zp_count == 1:
                self._running_zp = batch_zp.float()
            else:
                n = self._zp_count
                self._running_zp = self._running_zp + (batch_zp.float() - self._running_zp) / n


class MSEBatchedMatMulObserver(MSEObserver):
    """
    MatMul-aware MSE observer for the K-cache of grouped-query attention.

    Attention computes ``logits = Q @ K_rep^T`` where
    ``K_rep = K.repeat_interleave(kv_rep, dim=heads)``: each KV head h is
    consumed by the ``kv_rep`` attending Q heads ``j in [h*kv_rep, (h+1)*kv_rep)``.
    This observer minimizes the *exact* resulting logits error

        sum_h sum_{j attending h} || Q_j (K_h - Q(K_h))^T ||^2

    which, with ``E_h = K_h - Q(K_h)`` and per-KV-head Gram matrices
    ``H_h = sum_{j attending h} Q_j^T Q_j`` (D x D), equals
    ``sum_h sum_rows e^T H_h e``.

    The Q activations are passed via the ``weight`` kwarg of :meth:`collect`
    (shape ``[B, num_heads, S, D]``, post-RoPE), together with ``kv_rep``.
    Grams are recomputed every batch (Q changes per batch — no caching).

    Falls back to plain :class:`MSEObserver` behavior when no weight is given.
    """

    @staticmethod
    def _head_grams(q: torch.Tensor, kv_rep: int) -> torch.Tensor:
        """
        Per-KV-head Gram matrices H_h = sum_{j attending h} Q_j^T Q_j.

        Args:
            q: [B, num_heads, S, D] query activations.
            kv_rep: number of Q heads attending each KV head.

        Returns:
            [num_kv_heads, D, D] with num_kv_heads = num_heads // kv_rep.
        """
        qf = q.detach().float()
        # Per-Q-head Gram: [num_heads, D, D]
        g = torch.einsum("bhsd,bhse->hde", qf, qf)
        if kv_rep == 1:
            return g
        num_heads, D, _ = g.shape
        # Group attending Q heads per KV head and sum
        return g.reshape(num_heads // kv_rep, kv_rep, D, D).sum(dim=1)

    # ------------------------------------------------------------------
    # Scale search (per-KV-head weighted)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _search_optimal_scale(
        self, x: torch.Tensor, fixed_zp: Optional[torch.Tensor] = None,
        weight: Optional[torch.Tensor] = None, kv_rep: int = 1,
    ) -> torch.Tensor:
        """Search scale minimizing the exact GQA logits error (or plain MSE)."""
        if weight is None:
            return super()._search_optimal_scale(x, fixed_zp=fixed_zp)

        qmin = self.dtype.qmin
        qmax = self.dtype.qmax
        is_symmetric = self.qscheme.is_symmetric()

        # x: [B, kv_heads, S, D]
        B, H, S, D = x.shape
        grams = self._head_grams(weight, kv_rep)  # [H, D, D]
        assert grams.shape[0] == H, (grams.shape, x.shape)

        # Base scale from min/max of the whole tensor (same grid as MSEObserver)
        if is_symmetric:
            max_abs = x.abs().max().clamp(min=1e-12)
            base_scale = max_abs / qmax
        else:
            x_min = x.min()
            x_max = x.max()
            rng = (x_max - x_min).clamp(min=1e-12)
            base_scale = rng / (qmax - qmin)

        alphas = torch.linspace(
            1.0 / self.num_grid, 1.0, self.num_grid, device=x.device, dtype=torch.float32
        )
        scales = alphas * base_scale  # [G]

        if fixed_zp is not None:
            zp_val = float(fixed_zp)
            zps = None
        else:
            zp_val = None
            if not is_symmetric:
                zps = torch.round(qmin - x_min / scales).clamp(qmin, qmax).to(torch.int)  # [G]
            else:
                zps = None

        chunk = max(int(self.search_chunk_size), 1)
        err_sum = torch.zeros_like(scales)  # [G]
        N_total = 0
        for h in range(H):
            x_rows = x[:, h].reshape(-1, D).float()  # [B*S, D]
            N_total += x_rows.shape[0]
            err_sum += _chunked_scale_err_sum(
                x_rows, scales, grams[h], qmin, qmax,
                is_symmetric, zp_val, zps, chunk,
            )

        mse = err_sum / N_total
        assert not torch.isnan(mse).any()
        best_idx = mse.argmin()
        return scales[best_idx]

    # ------------------------------------------------------------------
    # Zero-point search (per-KV-head weighted)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _search_optimal_zp(
        self, x: torch.Tensor, scale: torch.Tensor,
        weight: Optional[torch.Tensor] = None, kv_rep: int = 1,
    ) -> torch.Tensor:
        """Search zp minimizing the exact GQA logits error with scale fixed."""
        if weight is None:
            return super()._search_optimal_zp(x, scale)

        qmin = self.dtype.qmin
        qmax = self.dtype.qmax

        B, H, S, D = x.shape
        grams = self._head_grams(weight, kv_rep)

        x_min = x.min()
        init_zp = int(torch.round(qmin - x_min / scale).clamp(qmin, qmax).item())

        half = self.num_zp_grid // 2
        lo = max(qmin, init_zp - half)
        hi = min(qmax, init_zp + half)
        zp_candidates = torch.arange(lo, hi + 1, device=x.device, dtype=torch.float32)  # [Z]

        chunk = max(int(self.search_chunk_size), 1)
        err_sum = torch.zeros_like(zp_candidates)  # [Z]
        N_total = 0
        for h in range(H):
            x_rows = x[:, h].reshape(-1, D).float()
            N_total += x_rows.shape[0]
            err_sum += _chunked_zp_err_sum(
                x_rows, scale, zp_candidates, grams[h], qmin, qmax, chunk
            )

        mse = err_sum / N_total
        best_idx = mse.argmin()
        return zp_candidates[best_idx]

    # ------------------------------------------------------------------
    # Alternating search (weighted)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _search_optimal_scale_and_zp(
        self, x: torch.Tensor,
        weight: Optional[torch.Tensor] = None, kv_rep: int = 1,
    ) -> tuple:
        scale = self._search_optimal_scale(x, fixed_zp=None, weight=weight, kv_rep=kv_rep)
        for _ in range(self.max_alternating_iters):
            zp = self._search_optimal_zp(x, scale, weight=weight, kv_rep=kv_rep)
            scale = self._search_optimal_scale(x, fixed_zp=zp, weight=weight, kv_rep=kv_rep)
        zp = self._search_optimal_zp(x, scale, weight=weight, kv_rep=kv_rep)
        return scale, zp

    # ------------------------------------------------------------------
    # ObserverBase interface
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _update_stats(self, x: torch.Tensor, **kwargs) -> None:
        """
        Update running min/max and search for the scale (and zp) minimizing
        the exact GQA attention-logits error.

        Kwargs:
            weight: Q activations [B, num_heads, S, D] (post-RoPE).
            kv_rep: number of Q heads attending each KV head.
        Falls back to plain MSEObserver behavior when weight is absent.
        """
        weight = kwargs.pop("weight", None)
        if weight is None:
            return None
        kv_rep = int(kwargs.pop("kv_rep", 1))

        # Update min/max (same as MSEObserver)
        if self.channel_axis is None:
            curr_min, curr_max = x.min(), x.max()
        else:
            curr_min, curr_max = channelwise_minmax(x, self.channel_axis)

        self.min_val = torch.minimum(self.min_val, curr_min)
        self.max_val = torch.maximum(self.max_val, curr_max)

        is_symmetric = self.qscheme.is_symmetric()

        if is_symmetric:
            batch_scale = self._search_optimal_scale(
                x, fixed_zp=None, weight=weight, kv_rep=kv_rep
            )
            batch_zp = torch.tensor(0.0, device=x.device)
        elif self.alternating_search:
            batch_scale, batch_zp = self._search_optimal_scale_and_zp(
                x, weight=weight, kv_rep=kv_rep
            )
        else:
            batch_scale = self._search_optimal_scale(
                x, fixed_zp=None, weight=weight, kv_rep=kv_rep
            )
            batch_zp = None

        # Merge scale with running scale (max-merge) — inherited strategy
        if torch.isinf(self._running_scale).any():
            self._running_scale = batch_scale
        else:
            if self.max_merge:
                self._running_scale = torch.maximum(self._running_scale, batch_scale)
            else:
                self._running_scale = batch_scale

        if batch_zp is not None:
            self._zp_count += 1
            if self._zp_count == 1:
                self._running_zp = batch_zp.float()
            else:
                n = self._zp_count
                self._running_zp = self._running_zp + (batch_zp.float() - self._running_zp) / n

