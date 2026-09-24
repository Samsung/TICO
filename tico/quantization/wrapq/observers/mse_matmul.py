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
        # Two-stage (global) state — used when max_merge=False
        self._stage: int = 1
        self._err_accumulator: Optional[torch.Tensor] = None  # [G, Z]
        self._grid_scales: Optional[torch.Tensor] = None      # [G]
        self._grid_zps: Optional[torch.Tensor] = None         # [Z]
        
        # Default to max-merge strategy (consistent with MSEObserver),
        # unless the caller explicitly passed max_merge via kwargs.
        if "max_merge" not in kwargs:
            self.max_merge = True

    def reset(self) -> None:
        super().reset()
        self._gram = None
        self._gram_key = None
        self._stage = 1
        self._err_accumulator = None
        self._grid_scales = None
        self._grid_zps = None

    # ------------------------------------------------------------------
    # Two-stage (global) strategy — used when max_merge=False
    # ------------------------------------------------------------------
    def prepare_stage2(self) -> None:
        """Build the fixed 2-D (scale, zp) grid from global min/max.

        Called between the two calibration passes.  After this call,
        :meth:`collect` switches to stage-2 mode (error accumulation on
        the fixed grid).

        When ``max_merge=True``, this observer is disabled instead — it
        already has its stats from pass 1 and should not double-collect.
        """
        if self.max_merge:
            self.enabled = False
            return

        qmin = self.dtype.qmin
        qmax = self.dtype.qmax
        is_symmetric = self.qscheme.is_symmetric()

        if is_symmetric:
            max_abs = torch.maximum(self.max_val.abs(), self.min_val.abs()).clamp(min=1e-12)
            base_scale = max_abs / qmax
        else:
            rng = (self.max_val - self.min_val).clamp(min=1e-12)
            base_scale = rng / (qmax - qmin)

        alphas = torch.linspace(
            1.0 / self.num_grid, 1.0, self.num_grid,
            device=self.min_val.device, dtype=torch.float32,
        )

        if self.channel_axis is None:
            # Per-tensor: grid_scales [G]
            self._grid_scales = alphas * base_scale  # [G]
            C = 1
        else:
            # Per-channel: grid_scales [C, G]
            self._grid_scales = (
                alphas[None, :] * base_scale[:, None]
            )  # [C, G]
            C = base_scale.shape[0]

        if is_symmetric:
            self._grid_zps = torch.zeros(
                1, dtype=torch.float32, device=self.min_val.device,
            )
        else:
            self._grid_zps = torch.arange(
                qmin, qmax + 1, dtype=torch.float32, device=self.min_val.device,
            )

        Z = len(self._grid_zps)
        if self.channel_axis is None:
            self._err_accumulator = torch.zeros(
                self.num_grid, Z,
                device=self.min_val.device, dtype=torch.float32,
            )  # [G, Z]
        else:
            self._err_accumulator = torch.zeros(
                C, self.num_grid, Z,
                device=self.min_val.device, dtype=torch.float32,
            )  # [C, G, Z]
        self._stage = 2

    @torch.no_grad()
    def _update_stats_global(self, x: torch.Tensor, **kwargs) -> None:
        """Two-stage update: stage 1 accumulates global min/max; stage 2
        accumulates per-grid errors on the fixed grid built by
        :meth:`prepare_stage2`.

        Unlike :class:`MSEBatchedMatMulObserver`, the Gram matrix is fixed
        (it depends on the consuming weight, not on per-batch activations),
        so stage 1 only needs min/max — no running Gram accumulation.
        """
        weight = kwargs.pop("weight", None)
        if weight is None:
            return

        # Update min/max (same as MSEObserver) — always, both stages
        if self.channel_axis is None:
            curr_min, curr_max = x.min(), x.max()
        else:
            curr_min, curr_max = channelwise_minmax(x, self.channel_axis)
        self.min_val = torch.minimum(self.min_val, curr_min)
        self.max_val = torch.maximum(self.max_val, curr_max)

        if self._stage == 1:
            return

        # Stage 2: accumulate per-grid errors.
        assert self._grid_scales is not None and self._grid_zps is not None
        assert self._err_accumulator is not None

        qmin = self.dtype.qmin
        qmax = self.dtype.qmax
        is_symmetric = self.qscheme.is_symmetric()

        zps = self._grid_zps  # [Z]
        Z = zps.shape[0]

        if self.channel_axis is None:
            # --- Per-tensor: accumulator [G, Z] ---
            scales = self._grid_scales  # [G]
            G = scales.shape[0]

            if weight.dim() == 2:
                # --- 2D weight: Gram-matrix trick (fast) ---
                # weight: [out_features, in_features], x: [..., in_features]
                # Gram = W^T W is fixed and can be cached.
                gram = self._get_gram(weight)  # [K_in, K_in]
                x_rows = x.reshape(-1, x.shape[-1]).float()  # [N, K_in]
                N, K_in = x_rows.shape

                # Chunk over rows to bound peak memory of [chunk, K_in, Z] intermediate.
                mem_per_row = K_in * Z * 4  # float32
                chunk = max(1, (64 << 20) // mem_per_row)

                for gi in range(G):
                    scale = scales[gi]
                    zps_col = zps.view(1, 1, Z)  # [1, 1, Z]
                    err_z = torch.zeros(Z, device=x.device, dtype=torch.float32)

                    for start in range(0, N, chunk):
                        end = min(start + chunk, N)
                        xc = x_rows[start:end]  # [chunk, K_in]
                        if is_symmetric:
                            q = torch.round(xc[:, :, None] / scale).clamp(-qmax, qmax)
                        else:
                            q = torch.round(xc[:, :, None] / scale + zps_col).clamp(qmin, qmax)
                        x_deq = (q - zps_col) * scale  # [chunk, K_in, Z]
                        e = xc[:, :, None] - x_deq     # [chunk, K_in, Z]
                        # Weighted error per zp: sum_n e_n^T H e_n
                        err_z += torch.einsum("ckz,kl,clz->z", e, gram, e)

                    self._err_accumulator[gi, :] += err_z
            else:
                # --- N-D weight: direct matmul (no fixed Gram possible) ---
                # weight: [..., S, K], x: [..., K, D]
                # The contraction is over K (last dim of weight, second-to-last
                # dim of x).  The weight changes per batch (e.g. attention
                # weights), so a single Gram cannot be precomputed.
                # error = ||weight @ (x - Q(x))||^2
                wf = weight.detach().float()
                xf = x.detach().float()
                x_3d = xf.reshape(-1, xf.shape[-2], xf.shape[-1])  # [N, K, D]
                w_3d = wf.reshape(-1, wf.shape[-2], wf.shape[-1])  # [N, S, K]
                N, K, D = x_3d.shape

                # Chunk over N to bound peak memory of [chunk, K, D, Z] intermediate.
                mem_per_row = K * D * Z * 4  # float32
                chunk = max(1, (64 << 20) // mem_per_row)

                for gi in range(G):
                    scale = scales[gi]
                    zps_col = zps.view(1, 1, 1, Z)  # [1, 1, 1, Z]
                    err_z = torch.zeros(Z, device=x.device, dtype=torch.float32)

                    for start in range(0, N, chunk):
                        end = min(start + chunk, N)
                        xc = x_3d[start:end]  # [chunk, K, D]
                        wc = w_3d[start:end]  # [chunk, S, K]
                        if is_symmetric:
                            q = torch.round(xc[:, :, :, None] / scale).clamp(-qmax, qmax)
                        else:
                            q = torch.round(xc[:, :, :, None] / scale + zps_col).clamp(qmin, qmax)
                        x_deq = (q - zps_col) * scale  # [chunk, K, D, Z]
                        e = xc[:, :, :, None] - x_deq  # [chunk, K, D, Z]
                        # Weighted error: ||w @ e||^2 for each zp
                        # einsum: w[chunk,S,K] @ e[chunk,K,D,Z] -> [chunk,S,D,Z]
                        err = torch.einsum("csk,ckdz->csdz", wc, e)
                        err_z += (err * err).sum(dim=(0, 1, 2))  # [Z]

                    self._err_accumulator[gi, :] += err_z
        else:
            # --- Per-channel: accumulator [C, G, Z] ---
            scales = self._grid_scales  # [C, G]
            C, G = scales.shape

            if weight.dim() == 2:
                # --- 2D weight: diagonal Gram approximation ---
                # H_diag[c] = ||W[:,c]||^2 makes the error separable per
                # channel: err_c = H_diag[c] * sum_n E_{n,c}^2
                gram = self._get_gram(weight)  # [K_in, K_in]
                gram_diag = gram.diagonal()  # [K_in] = [C]

                ca = self.channel_axis % x.dim()
                x_perm = x.movedim(ca, -1)  # [..., C]
                x_rows = x_perm.reshape(-1, x_perm.shape[-1]).float()  # [N, C]
                N = x_rows.shape[0]

                # Chunk over rows to bound peak memory of [chunk, C, Z] intermediate.
                mem_per_row = C * Z * 4  # float32
                chunk = max(1, (64 << 20) // mem_per_row)

                for gi in range(G):
                    scale_g = scales[:, gi]  # [C]
                    err_cz = torch.zeros(C, Z, device=x.device, dtype=torch.float32)

                    for start in range(0, N, chunk):
                        end = min(start + chunk, N)
                        xc = x_rows[start:end]  # [chunk, C]
                        if is_symmetric:
                            q = torch.round(
                                xc[:, :, None] / scale_g[None, :, None]
                            ).clamp(-qmax, qmax)
                            x_deq = q * scale_g[None, :, None]
                        else:
                            zps_col = zps.view(1, 1, Z)
                            q = torch.round(
                                xc[:, :, None] / scale_g[None, :, None] + zps_col
                            ).clamp(qmin, qmax)
                            x_deq = (q - zps_col) * scale_g[None, :, None]
                        e = xc[:, :, None] - x_deq  # [chunk, C, Z]
                        # Diagonal Gram: H_diag[c] * sum_n e_{n,c}^2
                        err_cz += gram_diag[:, None] * (e * e).sum(dim=0)  # [C, Z]

                    self._err_accumulator[:, gi, :] += err_cz
            else:
                # --- N-D weight: exact separable per channel ---
                # weight: [..., S, K], x: [..., K, D], channel_axis = D (last dim).
                # The contraction is over K (not the channel dim D), so the
                # error is exactly separable per channel.
                wf = weight.detach().float()
                xf = x.detach().float()
                x_3d = xf.reshape(-1, xf.shape[-2], xf.shape[-1])  # [N, K, D]
                w_3d = wf.reshape(-1, wf.shape[-2], wf.shape[-1])  # [N, S, K]
                N, K, D = x_3d.shape
                # C = D (channel axis is the last dim)

                # Chunk over N to bound peak memory of [chunk, K, D, Z] intermediate.
                mem_per_row = K * D * Z * 4  # float32
                chunk = max(1, (64 << 20) // mem_per_row)

                for gi in range(G):
                    scale_g = scales[:, gi]  # [D] = [C]
                    err_cz = torch.zeros(D, Z, device=x.device, dtype=torch.float32)

                    for start in range(0, N, chunk):
                        end = min(start + chunk, N)
                        xc = x_3d[start:end]  # [chunk, K, D]
                        wc = w_3d[start:end]  # [chunk, S, K]
                        if is_symmetric:
                            q = torch.round(
                                xc[:, :, :, None] / scale_g[None, None, :, None]
                            ).clamp(-qmax, qmax)
                            x_deq = q * scale_g[None, None, :, None]
                        else:
                            zps_col = zps.view(1, 1, 1, Z)
                            q = torch.round(
                                xc[:, :, :, None] / scale_g[None, None, :, None] + zps_col
                            ).clamp(qmin, qmax)
                            x_deq = (q - zps_col) * scale_g[None, None, :, None]
                        e = xc[:, :, :, None] - x_deq  # [chunk, K, D, Z]
                        # Weighted error: ||w @ e||^2, separable per channel D
                        # einsum: w[chunk,S,K] @ e[chunk,K,D,Z] -> [chunk,S,D,Z]
                        err = torch.einsum("csk,ckdz->csdz", wc, e)
                        err_cz += (err * err).sum(dim=(0, 1))  # [D, Z] = [C, Z]

                    self._err_accumulator[:, gi, :] += err_cz

    @torch.no_grad()
    def compute_qparams(self):
        """Pick best (scale, zp) from the error accumulator (global),
        or fall back to the parent implementation for max_merge.

        When per-channel, the accumulator is [C, G, Z] and the best
        (scale, zp) is selected independently per channel.
        """
        if self.max_merge:
            return super().compute_qparams()

        assert self._err_accumulator is not None
        Z = self._grid_zps.shape[0]

        if self.channel_axis is None:
            # Per-tensor: accumulator [G, Z]
            best_flat = self._err_accumulator.argmin().item()
            best_g = best_flat // Z
            best_z = best_flat % Z

            best_scale = self._grid_scales[best_g].float()
            best_zp = self._grid_zps[best_z].to(torch.int)
        else:
            # Per-channel: accumulator [C, G, Z]
            C, G, _ = self._err_accumulator.shape
            best_flat = self._err_accumulator.reshape(C, -1).argmin(dim=1)  # [C]
            best_g = best_flat // Z  # [C]
            best_z = best_flat % Z   # [C]

            best_scale = self._grid_scales[
                torch.arange(C, device=best_g.device), best_g
            ].float()
            best_zp = self._grid_zps[best_z].to(torch.int)

        self._cached_scale = best_scale
        self._cached_zp = best_zp
        return best_scale, best_zp

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
        if not self.max_merge:
            self._update_stats_global(x, **kwargs)
            return

        weight = kwargs.pop("weight", None)
        if weight is None:
            return
        
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

    Parameters
    ----------
    max_merge : bool
        ``True`` (default) — per-batch scale/zp search with max-merge
        across batches.  This is the original strategy inherited from
        :class:`MSEObserver`.

        ``False`` — two-pass strategy that exactly matches global
        calibration without storing raw activations:

        * Stage 1 (first calibration pass): accumulate a running Gram
          ``Σ_t Q_t^T Q_t`` and global min/max.  No scale search is done.
        * :meth:`prepare_stage2` (called between passes): build a fixed
          2-D ``(scale, zp)`` grid from the global min/max.
        * Stage 2 (second calibration pass): accumulate per-grid errors
          on the fixed grid using the accumulated Gram.
        * :meth:`compute_qparams`: pick the ``(scale, zp)`` pair with the
          minimum accumulated error.

        Memory cost: ``O(H·D² + G·Z)`` — no raw data stored between passes.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Two-stage (global) state
        self._running_grams: Optional[torch.Tensor] = None  # [H, D, D]
        self._stage: int = 1
        self._err_accumulator: Optional[torch.Tensor] = None  # [G, Z]
        self._grid_scales: Optional[torch.Tensor] = None  # [G]
        self._grid_zps: Optional[torch.Tensor] = None  # [Z]
        # Default to max-merge strategy (consistent with MSEObserver),
        # unless the caller explicitly passed max_merge via kwargs.
        if "max_merge" not in kwargs:
            self.max_merge = True

    def reset(self) -> None:
        super().reset()
        self._running_grams = None
        self._stage = 1
        self._err_accumulator = None
        self._grid_scales = None
        self._grid_zps = None

    def prepare_stage2(self) -> None:
        """Build the fixed 2-D (scale, zp) grid from global min/max.

        Called between the two calibration passes.  After this call,
        :meth:`collect` switches to stage-2 mode (error accumulation on
        the fixed grid using the accumulated running Gram).

        When ``max_merge=True``, this observer is disabled
        instead — it already has its stats from pass 1 and should not
        double-collect.
        """
        if self.max_merge:
            self.enabled = False
            return

        qmin = self.dtype.qmin
        qmax = self.dtype.qmax
        is_symmetric = self.qscheme.is_symmetric()

        if is_symmetric:
            max_abs = torch.maximum(self.max_val.abs(), self.min_val.abs()).clamp(min=1e-12)
            base_scale = max_abs / qmax
        else:
            rng = (self.max_val - self.min_val).clamp(min=1e-12)
            base_scale = rng / (qmax - qmin)

        alphas = torch.linspace(
            1.0 / self.num_grid, 1.0, self.num_grid,
            device=self.min_val.device, dtype=torch.float32,
        )

        if self.channel_axis is None:
            # Per-tensor: grid_scales [G]
            self._grid_scales = alphas * base_scale  # [G]
            C = 1
        else:
            # Per-channel: grid_scales [C, G]
            self._grid_scales = (
                alphas[None, :] * base_scale[:, None]
            )  # [C, G]
            C = base_scale.shape[0]

        if is_symmetric:
            self._grid_zps = torch.zeros(
                1, dtype=torch.float32, device=self.min_val.device,
            )
        else:
            self._grid_zps = torch.arange(
                qmin, qmax + 1, dtype=torch.float32, device=self.min_val.device,
            )

        Z = len(self._grid_zps)
        if self.channel_axis is None:
            self._err_accumulator = torch.zeros(
                self.num_grid, Z,
                device=self.min_val.device, dtype=torch.float32,
            )  # [G, Z]
        else:
            self._err_accumulator = torch.zeros(
                C, self.num_grid, Z,
                device=self.min_val.device, dtype=torch.float32,
            )  # [C, G, Z]
        self._stage = 2

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
        if not self.max_merge:
            self._update_stats_global_gram(x, **kwargs)
            return

        weight = kwargs.pop("weight", None)
        if weight is None:
            return
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

    # ------------------------------------------------------------------
    # Global-Gram (two-stage) strategy
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _update_stats_global_gram(self, x: torch.Tensor, **kwargs) -> None:
        """Two-stage update: stage 1 accumulates Gram + min/max; stage 2
        accumulates per-grid errors on the fixed grid built by
        :meth:`prepare_stage2`.
        """
        weight = kwargs.pop("weight", None)
        if weight is None:
            return
        kv_rep = int(kwargs.pop("kv_rep", 1))

        # Update min/max (same as MSEObserver) — always, both stages
        if self.channel_axis is None:
            curr_min, curr_max = x.min(), x.max()
        else:
            curr_min, curr_max = channelwise_minmax(x, self.channel_axis)
        self.min_val = torch.minimum(self.min_val, curr_min)
        self.max_val = torch.maximum(self.max_val, curr_max)

        # Compute per-KV-head Gram for this batch
        grams = self._head_grams(weight, kv_rep)  # [H, D, D]

        if self._stage == 1:
            # Accumulate running Gram
            if self._running_grams is None:
                self._running_grams = grams.clone()
            else:
                self._running_grams = self._running_grams + grams
            return

        # Stage 2: accumulate per-grid errors using the accumulated Gram.
        # Vectorise over the zero-point dimension to avoid a Python loop.
        assert self._grid_scales is not None and self._grid_zps is not None
        assert self._err_accumulator is not None

        qmin = self.dtype.qmin
        qmax = self.dtype.qmax
        is_symmetric = self.qscheme.is_symmetric()
        B, H, S, D = x.shape

        grams_global = self._running_grams  # [H, D, D]
        zps = self._grid_zps  # [Z]
        Z = zps.shape[0]

        if self.channel_axis is None:
            # --- Per-tensor: accumulator [G, Z] ---
            scales = self._grid_scales  # [G]
            G = scales.shape[0]

            # Chunk over rows to bound peak memory of the [chunk, D, Z] intermediate.
            mem_per_row = D * Z * 4  # float32
            chunk = max(1, (64 << 20) // mem_per_row)

            for h in range(H):
                x_rows = x[:, h].reshape(-1, D).float()  # [N, D]
                N = x_rows.shape[0]
                gram_h = grams_global[h]  # [D, D]

                for gi in range(G):
                    scale = scales[gi]
                    zps_col = zps.view(1, 1, Z)  # [1, 1, Z]
                    err_z = torch.zeros(Z, device=x.device, dtype=torch.float32)

                    for start in range(0, N, chunk):
                        end = min(start + chunk, N)
                        xc = x_rows[start:end]  # [chunk, D]
                        q = torch.round(xc[:, :, None] / scale + zps_col).clamp(qmin, qmax)  # [chunk, D, Z]
                        x_deq = (q - zps_col) * scale  # [chunk, D, Z]
                        e = xc[:, :, None] - x_deq  # [chunk, D, Z]
                        # Weighted error per zp: sum_n e_n^T H_h e_n
                        err_z += torch.einsum("cdz,de,cez->z", e, gram_h, e)

                    self._err_accumulator[gi, :] += err_z
        else:
            # --- Per-channel: accumulator [C, G, Z] ---
            # Diagonal Gram approximation: H_h_diag[d] = H_h[d,d] makes
            # the error separable per channel.
            scales = self._grid_scales  # [C, G]
            C, G = scales.shape
            grams_diag = grams_global.diagonal(dim1=-2, dim2=-1)  # [H, D]

            # Chunk over rows to bound peak memory of [chunk, C, Z] intermediate.
            mem_per_row = C * Z * 4  # float32
            chunk = max(1, (64 << 20) // mem_per_row)

            for h in range(H):
                x_rows = x[:, h].reshape(-1, D).float()  # [N, D] = [N, C]
                N = x_rows.shape[0]
                gram_h_diag = grams_diag[h]  # [D] = [C]

                for gi in range(G):
                    scale_g = scales[:, gi]  # [C]
                    err_cz = torch.zeros(C, Z, device=x.device, dtype=torch.float32)

                    for start in range(0, N, chunk):
                        end = min(start + chunk, N)
                        xc = x_rows[start:end]  # [chunk, C]
                        if is_symmetric:
                            q = torch.round(
                                xc[:, :, None] / scale_g[None, :, None]
                            ).clamp(-qmax, qmax)
                            x_deq = q * scale_g[None, :, None]
                        else:
                            zps_col = zps.view(1, 1, Z)
                            q = torch.round(
                                xc[:, :, None] / scale_g[None, :, None] + zps_col
                            ).clamp(qmin, qmax)
                            x_deq = (q - zps_col) * scale_g[None, :, None]
                        e = xc[:, :, None] - x_deq  # [chunk, C, Z]
                        # Diagonal Gram: H_diag[c] * sum_n e_{n,c}^2
                        err_cz += gram_h_diag[:, None] * (e * e).sum(dim=0)  # [C, Z]

                    self._err_accumulator[:, gi, :] += err_cz

    @torch.no_grad()
    def compute_qparams(self):
        """Pick best (scale, zp) from the error accumulator (global),
        or fall back to the parent implementation for max_merge.

        When per-channel, the accumulator is [C, G, Z] and the best
        (scale, zp) is selected independently per channel.
        """
        if self.max_merge:
            return super().compute_qparams()

        assert self._err_accumulator is not None
        Z = self._grid_zps.shape[0]

        if self.channel_axis is None:
            # Per-tensor: accumulator [G, Z]
            best_flat = self._err_accumulator.argmin().item()
            best_g = best_flat // Z
            best_z = best_flat % Z

            best_scale = self._grid_scales[best_g].float()
            best_zp = self._grid_zps[best_z].to(torch.int)
        else:
            # Per-channel: accumulator [C, G, Z]
            C, G, _ = self._err_accumulator.shape
            best_flat = self._err_accumulator.reshape(C, -1).argmin(dim=1)  # [C]
            best_g = best_flat // Z  # [C]
            best_z = best_flat % Z   # [C]

            best_scale = self._grid_scales[
                torch.arange(C, device=best_g.device), best_g
            ].float()
            best_zp = self._grid_zps[best_z].to(torch.int)

        self._cached_scale = best_scale
        self._cached_zp = best_zp
        return best_scale, best_zp


