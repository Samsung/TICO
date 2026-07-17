# Copyright IST-DASLab. 2025. (commit: 2d65066). GitHub repository.
# Retrieved from https://github.com/IST-DASLab/gptq. Licensed under the
# Apache License 2.0.

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

# https://github.com/IST-DASLab/gptq/blob/2d65066/gptq.py

import math
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tico.quantization.algorithm.gptq.quant import quantize, Quantizer
from tico.quantization.algorithm.gptq.utils import get_numerical_padding
from tico.quantization.algorithm.fpi_gptq.util import iterate_GPTQ

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def convtranspose2d_weights_to_conv2d_weights(layer, w) -> torch.Tensor:
    if layer.groups == 1:
        # the last two dimensions of w is (k_h, k_w) to get equivalent Conv2D we need to flip them to get `w_conv2D_equivalent_to_w[i, j] = w_conv[k_h - i - 1, k_w - j - 1]`
        # the first two dimensions of w is (input_channels, output_channels), so we need to transpose them as Conv2D weights should be in the (output_channels, input_channels) form
        # please see https://github.com/pytorch/pytorch/blob/d38164a545b4a4e4e0cf73ce67173f70574890b6/torch/nn/modules/conv.py#L1059-L1061 for additional info
        w_conv_transposed = w.transpose(1, 0).flip((-2, -1))
    else:
        # basically it's the same as for `layer.groups == 1` but groupwise
        in_channels, out_channels, kernel_h, kernel_w = layer.weight.shape
        out_channels *= layer.groups
        w_conv_transposed = torch.zeros(
            out_channels, in_channels // layer.groups, kernel_h, kernel_w
        )
        for i in range(0, layer.groups):
            w_conv_transposed[
                i
                * out_channels
                // layer.groups : (i + 1)
                * out_channels
                // layer.groups,
                :,
                :,
                :,
            ] = (
                w[
                    i
                    * in_channels
                    // layer.groups : (i + 1)
                    * in_channels
                    // layer.groups,
                    :,
                    :,
                    :,
                ]
                .transpose(1, 0)
                .flip((-2, -1))
            )

    return w_conv_transposed


def conv2d_weights_to_convtranspose2d_weights(orig_layer, w) -> torch.Tensor:
    # this is just an inverse of convtranspose2d_weights_to_conv2d_weights
    if orig_layer.groups > 1:
        in_channels, out_channels, _, _ = orig_layer.weight.shape
        out_channels *= orig_layer.groups
        w_conv_transposed = torch.zeros_like(orig_layer.weight)
        for i in range(0, orig_layer.groups):
            w_conv_transposed[
                i
                * in_channels
                // orig_layer.groups : (i + 1)
                * in_channels
                // orig_layer.groups,
                :,
                :,
                :,
            ] = (
                w[
                    i
                    * out_channels
                    // orig_layer.groups : (i + 1)
                    * out_channels
                    // orig_layer.groups,
                    :,
                    :,
                    :,
                ]
                .transpose(1, 0)
                .flip((-2, -1))
            )
    else:
        w_conv_transposed = w.transpose(1, 0).flip((-2, -1))

    return w_conv_transposed


def get_matmul_input_for_convtranspose2d(layer, inp):
    # Please see https://github.com/pytorch/pytorch/blob/d38164a545b4a4e4e0cf73ce67173f70574890b6/torch/nn/modules/conv.py#L996-L998 for padding
    strided_pad = (
        layer.dilation[0] * (layer.kernel_size[0] - 1) - layer.padding[0],
        layer.dilation[1] * (layer.kernel_size[1] - 1) - layer.padding[1],
    )

    # interleave input with zero rows and columns according to stride
    # Please see https://github.com/pytorch/pytorch/blob/d38164a545b4a4e4e0cf73ce67173f70574890b6/torch/nn/modules/conv.py#L991-L994 for more info
    inp_strided = torch.zeros(
        inp.shape[0],
        inp.shape[1],
        layer.stride[0] * (inp.shape[2] - 1) + 2 * strided_pad[0] + 1,
        layer.stride[1] * (inp.shape[3] - 1) + 2 * strided_pad[1] + 1,
        device=inp.device,
    )

    indices = torch.arange(0, inp.shape[2], device=inp.device)
    # insert original input values according to stride to meet https://github.com/pytorch/pytorch/blob/d38164a545b4a4e4e0cf73ce67173f70574890b6/torch/nn/modules/conv.py#L991-L994
    inp_strided[
        :,
        :,
        layer.stride[0] * indices + strided_pad[0],
        strided_pad[1] : -strided_pad[1] : layer.stride[1],
    ] = inp[:, :, indices, :]
    del inp
    inp = (
        inp_strided  # so the rest is just processing for Conv2D with transposed weights
    )

    # TODO reduce code duplication with Conv2D
    unfold = nn.Unfold(
        layer.kernel_size,
        dilation=layer.dilation,
        padding=(
            0,
            0,
        ),  # equivalent Conv2D has (0, 0) padding for input_strided as input
        stride=(1, 1),  # equivalent Conv2D has (1, 1) stride for input_strided as input
    )

    if layer.groups != 1:
        inp = inp.reshape(
            inp.size(0) * layer.groups,
            inp.size(1) // layer.groups,
            inp.shape[2],
            inp.shape[3],
        )  # inp.shape == (batch*groups, in_channels / groups, H, W) to meet Groupwise-wise Convolution, so that each group is colvolved with its own filter

    inp = unfold(inp).permute([1, 0, 2]).flatten(1)
    return inp


def estimate_max_singular_value(A, n_iter=30):
    n = A.shape[0]
    v = torch.randn(n, device=A.device, dtype=A.dtype)
    for _ in range(n_iter):
        v = A.T @ (A @ v)
        v = v / v.norm()
    return (A @ v).norm().item()

def estimate_sigma_min_lanczos(H, n_iter=15):
    """Estimate σ_min of SPD H via Lanczos. Only needs H @ v matmuls."""
    n = H.shape[0]
    device, dtype = H.device, H.dtype

    # Lanczos with full reorthogonalization
    V = torch.zeros(n, n_iter, device=device, dtype=dtype)
    alphas = torch.zeros(n_iter, device=device, dtype=dtype)
    betas = torch.zeros(n_iter, device=device, dtype=dtype)

    v = torch.randn(n, device=device, dtype=dtype)
    v = v / v.norm()
    V[:, 0] = v

    for j in range(n_iter):
        w = H @ V[:, j]                    # ← the only matmul with H
        alphas[j] = V[:, j] @ w
        w = w - alphas[j] * V[:, j]
        if j > 0:
            w = w - betas[j] * V[:, j-1]
        # Reorthogonalize against all previous
        w = w - V[:, :j+1] @ (V[:, :j+1].T @ w)
        if j < n_iter - 1:
            betas[j+1] = w.norm()
            if betas[j+1] < 1e-12:
                break
            V[:, j+1] = w / betas[j+1]

    # Build tridiagonal T and get its eigenvalues
    m = j + 1
    T = torch.diag(alphas[:m]) + torch.diag(betas[1:m], 1) + torch.diag(betas[1:m], -1)
    eigvals = torch.linalg.eigvalsh(T)
    return eigvals[0].item()  # ≈ σ_min(H)


def _build_block_tridiagonal(alphas, betas, k, b, device, dtype):
    """Build the (k*b × k*b) block-tridiagonal T from the first k Lanczos blocks.

    Args:
        alphas: list of k   diagonal  blocks (each b × b).
        betas:  list of k-1 off-diag blocks (each b × b).
        k:      number of completed Lanczos blocks.
        b:      block size.

    Returns:
        Dense symmetric block-tridiagonal matrix T.
    """
    dim_T = k * b
    T = torch.zeros(dim_T, dim_T, device=device, dtype=dtype)
    for j in range(k):
        s = j * b
        e = s + b
        T[s:e, s:e] = alphas[j]
        if j < k - 1:
            beta = betas[j]
            # T[j, j+1] = β_j^T  and  T[j+1, j] = β_j  (NOT the other way around)
            # because T_m = V_m^T H V_m is symmetric and
            #   (j, j+1) block = V_j^T H V_{j+1} = (V_{j+1}^T H V_j)^T = β_j^T
            #   (j+1, j) block = V_{j+1}^T H V_j = β_j
            T[s:e, e : e + b] = beta.T
            T[e : e + b, s:e] = beta
    return T


def estimate_sigma_min_randomized_blocked_lanczos(
    H,
    block_size: int = 4,
    n_iter: int = 15,
    use_sketch: bool = False,
    sketch_dim: int | None = None,
    tol: float = 1e-4,
    check_interval: int = 3,
):
    """Estimate σ_min of SPD H via randomized blocked Lanczos.

    Generalises the single-vector Lanczos (:func:`estimate_sigma_min_lanczos`)
    to a *block* of ``block_size`` vectors and optionally applies a random
    Gaussian sketch to reduce the effective dimension before running the
    iteration.

    Advantages over single-vector Lanczos:
      * BLAS-3 ``H @ V`` matmul (better cache / GPU utilisation).
      * Captures clustered / degenerate small eigenvalues more reliably.
      * Random sketching (when ``use_sketch=True``) reduces cost for very
        large *n*.

    Convergence-based adaptive stopping: the smallest Ritz value (σ_min
    estimate) is monitored every ``check_interval`` iterations starting from
    iteration 2.  When its relative change drops below ``tol``, the iteration
    stops early — saving time on well-conditioned matrices and avoiding
    wasted iterations on ill-conditioned ones where the estimate plateaus.

    Args:
        H:              Symmetric positive-definite matrix (n × n).
        block_size:     Number of vectors per block (b).
        n_iter:         Max number of block Lanczos iterations (total Krylov
                        dim = block_size × n_iter).
        use_sketch:     If True, apply a Gaussian random sketch to reduce
                        dimension.
        sketch_dim:     Sketch dimension l (only used when use_sketch=True).
                        Defaults to min(n, 4 * block_size * n_iter).
        tol:            Relative tolerance for convergence.  Iteration stops
                        when |Δσ_min| / (|σ_min| + ε) < tol.
        check_interval: Check convergence every N iterations (starting at j=2).

    Returns:
        Estimated σ_min(H) (float).
    """
    n = H.shape[0]
    device, dtype = H.device, H.dtype
    b = min(block_size, n)

    # ------------------------------------------------------------------ #
    # Optional random sketching:  project H → S H Sᵀ  (l × l, l ≪ n)
    # ------------------------------------------------------------------ #
    if use_sketch:
        l = sketch_dim if sketch_dim is not None else min(n, 4 * b * n_iter)
        l = min(l, n)
        # Gaussian random sketch matrix (l × n)
        S = torch.randn(l, n, device=device, dtype=dtype) / math.sqrt(l)
        # Sketched operator: H_s = S (H Sᵀ)   — only matmuls with H
        HS = H @ S.T                      # (n × l)
        H_s = S @ HS                      # (l × l)  — symmetric PSD
        # Symmetrise to kill round-off asymmetry
        H_s = 0.5 * (H_s + H_s.T)
        # Recurse on the smaller sketched operator (no further sketching)
        return estimate_sigma_min_randomized_blocked_lanczos(
            H_s,
            block_size=b,
            n_iter=n_iter,
            use_sketch=False,
            tol=tol,
            check_interval=check_interval,
        )

    # ------------------------------------------------------------------ #
    # Randomized blocked Lanczos with full reorthogonalisation
    # ------------------------------------------------------------------ #
    # Random start block → orthonormalise
    V_block = torch.randn(n, b, device=device, dtype=dtype)
    V_block, _ = torch.linalg.qr(V_block)          # columns orthonormal

    # Storage for all Krylov basis vectors:  (n × (n_iter * b))
    m_total = n_iter * b
    V_all = torch.zeros(n, m_total, device=device, dtype=dtype)
    V_all[:, :b] = V_block

    # Block tridiagonal pieces
    #   α[j] : b × b diagonal block   (j = 0 .. n_iter-1)
    #   β[j] : b × b off-diagonal block (j = 0 .. n_iter-2)
    alphas = [None] * n_iter
    betas = [None] * (n_iter - 1) if n_iter > 1 else []

    prev_block = None      # V_{j-1}
    prev_beta = None       # β_{j-1}

    actual_iter = 0
    prev_sigma_min = None   # for convergence-based adaptive stopping
    for j in range(n_iter):
        # W = H @ V_j   — single BLAS-3 matmul
        W = H @ V_block                              # (n × b)

        # α_j = V_jᵀ W
        alpha_j = V_block.T @ W                       # (b × b)
        alpha_j = 0.5 * (alpha_j + alpha_j.T)         # symmetrise
        alphas[j] = alpha_j

        # W -= V_j @ α_j
        W -= V_block @ alpha_j

        # W -= V_{j-1} @ β_{j-1}ᵀ
        if prev_block is not None and prev_beta is not None:
            W -= prev_block @ prev_beta.T

        # Full reorthogonalisation against ALL previous basis vectors
        if j > 0:
            V_prev = V_all[:, : j * b]                # (n × j*b)
            # Modified Gram-Schmidt (two passes for stability)
            for _pass in range(2):
                W -= V_prev @ (V_prev.T @ W)

        # QR of W → V_{j+1}, β_j
        if j < n_iter - 1:
            Q, R = torch.linalg.qr(W)                 # Q: (n × b), R: (b × b)
            # Detect rank deficiency (R has near-zero diagonal)
            diag_r = torch.diagonal(R)
            if diag_r.abs().min() < 1e-12:
                # Block has deflated — stop early
                actual_iter = j + 1
                break
            betas[j] = R[:b, :b]                      # upper-triangular β_j
            V_all[:, (j + 1) * b : (j + 2) * b] = Q
            prev_block = V_block
            prev_beta = betas[j]
            V_block = Q
            actual_iter = j + 2
        else:
            actual_iter = j + 1

        # --- Convergence check: monitor smallest Ritz value ---
        # Every check_interval iterations (starting at j=2), build a
        # partial T from the completed blocks and compute its smallest
        # eigenvalue.  Stop early if the relative change < tol.
        if j >= 2 and (j - 2) % check_interval == 0:
            k_check = j + 1   # number of completed diagonal blocks
            T_check = _build_block_tridiagonal(
                alphas, betas, k_check, b, device, dtype
            )
            eigvals_check = torch.linalg.eigvalsh(T_check)
            sigma_min_check = max(0.0, eigvals_check[0].item())
            if prev_sigma_min is not None:
                rel_change = abs(sigma_min_check - prev_sigma_min) / (
                    abs(sigma_min_check) + 1e-30
                )
                if rel_change < tol:
                    actual_iter = k_check
                    break
            prev_sigma_min = sigma_min_check

    # ------------------------------------------------------------------ #
    # Build block-tridiagonal T  (actual_iter * b  ×  actual_iter * b)
    # and compute its smallest eigenvalue
    # ------------------------------------------------------------------ #
    k = actual_iter
    T = _build_block_tridiagonal(alphas, betas, k, b, device, dtype)

    eigvals = torch.linalg.eigvalsh(T)
    # Ritz values can slightly undershoot the true smallest eigenvalue;
    # clamp to 0 since H is SPD (σ_min ≥ 0).
    return max(0.0, eigvals[0].item())  # ≈ σ_min(H)



def cg(matvec, b, max_iter=50, tol=1e-8):
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rs_old = (r * r).sum()
    for _ in range(max_iter):
        Ap = matvec(p)                    # ← the only matmul
        alpha = rs_old / (p * Ap).sum()
        x += alpha * p
        r -= alpha * Ap
        rs_new = (r * r).sum()
        if rs_new.sqrt() < tol:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x

def estimate_min_singular_value(A, max_iter=20, tol=1e-5):
    """
    Estimates the minimum singular value of matrix A 
    using inverse iteration
    """
    device = A.device
    n = A.shape[-1]
    
    # Initialize random vector
    v = torch.randn(n, 1, device=device, dtype=A.dtype)
    v = v / torch.norm(v)
    
    AtA = A.T @ A
    def AtA_matmul(x):
        return AtA @ x
    
    for _ in range(max_iter):
        # Solve (A^T A) x = v
        x = torch.linalg.lstsq(AtA, v).solution
        #x = cg(AtA_matmul, v, max_iter=100) #slow convergence on ill-conditioned matrices
        # Power iteration step
        v_new = x / torch.norm(x)
        
        # Check convergence
        if torch.norm(v_new - v) < tol:
            break
        v = v_new
        
    # Rayleigh quotient for the eigenvalue of (A^T A)^-1
    AtA_v = AtA @ v
    min_eigval = (v.T @ AtA_v).item()
    return torch.sqrt(torch.tensor(min_eigval))

def estimate_cond(A):
    fast = True
    if fast is True:
        s_max = estimate_max_singular_value(A)
        #s_min = estimate_min_singular_value(A) #slow
        #s_min = estimate_sigma_min_lanczos(A, n_iter=150)
        s_min = estimate_sigma_min_randomized_blocked_lanczos(A, block_size=8, n_iter=100)
        cond_ref = torch.linalg.cond(A)
        return s_max / s_min
    
    return torch.linalg.cond(A)

class GPTQ:
    """
    GPTQ quantization class supporting both standard GPTQ and GPTQv2.
    """
    def __init__(self, layer, **kwargs):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            W = W.flatten(1)  # reshaped to matrix (OUT_channels x the_rest)
        elif isinstance(self.layer, nn.ConvTranspose2d):
            W = convtranspose2d_weights_to_conv2d_weights(self.layer, W)
            W = W.flatten(1)

        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H: Optional[torch.Tensor] = torch.zeros(
            (self.columns, self.columns), device=self.dev
        )
        self.nsamples = 0
        self.quantizer: Quantizer = Quantizer()
        # GPTQv2: for tracking FP vs quantized input difference
        self.dXXT: Optional[torch.Tensor] = None
        self.native_inp: Optional[List[torch.Tensor]] = None
        self.kwargs = kwargs
        # Track batch index for weighted Hessian accumulation
        self.batch_id = 0
        # Sample weights for weighted Hessian accumulation
        self.weights: Optional[List[float]] = None
        # Hessian saturation tracking via effective rank (participation ratio)
        # r_eff = trace(H)² / ||H||_F²  ∈ [1, d]
        # r_eff measures how many eigenvalues are "active". When the relative
        # change in r_eff between batches drops below the threshold, the Hessian
        # shape has converged and more samples won't change GPTQ.
        # Unlike the previous Frobenius-norm approach, r_eff is dimension-agnostic:
        # the relative change |Δr_eff| / r_eff is comparable across model sizes.
        self._prev_r_eff: Optional[float] = None
        self.h_saturation: Optional[float] = None
        # dXXT saturation tracking (GPTQv2 only)
        self._prev_r_eff_dXXT: Optional[float] = None
        self.dXXT_saturation: Optional[float] = None
        # Per-matrix saturated flags
        self._h_saturated: bool = False
        self._dXXT_saturated: bool = False
        # Early stopping: when saturated, add_batch skips further processing
        self.saturated: bool = False
        self.saturation_threshold: Optional[float] = None
        self.saturation_min_batches: int = 4

    def add_batch(self, inp, out=None):
        """
        Add a batch of inputs to the Hessian approximation.
        
        For GPTQv2, also processes native_inp (FP inputs) and computes dXXT.
        Uses internal self.weights and self.batch_id for weighted Hessian accumulation.
        
        Args:
            inp: Input tensor
            out: Output tensor (unused)
        """
        # Early exit: Hessian has saturated, skip further processing
        if self.saturated:
            return

        # Apply per-sample weights before reshaping: multiply inp by sqrt(weight)
        # so that (inp * sqrt(w)) @ (inp * sqrt(w)).T = w * inp @ inp.T
        if self.weights is not None and isinstance(self.weights[self.batch_id], torch.Tensor):
            # weight shape: [batch_size], inp shape: [batch, ...]
            # broadcast sqrt(weight) to all dims except batch
            sqrt_w = self.weights[self.batch_id].sqrt().view(-1, *[1]*(inp.dim()-1)).to(inp.device)
            inp = inp * sqrt_w

        # Process native input for GPTQv2 (before reshaping inp)
        native_inp_processed = None
        if hasattr(self, "native_inp") and self.native_inp is not None and len(self.native_inp) > 0:
            native = self.native_inp.pop(0)
            if native is not None:
                native_inp_processed = native
            # Apply same per-sample weighting to native_inp
            if self.weights is not None and isinstance(self.weights[self.batch_id], torch.Tensor) and native_inp_processed is not None:
                sqrt_w = self.weights[self.batch_id].sqrt().view(-1, *[1]*(native_inp_processed.dim()-1))
                native_inp_processed = native_inp_processed * sqrt_w

            if len(native_inp_processed.shape) == 2:
                native_inp_processed = native_inp_processed.unsqueeze(0)
            if isinstance(self.layer, nn.Linear):
                if len(native_inp_processed.shape) > 2:
                    native_inp_processed = native_inp_processed.reshape((-1, native_inp_processed.shape[-1]))
                native_inp_processed = native_inp_processed.t()
        
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) > 2:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            padding = get_numerical_padding(self.layer)
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=padding,
                stride=self.layer.stride,
            )

            if self.layer.groups != 1:
                # the idea behind conversion of depthwise convolution to matmul is described here
                # https://discuss.pytorch.org/t/conv1d-implementation-using-torch-nn-functional-unfold/109643/2
                # although depthwise convolution is equal to a set of MatMuls
                # (please note `w.view(1, groups, out_channels // groups, -1)` in the reference above is not just w.flatten(1))
                # we can approximate groupwise Hessians with their mean
                # so that we will have just a single Hessian and the usual GPTQ applies
                inp = inp.reshape(
                    inp.size(0) * self.layer.groups,
                    inp.size(1) // self.layer.groups,
                    inp.shape[2],
                    inp.shape[3],
                )  # inp.shape == (batch*groups, in_channels / groups, H, W) to meet Groupwise-wise Convolution, so that each group is colvolved with its own filter

            inp = unfold(
                inp
            )  # inp.shape == (batch*groups, k_h*k_w*in_channels / groups, flattened_patches)
            inp = inp.permute(
                [1, 0, 2]
            )  # inp.shape == (k_h*k_w*in_channels / groups, batch * groups, flattened_patches)
            inp = inp.flatten(
                1
            )  # inp.shape == (k_h*k_w*in_channels / groups, batch * groups * flattened_patches)
            # so inp.matmul(inp.t()).shape == (k_x*k_y*in_channels / groups, k_x*k_y*in_channels / groups) == W.flatten(1)

        if isinstance(self.layer, nn.Conv1d):
            # nn.Conv1d is basically the same as nn.Conv2d so we can use the same idea as for nn.Conv2d
            # TODO reduce code duplication
            # represent conv1d as conv2d(1, k) on reshaped_input(batch, in_channels, 1, L)
            unfold = nn.Unfold(
                (1, self.layer.kernel_size[0]),
                dilation=(1, self.layer.dilation[0]),
                padding=(0, self.layer.padding[0]),
                stride=(1, self.layer.stride[0]),
            )
            if self.layer.groups != 1:
                # please see Conv2D for additional info
                inp = inp.reshape(
                    inp.size(0) * self.layer.groups,
                    inp.size(1) // self.layer.groups,
                    inp.shape[2],
                )  # inp.shape == (batch*groups, in_channels / groups, L) to meet Groupwise-wise Convolution, so that each group is colvolved with its own filter

            inp = inp.unsqueeze(
                -2
            )  # (batch*groups, in_channels / groups, L)->(batch*groups, in_channels / groups, 1, L), valid for Conv2D
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)

        if isinstance(self.layer, nn.ConvTranspose2d):
            inp = get_matmul_input_for_convtranspose2d(self.layer, inp)

        if isinstance(self.layer, nn.Conv3d):
            # adapted from https://discuss.pytorch.org/t/manual-implementation-of-unrolled-3d-convolutions/91021
            assert (
                self.layer.groups == 1
            )  # depthwise/groupwise are not supported currently
            assert all(dilation == 1 for dilation in self.layer.dilation)

            # inp is assumed to be (N, C_in, H, W, D)
            padding = get_numerical_padding(self.layer)
            if isinstance(padding, int):
                padding = (padding, padding, padding)
            if not all(item == 0 for item in padding):
                inp = F.pad(
                    inp,
                    pad=(
                        padding[2],
                        padding[2],
                        padding[1],
                        padding[1],
                        padding[0],
                        padding[0],
                    ),
                    mode="constant",
                    value=0,
                )
            krn_size = self.layer.kernel_size
            stride = self.layer.stride
            inp = (
                inp.unfold(2, krn_size[0], stride[0])
                .unfold(3, krn_size[1], stride[1])
                .unfold(4, krn_size[2], stride[2])
            )  # inp.shape = (N, C_in, ..patches... , krn_size[0], krn_size[1], krn_size[2])
            inp = inp.reshape(
                inp.shape[0], inp.shape[1], -1, krn_size[0] * krn_size[1] * krn_size[2]
            )  # inp.shape = (N, C_in, num_patches, krn_size[0] * krn_size[1] * krn_size[2])
            inp = inp.permute(
                [0, 2, 1, 3]
            )  # inp.shape = (N, num_patches, C_in, krn_size[0] * krn_size[1] * krn_size[2])
            inp = inp.reshape(
                inp.shape[0] * inp.shape[1], inp.shape[2] * inp.shape[3]
            ).T  # inp.shape =(C_in * krn_size[0] * krn_size[1] * krn_size[2], N * num_patches)

        self.nsamples += tmp
        inp = inp.double()
        
        # Get weight from internal weights list using batch_id
        weight = 1.0
        if self.weights is not None and not isinstance(self.weights[self.batch_id], torch.Tensor):
            weight = self.weights[self.batch_id]
            
        # Scale Hessian contribution by weight
        self.H += weight * inp.matmul(inp.t()).to(device=self.H.device, dtype=self.H.dtype)  # type: ignore[union-attr]
        # GPTQv2: Compute dXXT using native (FP) vs processed input difference
        if native_inp_processed is not None:
            if self.dXXT is None:
                self.dXXT = torch.zeros_like(self.H)
            
            native_inp_processed = native_inp_processed.double()
            dX = native_inp_processed.to(inp.device) - inp
            # Also scale dXXT by weight
            self.dXXT += weight * dX.matmul(inp.t()).float()
            del native, native_inp_processed
            native = native_inp_processed = None

        # --- Hessian saturation tracking via effective rank (participation ratio) ---
        # r_eff = trace(H)² / ||H||_F²  ∈ [1, d]
        # r_eff is the "participation ratio" — it counts how many eigenvalues
        # of H are effectively non-zero.  It is cheap to compute (one BLAS-3
        # Frobenius norm + one trace, both O(d²) but negligible vs. the
        # Hessian accumulation itself) and, crucially, the *relative* change
        # |Δr_eff| / r_eff is dimension-agnostic: the same threshold works
        # for models of different sizes because r_eff is always in [1, d].
        #
        # When the relative change in r_eff drops below the threshold, the
        # Hessian's eigenvalue distribution has converged — more calibration
        # samples won't meaningfully change the GPTQ solution.
        trace_H = self.H.trace().item()
        if self.saturation_threshold is not None and trace_H > 0:
            frob_H = self.H.float().norm().item()
            r_eff = (trace_H * trace_H) / (frob_H * frob_H + 1e-9)
            if self._prev_r_eff is not None:
                self.h_saturation = abs(r_eff - self._prev_r_eff) / (r_eff + 1e-30)
          #      print(
          #          f"  [GPTQ sat] layer={self.layer.__class__.__name__} "
          #          f"batch={self.batch_id} r_eff={r_eff:.4f} "
          #          f"h_saturation={self.h_saturation:.6e}"
          #      )
                # Check saturation threshold for H
                if (self.batch_id + 1 >= self.saturation_min_batches
                    and self.h_saturation < self.saturation_threshold
                ):
                    if not self._h_saturated:
                        self._h_saturated = True
                       #s print(
                       #s     f"  [GPTQ sat] layer={self.layer.__class__.__name__} "
                       #s     f"H saturated at batch={self.batch_id}, "
                       #s     f"r_eff={r_eff:.4f}"
                       #s )
            self._prev_r_eff = r_eff

        # --- dXXT saturation tracking (GPTQv2 only) ---
        # Same effective-rank metric applied to dXXT.  When GPTQv2 is enabled,
        # both H and dXXT must converge before early-stopping.
        if self.dXXT is not None and self.saturation_threshold is not None:
            trace_dXXT = self.dXXT.trace().item()
            if trace_dXXT > 0:
                frob_dXXT = self.dXXT.float().norm().item()
                r_eff_dXXT = (trace_dXXT * trace_dXXT) / (frob_dXXT * frob_dXXT + 1e-9)
                if self._prev_r_eff_dXXT is not None:
                    self.dXXT_saturation = abs(r_eff_dXXT - self._prev_r_eff_dXXT) / (r_eff_dXXT + 1e-30)
                 #   print(
                 #       f"  [GPTQ sat] layer={self.layer.__class__.__name__} "
                 #       f"batch={self.batch_id} r_eff_dXXT={r_eff_dXXT:.4f} "
                 #       f"dXXT_saturation={self.dXXT_saturation:.6e}"
                 #   )
                    if (self.batch_id + 1 >= self.saturation_min_batches
                        and self.dXXT_saturation < self.saturation_threshold
                    ):
                        if not self._dXXT_saturated:
                            self._dXXT_saturated = True
                            #print(
                            #    f"  [GPTQ sat] layer={self.layer.__class__.__name__} "
                            #    f"dXXT saturated at batch={self.batch_id}, "
                            #    f"r_eff_dXXT={r_eff_dXXT:.4f}"
                            #)
                self._prev_r_eff_dXXT = r_eff_dXXT

        # Set saturated when both H and dXXT (if used) have saturated
        if self._h_saturated:
            if self.dXXT is not None:
                self.saturated = self._dXXT_saturated
            else:
                self.saturated = True
            if self.saturated:
                print(
                    f"  [GPTQ sat] layer={self.layer.__class__.__name__} "
                    f"SATURATED at batch={self.batch_id}, stopping collection"
                )

        self.batch_id += 1

    def _adaptive_percdamp(
        self,
        H: torch.Tensor,
        user_percdamp: float,
        cond_threshold_good: float,
        verbose: bool,
    ) -> float:
        """Compute adaptive percdamp based on the Hessian condition number.

        Uses a piecewise-linear rule with an iterative binary-search fallback.
        Falls back to ``user_percdamp`` if the condition number is nan/inf.

        Note: callers should wrap this in a try/except to also fall back to
        ``user_percdamp`` in case of any exception.

        Args:
            H:                   Hessian matrix (double precision, pre-damping).
            user_percdamp:       User-provided damping factor to fall back to.
            cond_threshold_good: Condition number threshold below which minimal
                                 damping is used.
            verbose:             Whether to print diagnostic information.

        Returns:
            Selected percdamp value.
        """
        # Parameters for adaptive percdamp
        COND_THRESHOLD_GOOD = cond_threshold_good      # Below: use minimal damping
        COND_THRESHOLD_HIGH = 100000     # Above: use user percdamp
        COND_TARGET_MAX = COND_THRESHOLD_GOOD          # Maximum allowed condition number after damping
        MIN_PERCDAMP = 1e-06             # Minimal damping for good matrices
        MAX_PERCDAMP = 0.5               # Maximum damping for binary search

        # Define diag before use
        diag = torch.arange(self.columns, device=self.dev)
        diag_mean = torch.mean(torch.diag(H)).item()

        # Compute condition number of H (before damping)
        cond_H = estimate_cond(H)

        if math.isnan(cond_H) or math.isinf(cond_H):
            if verbose:
                print(
                    f"adaptive_percdamp: cond_H is {cond_H}, "
                    f"using user_percdamp={user_percdamp:.6f}"
                )
            return user_percdamp

        # Determine initial percdamp using piecewise rule
        if cond_H > COND_THRESHOLD_HIGH:
            # Extremely high condition number: use user-provided percdamp
            percdamp = user_percdamp
        elif cond_H < COND_THRESHOLD_GOOD:
            # Good matrices: use minimal damping
            percdamp = MIN_PERCDAMP
        else:
            # Between: linear interpolation between MIN_PERCDAMP and user_percdamp
            ratio = (cond_H - COND_THRESHOLD_GOOD) / (
                COND_THRESHOLD_HIGH - COND_THRESHOLD_GOOD
            )
            percdamp = MIN_PERCDAMP + (user_percdamp - MIN_PERCDAMP) * ratio

        # Apply damping and verify condition number
        damp = percdamp * diag_mean
        H_test = H.clone()
        H_test[diag, diag] += damp
        cond_after_damp = estimate_cond(H_test)

        # Binary search fallback if condition number still too high
        if cond_after_damp > COND_TARGET_MAX:
            low, high = MIN_PERCDAMP, MAX_PERCDAMP
            for _ in range(10):  # Max iterations for binary search
                mid = (low + high) / 2
                damp_test = mid * diag_mean
                H_test = H.clone()
                H_test[diag, diag] += damp_test
                cond_test = estimate_cond(H_test)

                if cond_test > COND_TARGET_MAX:
                    low = mid  # Need more damping
                else:
                    high = mid  # Can reduce damping

            percdamp = high

        if verbose:
            print(
                f"adaptive_percdamp: initial cond={cond_H:.2e}, "
                f"selected percdamp={percdamp:.6f}"
            )

        return percdamp

    def fasterquant(
        self,
        blocksize=128,
        percdamp=0.01,
        groupsize=-1,
        actorder=False,
        static_groups=False,
        verbose=False,
        just_quantize=False,
        adaptive_percdamp=False,
        cond_threshold_good=100000.0,
        use_iterate=False,
    ):
        """
        Perform GPTQ quantization.
        
        Args:
            blocksize: Block size for GPTQ
            percdamp: Damping factor for Hessian
            groupsize: Group size for groupwise quantization (-1 for no grouping)
            actorder: Whether to use activation ordering
            static_groups: Whether to use static groups
            verbose: Whether to print verbose output
            just_quantize: If True, only quantize weights without GPTQ optimization
            adaptive_percdamp: Whether to use adaptive percdamp based on condition number
            cond_threshold_good: Condition number threshold for good matrices in adaptive percdamp
        """
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            W = W.flatten(1)  # reshaped to matrix (OUT_channels x the_rest)
            if self.quantizer.sensitivity is not None:
                self.quantizer.sensitivity = self.quantizer.sensitivity.flatten(1)
        elif isinstance(self.layer, nn.ConvTranspose2d):
            W = convtranspose2d_weights_to_conv2d_weights(self.layer, W)
            conv2d_shape = W.shape
            W = W.flatten(1)  # reshaped to matrix (OUT_channels x the_rest)
            if self.quantizer.sensitivity is not None:
                self.quantizer.sensitivity = convtranspose2d_weights_to_conv2d_weights(
                    self.layer, self.quantizer.sensitivity
                )
                self.quantizer.sensitivity = self.quantizer.sensitivity.flatten(1)

        W = W.float()
        user_percdamp = percdamp  # save before adaptive_percdamp may modify it
        tick = time.time()
        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        H = self.H
        del self.H
        assert isinstance(H, torch.Tensor)
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        
        # GPTQv2: Zero out dead elements in dXXT
        if self.dXXT is not None:
            self.dXXT[:, dead] = 0

        if groupsize != -1 and self.quantizer.mse in {"mse_for_gptq", "smse_for_gptq"}:
            raise ValueError(
                "GPTQ-adjusted MSE currently does not support groupsize != -1"
            )

        if static_groups:
            import copy

            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i : (i + groupsize)], weight=True)
                groups.append(quantizer)

        perm = None
        invperm = None
        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)
            if self.dXXT is not None:
                self.dXXT = self.dXXT[perm][:, perm]

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        H = H.double()
        if verbose:
            cond_number = torch.linalg.cond(H)
            print("condition number init %.2e" % cond_number.item())
        
        # Adaptive percdamp: adjust damping based on Hessian condition number
        if adaptive_percdamp:
            try:
                percdamp = self._adaptive_percdamp(
                    H, user_percdamp, cond_threshold_good, verbose
                )
            except Exception as e:
                if verbose:
                    print(
                        f"adaptive_percdamp: exception {e}, "
                        f"falling back to user_percdamp={user_percdamp:.6f}"
                    )
                percdamp = user_percdamp

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        if verbose:
            cond_number = torch.linalg.cond(H)
            print("condition number damp %.2e" % cond_number.item())
            
        try:
            H = torch.linalg.cholesky(H)
            assert isinstance(H, torch.Tensor)
            H = torch.cholesky_inverse(H)
            H = torch.linalg.cholesky(H, upper=True).float()
            Hinv = H
        except torch._C._LinAlgError:
            if verbose:
                print(
                    f"Cholesky failed with percdamp={percdamp:.6f}, "
                    f"retrying with user_percdamp={user_percdamp:.6f}"
                )
            # Undo current damping and apply user_percdamp
            H[diag, diag] -= damp
            damp = user_percdamp * torch.mean(torch.diag(H))
            H[diag, diag] += damp
            # This will raise if it still fails — no further fallback
            H = torch.linalg.cholesky(H)
            assert isinstance(H, torch.Tensor)
            H = torch.cholesky_inverse(H)
            H = torch.linalg.cholesky(H, upper=True).float()
            Hinv = H
        
        # GPTQv2: Compute P correction matrix from dXXT
        P = None
        if self.dXXT is not None:
            alpha = 0.25
            P = alpha * ((self.dXXT @ Hinv.T).triu_(diagonal=1)) @ Hinv

        self.quantizer.update(W, Hinv, perm, P=P)
        #Q = self.quantizer.quantize(W)

        assert isinstance(Hinv, torch.Tensor)
        
        if use_iterate:
            # Use iterate_GPTQ approach (same as fpi_gptq.py)
            Q, W = iterate_GPTQ(
                self.quantizer.scale,
                self.quantizer.zero,
                self.quantizer.maxq,
                W,
                Hinv=Hinv,
                max_num_of_iters=min(50, self.columns),
                P=P,
            )
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if verbose:
                print("time %.2f" % (time.time() - tick))
                Losses = 0.5 * ((Q - W) / torch.diag(Hinv)) ** 2
                print("error", torch.sum(Losses).item())
        else:
         # Original block-based GPTQ loop
         for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            P1 = P[i1:i2, i1:i2] if P is not None else None

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(
                                W[:, (i1 + i) : (i1 + i + groupsize)], weight=True
                            )
                    else:
                        idx: torch.Tensor | int = i1 + i
                        if actorder:
                            idx = perm[idx]  # type: ignore[index]
                        self.quantizer = groups[idx // groupsize]

                q = quantize(
                    w.unsqueeze(1),
                    self.quantizer.scale,
                    self.quantizer.zero,
                    self.quantizer.maxq,
                ).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d**2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                # GPTQv2: Apply P correction
                if P1 is not None:
                    W1[:, i:] += w.unsqueeze(1).matmul(P1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
            # GPTQv2: Apply P correction to remaining weights
            if P is not None:
                W[:, i2:] += W1.matmul(P[i1:i2, i2:])

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if verbose:
            print("time %.2f" % (time.time() - tick))
            print("error", torch.sum(Losses).item())

        if actorder:
            Q = Q[:, invperm]

        if isinstance(self.layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if groupsize == -1:  # TODO support groupsize != -1
                Q[:, dead] = quantize(
                    self.layer.weight.flatten(1)[:, dead],
                    self.quantizer.scale,
                    self.quantizer.zero,
                    self.quantizer.maxq,
                )
        elif isinstance(self.layer, nn.ConvTranspose2d):
            if groupsize == -1:  # TODO support groupsize != -1
                Q[:, dead] = quantize(
                    convtranspose2d_weights_to_conv2d_weights(
                        self.layer, self.layer.weight.data
                    ).flatten(1)[:, dead],
                    self.quantizer.scale,
                    self.quantizer.zero,
                    self.quantizer.maxq,
                )
        else:
            if groupsize == -1:  # TODO support groupsize != -1
                Q[:, dead] = quantize(
                    self.layer.weight[:, dead],
                    self.quantizer.scale,
                    self.quantizer.zero,
                    self.quantizer.maxq,
                )

        assert (
            groupsize == -1 or torch.sum(dead) == 0
        )  # TODO `dead` elements should be RTN quantized for groupwise

        if isinstance(self.layer, nn.ConvTranspose2d):
            Q_conv2d = Q.reshape(conv2d_shape).to(self.layer.weight.data.dtype)
            self.layer.weight.data = conv2d_weights_to_convtranspose2d_weights(
                self.layer, Q_conv2d
            )
        else:
            self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(
                self.layer.weight.data.dtype
            )

    def free(self):
        self.H = None
        self.Losses = None
        self.Trace = None
        self.dXXT = None
        self._prev_r_eff = None
        self._prev_r_eff_dXXT = None
        self.native_inp = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
