# Copyright IST-DASLab. 2025. (commit: 2d65066). GitHub repository.
# Retrieved from https://github.com/IST-DASLab/gptq. Licensed under the
# Apache License 2.0.

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

# https://github.com/IST-DASLab/gptq/blob/2d65066/quant.py

import torch


def quantize(x, scale, zero, maxq):
    if maxq < 0:
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)


def soft_quantize(x, scale, zero, maxq, temperature=0.01):
    """Smooth approximation of :func:`quantize` using sigmoid instead of round.

    Args:
        x:          Input tensor.
        scale:      Quantization scale.
        zero:       Zero-point.
        maxq:       Maximum quantized value.
        temperature: Width of the sigmoid transition band.  At
                     ``temperature=0.01`` the output differs from
                     :func:`quantize` only within ±0.01 of each half-integer
                     boundary; outside that band the match is exact.

    Returns:
        Dequantized tensor (smooth approximation).
    """
    if maxq < 0:
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
    # x / scale + zero → continuous value q
    q = x / scale + zero
    # Soft round: floor(q) + sigmoid((frac - 0.5) / temperature)
    floor_q = torch.floor(q)
    frac = q - floor_q
    q_soft = floor_q + torch.sigmoid((frac - 0.5) / temperature)
    q_soft = torch.clamp(q_soft, 0, maxq)
    return scale * (q_soft - zero)

def iterate_GPTQ(
    scale, zero, maxq, W, Hinv, max_num_of_iters=50, P=None, quantize_fn=None
):
    if quantize_fn is None:
        quantize_fn = quantize

    cur_weights = W.clone().to(Hinv.dtype)

    mults = torch.pow(torch.diag(Hinv), -1)
    Hinv_U = torch.triu(Hinv, diagonal=1)
    P_U = torch.triu(P, diagonal=1) if P is not None else None

    init_weights = W.clone()
    for i in range(max_num_of_iters):
        cur_Q = quantize(cur_weights, scale, zero, maxq)

        d_W = torch.mul((cur_weights - cur_Q), mults)
        cur_weights = init_weights - torch.matmul(d_W, Hinv_U)
        # GPTQv2: Apply P correction
        if P_U is not None:
            cur_weights += torch.matmul(cur_Q, P_U)
        del d_W, cur_Q
        d_W = cur_Q = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del init_weights
    init_weights = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cur_Q = quantize_fn(cur_weights, scale, zero, maxq)

    return cur_Q, cur_weights


def iterate_GPTQ_batched(
    scale_all,
    zero_all,
    maxq,
    W,
    Hinv,
    max_num_of_iters=50,
    P=None,
    chunk_size=8,
    quantize_fn=None,
):
    """Batched version of :func:`iterate_GPTQ` for grid-search parallelisation.

    Runs ``iterate_GPTQ`` for *many* (scale, zero) pairs simultaneously,
    exploiting the fact that ``W``, ``Hinv`` and ``P`` are shared across all
    grid points — only ``scale`` and ``zero`` differ.

    The grid dimension is processed in chunks of ``chunk_size`` to bound
    memory.  Within each chunk the iteration loop uses batched matmuls
    (``[chunk, rows, cols] @ [cols, cols]``) which PyTorch broadcasts over
    the batch dimension, keeping the GPU saturated.

    Args:
        scale_all:   ``[n_grid, channels, 1]`` per-grid-point scales.
        zero_all:    ``[n_grid, channels, 1]`` per-grid-point zeros.
        maxq:        Maximum quantized value (scalar).
        W:           ``[channels, columns]`` shared weight matrix.
        Hinv:        ``[columns, columns]`` shared inverse Hessian.
        max_num_of_iters: Number of GPTQ correction iterations.
        P:           Optional GPTQv2 correction matrix ``[columns, columns]``.
        chunk_size:  Number of grid points to process simultaneously.
        quantize_fn: Optional quantize function (defaults to :func:`quantize`).

    Returns:
        ``(cur_Q, cur_weights)`` where both have shape
        ``[n_grid, channels, columns]``.
    """
    if quantize_fn is None:
        quantize_fn = quantize

    n_grid = scale_all.shape[0]
    dev = W.device

    # Precompute shared quantities once (identical for every grid point)
    mults = torch.pow(torch.diag(Hinv), -1)  # [columns]
    Hinv_U = torch.triu(Hinv, diagonal=1)  # [columns, columns]
    P_U = torch.triu(P, diagonal=1) if P is not None else None  # [columns, columns]

    results_Q = []
    results_W = []

    for s in range(0, n_grid, chunk_size):
        e = min(s + chunk_size, n_grid)
        sc = scale_all[s:e]  # [chunk, channels, 1]
        zo = zero_all[s:e]  # [chunk, channels, 1]
        k = e - s

        # Expand W to [chunk, channels, columns] — shared across grid points
        cur_weights = W.unsqueeze(0).expand(k, -1, -1).clone().to(Hinv.dtype)
        init_weights = W.unsqueeze(0).expand(k, -1, -1).clone()

        for _ in range(max_num_of_iters):
            # quantize broadcasts scale/zero over [chunk, channels, columns]
            cur_Q = quantize_fn(cur_weights, sc, zo, maxq)

            # d_W: [chunk, channels, columns]  (mults broadcasts on last dim)
            d_W = (cur_weights - cur_Q) * mults

            # Batched matmul: [chunk, ch, cols] @ [cols, cols] → [chunk, ch, cols]
            cur_weights = init_weights - torch.matmul(d_W, Hinv_U)
            if P_U is not None:
                cur_weights += torch.matmul(cur_Q, P_U)

            del d_W, cur_Q

        # Final quantization with the (optionally) custom quantize_fn
        cur_Q = quantize_fn(cur_weights, sc, zo, maxq)

        results_Q.append(cur_Q)
        results_W.append(cur_weights)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return torch.cat(results_Q, dim=0), torch.cat(results_W, dim=0)

