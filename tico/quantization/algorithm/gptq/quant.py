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

# https://github.com/IST-DASLab/gptq/blob/2d65066/quant.py

import torch
import torch.nn as nn

from tico.quantization.algorithm.fpi_gptq.util import iterate_GPTQ


def quantize(x, scale, zero, maxq):
    if maxq < 0:
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)


class Quantizer(nn.Module):
    def __init__(self, shape=1):
        super(Quantizer, self).__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(shape))
        self.register_buffer("zero", torch.zeros(shape))

    def configure(
        self,
        bits,
        perchannel=False,
        sym=True,
        mse=None,
        norm=2.4,
        grid=100,
        maxshrink=0.8,
        trits=False,
        sensitivity=None,
    ):
        self.maxq = torch.tensor(2**bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        self.sensitivity = sensitivity
        if trits:
            self.maxq = torch.tensor(-1)

    def _prepare_tensor(self, x, weight=False):
        """Prepare tensor for quantization by flattening according to per-channel setting.

        Args:
            x: Input tensor to prepare
            weight: Whether the tensor is a weight (affects flattening for activations)

        Returns:
            Tuple of (prepared tensor, original shape)
        """
        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)
        return x, shape

    def _compute_scale_zero_bounds(self, x):
        """Compute scale and zero bounds from tensor values.

        Args:
            x: Prepared tensor (flattened according to per-channel setting)

        Returns:
            Tuple of (scale, zero, xmin, xmax) computed from tensor bounds
        """
        dev = x.device
        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        if self.maxq < 0:
            scale = xmax
            zero = xmin
        else:
            scale = (xmax - xmin) / self.maxq
            if self.sym:
                zero = torch.full_like(scale, (self.maxq + 1) / 2)  # type: ignore[arg-type]
            else:
                zero = torch.round(-xmin / scale)

        return scale, zero, xmin, xmax

    def _reshape_scale_zero(self, shape, weight=False):
        """Reshape scale and zero tensors according to the original tensor shape.

        Args:
            shape: Original tensor shape before preparation
            weight: Whether the tensor is a weight (affects reshape for activations)
        """
        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)  # type: ignore[has-type]
            self.zero = self.zero.reshape(shape)  # type: ignore[has-type]
            return

        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        elif len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1))
        elif len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def _expand_for_per_tensor(self, shape, weight=False):
        """Expand scale and zero for per-tensor quantization.

        Args:
            shape: Original tensor shape before preparation
            weight: Whether the tensor is a weight
        """
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            assert isinstance(tmp, int)
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

    def find_params(self, x, weight=False):
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        x, shape = self._prepare_tensor(x, weight)

        self.scale, self.zero, xmin, xmax = self._compute_scale_zero_bounds(x)

        if (
            self.mse is not None
            and self.mse != "smse_for_gptq"
            and self.mse != "mse_for_gptq"
        ):
            self._optimize_mse(x, xmin, xmax)

        self._expand_for_per_tensor(shape, weight)
        self._reshape_scale_zero(shape, weight)

    def _compute_shrink_params(self, p, xmin, xmax):
        """Compute scale and zero for a shrink factor p.

        Args:
            p: Shrink factor (1 - i / grid)
            xmin: Minimum values per channel
            xmax: Maximum values per channel

        Returns:
            Tuple of (scale1, zero1) for the given shrink factor
        """
        xmin1 = p * xmin
        xmax1 = p * xmax
        scale1 = (xmax1 - xmin1) / self.maxq
        zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
        return scale1, zero1

    def _update_best_params(self, best, err, scale1, zero1):
        """Update best parameters if current error is lower.

        Args:
            best: Current best error values
            err: Current iteration error values
            scale1: Current iteration scale values
            zero1: Current iteration zero values

        Returns:
            Updated best error values
        """
        tmp = err < best
        if torch.any(tmp):
            best[tmp] = err[tmp]
            self.scale[tmp] = scale1[tmp]
            self.zero[tmp] = zero1[tmp]
        return best

    def _grid_search(self, x, xmin, xmax, compute_error_fn):
        """Common grid search loop for MSE optimization.

        Args:
            x: Prepared tensor
            xmin: Minimum values per channel
            xmax: Maximum values per channel
            compute_error_fn: Function that takes (x, scale1, zero1) and returns error tensor
        """
        dev = x.device
        best = torch.full([x.shape[0]], float("inf"), device=dev)
        for i in range(int(self.maxshrink * self.grid)):
            p = 1 - i / self.grid
            scale1, zero1 = self._compute_shrink_params(p, xmin, xmax)
            err = compute_error_fn(x, scale1, zero1)
            best = self._update_best_params(best, err, scale1, zero1)

    def _optimize_mse(self, x, xmin, xmax):
        """Optimize scale and zero using MSE-based grid search.

        Args:
            x: Prepared tensor
            xmin: Minimum values per channel
            xmax: Maximum values per channel
        """

        def compute_error(x, scale1, zero1):
            q = quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
            q -= x
            q.abs_()
            if self.mse == "smse":  # sensitivity weighted mse
                # in case sensitivity is a second order derivatives of some global loss
                # (q**2) * self.sensitivity is just a global loss change due to quantization.
                q = (q**2) * self.sensitivity.to(
                    q.device
                )  # estimate global target change
            else:
                assert self.mse == "mse"
                q.pow_(self.norm)
            return torch.sum(q, 1)

        self._grid_search(x, xmin, xmax, compute_error)

    def update(self, x, Hinv, perm):
        if self.mse is None or (
            self.mse != "smse_for_gptq" and self.mse != "mse_for_gptq"
        ):
            return

        shape = x.shape
        x, shape = self._prepare_tensor(x, weight=True)

        self.scale, self.zero, xmin, xmax = self._compute_scale_zero_bounds(x)

        sensitivity = None
        if self.sensitivity is not None:
            sensitivity = self.sensitivity.to(Hinv.dtype).to(x.device)
            if perm is not None:
                sensitivity = sensitivity[:, perm.to(x.device)]

        self._optimize_gptq_adjusted(x, Hinv, sensitivity, xmin, xmax)

        self._reshape_scale_zero(shape, weight=True)

        del sensitivity
        sensitivity = None

    def _optimize_gptq_adjusted(self, x, Hinv, sensitivity, xmin, xmax):
        """Optimize scale and zero using GPTQ-aware MSE/SMSE grid search.

        Args:
            x: Prepared tensor
            Hinv: Inverse Hessian matrix
            sensitivity: Sensitivity tensor for weighted MSE
            xmin: Minimum values per channel
            xmax: Maximum values per channel
        """
        num_of_iters = 15

        def compute_error(x, scale1, zero1):
            q, pre_q = iterate_GPTQ(
                scale1.unsqueeze(1),
                zero1.unsqueeze(1),
                self.maxq,
                x,
                Hinv,
                max_num_of_iters=num_of_iters,
            )
            if sensitivity is not None:
                assert self.mse == "smse_for_gptq"
                err = ((q - x) ** 2) * sensitivity.to(q.device)
            else:
                assert self.mse == "mse_for_gptq"
                err = ((q - pre_q) / torch.diag(Hinv)) ** 2

            err = err
            err = torch.sum(err, 1)
            return err

        self._grid_search(x, xmin, xmax, compute_error)

    def quantize(self, x):
        if self.ready():
            return quantize(x, self.scale, self.zero, self.maxq)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)
