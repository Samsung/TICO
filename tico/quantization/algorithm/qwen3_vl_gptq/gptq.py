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

import math
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tico.quantization.algorithm.gptq.quant import quantize, Quantizer

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def convtranspose2d_weights_to_conv2d_weights(
    layer: nn.ConvTranspose2d,
    w: torch.Tensor,
) -> torch.Tensor:
    """
    Convert ConvTranspose2d weights into the equivalent Conv2d weight layout.

    Args:
        layer: Original ConvTranspose2d layer.
        w: Weight tensor from the ConvTranspose2d layer.

    Returns:
        Weight tensor laid out like Conv2d weights.
    """
    if layer.groups == 1:
        return w.transpose(1, 0).flip((-2, -1))

    in_channels, out_channels, kernel_h, kernel_w = layer.weight.shape
    out_channels *= layer.groups

    w_conv = torch.zeros(
        out_channels,
        in_channels // layer.groups,
        kernel_h,
        kernel_w,
        device=w.device,
        dtype=w.dtype,
    )

    for group_idx in range(layer.groups):
        out_start = group_idx * out_channels // layer.groups
        out_end = (group_idx + 1) * out_channels // layer.groups
        in_start = group_idx * in_channels // layer.groups
        in_end = (group_idx + 1) * in_channels // layer.groups

        w_conv[out_start:out_end] = w[in_start:in_end].transpose(1, 0).flip((-2, -1))

    return w_conv


def conv2d_weights_to_convtranspose2d_weights(
    orig_layer: nn.ConvTranspose2d,
    w: torch.Tensor,
) -> torch.Tensor:
    """
    Convert equivalent Conv2d-layout weights back to ConvTranspose2d layout.

    Args:
        orig_layer: Original ConvTranspose2d layer.
        w: Weight tensor in Conv2d-equivalent layout.

    Returns:
        Weight tensor laid out like ConvTranspose2d weights.
    """
    if orig_layer.groups == 1:
        return w.transpose(1, 0).flip((-2, -1))

    in_channels, out_channels, _, _ = orig_layer.weight.shape
    out_channels *= orig_layer.groups

    w_conv_t = torch.zeros_like(orig_layer.weight)
    for group_idx in range(orig_layer.groups):
        in_start = group_idx * in_channels // orig_layer.groups
        in_end = (group_idx + 1) * in_channels // orig_layer.groups
        out_start = group_idx * out_channels // orig_layer.groups
        out_end = (group_idx + 1) * out_channels // orig_layer.groups

        w_conv_t[in_start:in_end] = w[out_start:out_end].transpose(1, 0).flip((-2, -1))

    return w_conv_t


def get_matmul_input_for_convtranspose2d(
    layer: nn.ConvTranspose2d,
    inp: torch.Tensor,
) -> torch.Tensor:
    """
    Convert ConvTranspose2d input into an unfolded matrix form compatible with
    GPTQ Hessian accumulation.

    Args:
        layer: ConvTranspose2d layer.
        inp: Input activation tensor of shape (N, C, H, W).

    Returns:
        Matrix shaped like (flattened_kernel_input_dim, num_samples).
    """
    strided_pad = (
        layer.dilation[0] * (layer.kernel_size[0] - 1) - layer.padding[0],
        layer.dilation[1] * (layer.kernel_size[1] - 1) - layer.padding[1],
    )

    inp_strided = torch.zeros(
        inp.shape[0],
        inp.shape[1],
        layer.stride[0] * (inp.shape[2] - 1) + 2 * strided_pad[0] + 1,
        layer.stride[1] * (inp.shape[3] - 1) + 2 * strided_pad[1] + 1,
        device=inp.device,
        dtype=inp.dtype,
    )

    indices = torch.arange(0, inp.shape[2], device=inp.device)
    inp_strided[
        :,
        :,
        layer.stride[0] * indices + strided_pad[0],
        strided_pad[1] : -strided_pad[1] : layer.stride[1],
    ] = inp[:, :, indices, :]

    unfold = nn.Unfold(
        kernel_size=layer.kernel_size,
        dilation=layer.dilation,
        padding=(0, 0),
        stride=(1, 1),
    )

    if layer.groups != 1:
        inp_strided = inp_strided.reshape(
            inp_strided.size(0) * layer.groups,
            inp_strided.size(1) // layer.groups,
            inp_strided.shape[2],
            inp_strided.shape[3],
        )

    return unfold(inp_strided).permute(1, 0, 2).flatten(1)


def _normalize_2d_padding(
    padding: int | tuple[int, int] | str,
) -> tuple[int, int]:
    """
    Normalize Conv2d padding into a tuple.

    Args:
        padding: Conv2d padding value.

    Returns:
        Tuple (pad_h, pad_w).

    Raises:
        NotImplementedError: If string padding other than 'valid' is used.
    """
    if isinstance(padding, str):
        if padding == "valid":
            return (0, 0)
        raise NotImplementedError(
            "Conv2d with string padding other than 'valid' is not supported by GPTQ."
        )

    if isinstance(padding, int):
        return (padding, padding)

    return padding


def _normalize_3d_padding(
    padding: int | tuple[int, int, int] | str,
) -> tuple[int, int, int]:
    """
    Normalize Conv3d padding into a tuple.

    Args:
        padding: Conv3d padding value.

    Returns:
        Tuple (pad_d, pad_h, pad_w).

    Raises:
        NotImplementedError: If string padding other than 'valid' is used.
    """
    if isinstance(padding, str):
        if padding == "valid":
            return (0, 0, 0)
        raise NotImplementedError(
            "Conv3d with string padding other than 'valid' is not supported by GPTQ."
        )

    if isinstance(padding, int):
        return (padding, padding, padding)

    return padding


def _conv3d_input_to_unfolded(
    layer: nn.Conv3d,
    inp: torch.Tensor,
) -> torch.Tensor:
    """
    Convert Conv3d input into a 2D matrix suitable for GPTQ Hessian updates.

    Args:
        layer: Conv3d layer.
        inp: Input tensor of shape (N, C, D, H, W).

    Returns:
        Matrix of shape (C * kD * kH * kW, N * num_patches).

    Raises:
        NotImplementedError: If grouped Conv3d or unsupported padding/dilation is used.
    """
    if layer.groups != 1:
        raise NotImplementedError("Grouped Conv3d is not supported in GPTQ.")

    if not all(d == 1 for d in layer.dilation):
        raise NotImplementedError("Conv3d with dilation != 1 is not supported in GPTQ.")

    padding = _normalize_3d_padding(layer.padding)

    if any(p != 0 for p in padding):
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

    k_d, k_h, k_w = layer.kernel_size
    s_d, s_h, s_w = layer.stride

    inp = inp.unfold(2, k_d, s_d).unfold(3, k_h, s_h).unfold(4, k_w, s_w)
    inp = inp.reshape(inp.shape[0], inp.shape[1], -1, k_d * k_h * k_w)
    inp = inp.permute(0, 2, 1, 3)
    inp = inp.reshape(inp.shape[0] * inp.shape[1], inp.shape[2] * inp.shape[3]).t()

    return inp


class GPTQ:
    """
    GPTQ helper for a single quantizable module.

    This class accumulates an approximate Hessian from calibration activations
    and then performs blockwise GPTQ quantization of the layer weights.
    """

    def __init__(self, layer: nn.Module):
        """
        Initialize GPTQ state for a single layer.

        Args:
            layer: Quantizable layer. Supported types are Linear, Conv1d,
                Conv2d, Conv3d, and ConvTranspose2d.
        """
        self.layer = layer
        self.dev = self.layer.weight.device

        w = layer.weight.data.clone()
        if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            w = w.flatten(1)
        elif isinstance(layer, nn.ConvTranspose2d):
            w = convtranspose2d_weights_to_conv2d_weights(layer, w)
            w = w.flatten(1)

        self.rows = w.shape[0]
        self.columns = w.shape[1]
        self.H: Optional[torch.Tensor] = torch.zeros(
            (self.columns, self.columns),
            device=self.dev,
        )
        self.nsamples = 0
        self.quantizer: Quantizer = Quantizer()

    @torch.no_grad()
    def add_batch(self, inp: torch.Tensor, out: torch.Tensor) -> None:
        """
        Accumulate Hessian statistics from one calibration batch.

        Args:
            inp: Layer input tensor.
            out: Layer output tensor. Present for interface consistency.
        """
        del out

        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)

        batch_size = inp.shape[0]

        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) > 2:
                inp = inp.reshape(-1, inp.shape[-1])
            inp = inp.t()

        elif isinstance(self.layer, nn.Conv2d):
            padding = _normalize_2d_padding(self.layer.padding)
            unfold = nn.Unfold(
                kernel_size=self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=padding,
                stride=self.layer.stride,
            )

            if self.layer.groups != 1:
                inp = inp.reshape(
                    inp.size(0) * self.layer.groups,
                    inp.size(1) // self.layer.groups,
                    inp.shape[2],
                    inp.shape[3],
                )

            inp = unfold(inp).permute(1, 0, 2).flatten(1)

        elif isinstance(self.layer, nn.Conv1d):
            unfold = nn.Unfold(
                kernel_size=(1, self.layer.kernel_size[0]),
                dilation=(1, self.layer.dilation[0]),
                padding=(0, self.layer.padding[0]),
                stride=(1, self.layer.stride[0]),
            )

            if self.layer.groups != 1:
                inp = inp.reshape(
                    inp.size(0) * self.layer.groups,
                    inp.size(1) // self.layer.groups,
                    inp.shape[2],
                )

            inp = inp.unsqueeze(-2)
            inp = unfold(inp).permute(1, 0, 2).flatten(1)

        elif isinstance(self.layer, nn.ConvTranspose2d):
            inp = get_matmul_input_for_convtranspose2d(self.layer, inp)

        elif isinstance(self.layer, nn.Conv3d):
            inp = _conv3d_input_to_unfolded(self.layer, inp)

        self.H *= self.nsamples / (self.nsamples + batch_size)
        self.nsamples += batch_size
        inp = math.sqrt(2.0 / self.nsamples) * inp.float()
        assert self.H is not None
        self.H += inp.matmul(inp.t()).to(self.H.device)

    @torch.no_grad()
    def fasterquant(
        self,
        blocksize: int = 128,
        percdamp: float = 0.01,
        groupsize: int = -1,
        actorder: bool = False,
        static_groups: bool = False,
        verbose: bool = False,
    ) -> None:
        """
        Run blockwise GPTQ quantization on the layer weights.

        Args:
            blocksize: Number of columns processed at a time.
            percdamp: Damping factor relative to the average Hessian diagonal.
            groupsize: Group size for grouped quantization. -1 disables grouping.
            actorder: Whether to reorder columns by Hessian diagonal magnitude.
            static_groups: Whether to precompute group quantizers.
            verbose: Whether to print timing/error details.
        """
        w = self.layer.weight.data.clone()

        if isinstance(self.layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            w = w.flatten(1)
            if self.quantizer.sensitivity is not None:
                self.quantizer.sensitivity = self.quantizer.sensitivity.flatten(1)

        elif isinstance(self.layer, nn.ConvTranspose2d):
            w = convtranspose2d_weights_to_conv2d_weights(self.layer, w)
            conv2d_shape = w.shape
            w = w.flatten(1)

            if self.quantizer.sensitivity is not None:
                self.quantizer.sensitivity = convtranspose2d_weights_to_conv2d_weights(
                    self.layer,
                    self.quantizer.sensitivity,
                )
                self.quantizer.sensitivity = self.quantizer.sensitivity.flatten(1)

        w = w.float()
        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(w, weight=True)

        h = self.H
        del self.H
        assert isinstance(h, torch.Tensor)

        dead = torch.diag(h) == 0
        h[dead, dead] = 1
        w[:, dead] = 0

        if static_groups:
            import copy

            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(w[:, i : i + groupsize], weight=True)
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(h), descending=True)
            w = w[:, perm]
            h = h[perm][:, perm]
            invperm = torch.argsort(perm)

        losses = torch.zeros_like(w)
        q_all = torch.zeros_like(w)

        damp = percdamp * torch.mean(torch.diag(h))
        diag = torch.arange(self.columns, device=self.dev)
        h[diag, diag] += damp
        h = torch.linalg.cholesky(h)
        h = torch.cholesky_inverse(h)
        h = torch.linalg.cholesky(h, upper=True)
        hinv = h

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            w1 = w[:, i1:i2].clone()
            q1 = torch.zeros_like(w1)
            err1 = torch.zeros_like(w1)
            losses1 = torch.zeros_like(w1)
            hinv1 = hinv[i1:i2, i1:i2]

            for i in range(count):
                w_col = w1[:, i]
                d = hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(
                                w[:, (i1 + i) : (i1 + i + groupsize)],
                                weight=True,
                            )
                    else:
                        idx: torch.Tensor | int = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]

                q_col = quantize(
                    w_col.unsqueeze(1),
                    self.quantizer.scale,
                    self.quantizer.zero,
                    self.quantizer.maxq,
                ).flatten()

                q1[:, i] = q_col
                losses1[:, i] = (w_col - q_col) ** 2 / d**2

                cur_err = (w_col - q_col) / d
                w1[:, i:] -= cur_err.unsqueeze(1).matmul(hinv1[i, i:].unsqueeze(0))
                err1[:, i] = cur_err

            q_all[:, i1:i2] = q1
            losses[:, i1:i2] = losses1 / 2
            w[:, i2:] -= err1.matmul(hinv[i1:i2, i2:])

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        if verbose:
            print(f"time {time.time() - tick:.2f}")
            print("error", torch.sum(losses).item())

        if actorder:
            q_all = q_all[:, invperm]

        if isinstance(self.layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if groupsize == -1:
                q_all[:, dead] = quantize(
                    self.layer.weight.flatten(1)[:, dead],
                    self.quantizer.scale,
                    self.quantizer.zero,
                    self.quantizer.maxq,
                )
        elif isinstance(self.layer, nn.ConvTranspose2d):
            if groupsize == -1:
                q_all[:, dead] = quantize(
                    convtranspose2d_weights_to_conv2d_weights(
                        self.layer,
                        self.layer.weight.data,
                    ).flatten(1)[:, dead],
                    self.quantizer.scale,
                    self.quantizer.zero,
                    self.quantizer.maxq,
                )
        else:
            if groupsize == -1:
                q_all[:, dead] = quantize(
                    self.layer.weight[:, dead],
                    self.quantizer.scale,
                    self.quantizer.zero,
                    self.quantizer.maxq,
                )

        assert (
            groupsize == -1 or torch.sum(dead) == 0
        ), "`dead` columns should be RTN-quantized for grouped quantization."

        if isinstance(self.layer, nn.ConvTranspose2d):
            q_conv2d = q_all.reshape(conv2d_shape).to(self.layer.weight.data.dtype)
            self.layer.weight.data = conv2d_weights_to_convtranspose2d_weights(
                self.layer,
                q_conv2d,
            )
        else:
            self.layer.weight.data = q_all.reshape(self.layer.weight.shape).to(
                self.layer.weight.data.dtype
            )

    def free(self) -> None:
        """
        Release temporary GPTQ state.
        """
        self.H = None
        self.Losses = None
        self.Trace = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
