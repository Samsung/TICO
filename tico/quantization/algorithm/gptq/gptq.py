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
from typing import Optional

import torch
import torch.nn as nn

from tico.quantization.algorithm.gptq.quant import quantize, Quantizer

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


class GPTQ:
    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d) or isinstance(self.layer, nn.Conv1d):
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

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) > 2:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
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

        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self,
        blocksize=128,
        percdamp=0.01,
        groupsize=-1,
        actorder=False,
        static_groups=False,
        verbose=False,
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d) or isinstance(self.layer, nn.Conv1d):
            W = W.flatten(1)  # reshaped to matrix (OUT_channels x the_rest)
        elif isinstance(self.layer, nn.ConvTranspose2d):
            W = convtranspose2d_weights_to_conv2d_weights(self.layer, W)
            conv2d_shape = W.shape
            W = W.flatten(1)  # reshaped to matrix (OUT_channels x the_rest)

        W = W.float()
        tick = time.time()
        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        H = self.H
        del self.H
        assert isinstance(H, torch.Tensor)
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if static_groups:
            import copy

            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i : (i + groupsize)], weight=True)
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        assert isinstance(H, torch.Tensor)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        assert isinstance(Hinv, torch.Tensor)
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

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
                            idx = perm[idx]
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
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if verbose:
            print("time %.2f" % (time.time() - tick))
            print("error", torch.sum(Losses).item())

        if actorder:
            Q = Q[:, invperm]

        if isinstance(self.layer, nn.Conv2d) or isinstance(self.layer, nn.Conv1d):
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
