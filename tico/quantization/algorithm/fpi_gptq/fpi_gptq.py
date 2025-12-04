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

from tico.quantization.algorithm.gptq.gptq import (
    conv2d_weights_to_convtranspose2d_weights,
    convtranspose2d_weights_to_conv2d_weights,
    get_matmul_input_for_convtranspose2d,
)

from tico.quantization.algorithm.gptq.quant import quantize, Quantizer


def iterate_GPTQ(scale, zero, maxq, W, Hinv, max_num_of_iters=50):

    cur_weights = W.clone()
    mults = torch.pow(torch.diag(Hinv), -1)
    Hinv_U = torch.triu(Hinv, diagonal=1)

    init_weights = W.clone()
    for _ in range(max_num_of_iters):
        cur_Q = quantize(cur_weights, scale, zero, maxq)

        d_W = torch.mul((cur_weights - cur_Q), mults)
        cur_weights = init_weights - torch.matmul(d_W, Hinv_U)
        del d_W, cur_Q
        d_W = cur_Q = None

    del init_weights
    init_weights = None

    cur_Q = quantize(cur_weights, scale, zero, maxq)

    return cur_Q, cur_weights


class FPI_GPTQ:
    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d) or isinstance(self.layer, nn.Conv1d):
            W = W.flatten(1)
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
        percdamp=0.01,
        verbose=False,
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d) or isinstance(self.layer, nn.Conv1d):
            W = W.flatten(1)
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

        # actorder
        perm = torch.argsort(torch.diag(H), descending=True)
        W = W[:, perm]
        H = H[perm][:, perm]
        invperm = torch.argsort(perm)

        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        assert isinstance(H, torch.Tensor)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        Q, W = iterate_GPTQ(
            self.quantizer.scale,
            self.quantizer.zero,
            self.quantizer.maxq,
            W,
            Hinv=Hinv,
            max_num_of_iters=min(
                50, self.columns
            ),  # we don't need to iterate more than self.columns
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if verbose:
            print("time %.2f" % (time.time() - tick))
            Losses = 0.5 * ((Q - W) / torch.diag(Hinv)) ** 2
            print("error", torch.sum(Losses).item())

        Q = Q[:, invperm]

        if isinstance(self.layer, nn.Conv2d) or isinstance(self.layer, nn.Conv1d):
            Q[:, dead] = quantize(
                self.layer.weight.flatten(1)[:, dead],
                self.quantizer.scale,
                self.quantizer.zero,
                self.quantizer.maxq,
            )
        elif isinstance(self.layer, nn.ConvTranspose2d):
            Q[:, dead] = quantize(
                convtranspose2d_weights_to_conv2d_weights(
                    self.layer, self.layer.weight.data
                ).flatten(1)[:, dead],
                self.quantizer.scale,
                self.quantizer.zero,
                self.quantizer.maxq,
            )
        else:
            Q[:, dead] = quantize(
                self.layer.weight[:, dead],
                self.quantizer.scale,
                self.quantizer.zero,
                self.quantizer.maxq,
            )

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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
