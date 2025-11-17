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


def iterate_GPTQ(scale, zero, maxq, W, Hinv, max_num_of_iters=50):

    max_num_of_iters = max(
        1, max_num_of_iters
    )  # constrain it to be at least one iteration

    cur_weights = W.clone()
    if len(Hinv.shape) == 2:
        mults = torch.pow(torch.diag(Hinv), -1)
        Hinv_U = torch.triu(Hinv, diagonal=1)
    else:
        mults = torch.pow(torch.diagonal(Hinv, dim1=-2, dim2=-1), -1)
        Hinv_U = torch.triu(Hinv, diagonal=1)

    init_weights = W.clone()
    for _ in range(max_num_of_iters):
        cur_Q = quantize(cur_weights, scale, zero, maxq)

        d_W = torch.mul((cur_weights - cur_Q), mults)
        if len(Hinv.shape) == 2:
            cur_weights = init_weights - torch.matmul(d_W, Hinv_U)
        else:
            # for i in range(Hinv.shape[0]):
            #    cur_weights[i] = init_weights[i] - torch.matmul(d_W[i], Hinv_U[i])
            mm = torch.bmm(d_W.unsqueeze(-2), Hinv_U).squeeze(-2)
            cur_weights = init_weights - mm

        del d_W, cur_Q
        d_W = cur_Q = None

    del init_weights
    init_weights = None

    cur_Q = quantize(cur_weights, scale, zero, maxq)

    return cur_Q, cur_weights


class FPI_GPTQ:
    def __init__(self, layer, quantize_convs_groupwise: bool = False):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)

        if isinstance(self.layer, nn.Conv1d):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]

        self.H: Optional[torch.Tensor] = None

        if (
            (isinstance(self.layer, nn.Conv2d) or isinstance(self.layer, nn.Conv1d))
            and quantize_convs_groupwise
            and self.layer.groups != 1
        ):
            self.H = torch.zeros(
                (self.layer.groups, self.columns, self.columns), device=self.dev
            )
        else:
            self.H = torch.zeros((self.columns, self.columns), device=self.dev)

        self.nsamples = 0
        self.quantizer: Quantizer = Quantizer()

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        processed = False

        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, nn.Conv1d):
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

            if self.layer.groups == 1:
                inp = unfold(inp)
                inp = inp.permute([1, 0, 2])
                inp = inp.flatten(1)
            elif len(self.H.shape) > 2:  # type: ignore[union-attr]
                # groupwise
                # H_0 = torch.zeros_like(self.H)
                self.H *= self.nsamples / (self.nsamples + tmp)
                self.nsamples += tmp

                # ind_range = inp.shape[1] // self.layer.groups
                # for i in range(self.layer.groups):
                #     inp_ = unfold(inp[:, i : i + ind_range, :, :])
                #     inp_ = inp_.permute([1, 0, 2])
                #     inp_ = inp_.flatten(1)
                #     H_0[i] += inp_.matmul(inp_.t())
                #     inp_ = math.sqrt(2 / self.nsamples) * inp_.float()
                #     self.H[i] += inp_.matmul(inp_.t())

                inp_ = inp.reshape(
                    inp.size(0) * self.layer.groups,
                    inp.size(1) // self.layer.groups,
                    inp.shape[2],
                    inp.shape[3],
                )
                inp_ = unfold(inp_)
                inp_ = math.sqrt(2 / self.nsamples) * inp_.float()
                # H_1 = torch.matmul(inp_, inp_.permute([0, 2, 1]))
                self.H += torch.matmul(inp_, inp_.permute([0, 2, 1]))
                processed = True
                #   H_1 = torch.zeros_like(H_0)
                #   inp_ = inp.reshape(
                #       inp.size(0) * self.layer.groups,
                #       inp.size(1) // self.layer.groups,
                #       inp.shape[2],
                #       inp.shape[3],
                #   )
                #   unfold = nn.Unfold(
                #       self.layer.kernel_size,
                #       dilation=self.layer.dilation,
                #       padding=self.layer.padding,
                #       stride=self.layer.stride,
                #   )
                #   # output size (batch_size, channels * \prod kernel_size, num_patches)
                #   inp_ = unfold(inp_)
                #   inp_ = inp_.permute([1, 0, 2])
                #   inp_ = inp_.flatten(1)
                #   inp_ = math.sqrt(2 / self.nsamples) * inp_.float()
                #   H_1 += torch.matmul(inp_, inp_.t())
                #   dist = torch.mean(torch.abs(H_0 - H_1)) / torch.max(torch.abs(H_1))
                #   assert(dist < 1.e-05) #it should be very low with math.sqrt(2 / self.nsamples) * reshaped_inp.float() turned off
            else:
                # the idea behind conversion of depthwise convolution to matmul is described here
                # https://discuss.pytorch.org/t/conv1d-implementation-using-torch-nn-functional-unfold/109643/2
                # although depthwise convolution is equal to a set of MatMul
                # (please note `w.view(1, groups, out_channels // groups, -1)` in the reference above it is not w.flatten(1))
                # it occures that for bits (>=4) we can approximate groupwise Hessians with their sum
                # so that we will have just a single Hessian and the usual GPTQ applies
                inp = inp.reshape(
                    inp.size(0) * self.layer.groups,
                    inp.size(1) // self.layer.groups,
                    inp.shape[2],
                    inp.shape[3],
                )
                inp = unfold(inp)
                inp = inp.permute([1, 0, 2])
                inp = inp.flatten(1)

        if not processed:
            self.H *= self.nsamples / (self.nsamples + tmp)
            self.nsamples += tmp
            inp = math.sqrt(2 / self.nsamples) * inp.float()
            self.H += inp.matmul(inp.t())

    def fasterquant_group_wise(
        self,
        percdamp=0.01,
        verbose=False,
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d) or isinstance(self.layer, nn.Conv1d):
            W = W.flatten(1)

        W = W.float()
        tick = time.time()
        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        H = self.H
        del self.H
        assert isinstance(H, torch.Tensor)

        meanH = torch.sum(H, dim=0)

        dead = torch.diagonal(H, dim1=-2, dim2=-1) == 0
        for i in range(H.shape[0]):
            H[i, dead[i], dead[i]] = 1
            W[i, dead[i]] = 0

        # actorder
        # perm = torch.argsort(torch.diag(meanH), descending=True)
        # W = W[:, perm]
        # H = H[:, perm, :][:, :, perm]
        # meanH=meanH[perm][:, perm]
        perm = torch.argsort(torch.diagonal(H, dim1=-2, dim2=-1), descending=True)
        for i in range(H.shape[0]):
            W[i] = W[i, perm[i]]
            H[i] = H[i][perm[i]][:, perm[i]]
        invperm = torch.argsort(perm)

        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diagonal(H, dim1=-2, dim2=-1), dim=-1)
        diag = torch.arange(self.columns, device=self.dev)
        for i in range(H.shape[0]):
            H[i, diag, diag] += damp[i]

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
            max_num_of_iters=min(50, self.columns),
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if verbose:
            print("time %.2f" % (time.time() - tick))
            Losses = 0.5 * ((Q - W) / torch.diagonal(Hinv, dim1=-2, dim2=-1)) ** 2
            print("error", torch.sum(Losses).item())

        #        Q = Q[:, invperm]
        for i in range(H.shape[0]):
            Q[i] = Q[i, invperm[i]]

        if isinstance(self.layer, nn.Conv2d):
            flattened = self.layer.weight.flatten(1)
            for i in range(H.shape[0]):
                if torch.sum(dead[i]) > 0:
                    Q[i, dead[i]] = quantize(
                        flattened[i, dead[i]],
                        self.quantizer.scale[i],
                        self.quantizer.zero[i],
                        self.quantizer.maxq,
                    )

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )

    def fasterquant(
        self,
        percdamp=0.01,
        verbose=False,
    ):
        if len(self.H.shape) == 3:  # type: ignore[union-attr]
            return self.fasterquant_group_wise(percdamp, verbose)

        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d) or isinstance(self.layer, nn.Conv1d):
            W = W.flatten(1)
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
            max_num_of_iters=min(50, self.columns),  # max_num_of_iters=50,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if verbose:
            print("time %.2f" % (time.time() - tick))
            Losses = 0.5 * ((Q - W) / torch.diag(Hinv)) ** 2
            print("error", torch.sum(Losses).item())

        Q = Q[:, invperm]

        if isinstance(self.layer, nn.Conv2d):
            Q[:, dead] = quantize(
                self.layer.weight.flatten(1)[:, dead],
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

        if isinstance(self.layer, nn.Conv2d):
            Q[:, dead] = quantize(
                self.layer.weight.flatten(1)[:, dead],
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

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )

    def free(self):
        self.H = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
