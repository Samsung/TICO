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
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)

        if isinstance(self.layer, nn.Conv1d):
            W = W.t()
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

            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)

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
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, nn.Conv1d):
            W = W.t()
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
            max_num_of_iters=50,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if verbose:
            print("time %.2f" % (time.time() - tick))
            Losses = 0.5 * ((Q - W) / torch.diag(Hinv)) ** 2
            print("error", torch.sum(Losses).item())

        Q = Q[:, invperm]

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )

    def free(self):
        self.H = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
