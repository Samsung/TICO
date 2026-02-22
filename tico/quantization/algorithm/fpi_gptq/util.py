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
