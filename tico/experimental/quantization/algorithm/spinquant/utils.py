# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the same directory of this source file.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.
# Adapted from https://github.com/Cornell-RelaxML/quip-sharp/blob/main/lib/utils/matmul_had.py

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

import torch

from tico.utils.errors import InvalidArgumentError


def is_pow2(n):
    return (n & (n - 1) == 0) and (n > 0)


@torch.no_grad()
def random_hadamard_matrix(size):
    # See https://cornell-relaxml.github.io/quip-sharp/ , Section "Randomized Hadamard Transformation"
    q = torch.randint(low=0, high=2, size=(size,)).to(torch.float64)
    q = q * 2 - 1
    q = torch.diag(q)

    n = q.shape[-1]
    assert is_pow2(n)

    # k is always 1 when `n` is power of 2
    # TODO: Support the case when the `n` is not power of 2
    k = 1

    # Fast Walsh-Hadaramd Transform(FWHT)
    in_ = q.clone().view(-1, n, 1)
    out_ = in_.clone()
    while in_.shape[1] > k:
        in_ = in_.view(in_.shape[0], in_.shape[1] // 2, 2, in_.shape[2])
        out_ = out_.view(in_.shape)
        out_[:, :, 0, :] = in_[:, :, 0, :] + in_[:, :, 1, :]
        out_[:, :, 1, :] = in_[:, :, 0, :] - in_[:, :, 1, :]
        out_ = out_.view(in_.shape[0], in_.shape[1], -1)
        (in_, out_) = (out_, in_)
    del out_

    return in_.view(q.shape) / torch.tensor(n).sqrt()


@torch.no_grad()
def random_orthogonal_matrix(size: int):
    random_matrix = torch.randn(size, size, dtype=torch.float64)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q


@torch.no_grad()
def get_orthogonal_matrix(size: int, mode="hadamard") -> torch.Tensor:
    if mode == "hadamard":
        return random_hadamard_matrix(size)
    if mode == "random":
        return random_orthogonal_matrix(size)
    raise InvalidArgumentError(f"Unknown rotation mode: {mode}")
