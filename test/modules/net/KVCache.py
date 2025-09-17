# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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

from test.modules.base import TestModuleBase


bs = bsz = 4
seq_len = slen = 16
n_kv_heads = 8
head_dim = 128
cache_len = 128
max_batch_size = 9
max_seq_len = 64
n_local_kv_heads = 8


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class RepeatKV(TestModuleBase):
    def __init__(self):
        super().__init__()

    def forward(self, x, n_rep: int) -> torch.Tensor:
        return repeat_kv(x, n_rep)

    def get_example_inputs(self):
        return (
            torch.randn(bs, n_kv_heads, seq_len, head_dim),
            2,
        ), {}


class KVCacheSlice(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.n_rep = 1
        self.register_buffer(
            "cache_k",
            torch.randn(
                [max_batch_size, n_local_kv_heads, max_seq_len, head_dim],
            ),
        )  # 8, 8, 64, 128
        self.register_buffer(
            "cache_v",
            torch.randn(
                [max_batch_size, n_local_kv_heads, max_seq_len, head_dim],
            ),
        )

    def forward(self, start_pos, seq_len):
        keys = self.cache_k[:bsz, :, : start_pos + seq_len, :]  # type: ignore[index]
        values = self.cache_v[:bsz, :, : start_pos + seq_len, :]  # type: ignore[index]

        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)
        return (keys, values, self.cache_k, self.cache_v)

    def get_example_inputs(self):
        start_pos = 8
        return (start_pos, seq_len), {}


class KVCacheUpdate(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.n_rep = 1
        self.register_buffer(
            "cache_k",
            torch.randn(
                [max_batch_size, n_local_kv_heads, max_seq_len, head_dim],
            ),
        )  # 8, 64, 8, 128
        self.register_buffer(
            "cache_v",
            torch.randn(
                [max_batch_size, n_local_kv_heads, max_seq_len, head_dim],
            ),
        )

    def forward(self, xk, xv, start_pos):
        self.cache_k[:bsz, :, start_pos : start_pos + seq_len, :] = xk  # type: ignore[operator]
        self.cache_v[:bsz, :, start_pos : start_pos + seq_len, :] = xv  # type: ignore[operator]

        keys = self.cache_k[:bsz, :, : start_pos + seq_len, :]  # type: ignore[index]
        values = self.cache_v[:bsz, :, : start_pos + seq_len, :]  # type: ignore[index]

        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)
        return (keys, values, self.cache_k, self.cache_v)

    def get_example_inputs(self):
        start_pos = 8
        return (
            torch.randn(bs, n_local_kv_heads, seq_len, head_dim),
            torch.randn(bs, n_local_kv_heads, seq_len, head_dim),
            start_pos,
        ), {}
