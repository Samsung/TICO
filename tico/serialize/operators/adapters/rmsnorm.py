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

from contextlib import contextmanager

import torch

from transformers.models.llama.modeling_llama import LlamaRMSNorm


def llama_rmsnorm_forward_adapter(self: LlamaRMSNorm, hidden_states: torch.Tensor):
    return torch.ops.circle_custom.rms_norm(
        hidden_states, self.weight, self.variance_epsilon
    )


@contextmanager
def patched_llama_rmsnorm():
    orig = LlamaRMSNorm.forward
    LlamaRMSNorm.forward = llama_rmsnorm_forward_adapter
    try:
        yield
    finally:
        LlamaRMSNorm.forward = orig
