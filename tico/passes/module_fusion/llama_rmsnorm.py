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

from transformers.models.llama.modeling_llama import LlamaRMSNorm

from .fusion_registry import register_fused_module


class FusedLlamaRMSNorm(LlamaRMSNorm):
    def __init__(self, original_rmsnorm: LlamaRMSNorm):
        super().__init__(
            original_rmsnorm.weight.shape[0], original_rmsnorm.variance_epsilon
        )
        with torch.no_grad():
            self.weight.copy_(original_rmsnorm.weight)

    def forward(self, hidden_states):
        return torch.ops.circle_custom.rms_norm(
            hidden_states, self.weight, self.variance_epsilon
        )


@register_fused_module(LlamaRMSNorm)
def create_fused_llama_rmsnorm(original_module: LlamaRMSNorm) -> FusedLlamaRMSNorm:
    return FusedLlamaRMSNorm(original_module)
