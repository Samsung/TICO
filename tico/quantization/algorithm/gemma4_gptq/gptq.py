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

"""
GPTQ core for Gemma4.

This module re-exports the GPTQ class from ``qwen3_vl_gptq.gptq`` because the
core GPTQ algorithm (Hessian accumulation + blockwise ``fasterquant``) is
identical across models.  The only difference is which layer types appear in
the model:

- Gemma4's vision patch embedder uses a ``nn.Linear`` (``input_proj``), so no
  ``Conv3d`` unfolding is needed in practice — but the GPTQ class handles
  ``nn.Linear`` natively and the Conv helpers are simply unused.
- All other quantizable layers in Gemma4 (attention projections, MLP
  projections, multimodal embedder projection, lm_head) are ``nn.Linear``.

If future Gemma4 variants introduce convolutional patch embeddings, the
shared GPTQ class already supports ``Conv1d``, ``Conv2d``, ``Conv3d``, and
``ConvTranspose2d``.
"""

from tico.quantization.algorithm.qwen3_vl_gptq.gptq import (  # noqa: F401
    GPTQ,
    convtranspose2d_weights_to_conv2d_weights,
    conv2d_weights_to_convtranspose2d_weights,
    get_matmul_input_for_convtranspose2d,
    _normalize_2d_padding,
    _normalize_3d_padding,
    _conv3d_input_to_unfolded,
)
