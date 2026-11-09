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

import torch


def make_text_mlp_fixture():
    """An identity-projection MLP computing x**2 * sigmoid(x), without RNG leaks."""
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextMLP

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = (
            Qwen3VLTextMLP(
                Qwen3VLTextConfig(hidden_size=4, intermediate_size=8, hidden_act="silu")
            )
            .cpu()
            .eval()
        )
    with torch.no_grad():
        identity = torch.eye(4)
        model.gate_proj.weight.copy_(torch.cat((identity, identity)))
        model.up_proj.weight.copy_(torch.cat((identity, identity)))
        model.down_proj.weight.copy_(torch.cat((identity, identity), dim=1) / 2)

    # Endpoints cover every input/projection channel's full [0, 1] range.
    # Sigmoid, SiLU, gated and output grids span [0, sigmoid(1)]; affine includes 0.
    calibration = torch.tensor(
        [
            [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]],
            [[0.25, 0.5, 0.75, 1.0], [1.0, 0.75, 0.5, 0.25]],
        ]
    )
    # Avoid half-grid ties, whose direction can depend on float32 scale rounding.
    hidden = torch.tensor([[[0.13, 0.34, 0.56, 0.78], [0.89, 0.67, 0.49, 0.31]]])
    return model, calibration, hidden
