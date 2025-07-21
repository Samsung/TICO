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
from transformers.models.bitnet.modeling_bitnet import BitNetConfig, BitNetModel

from test.utils import tag


@tag.use_onert
class BitNet(torch.nn.Module):
    """
    BitNet-b1.58 Decoder layer

    Constraints:
    1. Due to flatbuffer 2GB limitation, testing is performed with only 1 decoder layer
    2. Due to value mismatches in pretrained model weights, testing is performed with untrained ones
    """

    def __init__(self):
        super().__init__()
        torch.manual_seed(5)
        self.model = BitNetModel(config=BitNetConfig())

    def forward(self, *args, **kwargs):
        return self.model.layers[0](*args, **kwargs)

    def get_example_inputs(self):
        batch = 1
        seq_len = 21
        hidden_size = 2560

        hidden_states = torch.randn((batch, seq_len, hidden_size), dtype=torch.float32)
        position_ids = torch.tensor([[i for i in range(0, seq_len)]], dtype=torch.long)
        position_embeddings = (
            torch.rand((batch, seq_len, 128), dtype=torch.float32),
            torch.rand((batch, seq_len, 128), dtype=torch.float32),
        )
        return (
            hidden_states,
            {"position_ids": position_ids, "position_embeddings": position_embeddings},
        )
