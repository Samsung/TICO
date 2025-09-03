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

from tico.experimental.quantization.ptq.observers.affine_base import AffineObserverBase
from tico.experimental.quantization.ptq.utils.reduce_utils import channelwise_minmax


class MinMaxObserver(AffineObserverBase):
    """Plain min/max range tracker."""

    @torch.no_grad()
    def _update_stats(self, x: torch.Tensor) -> None:
        """
        Update running min/max with the incoming batch.

        Per-tensor: use global min/max.
        Per-channel: reduce all axes except the channel axis.
        """
        if self.channel_axis is None:
            curr_min, curr_max = x.min(), x.max()
        else:
            curr_min, curr_max = channelwise_minmax(x, self.channel_axis)

        # Broadcasting handles scalar-vs-vector cases
        self.min_val = torch.minimum(self.min_val, curr_min)
        self.max_val = torch.maximum(self.max_val, curr_max)
