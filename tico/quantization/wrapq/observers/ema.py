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

from tico.quantization.wrapq.observers.affine_base import AffineObserverBase
from tico.quantization.wrapq.utils.reduce_utils import channelwise_minmax


class EMAObserver(AffineObserverBase):
    """
    Exponential-Moving-Average min/max tracker.

    Why?
    -----
    • Smoother than raw MinMax (reduces outlier shock).
    • Much cheaper than histogram/MSE observers.

    The update rule follows the common "momentum" form:

        ema = momentum * ema + (1 - momentum) * new_value

    With momentum → 0: FAST adaptation, momentum → 1: SLOW adaptation.
    """

    def __init__(
        self,
        *,
        momentum: float = 0.9,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert 0.0 < momentum < 1.0, "momentum must be in (0, 1)"
        self.momentum = momentum

    @torch.no_grad()
    def _update_stats(self, x: torch.Tensor):
        if self.channel_axis is None:
            curr_min, curr_max = x.min(), x.max()
        else:
            curr_min, curr_max = channelwise_minmax(x, self.channel_axis)

        if (
            torch.isinf(self.min_val).any() and torch.isinf(self.max_val).any()
        ):  # first batch → hard init
            self.min_val, self.max_val = curr_min, curr_max
            return

        m = self.momentum
        self.min_val = m * self.min_val + (1 - m) * curr_min
        self.max_val = m * self.max_val + (1 - m) * curr_max
