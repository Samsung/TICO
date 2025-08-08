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


class PercentileObserver(AffineObserverBase):
    """
    Observer that clips extreme values to a percentile range before computing
    min/max. Useful for outlier suppression.
    """

    def __init__(self, *, percentile: float = 99.9, **kwargs):
        super().__init__(**kwargs)
        self.percentile = percentile

    @torch.no_grad()
    def _update_stats(self, x):
        q_low = (100.0 - self.percentile) / 200.0  # two-sided
        q_high = 1.0 - q_low

        if self.channel_axis is None:
            # ---------- per-tensor -----------------------------------
            low = torch.quantile(x.detach(), q_low)
            high = torch.quantile(x.detach(), q_high)
            self.min_val = torch.minimum(self.min_val, low)
            self.max_val = torch.maximum(self.max_val, high)

        else:
            # ---------- per-channel ----------------------------------
            cax = self.channel_axis % x.ndim
            # bring channel axis first, flatten the rest
            x_flat = x.transpose(0, cax).contiguous().view(x.size(cax), -1)

            low = torch.quantile(x_flat, q_low, dim=1)
            high = torch.quantile(x_flat, q_high, dim=1)

            # fold into running vectors
            self.min_val = torch.minimum(self.min_val, low)
            self.max_val = torch.maximum(self.max_val, high)
