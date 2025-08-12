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


def channelwise_minmax(x: torch.Tensor, channel_axis: int):
    """
    Compute per-channel (min, max) by reducing all axes except `channel_axis`.
    """
    channel_axis = channel_axis % x.ndim  # handle negative indices safely
    dims = tuple(d for d in range(x.ndim) if d != channel_axis)

    return x.amin(dim=dims), x.amax(dim=dims)
