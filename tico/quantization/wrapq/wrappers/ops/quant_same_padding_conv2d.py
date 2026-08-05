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

"""WrapQ support for ``tico.ops.SamePaddingConv2d`` modules."""

from tico.ops import SamePaddingConv2d
from tico.quantization.wrapq.wrappers.nn.quant_conv2d import QuantConv2d
from tico.quantization.wrapq.wrappers.registry import register


@register(SamePaddingConv2d)
class QuantSamePaddingConv2d(QuantConv2d):
    """Reuse Conv2d quantization while preserving Circle SAME padding."""
