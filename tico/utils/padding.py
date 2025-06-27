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

from tico.utils.errors import InvalidArgumentError

SAME = 0
VALID = 1


def is_valid_padding(padding: str | list):
    if isinstance(padding, str):
        return padding == "valid"

    if isinstance(padding, list):
        assert len(padding) == 2, "Padding should be a list of length 2."
        return padding == [0, 0]

    raise InvalidArgumentError("Invalid padding.")


def is_same_padding(
    padding: str | list, input_shape: list | torch.Size, output_shape: list | torch.Size
):
    if isinstance(padding, str):
        return padding == "same"

    if isinstance(padding, list):
        assert len(padding) == 2, "Padding should be a list of length 2."

        input_HW = tuple(input_shape[1:3])  # N H W C
        output_HW = tuple(output_shape[1:3])  # N H W C
        return input_HW == output_HW

    raise InvalidArgumentError("Invalid padding.")
