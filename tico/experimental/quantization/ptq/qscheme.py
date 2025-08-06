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

from enum import auto, Enum


class QScheme(Enum):
    # ───── Per-tensor ────────────
    PER_TENSOR_ASYMM = auto()
    PER_TENSOR_SYMM = auto()
    # ───── Per-channel ───────────
    PER_CHANNEL_ASYMM = auto()
    PER_CHANNEL_SYMM = auto()

    # helper
    def is_per_channel(self) -> bool:
        return self in {
            QScheme.PER_CHANNEL_ASYMM,
            QScheme.PER_CHANNEL_SYMM,
        }

    def is_symmetric(self) -> bool:
        return self in {
            QScheme.PER_TENSOR_SYMM,
            QScheme.PER_CHANNEL_SYMM,
        }

    def __str__(self) -> str:
        return self.name.lower()
