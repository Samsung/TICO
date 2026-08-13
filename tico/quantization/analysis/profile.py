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

"""Standard A/B/C/D/E quantization-ablation profiles."""

from __future__ import annotations

from enum import Enum


class QuantizationProfile(str, Enum):
    """Describe which quantization sites are active during one comparison.

    ``OUTPUT_ONLY`` (A)
        Quantize only the explicitly selected model-output domains.
    ``WEIGHT_ONLY`` (B)
        Quantize parameter observers while leaving activations and outputs in
        floating point.
    ``ACTIVATION_ONLY`` (C)
        Quantize internal activation observers while leaving parameters and
        explicitly selected model outputs in floating point.
    ``FULL`` (D)
        Enable every analysis site.
    ``INTERNAL_FULL`` (E)
        Enable every included site except explicitly selected model-output
        domains.
    """

    OUTPUT_ONLY = "A"
    WEIGHT_ONLY = "B"
    ACTIVATION_ONLY = "C"
    FULL = "D"
    INTERNAL_FULL = "E"

    @property
    def label(self) -> str:
        """Return a stable human-readable profile label."""
        labels = {
            QuantizationProfile.OUTPUT_ONLY: "output-only",
            QuantizationProfile.WEIGHT_ONLY: "weight-only",
            QuantizationProfile.ACTIVATION_ONLY: "activation-only",
            QuantizationProfile.FULL: "full",
            QuantizationProfile.INTERNAL_FULL: "internal-full",
        }
        return labels[self]
