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

"""Static PyTorch reconstruction of the MediaPipe palm detector."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from tico.ops import Concat, ResizeBilinear2d, SamePaddingConv2d
from tico.quantization import QuantStub
from torch import nn


class ConvNode(nn.Module):
    """Apply one regular, depthwise, VALID, or SAME Conv2d operation."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Create a Conv2d node from the static converted configuration."""
        super().__init__()
        padding = config.get("padding")
        if padding is None:
            padding = "same" if any(config.get("pad", ())) else "valid"
        if padding not in {"same", "valid"}:
            raise ValueError(f"Unsupported convolution padding: {padding!r}")

        conv_type = SamePaddingConv2d if padding == "same" else nn.Conv2d
        self.conv = conv_type(
            in_channels=int(config["in_channels"]),
            out_channels=int(config["out_channels"]),
            kernel_size=tuple(config["kernel_size"]),
            stride=tuple(config["stride"]),
            padding=0,
            dilation=tuple(config["dilation"]),
            groups=int(config["groups"]),
            bias=bool(config["has_bias"]),
        )

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Run the configured convolution."""
        return self.conv(input_)


class ChannelPadNode(nn.Module):
    """Apply constant zero padding after converting NHWC padding to NCHW order."""

    def __init__(self, pad: list[int]) -> None:
        """Store constant NCHW padding values."""
        super().__init__()
        self.pad = tuple(int(value) for value in pad)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Pad an NCHW tensor."""
        return F.pad(input_, self.pad)


class HandDetector(nn.Module):
    """Execute the converted static graph with NCHW input tensors."""

    def __init__(self, specification: dict[str, Any]) -> None:
        """Construct modules for every operation in the static specification."""
        super().__init__()
        self.specification = specification
        self.input_tensor = int(specification["inputs"][0])
        self.output_tensors = tuple(int(value) for value in specification["outputs"])
        self.operations = tuple(specification["operations"])
        self.input_quantizer = QuantStub()
        layers: list[nn.Module] = []
        for operation in self.operations:
            name = operation["name"]
            config = operation["config"]
            if name in {"CONV_2D", "DEPTHWISE_CONV_2D"}:
                layers.append(ConvNode(config))
            elif name == "PRELU":
                layers.append(nn.PReLU(int(config["channels"])))
            elif name == "MAX_POOL_2D":
                layers.append(
                    nn.MaxPool2d(
                        kernel_size=tuple(config["kernel_size"]),
                        stride=tuple(config["stride"]),
                    )
                )
            elif name == "PAD":
                layers.append(ChannelPadNode(config["pad"]))
            elif name == "RESIZE_BILINEAR":
                layers.append(
                    ResizeBilinear2d(
                        tuple(config["size"]),
                        align_corners=bool(config["align_corners"]),
                        half_pixel_centers=bool(config["half_pixel_centers"]),
                    )
                )
            elif name == "CONCATENATION":
                layers.append(Concat(dim=int(config["axis"])))
            else:
                layers.append(nn.Identity())
        self.layers = nn.ModuleList(layers)

    def _forward_core(
        self,
        input_: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the static detector graph from an already quantized NCHW tensor."""
        values: dict[int, torch.Tensor] = {self.input_tensor: input_}
        for operation, layer in zip(self.operations, self.layers):
            name = operation["name"]
            inputs = operation["inputs"]
            output = int(operation["outputs"][0])
            config = operation["config"]
            if name in {
                "CONV_2D",
                "DEPTHWISE_CONV_2D",
                "PRELU",
                "MAX_POOL_2D",
                "PAD",
                "RESIZE_BILINEAR",
            }:
                values[output] = layer(values[int(inputs[0])])
            elif name == "ADD":
                values[output] = values[int(inputs[0])] + values[int(inputs[1])]
            elif name == "RESHAPE":
                source = values[int(inputs[0])]
                if bool(config["nhwc_memory_order"]):
                    source = source.permute(0, 2, 3, 1)
                values[output] = source.reshape(tuple(config["shape"]))
            elif name == "CONCATENATION":
                values[output] = layer(tuple(values[int(index)] for index in inputs))
            else:
                raise RuntimeError(f"Unsupported converted operation: {name}")
        return values[self.output_tensors[0]], values[self.output_tensors[1]]

    def forward(self, input_: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the detector from an NCHW input tensor."""
        return self._forward_core(self.input_quantizer(input_))

    def forward_nhwc(self, input_: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize an NHWC input before converting it to the internal NCHW layout."""
        quantized = self.input_quantizer(input_)
        return self._forward_core(quantized.permute(0, 3, 1, 2))

    def get_example_inputs(self) -> tuple[torch.Tensor]:
        """Return the static NCHW example input used by direct model export."""
        return (torch.zeros(1, 3, 192, 192, dtype=torch.float32),)


class NHWCInputAdapter(nn.Module):
    """Expose an NHWC input ABI while preserving the NCHW detector implementation."""

    def __init__(self, detector: HandDetector) -> None:
        """Store the detector whose input boundary should be exported as NHWC."""
        super().__init__()
        self.detector = detector

    def forward(self, input_: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the wrapped detector from one NHWC input tensor."""
        return self.detector.forward_nhwc(input_)

    def get_example_inputs(self) -> tuple[torch.Tensor]:
        """Return the static NHWC example input used by Circle export."""
        return (torch.zeros(1, 192, 192, 3, dtype=torch.float32),)


def load_hand_detector(
    weights: str | Path,
    specification: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> HandDetector:
    """Construct the detector and load a converted state dictionary."""
    spec = json.loads(Path(specification).read_text(encoding="utf-8"))
    model = HandDetector(spec)
    state = torch.load(weights, map_location=map_location, weights_only=True)
    model.load_state_dict(state, strict=True)
    return model


def load_nhwc_hand_detector(
    weights: str | Path,
    specification: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> NHWCInputAdapter:
    """Load the detector and expose an NHWC input ABI for Circle export."""
    detector = load_hand_detector(
        weights,
        specification,
        map_location=map_location,
    )
    return NHWCInputAdapter(detector)
