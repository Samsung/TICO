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
from collections.abc import Mapping, Sequence
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


class ResizeBilinearTConv(nn.Module):
    """Exact 2x half-pixel RESIZE_BILINEAR as replicate padding plus TransposeConv.

    Circle RESIZE_BILINEAR with ``align_corners=False`` and
    ``half_pixel_centers=True`` at an exact 2x scale equals a fixed-weight
    ``ConvTranspose2d(kernel_size=4, stride=2, padding=3)`` over the input
    replicate-padded by one pixel. The padding is expressed with slice/concat
    so every operation lowers to quantizable Circle operators. A separable
    4x1 + 1x4 decomposition would halve the constant weight payload, but the
    NPU compiler may reject non-square TRANSPOSE_CONV kernels (HLAT failure),
    so the dense 4x4 kernel is kept.

    Because the interpolation weight is diagonal across channels, ``groups``
    splits the channel dimension into independent TRANSPOSE_CONV operators
    joined by one channel CONCATENATION. The result is bit-identical while
    the dense constant weight shrinks by the group factor.
    """

    def __init__(self, channels: int, *, groups: int = 1) -> None:
        """Create the fixed bilinear-interpolation transposed convolutions."""
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}.")
        if groups <= 0 or channels % groups != 0:
            raise ValueError(
                f"groups must evenly divide channels, got {channels}/{groups}."
            )
        self.group_channels = channels // groups
        kernel = torch.outer(
            torch.tensor([0.25, 0.75, 0.75, 0.25]),
            torch.tensor([0.25, 0.75, 0.75, 0.25]),
        )
        # The zero biases stay real parameters so bias quantization produces
        # fully quantized Circle operators.
        tconvs = []
        for _ in range(groups):
            tconv = nn.ConvTranspose2d(
                self.group_channels,
                self.group_channels,
                kernel_size=4,
                stride=2,
                padding=3,
                bias=True,
            )
            weight = torch.zeros(self.group_channels, self.group_channels, 4, 4)
            weight[range(self.group_channels), range(self.group_channels)] = kernel
            with torch.no_grad():
                tconv.weight.copy_(weight)
                tconv.bias.zero_()
            tconvs.append(tconv)
        self.tconvs = nn.ModuleList(tconvs)
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Upsample one NCHW tensor by exactly 2x in both spatial dimensions."""
        padded = torch.cat([input_[:, :, :1, :], input_, input_[:, :, -1:, :]], dim=2)
        padded = torch.cat([padded[:, :, :, :1], padded, padded[:, :, :, -1:]], dim=3)
        if len(self.tconvs) == 1:
            return self.tconvs[0](padded)
        size = self.group_channels
        outputs = [
            tconv(padded[:, index * size : (index + 1) * size])
            for index, tconv in enumerate(self.tconvs)
        ]
        return torch.cat(outputs, dim=1)


def lower_resize_bilinear_to_tconv(
    detector: "HandDetector",
    *,
    groups: int = 1,
) -> tuple[int, ...]:
    """Replace every 2x half-pixel RESIZE_BILINEAR layer with ResizeBilinearTConv.

    Call after loading weights; the replacement layers carry their own fixed
    interpolation weights. Returns the replaced operation positions.
    """
    device = next(detector.parameters()).device
    with torch.inference_mode():
        values = detector.forward_values(detector.get_example_inputs()[0].to(device))
    replaced: list[int] = []
    for position, operation in enumerate(detector.operations):
        if operation["name"] != "RESIZE_BILINEAR":
            continue
        config = operation["config"]
        if bool(config["align_corners"]) or not bool(config["half_pixel_centers"]):
            raise ValueError(
                "TransposeConv lowering requires align_corners=False and "
                "half_pixel_centers=True."
            )
        source = values[int(operation["inputs"][0])]
        out_h, out_w = (int(value) for value in config["size"])
        if (out_h, out_w) != (2 * source.shape[2], 2 * source.shape[3]):
            raise ValueError(
                "TransposeConv lowering requires an exact 2x resize, got "
                f"{tuple(source.shape[2:])} -> {(out_h, out_w)}."
            )
        detector.layers[position] = ResizeBilinearTConv(
            int(source.shape[1]), groups=groups
        ).to(device)
        replaced.append(position)
    if not replaced:
        raise RuntimeError("No RESIZE_BILINEAR operation was lowered.")
    return tuple(replaced)


def _is_width_concat_operation(
    operations: Sequence[Mapping[str, Any]],
    position: int,
) -> bool:
    """Return whether one channels-last Concat joins the width axis."""
    operation = operations[position]
    if operation["name"] != "CONCATENATION":
        raise ValueError(f"Operation {position} is not CONCATENATION.")
    ranks: set[int] = set()
    for raw_tensor_id in operation["inputs"]:
        tensor_id = int(raw_tensor_id)
        producer = next(
            (
                candidate
                for candidate in operations[:position]
                if tensor_id in tuple(int(value) for value in candidate["outputs"])
            ),
            None,
        )
        if producer is None or producer["name"] != "RESHAPE":
            return False
        shape = tuple(int(value) for value in producer["config"]["shape"])
        ranks.add(len(shape))
    if len(ranks) != 1:
        return False
    rank = ranks.pop()
    if rank < 2:
        return False
    axis = int(operation["config"]["axis"])
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        raise ValueError(f"Concat axis {axis} is invalid for rank {rank}.")
    return axis == rank - 2


class SpatialMeanNode(nn.Module):
    """Reduce one NCHW tensor over its spatial axes without keeping dims.

    The reduction runs in NHWC so the exported Circle MEAN keeps the TFLite
    axes ``[1, 2]``; layout optimization then cancels the internal permute
    against the producer's layout transition instead of wrapping the MEAN in
    NPU-unsupported NCHW transposes.
    """

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Average one rank-4 NCHW tensor over height and width."""
        return input_.permute(0, 2, 3, 1).mean(dim=(1, 2))


class HandDetector(nn.Module):
    """Execute the converted static graph with NCHW input tensors."""

    def __init__(self, specification: dict[str, Any]) -> None:
        """Construct modules for every operation in the static specification."""
        super().__init__()
        self.specification = specification
        self.input_tensor = int(specification["inputs"][0])
        self.output_tensors = tuple(int(value) for value in specification["outputs"])
        self.operations = tuple(specification["operations"])
        self.input_shape_nhwc = tuple(
            int(value) for value in specification.get("input_shape", (1, 192, 192, 3))
        )
        self.input_quantizer = QuantStub()
        layers: list[nn.Module] = []
        for position, operation in enumerate(self.operations):
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
            elif name == "RELU6":
                layers.append(nn.ReLU6())
            elif name == "LOGISTIC":
                layers.append(nn.Sigmoid())
            elif name == "MEAN":
                layers.append(SpatialMeanNode())
            elif name == "RESIZE_BILINEAR":
                layers.append(
                    ResizeBilinear2d(
                        tuple(config["size"]),
                        align_corners=bool(config["align_corners"]),
                        half_pixel_centers=bool(config["half_pixel_centers"]),
                    )
                )
            elif name == "CONCATENATION":
                layers.append(
                    Concat(
                        dim=int(config["axis"]),
                        allow_distinct_input_qparams=(
                            _is_width_concat_operation(
                                self.operations,
                                position,
                            )
                        ),
                    )
                )
            else:
                layers.append(nn.Identity())
        self.layers = nn.ModuleList(layers)

    def execute_segment(
        self,
        initial_values: Mapping[int, torch.Tensor],
        operation_positions: Sequence[int],
    ) -> dict[int, torch.Tensor]:
        """Execute selected static-graph operations from supplied tensor values."""
        values = {int(tensor_id): value for tensor_id, value in initial_values.items()}
        for position in operation_positions:
            if position < 0 or position >= len(self.operations):
                raise IndexError(f"Operation position {position} is out of range.")
            operation = self.operations[position]
            layer = self.layers[position]
            name = operation["name"]
            inputs = operation["inputs"]
            output = int(operation["outputs"][0])
            config = operation["config"]
            runtime_inputs = (
                inputs
                if name == "CONCATENATION"
                else inputs[:2]
                if name == "ADD"
                else inputs[:1]
            )
            missing = tuple(
                int(tensor_id)
                for tensor_id in runtime_inputs
                if int(tensor_id) not in values
            )
            if missing:
                raise KeyError(
                    f"Operation {position} ({name}) is missing input tensors {missing}."
                )
            if name in {
                "CONV_2D",
                "DEPTHWISE_CONV_2D",
                "PRELU",
                "MAX_POOL_2D",
                "PAD",
                "RELU6",
                "LOGISTIC",
                "MEAN",
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
        return values

    def forward_values(self, input_: torch.Tensor) -> dict[int, torch.Tensor]:
        """Run the complete graph and return every materialized NCHW tensor."""
        quantized = self.input_quantizer(input_)
        return self.execute_segment(
            {self.input_tensor: quantized},
            tuple(range(len(self.operations))),
        )

    def forward_nhwc_values(self, input_: torch.Tensor) -> dict[int, torch.Tensor]:
        """Run the complete graph from NHWC input and return all tensor values."""
        quantized = self.input_quantizer(input_).permute(0, 3, 1, 2)
        return self.execute_segment(
            {self.input_tensor: quantized},
            tuple(range(len(self.operations))),
        )

    def _forward_core(
        self,
        input_: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Run the static detector graph from an already quantized NCHW tensor."""
        values = self.execute_segment(
            {self.input_tensor: input_},
            tuple(range(len(self.operations))),
        )
        return tuple(values[tensor_id] for tensor_id in self.output_tensors)

    def forward(self, input_: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Run the detector from an NCHW input tensor."""
        return self._forward_core(self.input_quantizer(input_))

    def forward_nhwc(self, input_: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Quantize an NHWC input before converting it to the internal NCHW layout."""
        quantized = self.input_quantizer(input_)
        return self._forward_core(quantized.permute(0, 3, 1, 2))

    def get_example_inputs(self) -> tuple[torch.Tensor]:
        """Return the static NCHW example input used by direct model export."""
        batch, height, width, channels = self.input_shape_nhwc
        return (torch.zeros(batch, channels, height, width, dtype=torch.float32),)


class NHWCInputAdapter(nn.Module):
    """Expose an NHWC input ABI while preserving the NCHW detector implementation."""

    def __init__(self, detector: HandDetector) -> None:
        """Store the detector whose input boundary should be exported as NHWC."""
        super().__init__()
        self.detector = detector

    def forward(self, input_: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Run the wrapped detector from one NHWC input tensor."""
        return self.detector.forward_nhwc(input_)

    def get_example_inputs(self) -> tuple[torch.Tensor]:
        """Return the static NHWC example input used by Circle export."""
        return (torch.zeros(*self.detector.input_shape_nhwc, dtype=torch.float32),)


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
