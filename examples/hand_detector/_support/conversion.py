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

"""Convert a supported MediaPipe hand TFLite model into a static PyTorch model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from examples.hand_detector._support.tflite_flatbuffer import OperatorInfo, TFLiteModel

from examples.hand_detector.hand_detector import HandDetector


PADDING_SAME = 0
FUSED_ACTIVATION_NONE = 0
FUSED_ACTIVATION_RELU6 = 3


def _same_padding(
    input_size: int,
    output_size: int,
    kernel_size: int,
    stride: int,
    dilation: int,
) -> tuple[int, int]:
    """Return the before and after padding used by TFLite SAME convolution."""
    effective_kernel = (kernel_size - 1) * dilation + 1
    total = max((output_size - 1) * stride + effective_kernel - input_size, 0)
    before = total // 2
    return before, total - before


def _conv_config(
    model: TFLiteModel,
    operation: OperatorInfo,
    *,
    depthwise: bool,
) -> dict[str, Any]:
    """Build a PyTorch Conv2d configuration from one TFLite convolution."""
    input_tensor = model.tensors[operation.inputs[0]]
    weight_tensor = model.tensors[operation.inputs[1]]
    output_tensor = model.tensors[operation.outputs[0]]
    options = operation.options
    input_channels = int(input_tensor.shape[3])
    if depthwise:
        kernel_h, kernel_w = int(weight_tensor.shape[1]), int(weight_tensor.shape[2])
        output_channels = int(weight_tensor.shape[3])
        groups = input_channels
    else:
        output_channels = int(weight_tensor.shape[0])
        kernel_h, kernel_w = int(weight_tensor.shape[1]), int(weight_tensor.shape[2])
        groups = 1
    stride_h, stride_w = int(options["stride_h"]), int(options["stride_w"])
    dilation_h, dilation_w = int(options["dilation_h"]), int(options["dilation_w"])
    padding = "same" if int(options["padding"]) == PADDING_SAME else "valid"
    if padding == "same":
        top, bottom = _same_padding(
            int(input_tensor.shape[1]),
            int(output_tensor.shape[1]),
            kernel_h,
            stride_h,
            dilation_h,
        )
        left, right = _same_padding(
            int(input_tensor.shape[2]),
            int(output_tensor.shape[2]),
            kernel_w,
            stride_w,
            dilation_w,
        )
    else:
        left = right = top = bottom = 0
    return {
        "in_channels": input_channels,
        "out_channels": output_channels,
        "kernel_size": [kernel_h, kernel_w],
        "stride": [stride_h, stride_w],
        "dilation": [dilation_h, dilation_w],
        "groups": groups,
        "has_bias": len(operation.inputs) >= 3 and operation.inputs[2] >= 0,
        "padding": padding,
        "pad": [left, right, top, bottom],
    }


def _decode_constant_map(model: TFLiteModel) -> dict[int, np.ndarray[Any, Any]]:
    """Map DEQUANTIZE outputs to FP32 arrays and preserve direct constants."""
    constants: dict[int, np.ndarray[Any, Any]] = {}
    for operation in model.operators:
        if operation.name != "DEQUANTIZE":
            continue
        source = operation.inputs[0]
        constants[operation.outputs[0]] = model.tensor_array(source).astype(np.float32)
    for index, tensor in enumerate(model.tensors):
        if model.buffers[tensor.buffer_index]:
            constants.setdefault(index, model.tensor_array(index))
    return constants


def _convert_channel_pad(paddings_nhwc: np.ndarray[Any, Any]) -> list[int]:
    """Convert TFLite NHWC paddings to the order accepted by torch.nn.functional.pad."""
    if paddings_nhwc.shape != (4, 2):
        raise ValueError(f"Expected [4, 2] paddings, got {paddings_nhwc.shape}")
    n, h, w, c = paddings_nhwc.astype(np.int64).tolist()
    if n != [0, 0]:
        raise ValueError("Batch padding is not supported by this static converter")
    return [w[0], w[1], h[0], h[1], c[0], c[1]]


def _fully_connected_conv_config(
    model: TFLiteModel,
    operation: OperatorInfo,
) -> dict[str, Any]:
    """Express one spatial-map FULLY_CONNECTED as an equivalent 1x1 Conv2d."""
    input_tensor = model.tensors[operation.inputs[0]]
    weight_tensor = model.tensors[operation.inputs[1]]
    if len(input_tensor.shape) != 4:
        raise NotImplementedError(
            f"FULLY_CONNECTED at index {operation.index} expects a rank-4 "
            f"NHWC input, got {input_tensor.shape}."
        )
    if int(weight_tensor.shape[1]) != int(input_tensor.shape[3]):
        raise NotImplementedError(
            f"FULLY_CONNECTED at index {operation.index} mixes spatial and "
            "channel dimensions, which a 1x1 convolution cannot express."
        )
    return {
        "in_channels": int(input_tensor.shape[3]),
        "out_channels": int(weight_tensor.shape[0]),
        "kernel_size": [1, 1],
        "stride": [1, 1],
        "dilation": [1, 1],
        "groups": 1,
        "has_bias": len(operation.inputs) >= 3 and operation.inputs[2] >= 0,
        "padding": "valid",
        "pad": [0, 0, 0, 0],
    }


def build_specification(
    model: TFLiteModel,
) -> tuple[dict[str, Any], dict[int, np.ndarray[Any, Any]]]:
    """Build the JSON graph specification and decoded constant mapping."""
    constants = _decode_constant_map(model)
    operations: list[dict[str, Any]] = []
    synthetic_tensor = len(model.tensors)
    for operation in model.operators:
        if operation.name == "DEQUANTIZE":
            continue
        fused_activation = int(
            operation.options.get("fused_activation", FUSED_ACTIVATION_NONE) or 0
        )
        if fused_activation not in (FUSED_ACTIVATION_NONE, FUSED_ACTIVATION_RELU6):
            raise NotImplementedError(
                f"Operator {operation.index} uses unsupported fused "
                f"activation {fused_activation}."
            )
        if fused_activation == FUSED_ACTIVATION_RELU6 and operation.name not in {
            "CONV_2D",
            "DEPTHWISE_CONV_2D",
        }:
            raise NotImplementedError(
                f"Fused RELU6 on {operation.name} at index {operation.index} "
                "is not supported."
            )
        config: dict[str, Any] = {}
        if operation.name == "CONV_2D":
            config = _conv_config(model, operation, depthwise=False)
        elif operation.name == "DEPTHWISE_CONV_2D":
            config = _conv_config(model, operation, depthwise=True)
        elif operation.name == "FULLY_CONNECTED":
            config = _fully_connected_conv_config(model, operation)
            weight_index = int(operation.inputs[1])
            weight = constants[weight_index]
            # Store the weight in the OHWI Conv2d layout expected by the
            # shared CONV_2D parameter loader.
            constants[weight_index] = weight.reshape(
                weight.shape[0], 1, 1, weight.shape[1]
            )
        elif operation.name == "MEAN":
            axes = constants[operation.inputs[1]].astype(np.int64).reshape(-1).tolist()
            if sorted(axes) != [1, 2] or bool(operation.options.get("keep_dims")):
                raise NotImplementedError(
                    f"MEAN at index {operation.index} must reduce NHWC axes "
                    "[1, 2] without keep_dims."
                )
            config = {}
        elif operation.name == "LOGISTIC":
            config = {}
        elif operation.name == "PRELU":
            config = {"channels": int(model.tensors[operation.inputs[0]].shape[3])}
        elif operation.name == "MAX_POOL_2D":
            input_tensor = model.tensors[operation.inputs[0]]
            output_tensor = model.tensors[operation.outputs[0]]
            options = operation.options
            kernel_h, kernel_w = int(options["filter_h"]), int(options["filter_w"])
            stride_h, stride_w = int(options["stride_h"]), int(options["stride_w"])
            if int(options["padding"]) == PADDING_SAME:
                top, bottom = _same_padding(
                    int(input_tensor.shape[1]),
                    int(output_tensor.shape[1]),
                    kernel_h,
                    stride_h,
                    1,
                )
                left, right = _same_padding(
                    int(input_tensor.shape[2]),
                    int(output_tensor.shape[2]),
                    kernel_w,
                    stride_w,
                    1,
                )
                if any((left, right, top, bottom)):
                    raise NotImplementedError(
                        "This model requires padded max pooling, which is not expected"
                    )
            config = {
                "kernel_size": [kernel_h, kernel_w],
                "stride": [stride_h, stride_w],
            }
        elif operation.name == "PAD":
            config = {"pad": _convert_channel_pad(constants[operation.inputs[1]])}
        elif operation.name == "RESIZE_BILINEAR":
            size = constants[operation.inputs[1]].astype(np.int64).reshape(-1)
            config = {
                "size": [int(size[0]), int(size[1])],
                "align_corners": bool(operation.options["align_corners"]),
                "half_pixel_centers": bool(operation.options["half_pixel_centers"]),
            }
        elif operation.name == "RESHAPE":
            shape = constants[operation.inputs[1]].astype(np.int64).reshape(-1).tolist()
            config = {
                "shape": [int(value) for value in shape],
                "nhwc_memory_order": len(model.tensors[operation.inputs[0]].shape) == 4,
            }
        elif operation.name == "CONCATENATION":
            rank = len(model.tensors[operation.inputs[0]].shape)
            axis = int(operation.options["axis"])
            if rank == 4:
                axis = [0, 2, 3, 1][axis]
            config = {"axis": axis}
        elif operation.name == "ADD":
            config = {}
        else:
            raise NotImplementedError(
                f"Unsupported operator {operation.name} at index {operation.index}"
            )
        # FULLY_CONNECTED over a spatial map is stored as an equivalent 1x1
        # CONV_2D so the executor, quantizer, and exporter share one path.
        name = "CONV_2D" if operation.name == "FULLY_CONNECTED" else operation.name
        outputs = [int(value) for value in operation.outputs]
        if fused_activation == FUSED_ACTIVATION_RELU6:
            # Split the fused activation into an explicit RELU6 operation so
            # activation quantization observes the post-activation tensor.
            fused_output = outputs[0]
            outputs = [synthetic_tensor]
            synthetic_tensor += 1
        operations.append(
            {
                "index": operation.index,
                "name": name,
                "inputs": [int(value) for value in operation.inputs if value >= 0],
                "outputs": outputs,
                "config": config,
            }
        )
        if fused_activation == FUSED_ACTIVATION_RELU6:
            operations.append(
                {
                    "index": operation.index,
                    "name": "RELU6",
                    "inputs": [outputs[0]],
                    "outputs": [fused_output],
                    "config": {},
                }
            )
    input_tensor = model.tensors[int(model.inputs[0])]
    specification = {
        "format_version": 1,
        "source": model.path.name,
        "input_layout": "NCHW",
        "input_shape": [int(value) for value in input_tensor.shape],
        "inputs": [int(value) for value in model.inputs],
        "outputs": [int(value) for value in model.outputs],
        "operations": operations,
    }
    return specification, constants


def load_parameters(
    pytorch_model: HandDetector,
    specification: dict[str, Any],
    constants: dict[int, np.ndarray[Any, Any]],
) -> None:
    """Load converted TFLite constants into the generated PyTorch modules."""
    with torch.no_grad():
        for operation, layer in zip(specification["operations"], pytorch_model.layers):
            name = operation["name"]
            inputs = operation["inputs"]
            if name == "CONV_2D":
                weight = torch.from_numpy(constants[int(inputs[1])]).permute(0, 3, 1, 2)
                layer.conv.weight.copy_(weight)
                if layer.conv.bias is not None:
                    layer.conv.bias.copy_(torch.from_numpy(constants[int(inputs[2])]))
            elif name == "DEPTHWISE_CONV_2D":
                weight = torch.from_numpy(constants[int(inputs[1])]).permute(3, 0, 1, 2)
                layer.conv.weight.copy_(weight)
                if layer.conv.bias is not None:
                    layer.conv.bias.copy_(torch.from_numpy(constants[int(inputs[2])]))
            elif name == "PRELU":
                alpha = torch.from_numpy(constants[int(inputs[1])]).reshape(-1)
                layer.weight.copy_(alpha)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tflite", type=Path)
    parser.add_argument("--spec", type=Path, default=Path("hand_detector_spec.json"))
    parser.add_argument("--weights", type=Path, default=Path("hand_detector_float.pt"))
    return parser.parse_args()


def main() -> None:
    """Convert the TFLite graph and write a specification and state dictionary."""
    args = parse_args()
    tflite_model = TFLiteModel(args.tflite)
    specification, constants = build_specification(tflite_model)
    pytorch_model = HandDetector(specification)
    load_parameters(pytorch_model, specification, constants)
    args.spec.write_text(json.dumps(specification, indent=2), encoding="utf-8")
    torch.save(pytorch_model.state_dict(), args.weights)
    parameter_count = sum(parameter.numel() for parameter in pytorch_model.parameters())
    print(f"Wrote {args.spec}")
    print(f"Wrote {args.weights}")
    print(f"Parameters: {parameter_count:,}")


if __name__ == "__main__":
    main()
