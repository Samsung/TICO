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

"""Convert the MediaPipe palm-detector TFLite model into PyTorch artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from examples.hand_detector._support.conversion import (
    build_specification,
    load_parameters,
)
from examples.hand_detector._support.tflite_flatbuffer import TFLiteModel
from examples.hand_detector.hand_detector import HandDetector


DIRECTORY = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    """Parse source and destination artifact paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tflite", type=Path)
    parser.add_argument(
        "--spec",
        type=Path,
        default=DIRECTORY / "hand_detector_spec.json",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DIRECTORY / "hand_detector_float.pt",
    )
    return parser.parse_args()


def main() -> None:
    """Convert the source graph and save a specification and state dictionary."""
    args = parse_args()
    source = TFLiteModel(args.tflite)
    specification, constants = build_specification(source)
    model = HandDetector(specification)
    load_parameters(model, specification, constants)

    args.spec.parent.mkdir(parents=True, exist_ok=True)
    args.weights.parent.mkdir(parents=True, exist_ok=True)
    args.spec.write_text(json.dumps(specification, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), args.weights)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Wrote {args.spec}")
    print(f"Wrote {args.weights}")
    print(f"Parameters: {parameter_count:,}")


if __name__ == "__main__":
    main()
