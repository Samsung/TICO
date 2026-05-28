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

"""Case interfaces for wrapper-level quantization smoke checks."""

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class ForwardInput:
    """Container for positional and keyword arguments used by a smoke case."""

    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CaseAvailability:
    """Availability status for an optional smoke case dependency."""

    available: bool
    reason: str | None = None


class WrapperSmokeCase:
    """Base class for a reusable wrapper-level smoke case.

    Subclasses describe only module construction, input generation, reference
    execution, and optional export details. The common prepare-calibrate-convert,
    metric, plot, report, and Circle export flow is implemented by the runner.
    """

    name: str = ""
    description: str = ""
    tags: tuple[str, ...] = ()
    max_mean_abs_diff: float | None = None
    max_peir: float | None = None
    min_mean_abs_diff: float | None = None
    compare_reference_source: str = "reference"
    inplace_prepare: bool = False
    inplace_convert: bool = False

    def availability(self) -> CaseAvailability:
        """Return whether this case can run in the current environment."""
        return CaseAvailability(True)

    def ptq_config(self, cfg: Mapping[str, Any]) -> Any:
        """Build the PTQ config used to prepare the floating-point module."""
        from tico.quantization.config.ptq import PTQConfig

        return PTQConfig()

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build the mutable module to quantize and a floating-point reference."""
        raise NotImplementedError

    def after_prepare(self, prepared: torch.nn.Module, cfg: Mapping[str, Any]) -> None:
        """Apply case-specific tweaks after prepare and before calibration."""

    def prepare_model(
        self, model: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> torch.nn.Module:
        """Prepare a model for PTQ using the case configuration."""
        from tico.quantization import prepare

        return prepare(model, self.ptq_config(cfg), inplace=self.inplace_prepare)

    def convert_model(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> torch.nn.Module:
        """Convert a calibrated prepared model into quantized simulation mode."""
        from tico.quantization import convert

        return convert(prepared, inplace=self.inplace_convert)

    def calibration_inputs(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> list[ForwardInput]:
        """Return calibration inputs for the prepared module."""
        raise NotImplementedError

    def eval_input(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Return the input used for the post-convert numerical check."""
        inputs = self.calibration_inputs(prepared, cfg)
        if not inputs:
            raise ValueError(f"Case '{self.name}' did not produce evaluation input.")
        return inputs[0]

    def forward(self, module: torch.nn.Module, sample: ForwardInput) -> Any:
        """Run the quantized or prepared module for one sample."""
        return module(*sample.args, **dict(sample.kwargs))

    def reference_forward(
        self, reference: torch.nn.Module, sample: ForwardInput
    ) -> Any:
        """Run the floating-point reference module for one sample."""
        return self.forward(reference, sample)

    def output_tensor(self, output: Any) -> torch.Tensor:
        """Select the tensor used for parity metrics and plotting."""
        from .utils import first_tensor

        return first_tensor(output)

    def export_module(
        self, quantized: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> torch.nn.Module:
        """Return the module object to pass to tico.convert for Circle export."""
        return quantized

    def export_input(
        self, eval_sample: ForwardInput, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Return the input object to pass to tico.convert for Circle export."""
        return eval_sample

    def export_filename(self, cfg: Mapping[str, Any]) -> str:
        """Return the default Circle filename for this smoke case."""
        return f"{self.name}.q.circle"
