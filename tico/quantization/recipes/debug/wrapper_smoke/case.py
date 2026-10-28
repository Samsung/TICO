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

import inspect
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from typing import Any, cast, Mapping

import torch


@dataclass(frozen=True)
class ForwardInput:
    """Container for positional and keyword arguments used by a smoke case."""

    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExportInvocation:
    """An export adapter and the host inputs for its exact forward ABI."""

    module: torch.nn.Module
    inputs: ForwardInput

    def validate_static_decoder_inputs(self) -> None:
        """Check direct decoder ABIs before tracing, without executing the model.

        Signature binding handles positional production ABIs as well as named
        smoke inputs. Unrelated adapters retain their existing input contracts.
        A cache parameter denotes a cache-aware (decode or append-prefill) ABI;
        shared-KV consumers supply the complete key/value instead of past KV.
        """
        from tico.quantization.wrapq.utils.static_export import (
            require_static_decoder_inputs,
        )

        signature = inspect.signature(self.module.forward)
        if not {"hidden_states", "attention_mask", "position_embeddings"}.issubset(
            signature.parameters
        ):
            return
        bound = signature.bind(*self.inputs.args, **dict(self.inputs.kwargs))
        bound.apply_defaults()
        values = bound.arguments
        mask = values.get("attention_mask")
        rope = values.get("position_embeddings")
        require_static_decoder_inputs(mask, rope)
        mask = cast(torch.Tensor, mask)
        rope = cast(tuple[torch.Tensor, torch.Tensor], rope)
        hidden = values["hidden_states"]
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            raise ValueError("Static decoder hidden_states must have shape [B, Q, D].")
        batch, query, _ = hidden.shape
        if mask.ndim not in (3, 4) or mask.shape[0] != batch or mask.shape[-2] != query:
            raise ValueError(
                "Static decoder attention_mask must match hidden batch and query length."
            )
        if mask.ndim == 4 and mask.shape[1] != 1:
            raise ValueError(
                "Static decoder attention_mask must have a singleton head dimension."
            )
        cos, sin = rope
        if cos.shape != sin.shape or cos.ndim != 3 or cos.shape[:2] != (batch, query):
            raise ValueError(
                "Static decoder (cos, sin) must have matching [B, Q, head_dim] shapes."
            )
        owner = getattr(self.module, "wrapped", self.module)
        attention = getattr(owner, "self_attn", owner)
        attention = getattr(attention, "wrapped", attention)
        config = getattr(attention, "config", None)
        if config is not None and hidden.shape[-1] != config.hidden_size:
            raise ValueError(
                "Static decoder hidden size must match the wrapped configuration."
            )
        head_dim = getattr(attention, "head_dim", None)
        if head_dim is not None and cos.shape[-1] != head_dim:
            raise ValueError(
                "Static decoder RoPE must match the wrapped head dimension."
            )
        key_length = mask.shape[-1]
        shared = values.get("shared_key_value")
        if getattr(attention, "is_kv_shared_layer", False) and shared is None:
            raise ValueError(
                "Static shared-KV decoder export requires explicit shared KV tensors."
            )
        past_name = next(
            (
                name
                for name in ("past_key_value", "past_key_values")
                if name in signature.parameters
            ),
            None,
        )
        past = values.get(past_name) if past_name else None
        cache = shared if shared is not None else past
        if past_name is not None and cache is None:
            raise ValueError(
                "Static cache-aware decoder export requires explicit past or shared KV tensors."
            )
        if cache is None:
            if key_length != query:
                raise ValueError(
                    "Static prefill key length must equal query length without cache inputs."
                )
            return
        if (
            not isinstance(cache, (tuple, list))
            or len(cache) != 2
            or not all(isinstance(value, torch.Tensor) for value in cache)
        ):
            raise ValueError("Static decoder cache must be a (key, value) tensor pair.")
        key, value = cache
        kv_heads = getattr(attention, "num_kv_heads", None)
        if key.ndim == 4 and kv_heads is not None and key.shape[1] != kv_heads:
            raise ValueError(
                "Static decoder cache must match the wrapped KV head count."
            )
        expected_length = key_length if shared is not None else key_length - query
        if (
            key.ndim != 4
            or key.shape != value.shape
            or key.shape[0] != batch
            or key.shape[-2] != expected_length
            or key.shape[-1] != cos.shape[-1]
        ):
            raise ValueError(
                "Static decoder cache must match batch, key length and RoPE head dimension."
            )


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
    supports_circle_export: bool = True
    circle_export_unsupported_reason: str | None = None

    def availability(self) -> CaseAvailability:
        """Return whether this case can run in the current environment."""
        return CaseAvailability(True)

    def validate_config(self, cfg: Mapping[str, Any]) -> None:
        """Validate case-specific options before constructing a module."""

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

    def export_context(
        self,
        quantized: torch.nn.Module,
        eval_sample: ForwardInput,
        cfg: Mapping[str, Any],
    ) -> AbstractContextManager[None]:
        """Keep production static preparation active through adapter creation/tracing.

        Model-specific cases override this hook when their production export
        context owns temporary modes, shape profiles, or forward preparation.
        """
        return nullcontext()

    def export_invocation(
        self,
        quantized: torch.nn.Module,
        eval_sample: ForwardInput,
        cfg: Mapping[str, Any],
    ) -> ExportInvocation:
        """Assemble an adapter with its matching inputs as one export contract.

        Existing cases can keep their module/input hooks. Strict decoder cases
        may override this method to build host inputs from the actual wrapper.
        """
        return ExportInvocation(
            self.export_module(quantized, cfg), self.export_input(eval_sample, cfg)
        )

    def export_filename(self, cfg: Mapping[str, Any]) -> str:
        """Return the default Circle filename for this smoke case."""
        return f"{self.name}.q.circle"
