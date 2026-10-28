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

"""Explicit, temporary input preparation for static model export."""

from contextlib import contextmanager, ExitStack
from typing import Iterator

import torch
from torch import nn

from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase


class ExportCapabilityGuard:
    """Retain a stage's readiness owner without registering its unused weights.

    Stages may be constructed before freeze, so readiness is checked both here
    and at every entry. A plain floating stage must explicitly declare NO_QUANT;
    quantized stages always require the live WrapQ owner's capability protocol.
    """

    def __init__(
        self,
        owner: nn.Module,
        *capabilities: str,
        mode: Mode | None = None,
    ) -> None:
        self.owner = owner
        self.capabilities = capabilities
        self.mode = mode
        self.require()

    def require(self) -> None:
        if isinstance(self.owner, QuantModuleBase):
            self.owner.require_entry_available()
            for capability in self.capabilities:
                self.owner.require_capability(capability)
            if self.mode is not None and self.owner._mode is not self.mode:
                raise RuntimeError(
                    "Export stage mode does not match its capability owner; rebuild "
                    "the adapter after calibration or conversion."
                )
        elif self.mode not in (None, Mode.NO_QUANT) or (
            getattr(self.owner, "_mode", self.mode) is not Mode.NO_QUANT
        ):
            raise TypeError(
                "Quantized export requires a WrapQ capability owner; plain "
                "floating stages must explicitly declare Mode.NO_QUANT."
            )


def validate_static_length(length: object, capacity: int) -> int:
    """Validate an explicit profile without consulting calibration settings."""
    if isinstance(length, bool) or not isinstance(length, int):
        raise TypeError("Static export max_seq_len must be an integer.")
    if length < 1 or length > capacity:
        raise ValueError(
            f"Static export max_seq_len must be in [1, {capacity}], got {length}."
        )
    return length


def require_static_decoder_inputs(attention_mask, position_embeddings) -> None:
    """Reject missing host inputs instead of entering an eager fallback."""
    if not isinstance(attention_mask, torch.Tensor):
        raise ValueError("Static decoder export requires an explicit attention_mask.")
    if (
        not isinstance(position_embeddings, (tuple, list))
        or len(position_embeddings) != 2
        or not all(isinstance(value, torch.Tensor) for value in position_embeddings)
    ):
        raise ValueError("Static decoder export requires explicit (cos, sin) tensors.")


@contextmanager
def prepared_static_export(model: nn.Module, max_seq_len: int) -> Iterator[None]:
    """Install model-owned static inputs only for the duration of export.

    Owners implement ``_build_static_export_inputs(max_seq_len)``. Builders must
    not recalibrate, rewrap, or consult mutable eager rotary/cache state. Frozen
    quantized modules are shared, not copied. Temporary buffers are restored even
    when preparation, tracing, or serialization fails. This context is not safe
    for concurrent eager/export calls on the same model instance.
    """
    validate_static_length(max_seq_len, max_seq_len)
    owners = tuple(model.modules())
    for owner in owners:
        if getattr(getattr(owner, "_mode", None), "name", None) == "CALIB":
            raise ValueError("Static export cannot run while calibration is enabled.")
    with ExitStack() as stack:
        covered: set[int] = set()
        for owner in owners:
            if id(owner) in covered:
                continue
            builder = getattr(owner, "_build_static_export_inputs", None)
            if not callable(builder):
                continue
            # A full-model owner supplies positions to its layers; do not allocate
            # duplicate S-by-S masks for each nested decoder owner.
            covered.update(id(module) for module in owner.modules())
            if getattr(owner, "_static_export_inputs", None) is not None:
                raise ValueError("Nested static export preparation is not supported.")
            inputs = builder(max_seq_len)
            if not isinstance(inputs, nn.Module):
                raise TypeError("A static export input builder must return nn.Module.")
            existed = hasattr(owner, "_static_export_inputs")
            previous = getattr(owner, "_static_export_inputs", None)
            was_registered = "_static_export_inputs" in owner._modules

            def restore(
                owner=owner,
                existed=existed,
                previous=previous,
                was_registered=was_registered,
            ):
                if hasattr(owner, "_static_export_inputs"):
                    delattr(owner, "_static_export_inputs")
                if was_registered:
                    owner.add_module("_static_export_inputs", previous)
                elif existed:
                    setattr(owner, "_static_export_inputs", previous)

            stack.callback(restore)
            if existed:
                delattr(owner, "_static_export_inputs")
            owner.add_module("_static_export_inputs", inputs)
        yield
