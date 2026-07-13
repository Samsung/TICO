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

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Type

from tico.quantization.config.utils import auto_qscheme_for, dtype_is_unsigned
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.observers.base import ObserverBase
from tico.quantization.wrapq.observers.minmax import MinMaxObserver
from tico.quantization.wrapq.observers.mx import MXObserver
from tico.quantization.wrapq.qscheme import QScheme


def _resolve_qscheme(
    *,
    dtype: DType,
    qscheme: Optional[QScheme],
    context: str,
    obs_name: Optional[str] = None,
) -> QScheme:
    """Resolve and validate a dtype/qscheme pair."""
    resolved_qscheme = qscheme or auto_qscheme_for(dtype, obs_name)

    if dtype_is_unsigned(dtype) and resolved_qscheme.is_symmetric():
        raise ValueError(
            f"Invalid quantization spec at {context}: unsigned dtype "
            f"{dtype!r} cannot be paired with symmetric qscheme "
            f"{resolved_qscheme!r}."
        )

    return resolved_qscheme


@dataclass(frozen=True)
class QuantSpec:
    """
    Describe how a logical observer should be instantiated.

    QuantSpec is the public quantization policy unit. It can represent affine
    integer quantization through `dtype`/`qscheme` and backend-specific
    formats such as MX through observer-specific keyword arguments.
    """

    observer: Type[ObserverBase]
    dtype: Optional[DType] = None
    qscheme: Optional[QScheme] = None
    kwargs: Mapping[str, Any] = field(default_factory=dict)

    def to_kwargs(
        self,
        obs_name: Optional[str] = None,
        *,
        context: str = "QuantSpec",
        infer_qscheme: bool = True,
        mark_replace: bool = False,
    ) -> dict[str, Any]:
        """
        Return observer constructor kwargs represented by this spec.

        Args:
            obs_name: Logical observer name used for qscheme inference.
            context: Error context used for validation messages.
            infer_qscheme: If True, missing affine qscheme values are inferred
                immediately. If False, only explicitly configured qschemes are
                emitted so wrapper defaults may still participate in precedence.
            mark_replace: If True, add the internal
                ``__quant_spec_replace_role__`` marker to the returned mapping.
                ``QuantModuleBase`` consumes and removes the marker before
                constructing the observer. A marked QuantSpec is treated as a
                complete observer-level policy, so fields omitted from the
                role-level activation or weight QuantSpec are not inherited.
                Wrapper-provided defaults may still supply fields omitted from
                the explicit observer override.
        """
        out: dict[str, Any] = {"observer": self.observer}
        if mark_replace:
            out["__quant_spec_replace_role__"] = True
        out.update(dict(self.kwargs))

        if self.dtype is not None:
            out["dtype"] = self.dtype
            if self.qscheme is not None or infer_qscheme:
                out["qscheme"] = _resolve_qscheme(
                    dtype=self.dtype,
                    qscheme=self.qscheme,
                    context=context,
                    obs_name=obs_name,
                )
        elif self.qscheme is not None:
            out["qscheme"] = self.qscheme

        return out


def affine(
    dtype: DType,
    *,
    qscheme: Optional[QScheme] = None,
    observer: Type[ObserverBase] = MinMaxObserver,
    **kwargs: Any,
) -> QuantSpec:
    """
    Create an affine quantization spec.

    Args:
        dtype: Integer quantized dtype used by affine observers.
        qscheme: Optional quantization scheme. When omitted, it is inferred
            from the dtype and observer role.
        observer: Observer class used to collect stats and fake-quantize.
        **kwargs: Additional observer constructor keyword arguments.

    Returns:
        A QuantSpec that expands to affine observer kwargs.
    """
    if qscheme is not None:
        _resolve_qscheme(
            dtype=dtype,
            qscheme=qscheme,
            context="affine",
        )

    return QuantSpec(
        observer=observer,
        dtype=dtype,
        qscheme=qscheme,
        kwargs=kwargs,
    )


def mx(
    elem_format: str = "int8",
    *,
    axis: int = -1,
    shared_exp_method: str = "max",
    round: str = "nearest",
) -> QuantSpec:
    """
    Create an MX micro-scaling quantization spec.

    Args:
        elem_format: MX element format such as `"int8"` or `"fp8_e4m3"`.
        axis: Tensor axis used for shared-exponent grouping.
        shared_exp_method: Shared exponent selection method.
        round: Rounding mode passed to the MX backend.

    Returns:
        A QuantSpec that expands to MXObserver kwargs.
    """
    return QuantSpec(
        observer=MXObserver,
        kwargs={
            "elem_format": elem_format,
            "axis": axis,
            "shared_exp_method": shared_exp_method,
            "round": round,
        },
    )
