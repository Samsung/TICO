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

from typing import Final


MX_ELEM_FORMAT_TO_DTYPE: Final[dict[str, str]] = {
    "int8": "mxint8",
    "fp4": "mxfp4",
    "fp4_e2m1": "mxfp4",
    "fp6_e3m2": "mxfp6_e3m2",
    "fp6_e2m3": "mxfp6_e2m3",
    "fp8_e4m3": "mxfp8_e4m3",
    "fp8_e5m2": "mxfp8_e5m2",
}

MX_DTYPE_TO_ELEM_FORMAT: Final[dict[str, str]] = {
    dtype: elem_format
    for elem_format, dtype in MX_ELEM_FORMAT_TO_DTYPE.items()
    if elem_format != "fp4_e2m1"
}

SUPPORTED_MX_ELEM_FORMATS: Final[frozenset[str]] = frozenset(
    MX_ELEM_FORMAT_TO_DTYPE.keys()
)
SUPPORTED_MX_DTYPES: Final[frozenset[str]] = frozenset(MX_DTYPE_TO_ELEM_FORMAT.keys())


def mx_dtype_from_elem_format(elem_format: str) -> str:
    """Return the Circle quantized dtype string for an MX element format."""
    try:
        return MX_ELEM_FORMAT_TO_DTYPE[elem_format]
    except KeyError as exc:
        raise ValueError(f"Unsupported MX element format: {elem_format!r}") from exc


def elem_format_from_mx_dtype(dtype: str) -> str:
    """Return the MX element format encoded by a Circle quantized dtype string."""
    try:
        return MX_DTYPE_TO_ELEM_FORMAT[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported MX dtype: {dtype!r}") from exc


def is_mx_dtype(dtype: str) -> bool:
    """Return True when the dtype string denotes a supported MX dtype."""
    return dtype in SUPPORTED_MX_DTYPES


def normalize_mx_elem_format(elem_format: str) -> str:
    """Return the canonical MX element format spelling used by Circle metadata."""
    if elem_format == "fp4_e2m1":
        return "fp4"
    if elem_format in SUPPORTED_MX_ELEM_FORMATS:
        return elem_format
    raise ValueError(f"Unsupported MX element format: {elem_format!r}")


def assert_supported_mx_export_options(
    *,
    elem_format: str,
    shared_exp_method: str,
    round: str,
) -> None:
    """Validate MX fake-quant options that can be represented in Circle qparams.

    Circle tensor quantization metadata currently carries the MX dtype and the
    quantized dimension. It does not have fields for the shared-exponent method
    or the rounding mode, so this helper rejects non-default options before the
    Q-DQ pair is folded into tensor metadata.
    """
    normalize_mx_elem_format(elem_format)
    if shared_exp_method != "max":
        raise RuntimeError(
            "Circle MX export currently supports only shared_exp_method='max'. "
            f"Got {shared_exp_method!r}."
        )
    if round != "nearest":
        raise RuntimeError(
            "Circle MX export currently supports only round='nearest'. "
            f"Got {round!r}."
        )
