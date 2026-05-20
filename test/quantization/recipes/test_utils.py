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

import unittest
from dataclasses import dataclass

import torch

from tico.quantization.recipes.utils import (
    filter_dataclass_kwargs,
    qscheme_from_name,
    stage_payload,
    torch_dtype_from_name,
    wrapq_dtype_from_name,
)
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.qscheme import QScheme


@dataclass
class ExampleConfig:
    """Small dataclass used to test keyword filtering."""

    enabled: bool = True
    value: int = 0


class TestRecipeUtils(unittest.TestCase):
    def test_torch_dtype_from_name_accepts_common_aliases(self):
        """Torch dtype parsing should accept common FP aliases."""
        self.assertIs(torch_dtype_from_name("float32"), torch.float32)
        self.assertIs(torch_dtype_from_name("fp32"), torch.float32)
        self.assertIs(torch_dtype_from_name("float16"), torch.float16)
        self.assertIs(torch_dtype_from_name("bf16"), torch.bfloat16)

    def test_torch_dtype_from_name_rejects_unknown_dtype(self):
        """Unknown torch dtype names should raise a ValueError."""
        with self.assertRaises(ValueError):
            torch_dtype_from_name("int8")

    def test_wrapq_dtype_from_name_accepts_signed_and_unsigned_names(self):
        """WrapQ dtype parsing should convert textual dtypes to DType objects."""
        self.assertEqual(wrapq_dtype_from_name("int16"), DType.int(16))
        self.assertEqual(wrapq_dtype_from_name("uint4"), DType.uint(4))
        self.assertEqual(wrapq_dtype_from_name(8), DType.uint(8))

    def test_wrapq_dtype_from_name_uses_default_for_none(self):
        """A default dtype should be used when the input value is None."""
        self.assertEqual(
            wrapq_dtype_from_name(None, default=DType.int(8)),
            DType.int(8),
        )

    def test_wrapq_dtype_from_name_rejects_unknown_value(self):
        """Invalid WrapQ dtype names should raise a ValueError."""
        with self.assertRaises(ValueError):
            wrapq_dtype_from_name("float8")

    def test_qscheme_from_name_parses_supported_schemes(self):
        """QScheme parsing should accept the public config spellings."""
        self.assertEqual(
            qscheme_from_name("per_tensor_asymm"), QScheme.PER_TENSOR_ASYMM
        )
        self.assertEqual(qscheme_from_name("per_tensor_symm"), QScheme.PER_TENSOR_SYMM)
        self.assertEqual(
            qscheme_from_name("per_channel_asymm"), QScheme.PER_CHANNEL_ASYMM
        )
        self.assertEqual(
            qscheme_from_name("per_channel_symm"), QScheme.PER_CHANNEL_SYMM
        )

    def test_qscheme_from_name_rejects_unknown_scheme(self):
        """Unknown qscheme names should raise a ValueError."""
        with self.assertRaises(ValueError):
            qscheme_from_name("per_group_symm")

    def test_stage_payload_removes_control_keys(self):
        """Stage payload extraction should remove the runner control keys."""
        payload = stage_payload({"name": "ptq", "enabled": True, "weight_bits": 4})

        self.assertEqual(payload, {"weight_bits": 4})

    def test_filter_dataclass_kwargs_keeps_only_dataclass_fields(self):
        """Dataclass keyword filtering should drop unknown config keys."""
        kwargs = filter_dataclass_kwargs(
            ExampleConfig,
            {"enabled": False, "value": 3, "unknown": "drop"},
        )

        self.assertEqual(kwargs, {"enabled": False, "value": 3})
