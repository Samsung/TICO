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

from tico.quantization.config.utils import auto_qscheme_for, dtype_is_unsigned
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.qscheme import QScheme


class TestUtils(unittest.TestCase):
    def test_dtype_is_unsigned_true(self):
        self.assertTrue(dtype_is_unsigned(DType.uint(8)))
        self.assertTrue(dtype_is_unsigned(DType.uint(4)))

    def test_dtype_is_unsigned_false(self):
        self.assertFalse(dtype_is_unsigned(DType.int(8)))
        self.assertFalse(dtype_is_unsigned(DType.int(16)))

    def test_auto_qscheme_for_unsigned_activation(self):
        self.assertEqual(
            auto_qscheme_for(DType.uint(8), "act_in"),
            QScheme.PER_TENSOR_ASYMM,
        )

    def test_auto_qscheme_for_unsigned_weight(self):
        self.assertEqual(
            auto_qscheme_for(DType.uint(8), "weight"),
            QScheme.PER_CHANNEL_ASYMM,
        )

    def test_auto_qscheme_for_signed_dtype(self):
        self.assertEqual(
            auto_qscheme_for(DType.int(8), "act_in"),
            QScheme.PER_TENSOR_SYMM,
        )
        self.assertEqual(
            auto_qscheme_for(DType.int(16), "weight"),
            QScheme.PER_TENSOR_SYMM,
        )

    def test_auto_qscheme_for_default_obs_name(self):
        """
        When obs_name is None, it should behave like activation.
        """
        self.assertEqual(
            auto_qscheme_for(DType.uint(8), None),
            QScheme.PER_TENSOR_ASYMM,
        )
        self.assertEqual(
            auto_qscheme_for(DType.int(8), None),
            QScheme.PER_TENSOR_SYMM,
        )
