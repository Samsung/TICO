# Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

"""
Unit-tests for the `Mode` enumeration that governs the wrapper
finite-state-machine.

Checks
------
1. Enumeration contains exactly the three expected members.
2. Natural ordering follows definition order: NO_QUANT < CALIB < QUANT
   (the `auto()` values are monotonically increasing).
3. The `str(...)` representation returns the lower-case name
   ("no_quant", "calib", "quant").
"""

import unittest

from tico.experimental.quantization.wrapq.mode import Mode


class TestModeEnum(unittest.TestCase):
    def test_member_names(self):
        self.assertEqual(
            list(Mode.__members__.keys()),
            ["NO_QUANT", "CALIB", "QUANT"],
            msg="Mode enum must contain exactly NO_QUANT, CALIB, QUANT in that order",
        )

    def test_ordering(self):
        # auto() assigns consecutive integers starting from 1
        self.assertLess(Mode.NO_QUANT.value, Mode.CALIB.value)
        self.assertLess(Mode.CALIB.value, Mode.QUANT.value)

    def test_str_representation(self):
        self.assertEqual(str(Mode.NO_QUANT), "no_quant")
        self.assertEqual(str(Mode.CALIB), "calib")
        self.assertEqual(str(Mode.QUANT), "quant")
