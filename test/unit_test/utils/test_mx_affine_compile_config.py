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

"""Domain legalization is independent of equivalent layout boundary movement."""

import unittest

import tico


class MXAffineCompileConfigTest(unittest.TestCase):
    def test_default_and_missing_dictionary_key_are_disabled(self):
        for config in (
            tico.get_default_config(),
            tico.CompileConfigV1(),
            tico.CompileConfigV1.from_dict({"version": "1.0"}),
        ):
            with self.subTest(config=config):
                self.assertIs(config.get("legalize_mx_affine_domains"), False)
                self.assertIs(config.to_dict()["legalize_mx_affine_domains"], False)
                self.assertIs(config.get("move_affine_quantize_before_layout"), True)

    def test_constructor_get_set_and_dictionary_roundtrip(self):
        for enabled in (False, True):
            for move_layout in (False, True):
                with self.subTest(enabled=enabled, move_layout=move_layout):
                    config = tico.CompileConfigV1(
                        legalize_mx_affine_domains=enabled,
                        move_affine_quantize_before_layout=move_layout,
                    )
                    self.assertIs(config.get("legalize_mx_affine_domains"), enabled)
                    restored = tico.CompileConfigV1.from_dict(config.to_dict())
                    self.assertEqual(restored.to_dict(), config.to_dict())
                    restored.set("legalize_mx_affine_domains", not enabled)
                    self.assertIs(
                        restored.to_dict()["legalize_mx_affine_domains"], not enabled
                    )
                    self.assertIs(
                        restored.get("move_affine_quantize_before_layout"), move_layout
                    )


if __name__ == "__main__":
    unittest.main()
