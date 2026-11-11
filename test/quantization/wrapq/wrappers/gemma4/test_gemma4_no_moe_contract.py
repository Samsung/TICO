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

import types
import unittest

from tico.quantization.wrapq.wrappers.gemma4.utils import (
    assert_gemma4_dense_no_moe,
    assert_gemma4_e2b_no_moe,
)


class Gemma4NoMoeContractTest(unittest.TestCase):
    """Test the dense-only contract for the dense Gemma4 runtime."""

    def test_no_moe_contract_accepts_dense_config(self) -> None:
        """E2B and E4B shapes use the same structural dense guard."""
        for hidden, intermediate, layers, kv_heads, shared, double_wide in (
            (1536, 6144, 35, 1, 20, True),
            (2560, 10240, 42, 2, 18, False),
        ):
            with self.subTest(num_layers=layers):
                text_config = types.SimpleNamespace(
                    enable_moe_block=False,
                    hidden_size=hidden,
                    intermediate_size=intermediate,
                    num_hidden_layers=layers,
                    num_key_value_heads=kv_heads,
                    num_kv_shared_layers=shared,
                    use_double_wide_mlp=double_wide,
                )
                config = types.SimpleNamespace(text_config=text_config)
                assert_gemma4_dense_no_moe(config)
                assert_gemma4_e2b_no_moe(config)

    def test_legacy_helper_is_equivalent(self) -> None:
        """Assistant and external callers retain the old no-MoE symbol."""
        self.assertIs(assert_gemma4_e2b_no_moe, assert_gemma4_dense_no_moe)

    def test_no_moe_contract_rejects_moe_config(self) -> None:
        """MoE configs should be rejected by the no-MoE guard."""
        text_config = types.SimpleNamespace(enable_moe_block=True)
        config = types.SimpleNamespace(text_config=text_config)

        for guard in (assert_gemma4_dense_no_moe, assert_gemma4_e2b_no_moe):
            with self.assertRaisesRegex(ValueError, "dense decoder"):
                guard(config)

    def test_no_moe_contract_rejects_moe_modules(self) -> None:
        """Models containing router or expert modules should be rejected."""

        class Gemma4TextRouter:
            """Minimal class whose name matches the unsupported router type."""

        class FakeModel:
            """Minimal model-like object exposing a dense config and named modules."""

            config = types.SimpleNamespace(
                text_config=types.SimpleNamespace(enable_moe_block=False)
            )

            def named_modules(self):
                """Yield a fake MoE module to exercise the module-name guard."""
                yield "model.language_model.layers.0.router", Gemma4TextRouter()

        for guard in (assert_gemma4_dense_no_moe, assert_gemma4_e2b_no_moe):
            with self.assertRaisesRegex(ValueError, "Unexpected MoE module"):
                guard(FakeModel())


if __name__ == "__main__":
    unittest.main()
