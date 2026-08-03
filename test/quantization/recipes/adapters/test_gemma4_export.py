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

try:
    from quantization.recipes.optional_dependency_stubs import (
        install_optional_dependency_stubs,
    )
except ModuleNotFoundError:
    from optional_dependency_stubs import install_optional_dependency_stubs

install_optional_dependency_stubs()

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import tico.quantization.recipes.adapters.gemma4 as gemma4_mod
from tico.quantization.recipes.adapters.gemma4 import Gemma4Adapter
from tico.quantization.recipes.context import RecipeContext


class TestGemma4AdapterExport(unittest.TestCase):
    """Tests for config-driven Gemma4 checkpoint export."""

    def test_export_saves_configured_checkpoint_artifacts(self):
        """Checkpoint artifact aliases should use the shared export helper."""
        adapter = Gemma4Adapter()
        model = object()

        for artifact in ("ptq_checkpoint", "checkpoint"):
            with (
                self.subTest(artifact=artifact),
                tempfile.TemporaryDirectory() as tmpdir,
            ):
                ctx = RecipeContext(
                    cfg={
                        "export": {
                            "enabled": True,
                            "output_dir": tmpdir,
                            "artifacts": [artifact],
                        }
                    },
                    adapter=adapter,
                    model=model,
                )

                with patch.object(gemma4_mod, "save_checkpoint") as save_checkpoint:
                    adapter.export(ctx)

                save_checkpoint.assert_called_once_with(model, Path(tmpdir))

    def test_export_is_noop_when_disabled(self):
        """Disabled export should not write a checkpoint."""
        adapter = Gemma4Adapter()
        ctx = RecipeContext(
            cfg={
                "export": {
                    "enabled": False,
                    "output_dir": "./out/gemma4",
                    "artifacts": ["ptq_checkpoint"],
                }
            },
            adapter=adapter,
            model=object(),
        )

        with patch.object(gemma4_mod, "save_checkpoint") as save_checkpoint:
            adapter.export(ctx)

        save_checkpoint.assert_not_called()


if __name__ == "__main__":
    unittest.main()
