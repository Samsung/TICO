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

"""Check Pylint against real Transformers constructors and invalid keyword calls."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

from astroid import MANAGER, parse
from transformers import LlamaConfig

from infra.style.transformers_config_pylint import (
    _has_generated_constructor,
    _infer_config_constructor,
)


class TransformersConfigPylintTest(unittest.TestCase):
    """Keep inference consistent with generated and explicit runtime signatures."""

    @patch.dict(MANAGER.astroid_cache)
    def test_generated_signature_keeps_dataclass_fields_and_custom_initializers(
        self,
    ) -> None:
        """Retain field defaults and exclusions without replacing authored methods."""
        module = parse(
            textwrap.dedent("""\
                from dataclasses import dataclass, field
                from typing import ClassVar

                @dataclass
                class PreTrainedConfig:
                    inherited: int = 1

                class GeneratedConfig(PreTrainedConfig):
                    width: int = 32
                    metadata: ClassVar[str] = "config"
                    internal: int = field(default=0, init=False)

                class ExplicitConfig(PreTrainedConfig):
                    def __init__(self, width):
                        self.width = width
                """),
            module_name="transformers.configuration_utils",
        )
        generated = module["GeneratedConfig"]
        self.assertTrue(_has_generated_constructor(generated))
        _infer_config_constructor(generated)
        arguments = generated["__init__"].args
        self.assertEqual([argument.name for argument in arguments.args], ["self"])
        self.assertEqual(
            [argument.name for argument in arguments.kwonlyargs],
            ["inherited", "width"],
        )
        self.assertEqual([default.value for default in arguments.kw_defaults], [1, 32])
        self.assertEqual(arguments.kwarg, "kwargs")
        explicit = module["ExplicitConfig"]
        self.assertFalse(_has_generated_constructor(explicit))
        self.assertIsNone(explicit["__init__"].args.kwarg)

    def test_generated_fields_and_extension_keywords_match_runtime(self) -> None:
        """Accept library fields and extension keys while checking explicit signatures."""
        config = LlamaConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
        )
        self.assertEqual(config.hidden_size, 32)

        source = textwrap.dedent("""\
            from transformers import LlamaConfig

            valid = LlamaConfig(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                custom_extension=True,
            )

            class ExplicitConfig(LlamaConfig):
                def __init__(self, width=32):
                    super().__init__(hidden_size=width)

            class PlainConfig:
                def __init__(self, width=32):
                    self.width = width

            ExplicitConfig(width=16)
            PlainConfig(width=16)
            ExplicitConfig(invalid_keyword=True)
            PlainConfig(invalid_keyword=True)
            """)
        project_root = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "constructor_calls.py"
            fixture.write_text(source, encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pylint",
                    "--rcfile",
                    str(project_root / ".pylintrc"),
                    "--disable=all",
                    "--enable=unexpected-keyword-arg",
                    "--jobs=1",
                    "--output-format=json",
                    "--reports=no",
                    "--score=no",
                    str(fixture),
                ],
                cwd=project_root,
                check=False,
                capture_output=True,
                text=True,
                timeout=300,
            )
        self.assertEqual(completed.returncode, 2, completed.stdout + completed.stderr)
        messages = json.loads(completed.stdout)
        self.assertEqual(len(messages), 2, messages)
        self.assertEqual(
            {message["line"] for message in messages},
            {
                index
                for index, line in enumerate(source.splitlines(), start=1)
                if "(invalid_keyword=True)" in line
            },
        )
        self.assertTrue(
            all(message["symbol"] == "unexpected-keyword-arg" for message in messages)
        )


if __name__ == "__main__":
    unittest.main()
