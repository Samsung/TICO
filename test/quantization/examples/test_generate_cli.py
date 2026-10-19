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

import contextlib
import io
import unittest
from pathlib import Path

from tico.quantization.examples.generate import parse_args


class TestGenerateCLI(unittest.TestCase):
    def test_numeric_input_and_defaults(self):
        args = parse_args(["--bundle", "bundle", "--input-ids", "1, 2,0"])
        self.assertEqual(args.bundle, Path("bundle"))
        self.assertEqual(args.input_ids, [1, 2, 0])
        self.assertIsNone(args.prompt)
        self.assertEqual(args.max_new_tokens, 16)
        self.assertIsNone(args.output_json)

    def test_prompt_and_output(self):
        args = parse_args(
            [
                "--bundle",
                "bundle",
                "--prompt",
                " Hello ",
                "--max-new-tokens",
                "0",
                "--output-json",
                "result.json",
            ]
        )
        self.assertEqual(args.prompt, " Hello ")
        self.assertIsNone(args.input_ids)
        self.assertEqual(args.max_new_tokens, 0)
        self.assertEqual(args.output_json, Path("result.json"))

    def test_rejects_missing_or_ambiguous_input(self):
        cases: list[list[str]] = [[], ["--prompt", "Hello", "--input-ids", "1"]]
        for extra in cases:
            with self.subTest(extra=extra):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as caught:
                        parse_args(["--bundle", "bundle", *extra])
                self.assertEqual(caught.exception.code, 2)

    def test_rejects_malformed_numeric_input(self):
        for value in ("", "1,", "1,,2", "a", "1.0", "-1", "9223372036854775808"):
            with self.subTest(value=value):
                error = io.StringIO()
                with contextlib.redirect_stderr(error):
                    with self.assertRaises(SystemExit) as caught:
                        parse_args(["--bundle", "bundle", "--input-ids", value])
                self.assertEqual(caught.exception.code, 2)
                self.assertIn("--input-ids", error.getvalue())

    def test_rejects_negative_or_noninteger_step_count(self):
        for value in ("-1", "1.5", "many"):
            with self.subTest(value=value):
                error = io.StringIO()
                with contextlib.redirect_stderr(error):
                    with self.assertRaises(SystemExit):
                        parse_args(
                            [
                                "--bundle",
                                "bundle",
                                "--input-ids",
                                "1",
                                "--max-new-tokens",
                                value,
                            ]
                        )
                self.assertIn("must be a nonnegative integer", error.getvalue())

    def test_rejects_blank_prompt(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(["--bundle", "bundle", "--prompt", " \t"])


if __name__ == "__main__":
    unittest.main()
