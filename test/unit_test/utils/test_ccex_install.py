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

"""Tests for the ccex install command."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tico.utils.compat import torch_version_policy as policy


class CcexInstallTest(unittest.TestCase):
    """Verify package-index handling without invoking a real pip installation."""

    project_root: Path
    install_script: Path
    torch_family: str
    torch_version: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = Path(__file__).resolve().parents[3]
        cls.install_script = cls.project_root / "infra/scripts/install.sh"
        cls.torch_family = policy.DEFAULT_FAMILY
        cls.torch_version = policy.LATEST_STABLE_VERSION[cls.torch_family]

    def _run_install(
        self, *arguments: str
    ) -> tuple[subprocess.CompletedProcess[str], list[list[str]]]:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fake_bin = temp_path / "bin"
            fake_bin.mkdir()
            pip_call_log = temp_path / "pip-calls.log"
            pip_call_log.touch()
            torch_marker = temp_path / "torch-installed"

            fake_python = fake_bin / "python3"
            fake_python.write_text(
                self._fake_python_script(),
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            for executable in ("nvcc", "nvidia-smi"):
                path = fake_bin / executable
                path.write_text("#!/bin/bash\nexit 1\n", encoding="utf-8")
                path.chmod(0o755)

            environment = os.environ.copy()
            environment.update(
                {
                    "CCEX_PROJECT_PATH": str(self.project_root),
                    "FAKE_TORCH_FAMILY": self.torch_family,
                    "FAKE_TORCH_VERSION": self.torch_version,
                    "PATH": f"{fake_bin}{os.pathsep}{environment['PATH']}",
                    "PIP_CALL_LOG": str(pip_call_log),
                    "TORCH_MARKER": str(torch_marker),
                }
            )
            completed = subprocess.run(
                ["bash", str(self.install_script), *arguments],
                cwd=self.project_root,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            calls = [
                shlex.split(line)
                for line in pip_call_log.read_text(encoding="utf-8").splitlines()
            ]
            return completed, calls

    @staticmethod
    def _fake_python_script() -> str:
        real_python = shlex.quote(sys.executable)
        return rf"""#!/bin/bash
if [[ "$1" == "-m" && "$2" == "pip" ]]; then
  printf '%q ' "$@" >> "${{PIP_CALL_LOG}}"
  printf '\n' >> "${{PIP_CALL_LOG}}"
  for argument in "$@"; do
    case "${{argument}}" in
      torch|torch==*) touch "${{TORCH_MARKER}}" ;;
    esac
  done
  exit 0
fi

if [[ "$1" == "-" ]]; then
  if [[ -f "${{TORCH_MARKER}}" ]]; then
    printf '%s\t%s\t%s\tcpu\t0\n' \
      "${{FAKE_TORCH_VERSION}}" \
      "${{FAKE_TORCH_VERSION}}" \
      "${{FAKE_TORCH_FAMILY}}"
    exit 0
  fi
  exit 1
fi

exec {real_python} "$@"
"""

    def test_index_url_is_forwarded_to_every_pip_install(self) -> None:
        """Use one explicit index for Torch, requirements, and TICO itself."""
        index_url = "https://user:super-secret@packages.example.com/simple"

        completed, calls = self._run_install(
            "--torch_ver",
            self.torch_version,
            "--index-url",
            index_url,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(len(calls), 3)
        self.assertIn(f"torch=={self.torch_version}", calls[0])
        for call in calls:
            self.assertEqual(call[:3], ["-m", "pip", "install"])
            option_index = call.index("--index-url")
            self.assertEqual(call[option_index + 1], index_url)
        self.assertNotIn("download.pytorch.org", "\n".join(map(str, calls)))
        self.assertNotIn("super-secret", completed.stdout)
        self.assertNotIn("super-secret", completed.stderr)

    def test_default_index_behavior_is_unchanged(self) -> None:
        """Keep generated Torch indices and normal pip defaults without the option."""
        completed, calls = self._run_install("--torch_ver", self.torch_version)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(len(calls), 3)
        self.assertEqual(
            calls[0][calls[0].index("--index-url") + 1],
            "https://download.pytorch.org/whl/cpu",
        )
        self.assertNotIn("--index-url", calls[1])
        self.assertNotIn("--index-url", calls[2])

    def test_empty_index_url_is_rejected(self) -> None:
        """Reject an explicitly empty package-index value before running pip."""
        completed, calls = self._run_install("--index-url", "")

        self.assertEqual(completed.returncode, 1)
        self.assertEqual(calls, [])
        self.assertIn("requires a non-empty URL", completed.stderr)

    def test_help_describes_index_url(self) -> None:
        """Expose the option through the command's built-in help."""
        completed, calls = self._run_install("--help")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(calls, [])
        self.assertIn("--index-url URL", completed.stdout)


if __name__ == "__main__":
    unittest.main()
