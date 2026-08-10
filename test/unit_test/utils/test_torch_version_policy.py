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

"""Tests for the central PyTorch compatibility policy."""

from __future__ import annotations

import json
import re
import subprocess
import unittest
from pathlib import Path

from tico.utils.compat import torch_version_policy as policy


class TorchVersionPolicyTest(unittest.TestCase):
    """Keep installer and CI metadata internally consistent."""

    def test_policy_is_valid(self) -> None:
        """Validate all static policy invariants."""
        policy.validate_policy()

    def test_default_is_latest_supported_family(self) -> None:
        """Use the newest qualified stable family as the source default."""
        self.assertEqual(
            policy.DEFAULT_FAMILY,
            policy.SUPPORTED_STABLE_FAMILIES[-1],
        )

    def test_pr_smoke_matrix_does_not_repeat_default(self) -> None:
        """Keep PR compatibility jobs smaller than the scheduled matrix."""
        entries = policy.github_matrix("pr-smoke")["include"]
        versions = [entry["torch-version"] for entry in entries]

        self.assertNotIn(policy.DEFAULT_FAMILY, versions)
        self.assertIn(policy.SUPPORTED_STABLE_FAMILIES[0], versions)
        for candidate in policy.QUALIFICATION_CANDIDATE_FAMILIES:
            self.assertIn(candidate, versions)

    def test_scheduled_matrix_covers_all_tiers(self) -> None:
        """Run every supported family and nightly in scheduled CI."""
        entries = policy.github_matrix("scheduled-full")["include"]
        by_version = {entry["torch-version"]: entry for entry in entries}

        for family in policy.SUPPORTED_STABLE_FAMILIES:
            self.assertIn(family, by_version)
            self.assertFalse(by_version[family]["experimental"])
        for family in policy.QUALIFICATION_CANDIDATE_FAMILIES:
            self.assertIn(family, by_version)
            self.assertTrue(by_version[family]["experimental"])
        self.assertNotIn(policy.PINNED_NIGHTLY_SELECTOR, by_version)
        self.assertTrue(by_version[policy.LATEST_NIGHTLY_SELECTOR]["experimental"])

    def test_legacy_families_remain_installable_but_not_qualified(self) -> None:
        """Keep older source-install selections available on a best-effort basis."""
        for family in policy.LEGACY_INSTALLABLE_FAMILIES:
            self.assertTrue(policy.is_legacy_installable_family(family))
            self.assertTrue(policy.is_installable_family(family))
            self.assertFalse(policy.is_supported_family(family))
            self.assertIn(family, policy.LATEST_STABLE_VERSION)
            self.assertIn(family, policy.STABLE_CUDA_WHEELS)

    def test_project_declares_unbounded_torch_dependency(self) -> None:
        """Do not let package metadata reject unqualified PyTorch versions."""
        project_root = Path(__file__).resolve().parents[3]
        pyproject = (project_root / "pyproject.toml").read_text(encoding="utf-8")
        torch_dependencies = re.findall(
            r'^\s*"(torch(?:[<>=!~][^"]*)?)",\s*$',
            pyproject,
            flags=re.MULTILINE,
        )
        self.assertEqual(torch_dependencies, [policy.PACKAGE_TORCH_DEPENDENCY])

    def test_nightly_selectors_have_distinct_semantics(self) -> None:
        """Keep reproducible and moving nightly channels explicitly separate."""
        self.assertTrue(policy.is_nightly_selector(policy.PINNED_NIGHTLY_SELECTOR))
        self.assertTrue(policy.is_nightly_selector(policy.LATEST_NIGHTLY_SELECTOR))
        self.assertNotEqual(
            policy.PINNED_NIGHTLY_SELECTOR,
            policy.LATEST_NIGHTLY_SELECTOR,
        )

    def test_latest_nightly_installation_is_owned_by_ccex(self) -> None:
        """Keep the moving nightly resolver in ccex, not the composite action."""
        project_root = Path(__file__).resolve().parents[3]
        action = (
            project_root / ".github/actions/setup-tico-test/action.yml"
        ).read_text(encoding="utf-8")
        install = (project_root / "infra/scripts/install.sh").read_text(
            encoding="utf-8"
        )
        configure = (project_root / "infra/scripts/test_configure.sh").read_text(
            encoding="utf-8"
        )
        package_utils = (
            project_root / "infra/scripts/pytorch_package_utils.sh"
        ).read_text(encoding="utf-8")

        self.assertIn('./ccex install --dist --torch_ver "$TORCH_VERSION"', action)
        self.assertIn('./ccex configure test --torch_ver "$TORCH_VERSION"', action)
        self.assertNotIn("python3 -m pip", action)
        self.assertIn(
            "install_torch --pre --upgrade torch torchvision",
            install,
        )
        self.assertIn('NIGHTLY_MODE="latest"', configure)
        self.assertIn(
            "Keeping installed latest nightly torchvision",
            configure,
        )
        self.assertIn("python3 -m pip check", configure)
        self.assertNotIn("pytorch_pip_check", package_utils)
        self.assertNotIn("filter-nightly-pip-check", package_utils)

    def test_cli_matrix_is_compact_json(self) -> None:
        """Keep matrix output safe for one-line GitHub Actions outputs."""
        completed = subprocess.run(
            [
                "python3",
                policy.__file__,
                "ci-matrix",
                "scheduled-full",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertNotIn("\n", completed.stdout.rstrip("\n"))
        self.assertEqual(
            json.loads(completed.stdout),
            policy.github_matrix("scheduled-full"),
        )

    def test_shell_assignments_can_be_sourced(self) -> None:
        """Ensure Bash can consume the generated installer metadata."""
        script = (
            "set -euo pipefail\n"
            + policy.shell_assignments()
            + "\n"
            + 'test "$PYTORCH_DEFAULT_FAMILY" = "'
            + policy.DEFAULT_FAMILY
            + '"\n'
            + 'test "${#PYTORCH_LEGACY_INSTALLABLE_FAMILIES[@]}" -eq 5\n'
            + 'test "${#PYTORCH_SUPPORTED_FAMILIES[@]}" -eq 3\n'
            + 'test "$PYTORCH_PINNED_NIGHTLY_SELECTOR" = "'
            + policy.PINNED_NIGHTLY_SELECTOR
            + '"\n'
            + 'test "$PYTORCH_LATEST_NIGHTLY_SELECTOR" = "'
            + policy.LATEST_NIGHTLY_SELECTOR
            + '"\n'
            + 'test "${PYTORCH_LATEST_STABLE_VERSION['
            + policy.DEFAULT_FAMILY
            + ']}" = "'
            + policy.LATEST_STABLE_VERSION[policy.DEFAULT_FAMILY]
            + '"\n'
        )
        subprocess.run(["bash"], input=script, check=True, text=True)


if __name__ == "__main__":
    unittest.main()
