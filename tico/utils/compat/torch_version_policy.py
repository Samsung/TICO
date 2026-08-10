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

"""Central PyTorch version and CI policy for TICO.

Keep the policy in this module so that the installer, runtime warnings, tests,
and GitHub Actions all consume the same version window.

The package dependency intentionally remains unbounded so users can install TICO
into environments that already use older PyTorch releases. Installer acceptance,
qualified support, and CI coverage are separate policy tiers.

A newly released stable family should first be added to
``QUALIFICATION_CANDIDATE_FAMILIES``. After the scheduled compatibility job has
remained green for the qualification window, move it into
``SUPPORTED_STABLE_FAMILIES``, promote it to ``DEFAULT_FAMILY``, and move the
oldest supported family into ``LEGACY_INSTALLABLE_FAMILIES``.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
from collections.abc import Iterable, Mapping, Sequence
from typing import Final

DEFAULT_FAMILY: Final = "2.12"
LEGACY_INSTALLABLE_FAMILIES: Final = ("2.5", "2.6", "2.7", "2.8", "2.9")
SUPPORTED_STABLE_FAMILIES: Final = ("2.10", "2.11", "2.12")
QUALIFICATION_CANDIDATE_FAMILIES: Final = ("2.13",)
QUALIFICATION_WINDOW_DAYS: Final = 28
PACKAGE_TORCH_DEPENDENCY: Final = "torch"
PINNED_NIGHTLY_SELECTOR: Final = "nightly"
LATEST_NIGHTLY_SELECTOR: Final = "nightly-latest"
NIGHTLY_SELECTORS: Final = (
    PINNED_NIGHTLY_SELECTOR,
    LATEST_NIGHTLY_SELECTOR,
)

LATEST_STABLE_VERSION: Final[dict[str, str]] = {
    "2.5": "2.5.1",
    "2.6": "2.6.0",
    "2.7": "2.7.1",
    "2.8": "2.8.0",
    "2.9": "2.9.1",
    "2.10": "2.10.0",
    "2.11": "2.11.0",
    "2.12": "2.12.1",
    "2.13": "2.13.0",
}

# Values are ordered from newest to oldest. The installer selects the newest
# published wheel that does not exceed the detected host CUDA capability.
STABLE_CUDA_WHEELS: Final[dict[str, tuple[str, ...]]] = {
    "2.5": ("12.4", "12.1", "11.8"),
    "2.6": ("12.6", "12.4", "11.8"),
    "2.7": ("12.8", "12.6", "11.8"),
    "2.8": ("12.9", "12.8", "12.6"),
    "2.9": ("13.0", "12.8", "12.6"),
    "2.10": ("13.0", "12.8", "12.6"),
    "2.11": ("13.0", "12.8", "12.6"),
    "2.12": ("13.2", "13.0", "12.6"),
    "2.13": ("13.2", "13.0", "12.6"),
}

NIGHTLY_CUDA_FALLBACKS: Final = ("13.2", "13.0", "12.6")

INSTALLABLE_FAMILIES: Final = (
    *LEGACY_INSTALLABLE_FAMILIES,
    *SUPPORTED_STABLE_FAMILIES,
    *QUALIFICATION_CANDIDATE_FAMILIES,
)
MINIMUM_SUPPORTED_VERSION: Final = LATEST_STABLE_VERSION[SUPPORTED_STABLE_FAMILIES[0]]

_FAMILY_PATTERN = re.compile(r"^(\d+\.\d+)")


def version_family(version: str) -> str:
    """Return the ``major.minor`` family from a PyTorch version string."""
    match = _FAMILY_PATTERN.match(version)
    if match is None:
        raise ValueError(f"Unrecognized PyTorch version: {version}")
    return match.group(1)


def is_legacy_installable_family(family: str) -> bool:
    """Return whether a stable family is accepted on a best-effort basis."""
    return family in LEGACY_INSTALLABLE_FAMILIES


def is_supported_family(family: str) -> bool:
    """Return whether a stable family is in TICO's qualified support window."""
    return family in SUPPORTED_STABLE_FAMILIES


def is_candidate_family(family: str) -> bool:
    """Return whether a stable family is being qualified for future support."""
    return family in QUALIFICATION_CANDIDATE_FAMILIES


def is_installable_family(family: str) -> bool:
    """Return whether the source installer may explicitly install a family."""
    return family in INSTALLABLE_FAMILIES


def is_nightly_selector(selector: str) -> bool:
    """Return whether a CLI selector requests a PyTorch nightly channel."""
    return selector in NIGHTLY_SELECTORS


def _family_key(family: str) -> tuple[int, int]:
    major, minor = family.split(".", maxsplit=1)
    return int(major), int(minor)


def _validate_contiguous(families: Sequence[str], label: str) -> None:
    keys = [_family_key(family) for family in families]
    if keys != sorted(keys):
        raise ValueError(f"{label} must be sorted from oldest to newest: {families}")

    for previous, current in zip(keys, keys[1:]):
        if current[0] != previous[0] or current[1] != previous[1] + 1:
            raise ValueError(f"{label} must be contiguous: {families}")


def validate_policy() -> None:
    """Raise ``ValueError`` when policy metadata is internally inconsistent."""
    if len(set(NIGHTLY_SELECTORS)) != len(NIGHTLY_SELECTORS):
        raise ValueError("Nightly selectors must be unique")
    if any(selector in INSTALLABLE_FAMILIES for selector in NIGHTLY_SELECTORS):
        raise ValueError("Nightly selectors must not overlap stable family names")

    if len(SUPPORTED_STABLE_FAMILIES) != 3:
        raise ValueError("TICO must keep exactly three qualified stable families")
    if DEFAULT_FAMILY not in SUPPORTED_STABLE_FAMILIES:
        raise ValueError("DEFAULT_FAMILY must be a qualified stable family")
    if DEFAULT_FAMILY != SUPPORTED_STABLE_FAMILIES[-1]:
        raise ValueError("DEFAULT_FAMILY must be the newest qualified stable family")

    _validate_contiguous(
        LEGACY_INSTALLABLE_FAMILIES,
        "LEGACY_INSTALLABLE_FAMILIES",
    )
    _validate_contiguous(SUPPORTED_STABLE_FAMILIES, "SUPPORTED_STABLE_FAMILIES")
    _validate_contiguous(INSTALLABLE_FAMILIES, "INSTALLABLE_FAMILIES")

    tier_sets = (
        set(LEGACY_INSTALLABLE_FAMILIES),
        set(SUPPORTED_STABLE_FAMILIES),
        set(QUALIFICATION_CANDIDATE_FAMILIES),
    )
    overlaps = any(
        left.intersection(right)
        for index, left in enumerate(tier_sets)
        for right in tier_sets[index + 1 :]
    )
    if overlaps:
        raise ValueError("Legacy, supported, and candidate families must not overlap")

    if LEGACY_INSTALLABLE_FAMILIES:
        legacy_major, legacy_minor = _family_key(LEGACY_INSTALLABLE_FAMILIES[-1])
        if _family_key(SUPPORTED_STABLE_FAMILIES[0]) != (
            legacy_major,
            legacy_minor + 1,
        ):
            raise ValueError(
                "The supported window must immediately follow the legacy window"
            )

    if QUALIFICATION_CANDIDATE_FAMILIES:
        _validate_contiguous(
            QUALIFICATION_CANDIDATE_FAMILIES,
            "QUALIFICATION_CANDIDATE_FAMILIES",
        )
        supported_major, supported_minor = _family_key(SUPPORTED_STABLE_FAMILIES[-1])
        if _family_key(QUALIFICATION_CANDIDATE_FAMILIES[0]) != (
            supported_major,
            supported_minor + 1,
        ):
            raise ValueError(
                "The first candidate must immediately follow the newest "
                "supported family"
            )

    if PACKAGE_TORCH_DEPENDENCY != "torch":
        raise ValueError(
            "The package dependency must remain unbounded; support is enforced "
            "by policy and CI, not by pip resolution"
        )

    expected = set(INSTALLABLE_FAMILIES)
    if set(LATEST_STABLE_VERSION) != expected:
        raise ValueError(
            "LATEST_STABLE_VERSION keys must match all installable stable families"
        )
    if set(STABLE_CUDA_WHEELS) != expected:
        raise ValueError(
            "STABLE_CUDA_WHEELS keys must match all installable stable families"
        )

    for family, version in LATEST_STABLE_VERSION.items():
        if version_family(version) != family:
            raise ValueError(f"Version {version} does not belong to family {family}")
    for family, cuda_versions in STABLE_CUDA_WHEELS.items():
        if not cuda_versions:
            raise ValueError(f"No CUDA wheel candidates configured for torch {family}")


def _matrix_entry(
    torch_version: str,
    *,
    tier: str,
    experimental: bool,
) -> dict[str, object]:
    return {
        "torch-version": torch_version,
        "tier": tier,
        "experimental": experimental,
    }


def github_matrix(kind: str) -> dict[str, list[dict[str, object]]]:
    """Return a GitHub Actions matrix for one compatibility test tier."""
    include: list[dict[str, object]] = []

    if kind == "pr-smoke":
        oldest = SUPPORTED_STABLE_FAMILIES[0]
        if oldest != DEFAULT_FAMILY:
            include.append(
                _matrix_entry(oldest, tier="oldest-supported", experimental=False)
            )
        include.extend(
            _matrix_entry(family, tier="candidate", experimental=True)
            for family in QUALIFICATION_CANDIDATE_FAMILIES
        )
    elif kind == "scheduled-full":
        include.extend(
            _matrix_entry(family, tier="supported", experimental=False)
            for family in SUPPORTED_STABLE_FAMILIES
        )
        include.extend(
            _matrix_entry(family, tier="candidate", experimental=True)
            for family in QUALIFICATION_CANDIDATE_FAMILIES
        )
        include.append(
            _matrix_entry(
                LATEST_NIGHTLY_SELECTOR,
                tier="nightly-latest",
                experimental=True,
            )
        )
    elif kind == "release":
        include.extend(
            _matrix_entry(family, tier="supported", experimental=False)
            for family in SUPPORTED_STABLE_FAMILIES
        )
    else:
        raise ValueError(f"Unknown matrix kind: {kind}")

    return {"include": include}


def _shell_array(values: Iterable[str]) -> str:
    return "(" + " ".join(shlex.quote(value) for value in values) + ")"


def _shell_map(name: str, values: Mapping[str, str | Sequence[str]]) -> list[str]:
    lines = [f"declare -A {name}=("]
    for key, value in values.items():
        rendered = " ".join(value) if not isinstance(value, str) else value
        lines.append(f"  [{shlex.quote(key)}]={shlex.quote(rendered)}")
    lines.append(")")
    return lines


def shell_assignments() -> str:
    """Render policy metadata as Bash declarations for installer scripts."""
    lines = [
        f"PYTORCH_DEFAULT_FAMILY={shlex.quote(DEFAULT_FAMILY)}",
        "PYTORCH_SUPPORTED_FAMILIES=" + _shell_array(SUPPORTED_STABLE_FAMILIES),
        "PYTORCH_CANDIDATE_FAMILIES=" + _shell_array(QUALIFICATION_CANDIDATE_FAMILIES),
        "PYTORCH_LEGACY_INSTALLABLE_FAMILIES="
        + _shell_array(LEGACY_INSTALLABLE_FAMILIES),
        "PYTORCH_INSTALLABLE_FAMILIES=" + _shell_array(INSTALLABLE_FAMILIES),
        "PYTORCH_SUPPORTED_FAMILY_MIN=" + shlex.quote(SUPPORTED_STABLE_FAMILIES[0]),
        "PYTORCH_SUPPORTED_FAMILY_MAX=" + shlex.quote(SUPPORTED_STABLE_FAMILIES[-1]),
        "PYTORCH_INSTALLABLE_FAMILY_MIN=" + shlex.quote(INSTALLABLE_FAMILIES[0]),
        "PYTORCH_INSTALLABLE_FAMILY_MAX=" + shlex.quote(INSTALLABLE_FAMILIES[-1]),
        "PYTORCH_QUALIFICATION_WINDOW_DAYS="
        + shlex.quote(str(QUALIFICATION_WINDOW_DAYS)),
        "PYTORCH_PINNED_NIGHTLY_SELECTOR=" + shlex.quote(PINNED_NIGHTLY_SELECTOR),
        "PYTORCH_LATEST_NIGHTLY_SELECTOR=" + shlex.quote(LATEST_NIGHTLY_SELECTOR),
        "PYTORCH_NIGHTLY_SELECTORS=" + _shell_array(NIGHTLY_SELECTORS),
    ]
    lines.extend(_shell_map("PYTORCH_LATEST_STABLE_VERSION", LATEST_STABLE_VERSION))
    lines.extend(_shell_map("PYTORCH_STABLE_CUDA_WHEELS", STABLE_CUDA_WHEELS))
    lines.append(
        "PYTORCH_NIGHTLY_CUDA_FALLBACKS=" + _shell_array(NIGHTLY_CUDA_FALLBACKS)
    )
    return "\n".join(lines)


def _range_text(families: Sequence[str]) -> str:
    if len(families) == 1:
        return families[0]
    return f"{families[0]} ~ {families[-1]}"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate", help="Validate policy invariants")
    subparsers.add_parser("shell", help="Print Bash policy declarations")
    subparsers.add_parser("default-family", help="Print the default Torch family")
    subparsers.add_parser("supported-range", help="Print the qualified range")
    subparsers.add_parser("installable-range", help="Print the installer range")
    subparsers.add_parser(
        "package-dependency", help="Print the unbounded package dependency"
    )

    matrix_parser = subparsers.add_parser(
        "ci-matrix", help="Print a compact GitHub Actions matrix"
    )
    matrix_parser.add_argument(
        "kind",
        choices=("pr-smoke", "scheduled-full", "release"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the small CLI consumed by shell scripts and GitHub Actions."""
    args = _build_parser().parse_args(argv)
    validate_policy()

    if args.command == "validate":
        return 0
    if args.command == "shell":
        print(shell_assignments())
    elif args.command == "default-family":
        print(DEFAULT_FAMILY)
    elif args.command == "supported-range":
        print(_range_text(SUPPORTED_STABLE_FAMILIES))
    elif args.command == "installable-range":
        print(_range_text(INSTALLABLE_FAMILIES))
    elif args.command == "package-dependency":
        print(PACKAGE_TORCH_DEPENDENCY)
    elif args.command == "ci-matrix":
        print(json.dumps(github_matrix(args.kind), separators=(",", ":")))
    else:  # pragma: no cover - argparse rejects unknown commands.
        raise AssertionError(f"Unhandled command: {args.command}")
    return 0


validate_policy()

if __name__ == "__main__":
    raise SystemExit(main())
