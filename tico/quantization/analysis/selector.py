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

"""Composable selectors for WrapQ quantization sites."""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Callable, Iterable

from tico.quantization.analysis.profile import QuantizationProfile
from tico.quantization.wrapq.control import QuantizationSite, SiteRole


@dataclass(frozen=True)
class SiteSelector:
    """Wrap a composable predicate over ``QuantizationSite`` values."""

    predicate: Callable[[QuantizationSite], bool]
    description: str = "custom"

    def __call__(self, site: QuantizationSite) -> bool:
        return bool(self.predicate(site))

    def __and__(self, other: "SiteSelector") -> "SiteSelector":
        return SiteSelector(
            lambda site: self(site) and other(site),
            f"({self.description} and {other.description})",
        )

    def __or__(self, other: "SiteSelector") -> "SiteSelector":
        return SiteSelector(
            lambda site: self(site) or other(site),
            f"({self.description} or {other.description})",
        )

    def __invert__(self) -> "SiteSelector":
        return SiteSelector(lambda site: not self(site), f"not ({self.description})")

    @classmethod
    def all(cls) -> "SiteSelector":
        """Select every quantization site."""
        return cls(lambda _site: True, "all")

    @classmethod
    def none(cls) -> "SiteSelector":
        """Select no quantization sites."""
        return cls(lambda _site: False, "none")

    @classmethod
    def paths(cls, *patterns: str) -> "SiteSelector":
        """Select full site paths using shell-style glob patterns."""
        checked = _validated_patterns(patterns, "site path")
        return cls(
            lambda site: any(fnmatchcase(site.path, pattern) for pattern in checked),
            f"paths={checked}",
        )

    @classmethod
    def module_paths(cls, *patterns: str) -> "SiteSelector":
        """Select owner-module paths using shell-style glob patterns."""
        checked = _validated_patterns(patterns, "module path")
        return cls(
            lambda site: any(
                fnmatchcase(site.module_path, pattern) for pattern in checked
            ),
            f"module_paths={checked}",
        )

    @classmethod
    def fp_module_paths(cls, *patterns: str) -> "SiteSelector":
        """Select sites using their original floating-point module paths."""
        checked = _validated_patterns(patterns, "floating-point module path")
        return cls(
            lambda site: any(
                fnmatchcase(
                    getattr(site.module, "fp_name", None) or site.module_path,
                    pattern,
                )
                for pattern in checked
            ),
            f"fp_module_paths={checked}",
        )

    @classmethod
    def module_types(cls, *types: type) -> "SiteSelector":
        """Select sites whose owner is an instance of one of ``types``."""
        if not types:
            raise ValueError("At least one module type is required.")
        if any(not isinstance(module_type, type) for module_type in types):
            raise TypeError("Every module_types argument must be a type.")
        return cls(
            lambda site: isinstance(site.module, types),
            "module_types=" + str(tuple(module_type.__name__ for module_type in types)),
        )

    @classmethod
    def observer_names(cls, *names: str) -> "SiteSelector":
        """Select observers by their logical names."""
        checked = frozenset(_validated_patterns(names, "observer name"))
        return cls(
            lambda site: site.observer_name in checked,
            f"observer_names={tuple(sorted(checked))}",
        )

    @classmethod
    def roles(cls, *roles: SiteRole) -> "SiteSelector":
        """Select sites by logical role."""
        checked = frozenset(roles)
        if not checked:
            raise ValueError("At least one site role is required.")
        return cls(
            lambda site: site.role in checked,
            f"roles={tuple(sorted(role.value for role in checked))}",
        )


def _validated_patterns(values: Iterable[str], context: str) -> tuple[str, ...]:
    patterns = tuple(values)
    if not patterns or any(not value for value in patterns):
        raise ValueError(f"At least one non-empty {context} pattern is required.")
    return patterns


@dataclass(frozen=True)
class QuantizationBoundaries:
    """Define reusable parameter, output, and included analysis domains."""

    outputs: SiteSelector
    parameters: SiteSelector = SiteSelector.roles(SiteRole.PARAMETER)
    activations: SiteSelector | None = None
    included: SiteSelector = SiteSelector.all()

    def selector_for(self, profile: QuantizationProfile) -> SiteSelector:
        """Return the enabled-site selector for one standard profile."""
        if profile is QuantizationProfile.OUTPUT_ONLY:
            return self.included & self.outputs
        if profile is QuantizationProfile.WEIGHT_ONLY:
            return self.included & self.parameters
        if profile is QuantizationProfile.ACTIVATION_ONLY:
            activations = (
                self.activations
                if self.activations is not None
                else (~self.parameters & ~self.outputs)
            )
            return self.included & activations
        if profile is QuantizationProfile.FULL:
            return self.included
        raise ValueError(f"Unsupported quantization profile: {profile!r}")
