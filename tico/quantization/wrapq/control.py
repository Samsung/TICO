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

"""Runtime control of fake-quantization sites in WrapQ models."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable

from torch import nn

from tico.quantization.wrapq.observers.base import ObserverBase
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase


class SiteRole(str, Enum):
    """Classify an observer by its logical quantization role."""

    PARAMETER = "parameter"
    ACTIVATION_INPUT = "activation_input"
    ACTIVATION_OUTPUT = "activation_output"
    ACTIVATION = "activation"
    OTHER = "other"


_PARAMETER_NAMES = frozenset({"weight", "bias"})


def _observer_role(observer_name: str) -> SiteRole:
    leaf = observer_name.rsplit(".", maxsplit=1)[-1]
    if leaf in _PARAMETER_NAMES:
        return SiteRole.PARAMETER
    if leaf == "act_in" or leaf.endswith("_act_in"):
        return SiteRole.ACTIVATION_INPUT
    if leaf == "act_out" or leaf.endswith("_act_out"):
        return SiteRole.ACTIVATION_OUTPUT
    if leaf.startswith("act_") or "activation" in leaf:
        return SiteRole.ACTIVATION
    return SiteRole.OTHER


@dataclass(frozen=True)
class QuantizationSite:
    """Describe one observer-owned fake-quantization site."""

    path: str
    module_path: str
    observer_name: str
    role: SiteRole
    module: QuantModuleBase
    observer: ObserverBase


def iter_quantization_sites(model: nn.Module) -> Iterable[QuantizationSite]:
    """Yield every observer directly owned by a WrapQ module exactly once."""
    seen: set[int] = set()
    for module_path, module in model.named_modules():
        if not isinstance(module, QuantModuleBase):
            continue
        for observer_name, observer in module.named_observers(recurse=False):
            if id(observer) in seen:
                continue
            seen.add(id(observer))
            path = f"{module_path}.{observer_name}" if module_path else observer_name
            yield QuantizationSite(
                path=path,
                module_path=module_path,
                observer_name=observer_name,
                role=_observer_role(observer_name),
                module=module,
                observer=observer,
            )


SitePredicate = Callable[[QuantizationSite], bool]


def set_fake_quant_enabled(
    model: nn.Module,
    predicate: SitePredicate,
    enabled: bool,
) -> int:
    """Set fake quantization for matching sites and return the match count."""
    count = 0
    for site in iter_quantization_sites(model):
        if predicate(site):
            site.observer.set_fake_quant_enabled(enabled)
            count += 1
    return count


class FakeQuantState(AbstractContextManager["FakeQuantState"]):
    """Snapshot and restore per-observer fake-quantization switches."""

    def __init__(self, model: nn.Module) -> None:
        self._model = model
        self._states: dict[int, tuple[ObserverBase, bool]] = {
            id(site.observer): (site.observer, site.observer.fake_quant_enabled)
            for site in iter_quantization_sites(model)
        }

    def set_all(self, enabled: bool) -> None:
        """Set every observer in the snapshot to one fake-quantization state."""
        for observer, _ in self._states.values():
            observer.set_fake_quant_enabled(enabled)

    def set_where(self, predicate: SitePredicate, enabled: bool) -> int:
        """Set matching observers while retaining the original snapshot."""
        return set_fake_quant_enabled(self._model, predicate, enabled)

    def restore(self) -> None:
        """Restore all states captured at construction time."""
        for observer, enabled in self._states.values():
            observer.set_fake_quant_enabled(enabled)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.restore()
        return None
