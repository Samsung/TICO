# Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

import importlib
import inspect
import pkgutil
from abc import abstractmethod

import torch

from test.utils.tag import is_tagged


class TestRunnerBase:
    def __init__(self, test_name: str, nnmodule: torch.nn.Module):
        self.test_name = test_name
        assert hasattr(nnmodule, "get_example_inputs")
        assert isinstance(nnmodule.get_example_inputs(), tuple)  # type: ignore[operator]

        self.nnmodule = nnmodule
        self.example_inputs = nnmodule.get_example_inputs()  # type: ignore[operator]

        # Get tags
        self.skip: bool = is_tagged(self.nnmodule, "skip")
        self.skip_reason: str = getattr(self.nnmodule, "__tag_skip_reason", "")
        self.test_negative: bool = is_tagged(self.nnmodule, "test_negative")
        self.expected_err: str = getattr(self.nnmodule, "__tag_expected_err", "")
        self.use_onert: bool = is_tagged(self.nnmodule, "use_onert")

    @abstractmethod
    def make(self):
        pass

    @abstractmethod
    def _run(self):
        pass


class TestDictBuilderBase:
    def __init__(self, namespace: str):
        assert namespace.startswith("test.modules")
        self.namespace = namespace

    @property
    def submodules(self):
        # Return a list of submodules under the given namespace
        ret = [
            f"{self.namespace}.{m.name}"
            for m in pkgutil.iter_modules(
                importlib.import_module(self.namespace).__path__
            )
        ]

        return ret

    def _get_nnmodules(self, submodule: str):
        # Return a list of nn module classes from a submodule

        # Read all `torch.nn.Module` from a file
        nnmodule_classes = list(
            nnmodule_class
            for _, nnmodule_class in inspect.getmembers(
                importlib.import_module(submodule),
                lambda nnmodule_class: inspect.isclass(nnmodule_class)
                # Only include classes defined inside the module.
                and nnmodule_class.__module__ == submodule
                and issubclass(nnmodule_class, torch.nn.Module),
            )
        )

        # If any of the nnmodule_classes has a tag `__tag_target`, only those nnmodule_classes will be added
        target_only: bool = any(
            hasattr(nnmodule_cls, "__tag_target") for nnmodule_cls in nnmodule_classes
        )

        if target_only:
            nnmodule_classes = [
                nnmodule_cls
                for nnmodule_cls in nnmodule_classes
                if hasattr(nnmodule_cls, "__tag_target")
            ]

        return nnmodule_classes

    @abstractmethod
    def build(self, submodule):
        pass
