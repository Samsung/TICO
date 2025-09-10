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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List

from torch.export import ExportedProgram


@dataclass
class PassResult:
    modified: bool


class PassBase(ABC):
    """
    Base interface for passes.
    """

    @abstractmethod
    def call(self, exported_program: ExportedProgram, gm) -> PassResult:
        pass


class PassStrategy(Enum):
    # Run passes until there are no changes.
    UNTIL_NO_CHANGE = (1,)
    # Same as `UNTIL_NO_CHANGE` but it starts agian from the beginning.
    RESTART = (2,)


class PassManager:
    def __init__(
        self,
        passes: List[PassBase],
        strategy: PassStrategy = PassStrategy.RESTART,
    ):
        self.passes: List[PassBase] = passes
        self.strategy: PassStrategy = strategy

    def run(self, exported_program: ExportedProgram, graph_module):
        MAXIMUM_STEP_COUNT = 1000
        step = 0
        while True:
            modified = False
            for _pass in self.passes:
                # Automatically update the signatures of the input and output.
                # https://github.com/pytorch/executorch/issues/4013#issuecomment-2187161844
                with graph_module._set_replace_hook(
                    exported_program.graph_signature.get_replace_hook()
                ):
                    result = _pass.call(exported_program, graph_module)
                modified = modified or result.modified
                if modified and self.strategy == PassStrategy.RESTART:
                    break

            if not modified:
                break
            step += 1

            assert (
                step < MAXIMUM_STEP_COUNT
            ), f"Loop iterated for {MAXIMUM_STEP_COUNT} times. Circular loop is suspected in {self.passes}"
