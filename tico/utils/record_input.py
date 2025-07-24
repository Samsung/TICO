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

import copy

import inspect
from contextlib import contextmanager
from typing import Callable, List, Optional

import torch.nn as nn


class RecordingInput:
    def __init__(
        self,
        module: nn.Module,
        condition: Callable[[dict], bool] = lambda args_dict: True,
        *,
        input_to_remove: Optional[List[str]] = [],
    ):
        self.module = module
        self.forward_org = module.forward
        self.condition = condition
        self.input_to_remove = input_to_remove
        sig = inspect.signature(self.forward_org)
        self.args_names = [
            name for name in sig.parameters.keys() if name not in ("self", "kwargs")
        ]
        self.captured_input = ()

    def __enter__(self):
        def capture_and_forward(*args, **kwargs):
            args_dict = dict(zip(self.args_names, args))
            args_dict.update(kwargs)

            def populate_args(args_dict, filter):
                for key in filter:
                    args_dict.pop(key, None)
                args_tuple = tuple(
                    args_dict.get(name, None) for name in self.args_names
                )
                return copy.deepcopy(args_tuple)

            if self.condition(args_dict) and self.captured_input == ():
                self.captured_input = populate_args(args_dict, self.input_to_remove)

            return self.forward_org(*args, **kwargs)

        self.module.forward = capture_and_forward
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.module.forward = self.forward_org
