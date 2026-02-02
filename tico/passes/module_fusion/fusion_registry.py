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

from typing import Callable, Dict, List, Type

import torch.nn as nn

# Dict with original module classes as keys and fused module classes as values.
# The value can be the fused module class itself, or a factory function that
# takes the original module as an argument and creates a fused module instance
_FUSED_MODULE_MAPPING: Dict[Type[nn.Module], Callable[[nn.Module], nn.Module]] = {}


def register_fused_module(original_module_class: Type[nn.Module]):
    """
    Decorator to register an original module class and its corresponding factory that creates the fused module
    """

    def decorator(fused_module_factory: Callable[[nn.Module], nn.Module]):
        _FUSED_MODULE_MAPPING[original_module_class] = fused_module_factory
        return fused_module_factory

    return decorator


def get_fused_module_factory(
    original_module_class: Type[nn.Module],
) -> Callable[[nn.Module], nn.Module] | None:
    """
    Returns the fused module factory corresponding to the registered original module class
    """
    return _FUSED_MODULE_MAPPING.get(original_module_class)


def replace_modules_with_fused(
    model: nn.Module, target_module_classes: List[Type[nn.Module]]
):
    """
    Replaces all instances within the model that correspond to target_module_classes
    with their fused versions registered in the registry
    """
    replaced_count = 0
    for name, module in model.named_modules():
        if type(module) in target_module_classes:
            fused_module_factory = get_fused_module_factory(type(module))
            if fused_module_factory:
                parent_module_name = ".".join(name.split(".")[:-1])
                module_short_name = name.split(".")[-1]

                parent_module = model
                if parent_module_name:
                    for part in parent_module_name.split("."):
                        parent_module = getattr(parent_module, part)

                new_module = fused_module_factory(module)

                setattr(parent_module, module_short_name, new_module)
                replaced_count += 1
                print(
                    f"Replaced {name} ({type(module).__name__}) with {type(new_module).__name__}"
                )
            else:
                print(
                    f"Warning: No fused module factory registered for {type(module).__name__}. Skipping replacement of {name}."
                )

    if replaced_count > 0:
        print(f"Successfully replaced {replaced_count} module instances.")
    else:
        print("No target module instances found to replace.")
