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

from typing import Any, Optional

from tico.quantization.config.ptq import PTQConfig


def get_model_arg(
    qcfg: Optional[PTQConfig],
    *path: str,
    default: Any = None,
) -> Any:
    """
    Retrieve a nested value from PTQConfig.model_args.

    Args:
        qcfg: PTQ configuration object or None
        *path: Path components to navigate nested dictionary
        default: Default value to return if path is not found

    Returns:
        The value at the specified path or default if not found

    Raises:
        ValueError: If path components are accessed from non-dict

    Example:
        get_model_arg(qcfg, "vision", "grid_thw")
        get_model_arg(qcfg, "vision", "spatial_merge_size", default=2)
    """
    if qcfg is None:
        return default

    value: Any = qcfg.model_args
    for key in path:
        if not isinstance(value, dict):
            raise ValueError(
                f"PTQConfig.model_args path {'.'.join(path)} is invalid: "
                f"'{key}' is accessed from a non-mapping value."
            )
        if key not in value:
            return default
        value = value[key]

    return value
