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

import warnings

import torch
from packaging.version import Version

from tico.config import CompileConfigV1, get_default_config
from tico.utils.compat.torch_version_policy import (
    MINIMUM_SUPPORTED_VERSION,
    QUALIFICATION_CANDIDATE_FAMILIES,
    SUPPORTED_STABLE_FAMILIES,
    version_family,
)
from tico.utils.convert import convert, convert_from_exported_program, convert_from_pt2

__all__ = [
    "CompileConfigV1",
    "get_default_config",
    "convert",
    "convert_from_exported_program",
    "convert_from_pt2",
]

# THIS LINE IS AUTOMATICALLY GENERATED
__version__ = "0.2.0"

_torch_version = Version(torch.__version__)
_torch_family = version_family(str(_torch_version))
_supported_range = f"{SUPPORTED_STABLE_FAMILIES[0]} ~ {SUPPORTED_STABLE_FAMILIES[-1]}"

if _torch_version < Version(MINIMUM_SUPPORTED_VERSION):
    warnings.warn(
        f"TICO supports PyTorch families {_supported_range}; "
        f"detected torch {torch.__version__}. Upgrade to "
        f"torch>={MINIMUM_SUPPORTED_VERSION}.",
        stacklevel=2,
    )
elif not _torch_version.is_devrelease:
    if _torch_family in QUALIFICATION_CANDIDATE_FAMILIES:
        warnings.warn(
            f"PyTorch {_torch_family} is still a TICO qualification candidate. "
            "Use it for compatibility testing, not release builds.",
            stacklevel=2,
        )
    elif _torch_family not in SUPPORTED_STABLE_FAMILIES:
        warnings.warn(
            f"TICO supports PyTorch families {_supported_range}; "
            f"detected torch {torch.__version__}.",
            stacklevel=2,
        )
