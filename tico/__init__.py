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

MINIMUM_SUPPORTED_VERSION = "2.5.0"
SECURE_TORCH_VERSION = "2.6.0"

if Version(torch.__version__) < Version(MINIMUM_SUPPORTED_VERSION):
    warnings.warn(
        f"TICO officially supports torch>={MINIMUM_SUPPORTED_VERSION}. "
        f"You are using a lower version of torch ({torch.__version__}). "
        f"We highly recommend to upgrade torch>={MINIMUM_SUPPORTED_VERSION} to avoid unexpected behaviors."
    )

if Version(torch.__version__) < Version(SECURE_TORCH_VERSION):
    warnings.warn(
        f"Detected PyTorch version {torch.__version__}, which may include known security vulnerabilities. "
        f"We recommend upgrading to {SECURE_TORCH_VERSION} or later for better security.\n"
        "Upgrade command: pip install --upgrade torch\n"
        "For more details, see: https://pytorch.org/security"
    )
