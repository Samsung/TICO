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

# tico/__init__.py

from packaging.version import parse  
import torch
import warnings


MIN_FEATURE_VERSION = "2.0.0"      
RECOMMENDED_SECURE_VERSION = "2.6.0"  
__version__ = "0.1.0"
def _check_versions():
    
    torch_ver = parse(torch.__version__.split('+')[0])  
    min_ver = parse(MIN_FEATURE_VERSION)
    rec_ver = parse(RECOMMENDED_SECURE_VERSION)

    # Feature requirement (block if not met)
    if torch_ver < min_ver:
        raise RuntimeError(
            f"PyTorch ≥{MIN_FEATURE_VERSION} required (found: {torch.__version__}). "
            f"Required for torch.export support."
        )

    # Security recommendation (warn only if below recommended)
    if torch_ver < rec_ver:
        warnings.warn(
            f"Security Recommendation: PyTorch {torch.__version__} may contain vulnerabilities. "
            f"Upgrade to {RECOMMENDED_SECURE_VERSION}+ with: pip install --upgrade torch\n"
            "Details: https://pytorch.org/security",
            category=RuntimeWarning,
            stacklevel=2
        )

# Initialize (optional: only if you want checks on import)
_check_versions()

