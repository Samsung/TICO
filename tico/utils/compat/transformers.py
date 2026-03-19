# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
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

"""
Runtime capability-detection helpers for Hugging Face `transformers`.

Instead of branching on specific package versions such as
`transformers >= 5.x`, use these helpers to detect whether the exact
symbol or behavior required by the code is available at runtime.

Each probe is cached once per process with `functools.lru_cache`,
so repeated checks have negligible overhead.
"""

import functools
import importlib


@functools.lru_cache(maxsize=None)
def qwen3_vl_has_deepstack_model_output() -> bool:
    """
    Return whether Qwen3-VL exposes
    `BaseModelOutputWithDeepstackFeatures` in its modeling module.

    This wrapper only needs to know whether the structured return type is
    available. Using feature detection keeps the code resilient to
    backports, forward ports, and non-linear package versioning.

    Returns
    -------
    bool
        ``True`` if
        `transformers.models.qwen3_vl.modeling_qwen3_vl`
        defines `BaseModelOutputWithDeepstackFeatures`,
        otherwise ``False``.
    """
    try:
        module = importlib.import_module(
            "transformers.models.qwen3_vl.modeling_qwen3_vl"
        )
    except ImportError:
        return False

    return hasattr(module, "BaseModelOutputWithDeepstackFeatures")
