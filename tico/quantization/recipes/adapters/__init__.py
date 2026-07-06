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

from tico.quantization.recipes.adapters.base import ModelAdapter
from tico.quantization.recipes.adapters.gemma4 import Gemma4Adapter
from tico.quantization.recipes.adapters.llama import LlamaAdapter
from tico.quantization.recipes.adapters.qwen3_vl import Qwen3VLAdapter

_ADAPTERS = {
    "llama": LlamaAdapter(),
    "qwen3_vl": Qwen3VLAdapter(),
    "qwen3-vl": Qwen3VLAdapter(),
    "gemma4": Gemma4Adapter(),
}


def get_adapter(family: str) -> ModelAdapter:
    key = family.lower()
    if key not in _ADAPTERS:
        raise KeyError(f"Unknown model family: {family}. available={sorted(_ADAPTERS)}")
    return _ADAPTERS[key]
