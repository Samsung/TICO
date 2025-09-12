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

from dataclasses import dataclass

from tico.config.base import CompileConfigBase


@dataclass
class CompileConfigV1(CompileConfigBase):
    legalize_causal_mask_value: bool = False
    remove_constant_input: bool = False
    convert_lhs_const_mm_to_fc: bool = False
    convert_rhs_const_mm_to_fc: bool = True
    convert_single_batch_lhs_const_bmm_to_fc: bool = False
    convert_expand_to_slice_cat: bool = False

    def get(self, name: str):
        return super().get(name)

    def set(self, name: str, enabled: bool):
        super().set(name, enabled)

    def to_dict(self):
        return super().to_dict()

    @classmethod
    def from_dict(cls, config_dict: dict):
        return super().from_dict(config_dict)
