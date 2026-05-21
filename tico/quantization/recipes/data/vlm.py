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

from typing import Any

from tico.quantization.evaluation.vlm_eval_utils import get_calib_inputs


def build_vlm_calibration_inputs(
    *,
    processor: Any,
    dataset: str,
    n_samples: int,
    max_seq_len: int | None = None,
) -> list[dict]:
    return get_calib_inputs(
        dataset,
        processor,
        n_samples=n_samples,
        max_seq_len=max_seq_len,
    )
