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

import unittest
from types import SimpleNamespace

from tico.quantization.wrapq.utils.utils import get_model_arg


class TestGetModelArg(unittest.TestCase):
    def test_returns_default_when_qcfg_is_none(self):
        value = get_model_arg(None, "vision", "spatial_merge_size", default=2)
        self.assertEqual(value, 2)

    def test_returns_nested_value(self):
        qcfg = SimpleNamespace(
            model_args={
                "vision": {
                    "grid_thw": (1, 16, 16),
                    "visual_start_idx": 8,
                    "spatial_merge_size": 4,
                }
            }
        )

        self.assertEqual(
            get_model_arg(qcfg, "vision", "spatial_merge_size", default=2),  # type: ignore[arg-type]
            4,
        )
        self.assertEqual(
            get_model_arg(qcfg, "vision", "visual_start_idx", default=None),  # type: ignore[arg-type]
            8,
        )
        self.assertEqual(
            get_model_arg(qcfg, "vision", "grid_thw", default=None),  # type: ignore[arg-type]
            (1, 16, 16),
        )

    def test_returns_top_level_value_when_path_has_one_key(self):
        qcfg = SimpleNamespace(
            model_args={
                "vision": {
                    "grid_thw": (1, 16, 16),
                    "visual_start_idx": 8,
                    "spatial_merge_size": 4,
                },
                "text": {
                    "foo": "bar",
                },
            }
        )

        self.assertEqual(
            get_model_arg(qcfg, "vision", default=None),  # type: ignore[arg-type]
            {
                "grid_thw": (1, 16, 16),
                "visual_start_idx": 8,
                "spatial_merge_size": 4,
            },
        )
        self.assertEqual(
            get_model_arg(qcfg, "text", default=None),  # type: ignore[arg-type]
            {"foo": "bar"},
        )

    def test_returns_default_when_top_level_key_is_missing(self):
        qcfg = SimpleNamespace(
            model_args={
                "vision": {
                    "spatial_merge_size": 4,
                }
            }
        )

        value = get_model_arg(qcfg, "audio", default="missing")  # type: ignore[arg-type]
        self.assertEqual(value, "missing")

    def test_returns_default_when_key_is_missing(self):
        qcfg = SimpleNamespace(model_args={"vision": {}})

        value = get_model_arg(qcfg, "vision", "spatial_merge_size", default=2)  # type: ignore[arg-type]
        self.assertEqual(value, 2)

    def test_returns_default_when_intermediate_key_is_missing(self):
        qcfg = SimpleNamespace(model_args={})

        value = get_model_arg(qcfg, "vision", "spatial_merge_size", default=2)  # type: ignore[arg-type]
        self.assertEqual(value, 2)

    def test_raises_when_path_traverses_non_mapping(self):
        qcfg = SimpleNamespace(
            model_args={
                "vision": 123,
            }
        )

        with self.assertRaisesRegex(ValueError, "is invalid"):
            get_model_arg(qcfg, "vision", "spatial_merge_size", default=2)  # type: ignore[arg-type]

    def test_empty_path_returns_model_args(self):
        qcfg = SimpleNamespace(model_args={"vision": {"spatial_merge_size": 2}})

        value = get_model_arg(qcfg, default=None)  # type: ignore[arg-type]
        self.assertEqual(value, {"vision": {"spatial_merge_size": 2}})
