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

import unittest

import tico
import torch
from tico.utils.installed_packages import is_transformers_installed
from tico.utils.signature import flatten_dynamic_cache, ModelInputSpec


class SimpleModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 3)

    def forward(
        self,
        x0,
        x1,
        lin,
    ):
        z0 = x0 - x1
        z1 = self.linear(lin)
        return z0 + z1

    def get_example_inputs(self):
        return (
            torch.randn(2, 3),
            torch.randn(2, 3),
            torch.randn(2, 2),
        ), {}


class UtilsSignatureTest(unittest.TestCase):
    def setUp(self):
        m = SimpleModule()
        self.torch_model = m
        self.circle_model = tico.convert(m.eval(), *m.get_example_inputs())
        return

    def test_bind_check_success(self):
        spec = ModelInputSpec(self.circle_model.circle_binary)
        args = (torch.randn(2, 3, dtype=torch.float32),)
        kwargs = {
            "lin": torch.randn(2, 2, dtype=torch.float32),
            "x1": torch.randn(2, 3),
        }
        inputs = spec.bind(args, kwargs, check=True)

        assert len(inputs) == 3
        assert inputs[0].dtype == torch.float32
        assert inputs[1].dtype == torch.float32
        assert inputs[2].dtype == torch.float32

    def test_bind_type_check_fail(self):
        spec = ModelInputSpec(self.circle_model.circle_binary)
        args = (
            torch.randint(low=0, high=1000, size=(2, 3), dtype=torch.int64),
        )  # dtype mismatch
        kwargs = {
            "lin": torch.randn(2, 2, dtype=torch.float32),
            "x1": torch.randn(2, 3),
        }
        with self.assertRaisesRegex(
            TypeError, "type torch.int64 != expected torch.float32"
        ):
            spec.bind(args, kwargs, check=True)

    def test_bind_shape_check_fail(self):
        spec = ModelInputSpec(self.circle_model.circle_binary)
        args = (torch.randn(2, 3, dtype=torch.float32),)
        kwargs = {
            "lin": torch.randn(20, 20, dtype=torch.float32),
            "x1": torch.randn(2, 3),
        }  # shape mismatch
        with self.assertRaisesRegex(ValueError, "wrong dimension"):
            spec.bind(args, kwargs, check=True)

    def test_bind_missing_arg_fail(self):
        spec = ModelInputSpec(self.circle_model.circle_binary)
        args = (torch.randn(2, 3),)
        kwargs = {
            "x1": torch.randn(2, 3),
        }  # 'lin' is missing
        with self.assertRaisesRegex(ValueError, "arguments are not the same"):
            spec.bind(args, kwargs, check=True)

    def test_bind_too_many_positional_fail(self):
        spec = ModelInputSpec(self.circle_model.circle_binary)
        args = (
            torch.randn(2, 3),
            torch.randn(2, 3),
            torch.randn(2, 3),
            torch.randn(2, 3),
        )  # Too many args
        with self.assertRaisesRegex(ValueError, "arguments are not the same"):
            spec.bind(args, {}, check=True)

    def test_bind_multiple_values_fail(self):
        spec = ModelInputSpec(self.circle_model.circle_binary)
        args = (
            torch.randn(2, 3, dtype=torch.float32),  # x0
            torch.randn(2, 3, dtype=torch.float32),  # x1
        )
        kwargs = {
            "x1": torch.randn(2, 3),  # x1 !! multiple value for x1
            "lin": torch.randn(20, 20, dtype=torch.float32),
        }  # shape mismatch
        with self.assertRaisesRegex(ValueError, "arguments are not the same"):
            spec.bind(args, kwargs, check=True)

    def test_bind_tuple(self):
        spec = ModelInputSpec(self.circle_model.circle_binary)
        args = (
            torch.randn(
                2,
                3,
            ),
            (
                torch.randn(
                    2,
                    3,
                ),
                torch.randn(
                    2,
                    2,
                ),
            ),  # This tuple will be bound to x1, lin by flattening
        )
        inputs = spec.bind(args, {}, check=True)

        assert len(inputs) == 3
        assert inputs[0].shape == torch.Size([2, 3])
        assert inputs[1].shape == torch.Size([2, 3])
        assert inputs[2].shape == torch.Size([2, 2])

    def test_bind_multi_level_tuple(self):
        spec = ModelInputSpec(self.circle_model.circle_binary)
        args = (
            torch.randn(
                2,
                3,
            ),
            (
                (
                    (
                        (
                            torch.randn(
                                2,
                                3,
                            )
                        ),
                        torch.randn(
                            2,
                            2,
                        ),
                    )
                )
            ),  # This tuple will be bound to x1, lin by flattening
        )
        inputs = spec.bind(args, {}, check=True)

        assert len(inputs) == 3
        assert inputs[0].shape == torch.Size([2, 3])
        assert inputs[1].shape == torch.Size([2, 3])
        assert inputs[2].shape == torch.Size([2, 2])

    def test_bind_input_num_mismatch(self):
        spec = ModelInputSpec(self.circle_model.circle_binary)
        args = (torch.randn(2, 3, dtype=torch.float32),)
        kwargs = {
            "lin": torch.randn(2, 2, dtype=torch.float32),
        }
        with self.assertRaisesRegex(ValueError, "arguments are not the same"):
            spec.bind(args, kwargs, check=True)


@unittest.skipIf(not is_transformers_installed(), "transformers is not installed")
class DynamicCacheSignatureTest(unittest.TestCase):
    """
    `DynamicCache` inputs are flattened into one Circle input per cache tensor.
    The binding must follow the pytree flatten order and placeholder names used
    by `torch.export`, for both the layer-based and the legacy cache layout.
    """

    class CacheModule(torch.nn.Module):
        def forward(self, x, past_key_values):
            keys, values = past_key_values.update(x, x * 2, 0)
            return keys + values

    def setUp(self):
        from tico.utils.pytree_utils import (
            register_dynamic_cache,
            register_dynamic_layer,
        )
        from transformers.cache_utils import DynamicCache

        register_dynamic_cache()
        register_dynamic_layer()

        torch.manual_seed(0)
        self.x = torch.randn(1, 2, 3, 4)
        self.cache = DynamicCache()
        self.cache.update(torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4), 0)
        self.cache.update(torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4), 1)
        self.cache_tensors = [t for _, t in flatten_dynamic_cache(self.cache)]
        self.assertEqual(len(self.cache_tensors), 4)

        m = self.CacheModule().eval()
        self.circle_model = tico.convert(m, (self.x,), {"past_key_values": self.cache})

    def test_names_follow_export_flatten_order(self):
        spec = ModelInputSpec(self.circle_model.circle_binary)
        expected = ["x"] + [
            f"past_key_values_{suffix}"
            for suffix, _ in flatten_dynamic_cache(self.cache)
        ]
        self.assertEqual(spec.names, expected)

    def test_bind_cache_kwarg(self):
        spec = ModelInputSpec(self.circle_model.circle_binary)
        inputs = spec.bind((self.x,), {"past_key_values": self.cache}, check=True)

        self.assertEqual(len(inputs), 5)
        self.assertTrue(torch.equal(inputs[0], self.x))
        for bound, expected in zip(inputs[1:], self.cache_tensors):
            self.assertTrue(torch.equal(bound, expected))

    def test_bind_cache_positional(self):
        spec = ModelInputSpec(self.circle_model.circle_binary)
        inputs = spec.bind((self.x, self.cache), {}, check=True)

        self.assertEqual(len(inputs), 5)
        for bound, expected in zip(inputs[1:], self.cache_tensors):
            self.assertTrue(torch.equal(bound, expected))

    def test_bound_inputs_run(self):
        out = self.circle_model(self.x, past_key_values=self.cache)
        expected = self.CacheModule()(self.x, self.cache)
        torch.testing.assert_close(torch.from_numpy(out), expected)
