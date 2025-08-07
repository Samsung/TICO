import unittest

import tico
import torch
from tico.utils.signature import ModelInputSpec


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
        with self.assertRaises(TypeError):
            spec.bind(args, kwargs, check=True)

    def test_bind_shape_check_fail(self):
        spec = ModelInputSpec(self.circle_model.circle_binary)
        args = (torch.randn(2, 3, dtype=torch.float32),)
        kwargs = {
            "lin": torch.randn(20, 20, dtype=torch.float32),
            "x1": torch.randn(2, 3),
        }  # shape mismatch
        with self.assertRaises(ValueError):
            spec.bind(args, kwargs, check=True)

    def test_bind_missing_arg_fail(self):
        spec = ModelInputSpec(self.circle_model.circle_binary)
        args = (torch.randn(2, 3),)
        kwargs = {
            "x1": torch.randn(2, 3),
        }  # 'lin' is missing
        with self.assertRaises(ValueError):
            spec.bind(args, kwargs, check=True)

    def test_bind_too_many_positional_fail(self):
        spec = ModelInputSpec(self.circle_model.circle_binary)
        args = (
            torch.randn(2, 3),
            torch.randn(2, 3),
            torch.randn(2, 3),
            torch.randn(2, 3),
        )  # Too many args
        with self.assertRaises(ValueError):
            spec.bind(args, {}, check=True)
