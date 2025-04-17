import torch
from torch.export import Dim

from test.utils import tag


class SingleInputDynSimpleAdd(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        z = x + y
        return z

    def get_example_inputs(self):
        return (torch.ones(4, 5, 6), torch.ones(1, 1, 1))

    def get_input_dynamic_shapes(self):
        return (4, Dim("d2"), Dim("d3")), (1, 1, 1)


class TwoInputsDynSimpleAdd(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        z = x + y
        return z

    def get_example_inputs(self):
        return (torch.ones(1, 2, 3), torch.ones(1, 2, 3))

    def get_input_dynamic_shapes(self):
        return (1, Dim("d2"), Dim("d3")), (1, Dim("d2"), Dim("d3"))


@tag.test_negative(expected_err=f"Failed running call_function")
class DynSimpleAddNotMatchedShape(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        z = x + y
        return z

    def get_example_inputs(self):
        return (torch.ones(1, 2, 3), torch.ones(1, 4, 3))

    def get_input_dynamic_shapes(self):
        return (1, Dim("d"), 3), (1, Dim("d"), 3)
