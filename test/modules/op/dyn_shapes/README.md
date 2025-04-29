It can be accessed as a module as `test.modules.op.dyn_shapes*`

## How to test models with inputs which have dynamic shapes?

The folder contains tests for single-op models that have dynamic inputs.
Such test requires adding additional method `get_input_dynamic_shapes` to a test class inheriting from `nn.Module`.
The format of value returned by `get_input_dynamic_shapes` should match an `dynamic_shapes` argument of [torch.export](https://pytorch.org/docs/stable/export.html) function.


### An example:
```py
from torch.export import Dim

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
```
