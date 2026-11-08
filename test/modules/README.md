# Test target modules

Test targets are imported under `test.modules.*` and collected by the
[conversion test harness](../README.md#conversion-harness). See the
[test documentation index](../README.md) for setup, execution, and debugging.

## Module guides

| Modules | Import namespace | Scope |
|---|---|---|
| [Operator modules](op/) | `test.modules.op.*` | Focused operator conversion tests. |
| [Network modules](net/) | `test.modules.net.*` | Composed network tests. |
| [Model tests](../../docs/development.md#model-tests) | `test.modules.model.*` | Opt-in model-family tests under [`model/`](model/), with separate dependencies. |

Each target provides `get_example_inputs()`, which returns `(args, kwargs)`:
`args` is a tuple of positional inputs and `kwargs` is a dictionary. The
[discovery contract](../../docs/system_test.md#41-discover-a-test-module) describes
optional shape, compile, and expected-output hooks.

## How to use `target` decorator?

By default, all `torch.nn.Module` subclasses defined in a Python file are test
targets. If any class in the file has the `@target` decorator from
`test.support.tag`, only the decorated classes are selected. Use
[`@skip`](#test-target-instantiation-strategy-test-all-skip-some) to disable an
individual selected case with a reason.

### Example 1. net.Llama2

`@target` decorator is used.

```py
class TransformerBlock(nn.Module): # <--- Not tested
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.use_kv_cache = args.use_kv_cache
        self.n_heads = args.n_heads

@tag.target
class Transformer(nn.Module): # <--- Tested
    def __init__(self, params: ModelArgs = ModelArgs()):
        super().__init__()
```

### Example 2.

`@target` decorator is not used for any module in the file. All torch.nn.Module subclasses are test targets.

```py
class SimpleSqueeze(torch.nn.Module):  # <--- Tested
    def __init__(self):
        super().__init__()

    def forward(self, x):
        z = torch.squeeze(x)
        return z

    def get_example_inputs(self):
        torch.manual_seed(1234)
        return (torch.randn(2, 1, 2, 1, 2),), {}


class SimpleSqueezeWithDims(torch.nn.Module):  # <--- Tested
    def __init__(self):
        super().__init__()

    def forward(self, x):
        z = torch.squeeze(x, dim=(1, 3))
        return z

    def get_example_inputs(self):
        torch.manual_seed(1234)
        return (torch.randn(2, 1, 2, 1, 2),), {}
```

## Test target instantiation strategy: test-all, skip-some

Use `@skip(reason=...)` from `test.support.tag` to mark a selected module as a
skipped test. Keep the reason specific to the unsupported feature or environment
constraint. The builder still calls `get_example_inputs()` while constructing a
skipped test, so that method must remain inexpensive and return valid `(args, kwargs)`.

```py
import torch

from test.modules.base import TestModuleBase
from test.support.tag import skip


class SimpleAdd(TestModuleBase):
    def forward(self, x, y):
        return (x + y,)

    def get_example_inputs(self):
        return (torch.ones(1), torch.ones(1)), {}


@skip(reason="Example of an explicitly disabled test case")
class SkippedAdd(SimpleAdd):
    pass
```
