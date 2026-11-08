# Test target modules

Test targets are imported under `test.modules.*` and collected by the
[conversion test harness](../pt2_to_circle_test/README.md). See the
[test documentation index](../README.md) for setup, execution, and debugging.

## Module guides

| Guide | Scope |
|---|---|
| [Operator modules](op/README.md) | Focused operator conversion tests and skip tags. |
| [Network modules](net/README.md) | Composed network tests under `test.modules.net`. |
| [Model tests](../../docs/development.md#model-tests) | Opt-in model-family tests under [`model/`](model/), with separate dependencies. |

## How to use `target` decorator?

By default, all torch.nn.Module subclasses in the python script are assumed to be test targets.
If `@target` decorator is set for any torch.nn.Module subclass in the file, only those modules will be tested.

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
        return (torch.randn(2, 1, 2, 1, 2),)


class SimpleSqueezeWithDims(torch.nn.Module):  # <--- Tested
    def __init__(self):
        super().__init__()

    def forward(self, x):
        z = torch.squeeze(x, dim=(1, 3))
        return z

    def get_example_inputs(self):
        torch.manual_seed(1234)
        return (torch.randn(2, 1, 2, 1, 2),)
```
