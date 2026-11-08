# TICO tests

This directory contains TICO's automated tests and performance benchmarks.

## Test documentation

| Task | Guide |
|---|---|
| Set up the test environment and run or filter tests | [Development guide](../docs/development.md) |
| Understand test layers, validation criteria, and CI coverage | [System Test Guide](../docs/system_test.md) |
| Add operator or network conversion test modules | [Test target modules](modules/README.md) |
| Follow the conversion and comparison flow | [Conversion harness](pt2_to_circle_test/README.md) |
| Run opt-in model-family tests with their dependencies | [Model tests](../docs/development.md#model-tests) |
| Check numerical equivalence of Circle rewrites | [Circle value-test support](support/circle/README.md) |
| Measure conversion, serialization, or Circle pass performance | [Performance benchmarks](performance/README.md) |

## How to debug using vscode?

1. Add below configuration to `.vscode/launch.json`.
2. Make a breakpoint on the line you want.
3. Run the configuration with `Run and Debug`.

**TIP** Install this project in editable mode (pip install -e) for interactive test.

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug unit test (pt2_to_circle)",
            "type": "debugpy",
            "request": "launch",
            "module": "unittest",
            "args": ["test.pt2_to_circle_test.test_pt2_to_circle"],
            "cwd": "${workspaceFolder}",
            "console": "integratedTerminal",
            "justMyCode": false,
        },
    ]
}
```
