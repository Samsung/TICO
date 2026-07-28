# Circle value-test support

This directory contains test-only infrastructure for checking numerical equivalence of
Circle-to-Circle rewrites without `circle-interpreter` or `onert`.

## Components

- `builder.py` creates small, serializable `circle.ModelT` fixtures.
- `evaluator.py` evaluates a deliberately limited operator subset with NumPy and records
  every intermediate tensor value.
- `value_test.py` provides reusable assertions for serialization round trips, pass
  equivalence, graph-interface preservation, and extraction-boundary equivalence.

The reference evaluator currently supports `ADD`, `SUB`, `MUL`, `RESHAPE`, and
`TRANSPOSE`. Unsupported operators, fused activations, optional inputs, external
buffers, and unsupported tensor types fail explicitly. The evaluator is not a
production Circle runtime.

Run the value tests with:

```bash
./ccex test -k circle.value
```
