# Circle value-test support

This directory contains test-only infrastructure for checking numerical equivalence of
Circle-to-Circle rewrites.

## Components

- `builder.py` creates small, serializable `circle.ModelT` fixtures. Besides the
  convenience builders (`add`, `mul`, `reshape`, ...), `activation()`,
  `quantized_constant()`, and `operator()` can express any builtin operator with its
  Object API options table.
- `evaluator.py` wraps `tico.circle.runtime.CircleReferenceRuntime` for value tests:
  `evaluate()` runs one document with `trace=True` and returns every materialized
  tensor value together with the graph outputs. Virtual full-table dequantization
  values used by selected-row execution are not materialized or included in the trace.
- `value_test.py` provides reusable assertions for serialization round trips, pass
  equivalence, graph-interface preservation, and extraction-boundary equivalence.

The supported operator set and error contracts are those of the
[reference runtime](../../../tico/circle/README.md#reference-runtime). Unsupported
operators, absent required operands, unresolved external buffer offsets in Object API
models, and unsupported tensor types fail explicitly. Loading a complete Circle binary
through `CircleDocument` resolves appended payloads before evaluation.

Run the value tests with:

```bash
./ccex test -k circle.value
```
