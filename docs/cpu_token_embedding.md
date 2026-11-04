# CPU token embedding with explicit quantization

Token embedding can execute on the CPU before passing quantized activations to a
decoder. A vocabulary table is large, while each invocation usually selects only a
small subset of its rows. The implementation therefore separates the graph's
quantization semantics from the CPU's memory and execution strategy.

This document describes constant UINT8 tables with asymmetric per-row quantization
(axis 0) and a per-tensor INT16 output boundary. Quantization settings remain part
of the model's configuration and calibration contract. See the
[system design](design.md#10-quantization-integration) for the conversion pipeline
and [reference execution modes](../tico/circle/README.md#quantized-models) for
runtime arithmetic and backend support.

## Why the graph contains explicit conversions

GATHER selects values; it does not convert between quantization domains. Copying
integer codes from a row-quantized table into an activation tensor with different
qparams would change their meaning. Weight dequantization and activation
quantization must therefore appear as separate Circle operations:

```text
Wq: UINT8 [V, D], row scales and zero points
  -> DEQUANTIZE
  -> GATHER(integer token IDs)
  -> Q_A -> DQ_A
  -> optional MUL(dequantized, fake-quantized scale)
  -> Q_B: per-tensor INT16
```

The MUL and its surrounding boundaries depend on the exported model. When no
post-processing is needed, the embedding's activation Quantize is the output
boundary. Qparams belong to integer tensors; the internal FLOAT32 tensors carry
no qparams. Token IDs are unquantized indices.

Standard Circle operators make domain changes visible to optimizers, validators
and other runtimes. They also let the CPU reuse the existing quantization and
arithmetic kernels without defining a special embedding opcode or extending
GATHER's semantics.

## Why the CPU restores selected rows

The full-table DEQUANTIZE describes the graph's values, but need not materialize
its complete result. At preparation time the reference runtime recognizes
`DEQUANTIZE(constant table) -> GATHER` from serialized tensors, qparams and operator
options. Each invocation then executes:

```text
row_q = Wq[token_ids]
row_fp32 = (float32(row_q) - zero_point[token_ids]) * scale[token_ids]
```

The row parameters broadcast over the embedding width. Gathering codes and their
parameters before dequantization is equivalent to selecting from the dequantized
table, because each element depends only on its row's code, scale and zero point.
Existing bounds checks still apply.

The plan is built once and does not depend on model names, tensor names or Python
export metadata. The full FLOAT32 table remains virtual, including in trace mode.
For N selected tokens, temporary row storage scales with O(ND), while cached row
qparams require O(V) metadata. The original UINT8 table remains the weight storage;
no full FLOAT32, INT16 or scaled-weight copy is needed for native embedding.

FP32 row restoration reuses the reference arithmetic contract. Subsequent Q/DQ and
MUL operations remain separate so their rounding and saturation are preserved;
a fixed-point or wider fusion would need to reproduce those same boundaries.

## Numerical and storage contracts

Intermediate quantization is observable. In general,
`Q_B(DQ_A(Q_A(x)))` differs from `Q_B(x)` because Q_A rounds and clips. A downstream
domain change must retain Q_A and add a conversion rather than retarget it:

```text
GATHER(float) -> Q_A -> optional layout -> Quantize(A -> B) -> Concat[B]
```

The same rule applies around scale multiplication. If the multiplier is
fake-quantized, its quantized value must be used. Folding the raw scale into the
weight or moving Q_A across MUL can alter rounding and clipping.

A token table may also serve an LM head. Keeping the original codes, row qparams
and layout immutable preserves that consumer's meaning and allows existing buffer
sharing. Logical weight tying does not imply physical sharing between separately
packaged artifacts. Embedding scaling is applied to selected activations, not to
shared weight storage.

The decoder boundary retains its configured dtype, shape/layout, scale, zero point
and range. Internal CPU FLOAT32 buffers do not replace the INT16 interface.
Producer and consumer calibration must already agree at a staged boundary;
this execution optimization does not reconcile independently calibrated grids.

## Export-pass invariants

[PreserveEmbeddingQuantization](../tico/quantization/passes/preserve_embedding_quantization.py)
identifies the supported embedding path before Q/DQ folding. Its export-only
markers distinguish a preserved boundary (`EXPLICIT_EMBEDDING` on Quantize) from a
real float intermediate (`FLOAT_EMBEDDING`). Backward propagation must leave the
preserved Quantize's arguments and qparam metadata intact, including through layout
paths. Unprotected private quantizers retain their existing retarget policy.

Circle QUANTIZE saturates to its storage range. The supported reduced INT16 lower
bound of -32767 therefore requires a float clamp before Quantize, serialized as
MAXIMUM. Circle's min/max calibration metadata is not a substitute for that clamp.
Clamp insertion completes before internal DQs are classified on the updated graph.
Generated clamps belong to the CPU region and remain qparam-free; repeated pass
execution recognizes those owned nodes without adding duplicate clamps or removing
their input DQ. Ordinary quantized-clamp legalization remains applicable outside
this region.

Weight-DQ removal keeps the embedding's float connection while other quantized
consumers retain the original integer weight. Constant folding must also keep the
row-wise table compressed when it feeds a dynamic lookup, including with the HEAVY
profile. These policies are local to the embedding contract; they do not disable
folding or quantized-tensor validation globally.

## Supported execution pattern

The [selected-row planner](../tico/circle/runtime/rowwise_gather.py) requires:

- A constant, static rank-2 UINT8 table with axis-0 qparams: one finite positive
  scale and one UINT8-range zero point per vocabulary row.
- GATHER axis 0 (or equivalent -2), `batchDims=0`, unquantized INT32/INT64 indices
  of any rank, and an unquantized FLOAT32 result. Actual input and output shapes
  must satisfy the serialized tensor contracts.
- Only compatible GATHER consumers of the table DQ, with no graph output exposing
  the full dequantized table. Multiple GATHERs can share that virtual result;
  other consumers of the raw integer weight remain independent.

Unsupported target contracts produce a diagnostic instead of silently restoring
an entire table. Generic non-target graphs keep their existing execution path.
The export policy retains per-tensor UINT8/INT16 activation boundaries and supports
static quantized scale operands. A fake-quantized dynamic MUL side operand is
diagnosed as unsupported. `enable_rowwise_gather=False` permits
ordinary full-DQ execution for small-model debugging and numerical oracles.

The optimization is implemented in the built-in reference runtime's NATIVE mode.
FAKE_QUANTIZE remains a semantic evaluation mode: another consumer such as FC may
use its existing full-weight dequantization cache. External CPU runtimes need their
own selected-row execution plan and support for the serialized DQ/Q/clamp sequence.
Reference execution does not establish production-runtime or device support.

## Regression coverage

The tests exercise the production export pipeline, Circle save/load and reference
execution. Relevant coverage includes:

- [Explicit embedding export](../test/quantization/passes/test_explicit_embedding_export.py):
  integer output contracts, intermediate quantization, scale operands and tied weights.
- [Quantize boundary preservation](../test/quantization/passes/test_explicit_embedding_retarget.py):
  Concat/layout consumers and the unprotected retarget control.
- [Reduced-range preservation](../test/quantization/passes/test_explicit_embedding_reduced_range.py):
  float DQ/clamp edges, saturation, pass idempotence and traversal-order independence.
- [Selected-row execution](../test/unit_test/circle/runtime/test_rowwise_gather.py):
  row qparams, index bounds, sharing and loading/execution allocation checks.
