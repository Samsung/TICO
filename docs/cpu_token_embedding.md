# CPU token embedding with explicit quantization

Token embedding can execute on the CPU before passing quantized activations to a
decoder. A vocabulary table is large, while each invocation usually selects only a
small subset of its rows. The implementation therefore separates the graph's
quantization semantics from the CPU's memory and execution strategy.

The supported lookup uses a constant UINT8 table with asymmetric per-row
quantization (axis 0). Activation boundaries can be per-tensor UINT8 or INT16;
post-processing can also expose a real FLOAT32 output. Quantization settings remain
part of each model's configuration and calibration contract. See the
[system design](design.md#10-quantization-integration) for the conversion pipeline
and [reference execution modes](../tico/circle/README.md#quantized-models) for
runtime arithmetic and backend support.

## Select the export policy

`CompileConfigV1.preserve_embedding_quantization` defaults to `False`. Its meaning
is: Preserve explicit quantization boundaries in supported embedding paths during
Circle export. Omitting the option or setting it to `False` keeps the existing
embedding export policy. Set it explicitly to `True` to select the policy described
below; model family, CPU execution and the presence of an embedding do not enable
it automatically.

```python
import tico

config = tico.CompileConfigV1(
    preserve_embedding_quantization=True,
)
circle_model = tico.convert_from_exported_program(
    exported_program,
    config=config,
)
```

The same `config=` argument is accepted by `convert`, `convert_from_pt2`, and the
corresponding file-output APIs. The public `pt2-to-circle` CLI reads the option
through its existing YAML configuration path:

```yaml
version: "1.0"
preserve_embedding_quantization: true
```

```bash
pt2-to-circle -i embedding.pt2 -o embedding.circle --config compile.yaml
```

The Llama, Qwen3-VL and Gemma4 recipe exporter exposes the same opt-in:

```bash
python -m tico.quantization.examples.export \
  --config tico/quantization/examples/configs/llama_export.yaml \
  --checkpoint quantized_model.pt \
  --preserve-embedding-quantization
```

Use the corresponding Qwen3-VL or Gemma4 recipe config for those model families.
Recipe YAML can instead set `export.preserve_embedding_quantization: true` in its
`export` mapping. Recipe helpers accept
`compile_config=tico.CompileConfigV1(preserve_embedding_quantization=True)`.
The recipe convenience flag also defaults to `False`.

This compile option selects Circle export policy. It does not change calibration,
observers, activation dtype or runtime execution mode. It applies only to the
supported paths and dtype contracts documented here. It does not add support for
other embedding tables or general native quantized FC execution.

Explicit embedding conversion examples, integration tests and export benchmarks
must opt in. A runtime benchmark that constructs an explicit Circle graph directly
already supplies that graph's Q/DQ contract and does not run the compile pipeline.
Runtime selected-row execution remains controlled independently by the serialized
graph and runtime settings such as `enable_rowwise_gather`.

## Why the graph contains explicit conversions

GATHER selects values; it does not convert between quantization domains. Copying
integer codes from a row-quantized table into an activation tensor with different
qparams would change their meaning. In the supported explicit embedding path,
weight dequantization and activation quantization are separate Circle operations.
Without post-processing, the canonical graph is:

```text
Wq: UINT8 [V, D], domain W (per-row scale/zero point)
  -> DEQUANTIZE using W
  -> GATHER(FLOAT32 values, integer token IDs)
  -> QUANTIZE into domain A
  -> activation: configured per-tensor UINT8 or INT16
```

Here Q_A denotes the activation Quantize into domain A and is the output boundary
when no post-processing is needed. A scaled embedding may instead continue from
GATHER as follows:

```text
GATHER(FLOAT32)
  -> Q_A -> DQ_A
  -> MUL(dequantized, fake-quantized scale)
  -> Q_B: configured per-tensor UINT8 or INT16
```

The MUL and its surrounding boundaries depend on the exported model; they are not
required for every embedding. Existing intermediate Q/DQ boundaries are preserved.
Other supported post-processing can expose a real FLOAT32 output, as described in
[model adapters and stage interfaces](#model-adapters-and-stage-interfaces).
Qparams belong to integer tensors; the internal FLOAT32 tensors carry no qparams.
Token IDs are unquantized INT32/INT64 indices, independently of the table and
activation dtypes. Even UINT8-to-UINT8 needs an explicit conversion when the table
has per-row qparams and the activation has a different per-tensor grid.

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
MUL and feature-transform operations remain separate so their rounding and saturation are preserved;
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
and range. Producer and consumer domains must be compared before passing integer
codes between stages. When they differ, the host must perform an explicit domain
conversion that preserves the required rounding and saturation. The current
integration tests express this handoff as dequantization followed by quantization
into the consumer's frozen grid; the contract does not require a materialized
FLOAT32 intermediate. Dtype equality alone is insufficient. The selected-row
optimization does not align grids or automatically perform this handoff.

## Export-pass invariants

[PreserveEmbeddingQuantization](../tico/quantization/passes/preserve_embedding_quantization.py)
runs only when `preserve_embedding_quantization=True` and identifies the supported
embedding path immediately before Q/DQ folding. When disabled, it is absent from
the pass list and adds no graph nodes or metadata. Its export-only
markers distinguish a preserved boundary (`EXPLICIT_EMBEDDING` on Quantize) from a
real float intermediate (`FLOAT_EMBEDDING`). Backward propagation must leave the
preserved Quantize's arguments and qparam metadata intact, including through layout
paths. Unprotected private quantizers retain their existing retarget policy.

The float continuation includes reshape/permute/slice layouts, static scale MUL,
and concat that joins float embedding branches or inserts a genuinely float side
input. Without a float side input, a concat with an independent quantized branch
keeps the ordinary integer concat policy; a float edge entering it receives an
explicit Quantize. Mixed concats that also contain an independent Q/DQ side branch
are diagnosed as unsupported. Membership in the region alone does not turn an
integer Quantize output or integer layout into float. Forward and backward
propagation respect marked float values instead of inventing qparams. Independent
quantized consumers of a shared float side input retain their own explicit Q.
This includes separate branches that reach Q through a chain of float layouts.
The side-input layout closure is classified before those terminal quantizers are
collected, preserving each branch's original grid independently of traversal or
consumer creation order.

When an embedding float layout feeds an ordinary quantized FC outside the CPU
region, its input must retain the earlier activation domain A. The rewrite requires
the FC's existing weight DQ and output quantization contract; genuine float FC
consumers retain their float inputs. The output contract can be a per-tensor Q
after a reshape/permute chain: these layouts already propagate B back to the FC
in the existing pipeline. The search follows only their data-input edges and
stops at arithmetic, concat, DQ and other operations. Output slice/slice_copy are
excluded because backward domain propagation does not support them yet, even
though they remain supported CPU input/side layouts. This detection establishes
that an output contract exists; it never supplies the input domain A. For a
layout-only path from `DQ_A(Q_A)`, the FC edge uses the same layouts applied to
Q_A's integer codes. A shared float consumer keeps the original DQ/layout path. This commuting
rewrite adds no rounding or clipping and leaves the FC weight and output domain B
intact; B is not an input-domain fallback. The integer layout copies have genuine
integer inputs and A qparams, while the original float layouts remain qparam-free.
The rewrite does not infer an input domain through arithmetic or mixed concat.

A bias-free, constant square linear feature transform can continue the CPU region.
Its weight DQ and input/output quantizers remain explicit, allowing the existing
FLOAT32 FC kernel to execute a rotation. The embedding table itself is excluded
from this rule, including a square tied LM-head table. This is a local continuation
policy, not a change to general FC quantization or a test for matrix orthogonality.

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

Other table dtypes, per-tensor or non-row table quantization, nonconstant tables,
and general linear/Conv continuations are outside this selected-row contract.

The optimization is implemented in the built-in reference runtime's NATIVE mode.
FAKE_QUANTIZE remains a semantic evaluation mode: another consumer such as FC may
use its existing full-weight dequantization cache. External CPU runtimes need their
own selected-row execution plan and support for the serialized DQ/Q/clamp sequence.
Reference execution does not establish production-runtime or device support.

## Model adapters and stage interfaces

The small-config integration tests explicitly opt in to this export policy and use
the real model classes, WrapQ prepare/calibrate/convert, production export adapters,
Circle files, and fresh
reference runtime instances. They require a compatible Transformers installation,
but no checkpoint download or accelerator.

| Path | Matching contract | Circle file export/reload | NATIVE embedding values | Stage connection | Full checkpoint / device |
| --- | --- | --- | --- | --- | --- |
| Llama token | UINT8 row-wise table; UINT8 or INT16 activation | Both activation dtypes | Exact integer oracle, decode and multiple tokens | INT16 producer/first-layer input grids agree; first layer executes in FAKE_QUANTIZE | Unverified |
| Qwen3-VL token | UINT8 row-wise table; INT16 activation; rotation off/on | Both rotation settings | Exact integer oracle and selected rows | Explicit host conversion to independently frozen decode grid; first layer executes in FAKE_QUANTIZE | Unverified |
| Qwen3-VL multimodal | Same text table; FLOAT32 image input; fixed contiguous insertion span | Rotation off/on, including a separately exported zero-visual-row profile | FLOAT32 without rotation; INT16 with rotation; explicit-boundary oracle | Vision output grid inspected and FAKE_QUANTIZE vision run; host DQ to image input; explicit conversion to prefill input and FAKE_QUANTIZE first layer | Unverified |
| Gemma4 scaled word embedding | UINT8 row-wise table; INT16 activation and quantized static scale | Existing real scaled-wrapper export regression | Exact explicit Q/DQ/MUL/Q oracle | Full decoder connection not covered here | Unverified |

Qwen rotation tests install a deterministic nonidentity orthogonal Hadamard R1
through production SpinQuant before PTQ. They preserve text activation Q/DQ,
rotation input Q/DQ, quantized rotation-weight DQ, and rotation output Q. Multimodal
insertion occurs after text Q/DQ and before rotation, so the combined text/image
tensor is rotated once. The unrotated multimodal adapter has no final Quantize:
its FLOAT32 output is intentional. Llama rotation is not covered by these tests.

The multimodal adapter takes `input_ids: INT64[1,S]` and
`image_embeds: FLOAT32[V,D]`, replacing the contiguous span beginning at its static
`visual_start_idx`. It consumes every supplied visual row in order. It has no
valid-count or padding-mask input: sentinel rows within V are inserted too. Sequence
padding outside that span retains the text embedding. A zero-image input requires
a separately exported V=0 profile, rather than changing an existing artifact's
shape. Disjoint image spans and masked visual capacity are not this adapter's ABI.
The vision artifact's INT16 features require host dequantization before this
FLOAT32 image input.

The reference NATIVE API returns raw integer codes for quantized outputs, without
automatic output DQ or input Q. Tests inspect serialized qparams and compare raw
codes and their explicitly dequantized values against a frozen pre-export oracle.
They also compare with full-table DQ execution on tiny fixtures. Circle reference
quantization rounds halfway values away from zero; WrapQ fake quantization uses
ties-to-even. These contracts are compared separately; halfway differences are
not hidden by a wider tolerance. Rotation uses the same FP32 operation order as
the reference FC contract.

Quantized decoder and vision arithmetic is not generally implemented in NATIVE
mode. First-layer/vision FAKE_QUANTIZE execution establishes the tested handoff
and shape contract, not native integer decoder support or full-model parity.
The Llama bundle loader also rejects quantized stages in NATIVE mode, and the Qwen
debug host runtime uses eager wrappers rather than executing these Circle files.
Production host integration and ONERT/TV selected-row support remain separate work.

Tied source parameters and frozen observer state are checked before and after
export/execution; Llama also checks a separately exported LM head. Existing shared
table/LM-head tests cover consumers within one graph. Qwen token and multimodal
Circle files each contain their own UINT8 table; their independent runtimes load
that storage separately. Selected-row execution avoids restored table copies
within an artifact, not duplicated packaging across files.

## Regression coverage

The explicit embedding tests use `preserve_embedding_quantization=True` through
the production export pipeline, Circle save/load and reference execution. Direct
preservation-pass unit tests exercise the pass independently of compile settings.
Relevant coverage includes:

- [Explicit embedding export](../test/quantization/passes/test_explicit_embedding_export.py):
  integer output contracts, intermediate quantization, scale operands and tied weights.
- [Quantize boundary preservation](../test/quantization/passes/test_explicit_embedding_retarget.py):
  Concat/layout consumers and the unprotected retarget control.
- [Reduced-range preservation](../test/quantization/passes/test_explicit_embedding_reduced_range.py):
  float DQ/clamp edges, saturation, pass idempotence and traversal-order independence.
- [Float continuation legality](../test/quantization/passes/test_embedding_float_continuation.py):
  insertion/layout/feature-transform execution, shared inputs, integer continuations,
  idempotence, and nearby unsupported or ordinary quantized patterns.
- [Ordinary FC boundaries](../test/quantization/passes/test_embedding_layout_consumer_boundary.py)
  and [shared float layout branches](../test/quantization/passes/test_embedding_shared_float_layout.py):
  serialized input domains, shared consumers, layout chains, independent Q grids,
  FC output quantization through nonidentity permute/reshape, and repeated
  preservation/folding. Ordinary quantized FC values use
  FAKE_QUANTIZE; its existing NATIVE unsupported diagnostic remains in force.
- [Selected-row execution](../test/unit_test/circle/runtime/test_rowwise_gather.py):
  row qparams, index bounds, sharing and loading/execution allocation checks.
- Real [Llama](../test/quantization/recipes/test_llama_explicit_embedding_integration.py),
  [Qwen token](../test/quantization/recipes/test_qwen_token_embedding_integration.py),
  and [Qwen multimodal](../test/quantization/recipes/test_qwen_multimodal_embedding_integration.py)
  adapter exports: frozen-grid oracles, rotation, insertion, stage handoff, and
  loading/execution hooks that reject full embedding-table DQ/cache creation.

Run the focused CPU coverage in an installed test environment:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest \
  test.quantization.recipes.test_llama_explicit_embedding_integration \
  test.quantization.recipes.test_qwen_token_embedding_integration \
  test.quantization.recipes.test_qwen_multimodal_embedding_integration \
  test.quantization.passes.test_explicit_embedding_export \
  test.quantization.passes.test_explicit_embedding_retarget \
  test.quantization.passes.test_explicit_embedding_reduced_range \
  test.quantization.passes.test_embedding_float_continuation \
  test.quantization.passes.test_embedding_layout_consumer_boundary \
  test.quantization.passes.test_embedding_shared_float_layout \
  test.quantization.passes.test_preserve_embedding_quantization \
  test.unit_test.circle.runtime.test_rowwise_gather
```

Allocation checks observe DQ kernel shapes, serialized constants, the selected-row
plan and runtime caches; they do not measure every NumPy allocation or device RSS.
The small oracle deliberately permits full-table DQ. Restoring a separate D-by-D
rotation matrix is also distinct from restoring the V-by-D vocabulary table.
