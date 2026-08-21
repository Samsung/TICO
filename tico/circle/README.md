# Circle artifact tools

`tico.circle` provides a reusable Python library and command-line interface for inspecting and transforming
 exported Circle model artifacts. It operates **after** TICO has serialized a `torch.export.ExportedProgram`
 into Circle, so it is intentionally separate from the existing `tico.passes` package, whose passes operate
 on PyTorch IR.

## Architecture

```text
.circle bytes
    │
    ▼
tico.circle.io
    │  generated circle_schema Object API
    ▼
CircleDocument
    ├── CircleGraph             producer/consumer/traversal index
    ├── verify_document()       internal consistency checks
    ├── inspect                 stable summaries and text output
    ├── operations.extract      workflow-level graph extraction
    └── passes                  composable Circle-to-Circle rewrites
            ├── compatibility     optional legacy custom-op recovery
            ├── legalize          optional dynamic-FC lowering
            ├── optimize          generic passes plus optional pattern fusions
            └── finalize          one-shot index compaction
```

## What verification means

Verification is a **static internal-consistency check** over the parsed Circle object
model.

### Checks currently performed

| Area | Checks | Result |
|---|---|---|
| Model containers | At least one subgraph exists; buffer 0 exists and is empty; the operator-code vector exists | Error on violation |
| Index integrity | Subgraph I/O, tensor buffers, operator opcodes, operator tensor lists, signature mappings, metadata buffers, and control-flow subgraph references are in range | Error on violation |
| Dataflow | A tensor has at most one producer; every consumed tensor and graph output is produced, declared as an input, or backed by a constant buffer | Error on violation |
| Tensor interface | `shape` and a non-empty `shapeSignature` have the same rank | Error on violation |
| Signatures | Each signature points to an existing subgraph, and mapped inputs/outputs are actual inputs/outputs of that subgraph | Error on violation |
| Graph hygiene | Duplicate I/O indices, duplicate tensor names, duplicate signature keys, inputs with producers, and unused tensors/buffers/operator codes | Warning |

Each finding includes a severity, a stable issue code, and an object path. For example:

```text
ERROR [UNDEFINED_INPUT] model.subgraphs[0].tensors[7]: Tensor 7 is consumed by
operators [3] but has no producer and is not an input or constant.
```

Loading and verification are separate stages. Malformed FlatBuffer bytes that cannot be
parsed fail during `CircleDocument.load()`; `verify()` checks consistency after parsing.

## Python API

### Load, inspect, verify, and save

```python
from tico.circle import CircleDocument
from tico.circle.inspect import format_document

model = CircleDocument.load("model.circle")
print(format_document(model, include_tensors=True, include_operators=True))

report = model.verify(raise_on_error=False)
for issue in report.issues:
    print(issue.format())

model.save("model.copy.circle")
```

`CircleDocument` owns a mutable generated `ModelT` object. Use `clone()` before a transformation when the original document must remain unchanged.

```python
copy = model.clone()
assert copy.model is not model.model
```

### Extract operators by index

Operator ranges are inclusive in the CLI. The Python API accepts explicit indices.

```python
from tico.circle.operations import extract_by_operator_indices

result = extract_by_operator_indices(
    model,
    operator_indices=range(20, 65),
    subgraph_index=0,
)
result.document.save("attention.circle")

# Tensor indices before and after compaction are both available.
print(result.source_boundary)
print(result.boundary)
```

`source_boundary` uses tensor indices from the input model. `boundary` uses the
compacted tensor indices in `result.document`.

Extraction computes a new graph boundary from the selected region:

1. A non-constant tensor produced outside the region and consumed inside it becomes a graph input.
2. A tensor produced inside the region and consumed outside it becomes a graph output.
3. An original graph output produced inside the region remains an output.
4. A terminal selected tensor with no selected consumer becomes an output.
5. Constant tensors remain internal and retain their referenced buffers.
6. Dead operators, tensors, buffers, and operator codes are removed after boundary reconstruction.

### Extract paths between tensor names

Tensor selectors are regular expressions. A source tensor starts forward reachability; a destination
 tensor starts backward reachability. With both boundaries present, extraction keeps operators in the
 intersection, which corresponds to operators on directed paths between the boundaries.

```python
from tico.circle.operations import extract_by_tensor_patterns

result = extract_by_tensor_patterns(
    model,
    from_patterns=(r"^tico::args_0$",),
    to_patterns=(r"self_attn_o_proj.*",),
    subgraph_index=0,
)
result.document.save("attention.circle")
```

### Run optimization and cleanup passes

```python
from tico.circle.passes import (
    CanonicalizeEquivalentOpsPass,
    CirclePassManager,
    CommonSubexpressionEliminationPass,
    FoldConstantsPass,
    FuseLinearOpsPass,
    EliminateIdentityOpsPass,
    SimplifyViewOpsPass,
)
from tico.circle.passes.cleanup import (
    CompactIndicesPass,
    DeadCodeEliminationPass,
)

pipeline = CirclePassManager(
    [
        CanonicalizeEquivalentOpsPass(),
        FoldConstantsPass(),
        SimplifyViewOpsPass(),
        EliminateIdentityOpsPass(),
        FuseLinearOpsPass(),
        CommonSubexpressionEliminationPass(),
        DeadCodeEliminationPass(),
        CompactIndicesPass(),
    ]
)
result = pipeline.run(model)
print(result.changes)
model.save("model.optimized.circle")
```

`FoldConstantsPass` folds supported operators to a fixed point while preserving
existing output tensor indices and contracts. The first evaluator set covers `ADD`,
`MUL`, `CAST`, `RESHAPE`, `SHAPE`, `SQUEEZE`, and `GATHER`. Arithmetic folding is
limited to conservative dense cases, while exact quantized view operations may retain
their original qparams. Configurable storage and compute budgets prevent excessive
compile-time work or model growth. Newly dead producers are removed by default.

`CanonicalizeEquivalentOpsPass` reduces equivalent operator forms to a canonical
vocabulary. It converts static `EXPAND_DIMS`, one-input `PACK`, `SQUEEZE`, view-only
`STRIDED_SLICE`, and unit-dimension-only `TRANSPOSE` to `RESHAPE`; zero-valued
`PADV2` to `PAD`; and equal-size `SPLIT_V` to `SPLIT`.

`SimplifyViewOpsPass` rewires identity `RESHAPE` and `TRANSPOSE`, composes
compatible view chains, and moves `RESHAPE` after supported unary, scalar-binary, and
keep-dims `MEAN` operations. It intentionally leaves operators made unreachable by
rewiring in the graph. Schedule `DeadCodeEliminationPass` after it, followed by
`CompactIndicesPass`, to remove and compact those dead objects.

`EliminateIdentityOpsPass` removes contract-preserving `ADD` with an exact zero,
same-type `CAST`, full-range `SLICE`, identity `STRIDED_SLICE`, and one-output
`SPLIT` or `SPLIT_V`. Quantized arithmetic is kept conservative because fixed-point
requantization may remain observable even when real-number algebra suggests an
identity.

`FuseLinearOpsPass` absorbs supported static FLOAT32 affine patterns into
`FULLY_CONNECTED`, `CONV_2D`, `DEPTHWISE_CONV_2D`, and `TRANSPOSE_CONV` parameters.
It folds channel-wise post-linear `ADD`, `SUB`, and `MUL`; pre-FC affine input
transforms; decomposed BatchNorm-style `SUB`/`MUL`/`ADD` chains; and sums of two
compatible FC branches with a shared input. Existing weight and bias tensors are never
mutated in place, so shared parameters remain valid.

The pass replaces only the matched anchor operator. Superseded linear and affine
operators remain structurally valid but unreachable until an external
`DeadCodeEliminationPass` removes them. Run `CompactIndicesPass` afterward to remove
unused tensors, buffers, and operator codes. The initial implementation intentionally
skips quantized, sparse, variable, dynamic, non-finite, or unsupported broadcast
patterns. Floating-point parameter fusion is algebraically equivalent in real
arithmetic but may change rounding because it reassociates operations. Python callers
can set `LinearFusionPolicy(allow_float_reassociation=False)` to disable these
rewrites in a strict floating-point pipeline.

`CommonSubexpressionEliminationPass` reuses the outputs of structurally identical
pure operators. Its expression key includes the effective operator code and version,
ordered input identities, serialized options, intermediate contracts, and complete
output contracts including quantization metadata. Stateful, variable, control-flow,
random, and custom operators are skipped conservatively. Duplicate graph-output
producers are preserved so public tensor identities and names remain stable. Schedule
`DeadCodeEliminationPass` and `CompactIndicesPass` after CSE to remove superseded
operators and objects.

The built-in O1 preset runs the canonicalization, simplification, fusion, constant
folding, CSE, and dead-code passes in complete rounds until no pass reports a change, then
runs index compaction exactly once:

```python
from tico.circle.passes import create_o1_pipeline

result = create_o1_pipeline().run(model)
print(result.changes)
```

The canonicalization and rank-changing view rules reject dynamic, sparse, variable,
or unsupported per-axis-quantized patterns rather than guessing their semantics.
Contract-exact no-op rules may still handle dynamic tensors when the operator remains
an identity for every runtime shape.

By default, `CirclePassManager` verifies the document after every pass. 
Set `CirclePassContext(verify_after_each_pass=False)` only when a multi-step 
transformation intentionally has a temporary invalid state and performs 
explicit verification at the end.

### Optimization sessions and atomic rewrites

`CirclePassContext` retains one model-scoped `CircleOptimizationSession` for the
lifetime of a pass pipeline. The session caches `CircleGraph` producer/consumer
indexes by subgraph revision and lets builders that use the same tensor-type registry
and object factory share one canonical `ConstantPool`. A committed mutation advances
the affected revision and invalidates only the corresponding graph cache.

`CircleRulePass` applies each matched `CircleRewriteRule` in a
`CircleMutationTransaction`. Builders and tensor-use replacement helpers join the
active transaction automatically. If `apply()` raises or leaves the scope without
committing, appended buffers, operator codes, tensors, and operators are discarded;
captured operators and tensors, subgraph interfaces, and signature mappings are
restored. A custom rule that mutates an existing buffer payload directly must first
register that buffer with `current_mutation().watch_buffer(...)`.

Passes that still mutate the Object API directly remain supported. `CirclePassManager`
invalidates cached analyses when such a pass reports a change, and conservatively
rebuilds session state when a pass fails. Standalone code that mutates outside a pass
manager should call `context.session(document).mark_modified(...)` before reusing
cached analyses.

### Rule and pipeline scheduling

`CircleRulePass` uses a deterministic local worklist. After a rewrite it
revisits the captured producer/consumer neighborhood instead of rescanning the
complete graph prefix. Once local work is exhausted, one ordered full-graph
sweep validates that no non-local match was missed.

O1 uses `CirclePassStrategy.UNTIL_NO_CHANGE`: every optimization pass runs in
order for a complete round, and another round starts only when at least one
pass changed the document. The legacy `RESTART` strategy remains available for
explicitly custom pipelines and for scheduler comparison benchmarks.

## Command-line interface

The package installs one executable with subcommands:

```bash
tico-circle --help
```

All diagnostics are written to standard error. Binary Circle output can
 therefore be safely written to standard output and piped into another command.

### Inspect

```bash
tico-circle inspect model.circle

tico-circle inspect model.circle \
  --subgraph 0 \
  --tensors \
  --operators

tico-circle inspect model.circle --json
```

### Verify

```bash
tico-circle verify model.circle

tico-circle verify model.circle --warnings-as-errors
```

The command performs the internal-consistency checks described in
[What verification means](#what-verification-means). It exits with status `1` when an
error is found. Warnings normally keep status `0`; `--warnings-as-errors` changes that
behavior for stricter CI use.

Verification also runs automatically:

- after graph extraction, unless `--no-verify` is used
- after each optimization pass and at pipeline completion, unless `--no-verify` is used
- during `tico-circle inspect --verify`

From Python, `CircleDocument.verify()` raises `CircleVerificationError` on structural
errors by default. Pass `raise_on_error=False` to inspect a `VerificationReport` without
raising.

### Extract by operator index

```bash
tico-circle extract model.circle \
  --subgraph 0 \
  --ops 20-64 \
  -o attention.circle
```

Multiple inclusive ranges and individual indices are supported:

```bash
tico-circle extract model.circle \
  --ops 0-10,15,20-24 \
  -o region.circle
```

A colon can also delimit an inclusive range, for example `20:64`.

### Extract by tensor boundary

```bash
tico-circle extract model.circle \
  --from-tensor '^tico::args_0$' \
  --to-tensor 'self_attn_o_proj.*' \
  -o attention.circle
```

`--from-tensor` and `--to-tensor` may each be repeated. Add `--full-match` to use full regular-expression 
matching instead of search semantics.

### Keep other subgraphs

Extraction produces a single-subgraph model by default. Use `--keep-other-subgraphs` to retain the others.
Tensor cleanup is limited to the selected subgraph, while model-global buffer and operator-code compaction
 may remap references in every retained subgraph.

```bash
tico-circle extract merged.circle \
  --subgraph 1 \
  --ops 0-40 \
  --keep-other-subgraphs \
  -o merged.partial.circle
```

Global buffers are compacted across all retained subgraphs. A buffer shared by multiple retained subgraphs
 remains shared and is stored once.

### Signature policy

The default extraction policy drops signatures for the rewritten subgraph because newly introduced graph
 boundaries usually do not have a complete source signature mapping.

Use `--preserve-compatible-signatures` to keep a signature only when its input and output tensor sets
 exactly equal the extracted graph inputs and outputs.

```bash
tico-circle extract model.circle \
  --ops 0-100 \
  --preserve-compatible-signatures \
  -o model.extracted.circle
```

Signatures for untouched subgraphs remain intact when `--keep-other-subgraphs` is used.

### Optimize

```bash
tico-circle optimize model.circle \
  --passes simplify-view-ops,dce,compact \
  -o model.optimized.circle
```

Available passes:

| Name | Implementation | Behavior |
|---|---|---|
| `canonicalize-equivalent-ops` | `CanonicalizeEquivalentOpsPass` | Replaces equivalent operator spellings with canonical `RESHAPE`, `PAD`, or `SPLIT` forms |
| `cse` | `CommonSubexpressionEliminationPass` | Reuses structurally identical pure expressions while preserving graph-output tensor identities |
| `eliminate-identity-ops` | `EliminateIdentityOpsPass` | Removes operators that preserve the complete input tensor contract |
| `eliminate-transpose-bounded-layout-region` | `EliminateTransposeBoundedLayoutRegionPass` | Rewrites supported Transpose-bounded layout regions into the source layout |
| `fold-constants` | `FoldConstantsPass` | Folds supported constant operators with the selected `basic` or `heavy` evaluator profile |
| `fuse-composite-ops` | `FuseCompositeOpsPass` | Recognizes supported composite activation, normalization, and arithmetic patterns |
| `fuse-legacy-fc-gelu-fc` | `FuseLegacyFCGeluFCPass` | Recognizes the optional legacy FC-Erf GELU pattern |
| `fuse-linear-ops` | `FuseLinearOpsPass` | Folds safe static FLOAT32 affine patterns into linear weights and biases |
| `fuse-transpose-conv-slice` | `FuseTransposeConvSlicePass` | Fuses a supported static TransposeConv-Slice spatial pattern |
| `legalize-dynamic-fully-connected` | `LegalizeDynamicFullyConnectedPass` | Lowers supported dynamic-weight FLOAT32 FullyConnected operators |
| `resolve-legacy-custom-ops` | `ResolveLegacyCustomOpsPass` | Recovers selected former TensorFlow custom operators as Circle builtins |
| `simplify-arithmetic` | `SimplifyArithmeticPass` | Applies policy-controlled arithmetic canonicalization and simplification |
| `simplify-reduction-ops` | `SimplifyReductionOpsPass` | Simplifies supported consecutive or layout-adjacent reduction patterns |
| `simplify-view-ops` | `SimplifyViewOpsPass` | Rewires, composes, and safely moves compatible `RESHAPE` and `TRANSPOSE` views |
| `dce` | `DeadCodeEliminationPass` | Removes unreachable pure operators while preserving observable effects and protected graph inputs |
| `compact` | `CompactIndicesPass` | Removes unused tensors, buffers, and operator codes and remaps all supported references |

Dead-code elimination treats stateful, non-deterministic, custom, variable, and
subgraph-referencing operators as roots. Input pruning retains signature-bound inputs
and the complete input arity of subgraphs referenced by call or control-flow operators.

Run the built-in O1 pipeline with:

```bash
tico-circle optimize model.circle \
  --preset o1 \
  -o model.o1.circle
```

O1 owns its round-based fixed-point scheduling, so `--strategy` cannot be combined with `--preset`.
Use `--passes` and `--strategy` instead when a custom pass sequence is required.

Select the heavy evaluator profile or enable optional O1 phases explicitly:

```bash
tico-circle optimize model.circle \
  --preset o1 \
  --constant-folding-profile heavy \
  --fuse-transpose-conv-slice \
  --legalize-dynamic-fully-connected \
  --resolve-legacy-custom-ops \
  --fuse-legacy-fc-gelu-fc \
  -o model.o1.extended.circle
```

The constant-folding profile is an optimization option. Dynamic FullyConnected lowering
belongs to legalization, while custom-op recovery and the legacy FC-GELU-FC matcher are
compatibility options. These optional O1 flags require `--preset o1`; each underlying
transformation can also be selected directly by its canonical `--passes` name.

`--passes` defaults to `dce,compact`. View simplification is intentionally split
across three passes: `simplify-view-ops` rewires dataflow, `dce` removes newly dead
operators, and `compact` removes and remaps unused tensors, buffers, and operator codes.

Fold supported constant subgraphs and compact the newly unused objects with:

```bash
tico-circle optimize model.circle \
  --passes fold-constants,compact \
  -o model.constant-folded.circle
```

Canonicalize and simplify equivalent view and identity patterns with round scheduling:

```bash
tico-circle optimize model.circle \
  --passes canonicalize-equivalent-ops,simplify-view-ops,eliminate-identity-ops,dce,compact \
  --strategy until-no-change \
  -o model.simplified.circle
```

Round scheduling lets a later rewrite expose an earlier canonicalization candidate
during the next complete pass round. Use `restart` only for an explicitly legacy
custom pipeline or scheduler comparison.

Fuse static FLOAT32 linear and affine chains, then remove the superseded branches with:

```bash
tico-circle optimize model.circle \
  --passes fuse-linear-ops,dce,compact \
  --strategy until-no-change \
  -o model.linear-fused.circle
```

The constant-folding pass preserves existing output tensor indices, so graph outputs
and signature output mappings remain stable. It skips dynamic contracts, external
buffers, unsafe integer overflow, non-zero fused activations, unsupported qparams, and
zero-element outputs that cannot yet be represented as owned constants. Signature-bound
graph inputs are retained even when a metadata-only fold removes their data dependence.

`EliminateTransposeBoundedLayoutRegionPass` moves a region into the source layout when
all external data inputs cross one Transpose permutation and all external data outputs
cross its inverse. The registered layout-invariant operator families are:

- unary: `ABS`, `CAST`, `CEIL`, `COS`, `DEQUANTIZE`, `ELU`, `EXP`, `FLOOR`,
  `LEAKY_RELU`, `LOG`, `LOGICAL_NOT`, `LOGISTIC`, `NEG`, `QUANTIZE`, `RELU`,
  `RELU6`, `RELU_N1_TO_1`, `RSQRT`, `SIN`, `SQRT`, `SQUARE`, `TANH`, and
  `ZEROS_LIKE`
- binary without broadcasting: `ADD`, `DIV`, `EQUAL`, `FLOOR_DIV`, `FLOOR_MOD`,
  `GREATER`, `GREATER_EQUAL`, `LESS`, `LESS_EQUAL`, `LOGICAL_AND`, `LOGICAL_OR`,
  `MAXIMUM`, `MINIMUM`, `MUL`, `NOT_EQUAL`, `POW`, `SQUARED_DIFFERENCE`, and `SUB`
- variadic without broadcasting: `ADD_N`
- axis option remapping: `CONCATENATION`
- constant padding-row remapping: `PAD`, `PADV2`, and `MIRROR_PAD`
- constant rank-vector remapping: `TILE` and `SLICE`
- multi-output axis-constant remapping: `SPLIT` and `SPLIT_V`

Unary inputs and outputs must have the same shape. Binary and variadic operators require
all data inputs and outputs to have exactly the same shape, so broadcasting remains a
region boundary. `CONCATENATION` remaps its normalized axis. `PADV2` preserves its
scalar padding value, and `MIRROR_PAD` preserves its reflection mode. `TILE` requires a
static INT32 multiples vector. `SLICE` requires static INT32 begin and size vectors and
supports `-1` only in the size vector. `SPLIT` requires a static INT32 axis, equal-size
outputs, and a matching `numSplits` option. `SPLIT_V` requires static INT32 axis and
size-splits constants, permits at most one inferred `-1` size, and validates every
output shape. Rank-changing or unsupported axis-sensitive operators such as `PRELU`,
`RESHAPE`, and `SOFTMAX` remain region boundaries.

Run the bounded-region pass with round scheduling and cleanup:

```bash
tico-circle optimize model.circle \
  --passes eliminate-transpose-bounded-layout-region,simplify-view-ops,dce,compact \
  --strategy until-no-change \
  -o model.optimized.circle
```

### Standard input and output

Use `-` for a binary stream:

```bash
tico-circle extract model.circle --ops 0-100 -o - \
  | tico-circle optimize - \
      --passes simplify-view-ops,dce,compact \
      -o output.circle
```

Do not redirect `inspect` text into a Circle transformation command; `inspect` writes text by design.

## Graph and index handling

Circle uses several independent index spaces:

- model-global buffer indices
- model-global operator-code indices
- model-global subgraph indices
- subgraph-local tensor indices
- signature tensor-map indices into one subgraph

`compact_model()` updates the supported references together rather than deleting individual objects in isolation.
It preserves buffer 0 and keeps model metadata buffers. It also remaps signature tensor maps after tensor compaction.

The extraction workflow refuses to remove a subgraph when a retained operator's Object API options refer to 
that subgraph. This prevents silently producing invalid `IF`, `WHILE`, `CALL_ONCE`, or similar control-flow models.

## Writing a new Circle pass

Implement `CirclePass.run()` and return a `CirclePassResult`.

```python
from tico.circle.passes import CirclePass, CirclePassResult


class RenameDescriptionPass(CirclePass):
    """Set a stable model description."""

    def run(self, document, context):
        if document.model.description == "optimized":
            return CirclePassResult(modified=False)
        document.model.description = "optimized"
        return CirclePassResult(modified=True, changes=1)
```

Pass requirements:

- place the implementation in the semantic package that owns its responsibility:
  `canonicalize`, `simplify`, `fold`, `fuse`, `legalize`, `compatibility`, or `cleanup`
- mutate only the supplied `CircleDocument` and report `modified=True` only when model
  state changed
- use `CircleRewriteRule` for local patterns so the shared worklist, optimization
  session, and atomic mutation transaction can manage invalidation and rollback
- use `CircleGraph` or the session graph cache instead of rebuilding producer and
  consumer maps independently
- preserve Circle structural invariants at pass boundaries unless verification is
  explicitly disabled by the caller
- use the helpers in `rewrite.py` whenever deleting or remapping indexed objects
- register one canonical CLI name when the pass is user-selectable, and update
  `presets.py` only when a built-in pipeline should schedule it
- add positive and close non-matching tests; include multi-subgraph, shared-buffer,
  effect, signature, rollback, and fixed-point cases when applicable

## Testing

Run the Circle tool unit tests:

```bash
./ccex test -k circle
```

The tests include a schema-independent Object API fixture so graph, selection, rewrite, verification, pass scheduling,
 and extraction behavior can be tested without generating binary fixtures. When `circle-schema` and `flatbuffers` are
 installed, an additional integration test serializes and deserializes a minimal generated `ModelT`.

Important test scenarios include:

- graph producer and consumer indexing
- operator and tensor-boundary selection
- redundant Reshape and inverse Transpose elimination
- dead branch elimination
- signature tensor-map remapping
- shared buffer preservation across two subgraphs
- single-subgraph extraction from a multi-subgraph model
- compatible and incompatible signature handling
- invalid buffer, tensor, operator-code, signature, and subgraph references
- atomic file writes and binary stream I/O
- generated Object API NumPy-vector round trips
- scalar and vector control-flow subgraph reference remapping
- metadata buffer preservation and remapping
- constant-fold fixed points, budgets, overflow rejection, and multi-output rollback
- equivalent-op canonicalization with static constants and output-contract checks
- no-op removal with graph-output and signature remapping
- identity and chained view simplification with malformed-pattern rejection
- generated Circle round-trip value preservation for representative semantic rewrites
- atomic mutation rollback and optimization-session invalidation
- non-empty O1 idempotence and scheduler-equivalence coverage

## Current limitations

This implementation performs structural Circle rewrites, bounded constant evaluation,
and the documented optional legalizations. It does not perform numerical equivalence
testing, runtime execution, general shape inference, backend capability validation,
or comprehensive target-specific legalization.

Additional limitations:

- A constant is recognized by inline buffer data or a non-zero external buffer offset/size. A zero-sized constant
 with no payload metadata may be conservatively promoted to an extracted graph input. Constant folding skips
 zero-element outputs until graph-level constant ownership can distinguish them from absent storage.
- Tensor name selectors rely on names being present and stable. Operator indices remain useful for debugging
 but may change after any rewrite.
- Signature synthesis is intentionally not attempted when extraction creates new boundaries; only an exactly
 compatible source signature can be retained.
- Control-flow references are discovered from scalar `*SubgraphIndex` fields, vector `*SubgraphIndices` fields,
 and `CallOptions.subgraph`. A new schema option with a different naming convention must be added to the reference walker.
- Structural verification does not guarantee that a runtime accepts the model or that outputs are numerically equivalent.

## Pass taxonomy

Circle transformations are grouped by semantic responsibility under
`tico.circle.passes.optimization`:

- `canonicalize`: reduce equivalent operator spellings to a canonical form
- `simplify`: remove identities and simplify views, arithmetic, reductions,
  and layout regions
- `fold`: evaluate constant subgraphs under an explicit evaluator profile
- `fuse`: combine generic composite, linear, and spatial patterns
- `legalize`: lower representations that are not directly executable
- `compatibility`: recover legacy ONE or frontend-specific graph patterns

The temporary `canon`, `fusion`, and `remove` forwarding packages were
removed after the taxonomy migration. Import the semantic packages above;
deprecated class and CLI spellings are no longer accepted.

Constant folding uses one pass with an explicit profile instead of separate
basic and heavy pass implementations:

```python
from tico.circle.passes import ConstantFoldingProfile, FoldConstantsPass

folding = FoldConstantsPass(profile=ConstantFoldingProfile.HEAVY)
```

### Benchmark full-model O1 scheduling

A caller-provided full Circle artifact can compare the former restart
scheduler with the round scheduler while requiring byte-identical output:

```bash
python3 -m test.performance.benchmark_circle_optimizer \
  model.circle \
  --repeat 3
```

The benchmark does not download or commit model artifacts. It reports elapsed
time, pass executions, output size, and SHA-256.
