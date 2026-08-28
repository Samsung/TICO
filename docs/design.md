# TICO System Design

TICO converts PyTorch programs represented as `torch.export.ExportedProgram` into
Circle FlatBuffers. The implementation uses PyTorch's exported ATen graph as its
working IR, applies a fixed legalization/optimization pipeline, conditionally legalizes
quantized graphs, and serializes each remaining operator through a registered Circle
visitor.

This document describes the behavior implemented on the current `main` branch. It is
not a future architecture proposal.

## Contents

- [1. Scope and non-goals](#1-scope-and-non-goals)
- [2. End-to-end architecture](#2-end-to-end-architecture)
- [3. Public API layer](#3-public-api-layer)
- [4. Package responsibilities](#4-package-responsibilities)
- [5. Conversion pipeline](#5-conversion-pipeline)
- [6. Pass manager and graph invariants](#6-pass-manager-and-graph-invariants)
- [7. Compile configuration](#7-compile-configuration)
- [8. Circle serialization](#8-circle-serialization)
- [9. Shapes, interfaces, and runtime binding](#9-shapes-interfaces-and-runtime-binding)
- [10. Quantization integration](#10-quantization-integration)
- [11. Circle artifact layer](#11-circle-artifact-layer)
- [12. Validation, errors, and diagnostics](#12-validation-errors-and-diagnostics)
- [13. Extension points](#13-extension-points)
- [14. Current limitations](#14-current-limitations)
- [15. Implementation source map](#15-implementation-source-map)

## 1. Scope and non-goals

### Scope

TICO currently provides:

- `nn.Module` to Circle conversion through `torch.export.export()`
- in-memory `ExportedProgram` to Circle conversion
- saved `.pt2` to Circle conversion
- graph decomposition, legalization, canonicalization, and selected optimizations
- conditional handling of graphs that already contain quantization operations
- Circle serialization through ATen-target-specific visitors
- a Python `CircleModel` wrapper for saving, loading, and local execution
- post-serialization Circle inspection, verification, extraction, optimization, and cleanup tools
- a separate model quantization subsystem with public `prepare()` and `convert()` APIs
- config-driven quantization recipes for model-family workflows

### Non-goals of the core converter

The core conversion API does not:

- train a model or support training graphs
- quantize an arbitrary floating-point model by itself
- provide an unrestricted PyTorch interpreter
- guarantee support for every ATen operator or arbitrary Python control flow
- compile or schedule the result for a particular NPU backend
- prove numerical parity or backend compatibility merely by producing valid Circle bytes
- currently generate multiple Circle subgraphs from one `ExportedProgram`

## 2. End-to-end architecture

```text
PyTorch nn.Module
    │
    │ tico.convert(...)
    ▼
torch.export.export
    │
    ├──────────────────────────────┐
    │                              │
Saved .pt2                    In-memory ExportedProgram
    │ torch.export.load             │
    └──────────────┬───────────────┘
                   ▼
      convert_exported_module_to_circle
                   │
                   ├─ fake-quant decomposition
                   ├─ PyTorch decomposition with preserved operators
                   ├─ fixed-point legalization/optimization passes
                   ├─ relaxed-invariant cleanup
                   ├─ conditional quantization passes
                   ├─ supported-target and training-op checks
                   ▼
             Circle serializer
                   │
                   ▼
               Circle bytes
                   │
                   ├─ CircleModel.save/load/__call__
                   └─ tico.circle inspect/verify/extract/optimize
```

The working graph remains an `ExportedProgram`; TICO does not introduce a separate
custom middle IR between PyTorch export and Circle serialization.

## 3. Public API layer

The top-level `tico` package exports:

```python
from tico import (
    CompileConfigV1,
    convert,
    convert_from_exported_program,
    convert_from_pt2,
    get_default_config,
)
```

### `convert`

```python
convert(
    mod: torch.nn.Module,
    args: tuple,
    kwargs: dict | None = None,
    dynamic_shapes: dict | tuple | None = None,
    strict: bool = True,
    config: CompileConfigBase = get_default_config(),
) -> CircleModel
```

Behavior:

1. Warns fatally when the module reports `training=True`.
2. Calls `torch.export.export()` under `torch.no_grad()`.
3. Runs the common `ExportedProgram` conversion path.
4. Returns a `CircleModel` containing the serialized bytes.

### `convert_from_exported_program`

Accepts an existing `ExportedProgram`, skips `torch.export.export()`, and runs the same
legalization, validation, and serializer pipeline.

### `convert_from_pt2`

Loads an exported program with `torch.export.load()` and runs the same conversion path.
The `pt2-to-circle` executable is a file-oriented wrapper around this behavior and
supports a versioned YAML compile configuration.

## 4. Package responsibilities

| Package | Current responsibility |
|---|---|
| `tico/config/` | Versioned core compile configuration. |
| `tico/utils/convert.py` | Public conversion orchestration and the authoritative pass order. |
| `tico/passes/` | Rewrites over PyTorch `ExportedProgram` / FX graphs. |
| `tico/serialize/` | Circle graph construction, tensor/buffer encoding, dtype/shape mapping, and operator visitors. |
| `tico/serialize/operators/` | Registered ATen-overload-to-Circle operator lowering. |
| `tico/interpreter/` | In-process execution of supported one-subgraph Circle models. |
| `tico/circle/` | Post-serialization Circle document APIs, verification, extraction, and Circle-to-Circle passes. |
| `tico/quantization/` | Model quantization APIs, algorithms, WrapQ infrastructure, graph quantization passes, recipes, evaluation, export, and analysis. |
| `test/modules/` | Small PyTorch programs used by generated conversion/parity tests. |
| `test/unit_test/` | Focused core conversion and artifact-tool tests. |
| `test/quantization/` | Quantization-specific tests. |

A key boundary is the IR being transformed:

- `tico/passes/` operates before serialization on PyTorch exported graphs.
- `tico/circle/passes/` operates after serialization on Circle object-model documents.

## 5. Conversion pipeline

The authoritative implementation is
`convert_exported_module_to_circle()` in `tico/utils/convert.py`.

### 5.1 Input assumptions

The input is an `ExportedProgram` whose graph can be decomposed to supported core ATen
patterns. Placeholder order and `graph_signature.input_specs` must remain aligned.
Tensor-producing nodes are expected to carry usable `meta["val"]` shape and dtype
information before serialization.

### 5.2 Fake-quant decomposition

Before the normal PyTorch decomposition stage, TICO runs:

```text
DecomposeFakeQuantize
DecomposeFakeQuantizeTensorQParams
```

This exposes quantization behavior in forms that later decomposition, qparam
propagation, and Circle serialization understand.

### 5.3 PyTorch decomposition

TICO then calls `ExportedProgram.run_decompositions()` through a version-adapted helper.
Selected operators are deliberately preserved rather than decomposed because TICO has
specific legalization or serializer handling for them. The preserved set includes
convolution variants, selected activations and normalizations, linear, nearest-neighbor
upsampling, and RMS normalization.

When `TICO_GRAPH_DUMP` is set, the first graph snapshot is written after this stage.

### 5.4 Main legalization and optimization bundle

The main `PassManager` currently runs the following ordered bundle. Because the default
strategy is `RESTART`, a successful rewrite restarts scanning from the first pass.

```text
FillMetaVal
ExtractDtypeKwargsPass
RemoveNop
LowerCopy
ConvertGatherToGatherNd
ConvertSymSizeToCircleShape
ConvertLayoutOpToReshape
RestoreLinear
ConvertToReLU6
DecomposeAddmm
DecomposeSliceScatter
DecomposeGroupNorm
DecomposeBatchNorm
DecomposeGroupedConv2d
CastATenWhereArgType
ConvertRepeatToExpandCopy
RemoveRedundantPermutePasses
RemoveRedundantAssertionNodes
RemoveRedundantExpand
RemoveRedundantSlice
FuseRedundantReshapeToMean
RemoveRedundantViewPasses
RemoveRedundantToCopy
MergeConsecutiveCat
CastMixedTypeArgs(preserve_ep_invariant=True)
ConstPropPass
SegmentIndexSelectConst
LegalizeCausalMaskValue(config-gated)
ConvertExpandToSliceCat(config-gated)
ConvertMatmulToLinear(config-gated variants)
LowerToResizeNearestNeighbor
LegalizePreDefinedLayoutOperators
LowerPow2ToMul
ConvertConv1dToConv2d
ConvertConv3dToConv2d
LowerToSlicePasses
FuseLeadingUnsqueezeReshape
CastClampMixedTypeArgs
EliminateRankRoundTripRegion(enabled=True)
```

This list intentionally mixes legalization and optimization today; the implementation
contains a TODO to separate those concerns more explicitly. Adding a pass class does
not automatically schedule it. It must be inserted into this explicit pipeline.

### 5.5 Relaxed-invariant cleanup

After the main bundle, TICO runs:

```text
FillMetaVal
CastMixedTypeArgs(preserve_ep_invariant=False)
```

The code explicitly permits the strict `ExportedProgram` invariant to be relaxed at
this point; graph constants may exist without being lifted back into placeholders.
Serializer and subsequent passes must therefore handle the resulting representation.

The second graph snapshot is emitted after this phase when graph dumping is enabled.

### 5.6 Conditional quantization bundle

TICO detects whether the graph contains quantization operations. Only then it runs:

```text
FoldQuantOps
RemoveWeightDequantOp
PropagateQParamForward
PropagateQParamBackward
LegalizeQuantizedClamp
QParamSafeConstPropPass
QuantizeBias
RemoveUnusedPlaceholder
InsertQuantizeOnDtypeMismatch
```

`LegalizeQuantizedClamp` converts constant Clamp bounds to the activation's
per-tensor integer domain, aligns the Clamp input to that same scale and zero point,
and removes bounds that cover the complete quantized dtype range. Backend-specific
Clamp lowering and Linear fusion remain outside the core converter.

It then reports missing qparams non-strictly, with a specific exception for
`split_with_sizes` because qparams are attached to its `getitem` result nodes.

This phase legalizes and completes a graph that has already been prepared as quantized.
It is not a replacement for model calibration or the public quantization workflow.

The third graph snapshot is emitted after this phase when enabled.

### 5.7 Final checks and serialization

Before serialization, TICO:

1. Checks every remaining `call_function` target against registered serializer visitors,
   allowing `operator.getitem` as a multiple-output bookkeeping operation.
2. Rejects training operators such as `aten.dropout` and `aten.native_dropout`.
3. Calls `build_circle()` to produce a `CIR0` FlatBuffer.

## 6. Pass manager and graph invariants

### Pass interface

Core PyTorch-IR passes implement:

```python
class PassBase(ABC):
    def call(self, exported_program: ExportedProgram) -> PassResult:
        ...

@dataclass
class PassResult:
    modified: bool
```

`modified` is part of the scheduler contract. A pass that changes the graph must report
it accurately unless it intentionally guarantees a one-shot transformation and has a
well-documented reason not to restart.

### Scheduling strategies

- `PassStrategy.RESTART` is the default. After a modification, execution resumes from
  the first pass in the bundle.
- `PassStrategy.UNTIL_NO_CHANGE` completes the bundle before starting another
  iteration.
- The manager fails after 1,000 changing iterations to detect circular rewrite loops.

### Graph signature maintenance

Each pass call runs under the `ExportedProgram` graph-signature replacement hook. Node
replacement must still preserve these invariants:

- Placeholder node order matches `graph_signature.input_specs`.
- User inputs, parameters, buffers, constants, and outputs retain valid bindings.
- New tensor nodes carry correct shape/dtype metadata.
- Dead nodes are removed when the rewrite makes them unreachable.
- Graph linting and recompilation follow structural changes where required.

### Serializer-facing invariant

At the end of conversion:

- Every non-`getitem` `call_function` target has a registered `NodeVisitor`.
- Tensor-producing nodes have serializable dense values and metadata.
- User inputs and outputs resolve to registered Circle tensors.
- Circle shapes and shape signatures are internally consistent.
- Quantized tensors carry qparam metadata in a representation understood by the
  serializer.

## 7. Compile configuration

`CompileConfigFactory` currently supports version `1.0`, implemented by
`CompileConfigV1`.

| Field | Default | Consumed by |
|---|---:|---|
| `legalize_causal_mask_value` | `False` | `LegalizeCausalMaskValue` |
| `remove_constant_input` | `False` | Circle input registration in `build_circle()` |
| `convert_lhs_const_mm_to_fc` | `False` | `ConvertMatmulToLinear` |
| `convert_rhs_const_mm_to_fc` | `True` | `ConvertMatmulToLinear` |
| `convert_single_batch_lhs_const_bmm_to_fc` | `False` | `ConvertMatmulToLinear` |
| `convert_expand_to_slice_cat` | `False` | `ConvertExpandToSliceCat` |
| `eliminate_rank_round_trip` | `False` | Currently not consumed; the pass is instantiated with `enabled=True`. |

`CompileConfigBase.from_dict()` applies only keys already present on the dataclass. The
current implementation ignores unknown fields rather than rejecting them. Configuration
review and tests should therefore catch misspellings and obsolete keys.

Conversion behavior that affects semantics or backend compatibility should be exposed
through an explicit configuration field only when the pipeline actually consumes that
field. Keep the dataclass, YAML examples, implementation wiring, and tests synchronized.

## 8. Circle serialization

`build_circle()` constructs a generated Circle object model and packs it with
FlatBuffers.

### 8.1 Model construction

The current serializer:

- creates one `CircleSubgraph`
- reserves buffer 0 for tensors without embedded data
- exports graph tensors and constants
- registers `InputKind.USER_INPUT` values as graph inputs
- optionally excludes `ConstantArgument` inputs
- always excludes `None` constant inputs
- registers non-`None` user outputs
- emits one Circle operator for each supported non-`getitem` call-function node
- validates tensor shapes before packing
- writes the `CIR0` file identifier

### 8.2 Operator visitors

`NodeVisitor` subclasses register one or more ATen overload targets with the
`register_node_visitor` decorator. The registry provides both:

- the target-to-visitor mapping used during serialization
- the supported-target set used by pre-serialization validation

A newly added visitor is not useful unless its module is imported by the serializer
operator package so its registration side effect occurs.

### 8.3 Tensor data and shared storage

Parameters, buffers, and lifted constants are copied to CPU, made contiguous, and
encoded into Circle buffers. The serializer tracks tensor identity using device,
storage pointer, storage offset, shape, stride, dtype, layout, and qparam identity.
This allows genuinely shared storage, such as tied embedding and LM-head weights, to
reuse one Circle tensor without deduplicating unrelated cloned tensors that merely have
equal values.

Empty tensors and tensors without a stable nonzero data pointer are intentionally not
shared through this mechanism.

### 8.4 Quantized tensor encoding

When a node carries TICO qparam metadata, the Circle tensor dtype and quantization
record are derived from that metadata. The serializer supports regular integer types as
well as project-specific quantized string dtypes used by current workflows, including
`uint4`, `mxint8`, and `mxfp4`. Packed `uint4` data is encoded before buffer insertion.

## 9. Shapes, interfaces, and runtime binding

### Static and symbolic shapes

A static PyTorch shape becomes a Circle `shape` with no `shapeSignature`.

When a dimension is a `torch.SymInt`:

- Circle `shape` stores `1` as a concrete placeholder.
- Circle `shapeSignature` stores `-1` for that dimension.
- Static dimensions are repeated in both vectors.

Shape validation requires equal ranks and requires every dynamic `-1` signature entry
to correspond to placeholder value `1`.

### Model input binding

`ModelInputSpec` reads the one-subgraph Circle interface and binds user arguments in
serialized input order. It:

- flattens nested list/tuple positional values
- skips `None`
- flattens nested keyword values into generated names
- understands Hugging Face `DynamicCache` when Transformers provides it
- converts supported scalar values to tensors
- checks input count, dtype, rank, and static dimensions
- permits dimensions marked `-1` in the shape signature

### `CircleModel`

`CircleModel` owns raw bytes and provides:

```python
CircleModel.save(path)
CircleModel.load(path)
CircleModel(*args, **kwargs)
```

The built-in inference path currently asserts one subgraph. It returns one NumPy array
for one output and a list for multiple outputs.

The end-to-end test harness can instead execute through `onert`. For dynamic-shape
models it updates the runtime tensor information from concrete input shapes before
inference.

## 10. Quantization integration

Quantization has two related but distinct layers.

### 10.1 Public model quantization API

`tico.quantization` exports:

```python
from tico.quantization import prepare, convert, QuantStub
```

The lifecycle is:

```text
prepare(model, quant_config, args, kwargs)
    -> calibration or algorithm statistics collection
    -> convert(prepared_model)
```

`prepare()` chooses a quantizer through the quantizer registry and stores it on the
prepared model. `convert()` retrieves that quantizer; it does not accept the
configuration again. GPTQ currently requires in-place conversion because deep copying
would break its calibration catcher restoration.

### 10.2 Conversion-time quantized-graph legalization

After a model has been quantized or instrumented to contain supported quantization
operations, the core Circle conversion path detects those operations and performs
qparam folding, propagation, bias quantization, safe constant propagation, placeholder
cleanup, and dtype-bridge insertion.

These graph passes do not perform calibration and do not select a quantization
algorithm.

### 10.3 Recipe architecture

The recipe layer keeps end-to-end model workflows separate by responsibility:

- model-family behavior: `recipes/adapters/`
- algorithm stages: `recipes/stages/`
- calibration data: `recipes/data/`
- evaluation: `recipes/evaluation/`
- artifact export: `recipes/export/`
- debugging and parity: `recipes/debug/`
- user-selectable workflows: YAML presets under `examples/configs/`

See the [Quantization Recipes Developer Guide](../tico/quantization/recipes/README.md)
for the current package contracts.

## 11. Circle artifact layer

`tico.circle` operates on serialized Circle bytes and is intentionally independent from
the exported-graph pass pipeline.

```text
Circle bytes
    -> CircleDocument
        ├── inspect summaries
        ├── static verification
        ├── operator/tensor-boundary extraction
        └── CirclePassPipeline
              ├── compatibility   optional legacy custom-op recovery
              ├── legalize        optional dynamic-FC lowering
              ├── optimize        round-based generic and pattern rewrites
              │     ├── canonicalize
              │     ├── simplify
              │     ├── fold
              │     ├── fuse
              │     ├── CSE
              │     └── DCE
              └── compact         one-shot index compaction
```

Circle transformations are grouped by semantic responsibility under
`tico/circle/passes/optimization/`: `canonicalize`, `simplify`, `fold`, `fuse`,
`legalize`, and `compatibility`. Global CSE remains separate, while dead-code
elimination and index compaction live under `passes/cleanup/`.

The built-in O1 pipeline uses `CirclePassStrategy.UNTIL_NO_CHANGE` for the optimize
phase: all optimization passes complete one ordered round, and another round starts
only if at least one pass changed the document. Legacy custom-op recovery and dynamic
FullyConnected legalization create optional one-shot phases before optimization.
The compatibility-owned FC-GELU-FC recognizer is instead scheduled inside the
optimization phase at its pattern-sensitive position. `CompactIndicesPass` runs
exactly once in the final `compact` phase. The legacy `RESTART` scheduler remains available
for explicit custom pipelines and scheduler-comparison benchmarks.

A `CirclePassContext` owns a model-scoped optimization session. The session caches
producer/consumer analyses by subgraph revision and shares a canonical constant pool.
Local rewrite rules run through a deterministic neighborhood worklist and an atomic
mutation transaction, so failed rewrites roll back appended objects and watched
state. Direct Object API passes remain supported, but must report modifications
accurately so cached analyses are invalidated.

The artifact verifier checks structural and referential consistency, including
containers, indices, dataflow, tensor interfaces, signatures, and control-flow
subgraph references. Dead-code elimination preserves signature-bound and caller-owned
inputs and treats stateful, non-deterministic, custom, variable, and
subgraph-referencing operators as observable roots.

Verification does not execute inference or validate numerical parity or target-backend
support. Those concerns belong to runtime tests and backend compilation tests.

## 12. Validation, errors, and diagnostics

### Unsupported target validation

Before serialization, all remaining function targets are compared with the visitor
registry. TICO logs each unsupported operator and its source stack trace when present,
then raises `NotYetSupportedError`.

### Training validation

`convert()` reports a fatal message for a module still in training mode. The final
graph check also rejects known training operators, currently dropout variants.

### Shape validation

The serializer validates Circle tensor shapes and shape signatures before packing.
The runtime input binder separately validates user input count, dtype, rank, and static
dimensions.

### Debugging

- `TICO_LOG=4` enables debug logs and instrumented graph/constant diffs.
- `TICO_GRAPH_DUMP=1` writes post-decomposition, post-legalization, and
  post-quantization FX graph PNGs under `.tico_tmp/session_<timestamp>/`.
- `tico-circle inspect` shows serialized tensor/operator interfaces.
- `tico-circle verify` checks static Circle consistency.

## 13. Extension points

### Add a new exported-graph pass

1. Implement `PassBase` in `tico/passes/`.
2. Define precise matching preconditions and return `PassResult.modified` correctly.
3. Preserve graph signature and metadata invariants.
4. Add the pass explicitly to the correct position in `tico/utils/convert.py`.
5. Add focused unit tests, including non-matching cases.
6. Add an end-to-end module test when the serialized graph or numerical behavior is
   affected.

### Add a new Circle operator lowering

1. Add a `NodeVisitor` under `tico/serialize/operators/`.
2. Register every supported ATen overload.
3. Ensure the module is imported by the operator package.
4. Encode inputs, outputs, attributes, and opcode data through existing graph helpers.
5. Add visitor/serializer unit tests and a PyTorch-to-Circle parity test.

### Add a quantizer or quantization workflow

1. Define an algorithm configuration derived from the quantization `BaseConfig`.
2. Implement and register the `BaseQuantizer`.
3. Preserve the public `prepare`/statistics/`convert` lifecycle.
4. Put model-specific behavior in recipe adapters and algorithm behavior in stages.
5. Add deterministic unit coverage and the smallest useful recipe smoke test.

### Add a Circle-to-Circle pass

1. Put the transformation in the semantic package that owns its responsibility:
   `canonicalize`, `simplify`, `fold`, `fuse`, `legalize`, `compatibility`, or
   `cleanup`.
2. Use `CircleRewriteRule` and the shared mutation/session infrastructure for local
   rewrites; reserve a direct `CirclePass` for graph-global algorithms.
3. Preserve tensor contracts, signatures, subgraph references, shared buffers, and
   observable effects.
4. Register a canonical CLI name in `tico/circle/cli/main.py` when the pass is
   user-selectable, and add it to `tico/circle/passes/presets.py` only when a built-in
   preset should schedule it.
5. Add focused positive and close non-matching tests, plus multi-subgraph or
   global-resource coverage when applicable.

See [`tico/circle/README.md`](../tico/circle/README.md#writing-a-new-circle-pass) for
the detailed rewrite and testing contract.

## 14. Current limitations

- Core serialization emits one subgraph.
- Built-in `CircleModel` inference also supports one subgraph.
- Only registered ATen overloads can reach serialization.
- Dynamic shape signatures are preserved, but dynamic execution depends on the chosen
  runtime; the test harness uses `onert`.
- The main exported-graph pass schedule is a fixed list in `tico/utils/convert.py`, not
  a plugin-discovered pipeline.
- Legalization and optimization passes are currently combined in one main bundle.
- `CompileConfigV1.eliminate_rank_round_trip` is declared but not wired to the pass;
  the pass is currently always enabled.
- Unknown YAML configuration keys are ignored by the current dataclass loader.
- Successful Circle serialization does not imply compatibility with a particular NPU
  compiler.
- The default runtime and tests do not cover every dtype supported by specialized
  quantized serialization.

## 15. Implementation source map

Use these files as the source of truth when updating this document:

| Topic | Source |
|---|---|
| Public exports and minimum Torch warning | `tico/__init__.py` |
| Conversion APIs and pass order | `tico/utils/convert.py` |
| Pass interface and scheduling | `tico/utils/passes.py` |
| Compile configuration | `tico/config/base.py`, `tico/config/v1.py`, `tico/config/factory.py` |
| Circle construction | `tico/serialize/circle_serializer.py` |
| Tensor/buffer representation | `tico/serialize/circle_graph.py` |
| Shape and dtype mapping | `tico/serialize/circle_mapping.py` |
| Operator registry | `tico/serialize/operators/node_visitor.py` |
| Input binding and dynamic signatures | `tico/utils/signature.py` |
| Built-in runtime | `tico/utils/model.py`, `tico/interpreter/` |
| Quantization public API | `tico/quantization/public_interface.py` |
| Circle artifact APIs and pass taxonomy | `tico/circle/README.md`, `tico/circle/passes/optimization/` |
| Circle optimization presets and scheduling | `tico/circle/passes/presets.py`, `tico/circle/passes/manager.py` |
| Circle CLI pass registry | `tico/circle/cli/main.py` |
| Source tooling and CI | `infra/`, `.github/workflows/check-pr.yaml` |
