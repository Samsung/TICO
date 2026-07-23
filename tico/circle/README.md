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
            ├── DeadCodeEliminationPass
            └── CompactIndicesPass
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

### Run cleanup passes

```python
from tico.circle.passes import CirclePassManager
from tico.circle.passes.cleanup import (
    CompactIndicesPass,
    DeadCodeEliminationPass,
)

pipeline = CirclePassManager(
    [
        DeadCodeEliminationPass(),
        CompactIndicesPass(),
    ]
)
result = pipeline.run(model)
print(result.changes)
model.save("model.cleaned.circle")
```

By default, `CirclePassManager` verifies the document after every pass. 
Set `CirclePassContext(verify_after_each_pass=False)` only when a multi-step 
transformation intentionally has a temporary invalid state and performs 
explicit verification at the end.

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
  --passes dce,compact \
  -o model.cleaned.circle
```

Available first-stage passes:

| Name | Implementation | Behavior |
|---|---|---|
| `dce` | `DeadCodeEliminationPass` | Removes operators that cannot contribute to graph outputs and prunes unused graph inputs |
| `compact` | `CompactIndicesPass` | Removes unused tensors, buffers, and operator codes and remaps all supported references |

### Standard input and output

Use `-` for a binary stream:

```bash
tico-circle extract model.circle --ops 0-100 -o - \
  | tico-circle optimize - --passes dce,compact -o output.circle
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

- mutate only the supplied `CircleDocument`
- report `modified=True` only when model state changed
- preserve Circle structural invariants at pass boundaries unless verification is explicitly disabled by the caller
- use `CircleGraph` instead of rebuilding producer and consumer maps independently
- use the helpers in `rewrite.py` whenever deleting indexed objects
- add unit tests that include multi-subgraph and shared-buffer cases when the pass touches global state

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

## Current limitations

This first implementation performs structural Circle transformations. It does not perform numerical equivalence
 testing, runtime execution, shape inference, or target-specific operator legalization.

Additional limitations:

- A constant is recognized by inline buffer data or a non-zero external buffer offset/size. A zero-sized constant
 with no payload metadata may be conservatively promoted to an extracted graph input.
- Tensor name selectors rely on names being present and stable. Operator indices remain useful for debugging
 but may change after any rewrite.
- Signature synthesis is intentionally not attempted when extraction creates new boundaries; only an exactly
 compatible source signature can be retained.
- Control-flow references are discovered from scalar `*SubgraphIndex` fields, vector `*SubgraphIndices` fields,
 and `CallOptions.subgraph`. A new schema option with a different naming convention must be added to the reference walker.
- Structural verification does not guarantee that a runtime accepts the model or that outputs are numerically equivalent.
