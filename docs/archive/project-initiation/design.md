# System Design Document

> Torch IR → Circle conversion & optimization library with pluggable passes and automated validation

## Table of Contents

- [0. Decision Log (Design Summary)](#0-decision-log-design-summary)
- [1. Introduction](#1-introduction)
  - [1.1 Purpose](#11-purpose)
  - [1.2 Scope](#12-scope)
  - [1.3 Non-Goals](#13-non-goals)
- [2. Requirements Traceability](#2-requirements-traceability)
- [3. Software Structure Design](#3-software-structure-design)
  - [3.1 High-Level Architecture](#31-high-level-architecture)
  - [3.2 Core Modules](#32-core-modules)
  - [3.3 Configuration](#33-configuration)
  - [3.4 Invariants](#34-invariants)
- [4. Behavior Design](#4-behavior-design)
  - [4.1 End-to-End Conversion Flow](#41-end-to-end-conversion-flow)
  - [4.2 Pass Scheduling & Priority](#42-pass-scheduling--priority)
  - [4.3 Dynamic Shapes Behavior](#43-dynamic-shapes-behavior)
  - [4.4 Error Handling & Logging](#44-error-handling--logging)
- [5. Data & Interface Design](#5-data--interface-design)
  - [5.1 Public APIs](#51-public-apis)
  - [5.2 Pass Interface](#52-pass-interface-conceptual)
  - [5.3 Configuration Schema](#53-configuration-schema-excerpt)
- [6. Performance Design](#6-performance-design)
- [7. Reliability & Accuracy](#7-reliability--accuracy)
- [8. Security & Compliance](#8-security--compliance)

## 0. Decision Log (Design Summary)

- **IR Strategy**: Use **PyTorch ExportedProgram (core ATen)** as the working IR. No separate custom mid-IR.
- **Pass Architecture**: Separate **legalization** and **optimization**, but allow interleaving under a **Pass Manager** with scheduling & invariants.
- **Quantization**: Post-Training Quantization (PTQ) is **detected and applied conditionally** (fold/propagate qparams, bias quantization, dtype mismatch fix).
- **Validation**: Multi-stage validation (unsupported ops / training ops / invariant checks) + E2E inference parity check against PyTorch outputs.
- **Testing**: Container-friendly, **model-by-model isolated** runtime, selectable circle runtimes (`circle-interpreter` / `onert`), golden tests and tolerance.
- **Extensibility**: Plugin-style registration for passes, quantizers, runtimes, and config flags; clear contract for pass I/O and invariants.


## 1. Introduction

### 1.1 Purpose

Define the software structure and behavior for TICO, a Python library that converts PyTorch modules to Circle while applying optional optimizations and quantization, and verifying accuracy and NPU-compatibility.

### 1.2 Scope

- **Structure Design**: layers, modules, pass interfaces, configuration, extensibility.
- **Behavior Design**: end-to-end pipelines, pass scheduling, error handling, validation, and test flows.

### 1.3 Non-Goals

- Building an in-house mid-IR other than PyTorch `ExportedProgram` (EP).
- Implementing a training-time quantization or compiler backend; TICO targets **export + legalization + validation**.


## 2. Requirements Traceability

| Design Area | Requirement IDs |
|---|---|
| Direct conversion to Circle | RF-1 |
| Optimization pipeline | RF-2, RNF-1, RNF-2 |
| Quantization integration support | RF-3 |
| Debug/verification tools | RF-4, RNF-3 |
| NPU compatibility tests | RF-5 |
| Performance goals | RNF-1, RNF-2 |
| Maintainability/extensibility | RNF-4, RNF-5, RNF-6 |
| Operating env/compliance | RNF-7, RNF-8 |


## 3. Software Structure Design

### 3.1 High-Level Architecture

```
User Code
│
├── tico.convert() / convert_from_pt2() / convert_from_exported_program()
│
├── Export (torch.export.export)  ──▶  ExportedProgram (core ATen)
│
├── Pass Manager (Pipeline)
│     ├─ Pre-Edge Passes          (decompositions)
│     ├─ Legalize/Optimize Passes (EP-invariant preserved)
│     ├─ Legalize/Optimize Passes (EP-invariant relaxed)
│     └─ Quantization Passes      (conditional)
│
├── Validation & Checks
│     ├─ Unsupported Target Check
│     ├─ Training Op Check
│     └─ Invariant Check (selective)
│
└── Circle Builder ──▶ CircleModel (bytes + helpers)
└─ Optional: runtime inference (circle-interpreter / onert)
```

### 3.2 Core Modules

- **API Layer**
  - `convert(mod, args, kwargs, dynamic_shapes, strict, config) -> CircleModel`
  - `convert_from_exported_program(exported_program, config) -> CircleModel`
  - `convert_from_pt2(pt2_path, config) -> CircleModel`

- **Export & IR Layer**
  - Uses **PyTorch `ExportedProgram`** (core ATen).  
  - *Invariant A (EP-invariant)*: before certain legalization, constants are **lifted** (no inline constant tensors as non-placeholders).
  - After some legalization, EP-invariant can be **temporarily relaxed** (inline constants allowed) to unlock optimizations.

- **Pass System**
  - `PassManager(passes=[...])` executes passes in order.
  - Pass types: **Decomposition**, **Legalization**, **Optimization**, **Quantization**, **ConstProp**, **Shape/Type Cast**, **Fusion**, **Dead-Node Elimination**, etc.
  - Examples:
    - Decompose: `DecomposeFakeQuantize*`, `DecomposeBatchNorm/GroupNorm/GroupedConv2d`, `DecomposeSliceScatter`, `DecomposeAddmm`
    - Canonicalization/Relayout: `ConvertLayoutOpToReshape`, `ConvertRepeatToExpandCopy`, `LowerToResizeNearestNeighbor`, `ConvertConv1dToConv2d`
    - Cleanups/Fusions: `RemoveNop`, `RemoveRedundant*`, `FuseLeadingUnsqueezeReshape`, `MergeConsecutiveCat`, `FuseRedundantReshapeToMean`
    - Math: `LowerPow2ToMul`, `Cast*MixedTypeArgs`, `CastATenWhereArgType`, `ConstPropPass`, `SegmentIndexSelectConst`
    - Matmul→Linear: `ConvertMatmulToLinear(...)` with config toggles
    - Mask legalization: `LegalizeCausalMaskValue(enabled=...)`
    - Slicing lowering: `*LowerToSlicePasses()`
  - Quantization: `FoldQuantOps`, `RemoveWeightDequantOp`, `PropagateQParamForward/Backward`, `QuantizeBias`, `InsertQuantizeOnDtypeMismatch`

- **Circle Builder**
  - `build_circle(exported_program, config) -> bytes`, then `CircleModel(bytes)`.

- **Validation & Checks**
  - `check_unsupported_target`, `check_training_ops`
  - Final parity validation in tests: `validate_result(torch_result, circle_result, **tolerance)`

- **Testing Framework**
  - `NNModuleTest(TestRunnerBase)` orchestrates:
    - optional **PT2 file stage** vs direct EP path
    - dynamic shape validations
    - runtime selection (`circle-interpreter`, `onert`)
    - golden comparison & tolerances
  - `NormalTestDictBuilder` auto-discovers `TestModuleBase` subclasses and builds test dicts.

### 3.3 Configuration

- **Compile Config (example toggles)**
  - `legalize_causal_mask_value: bool`
  - `convert_expand_to_slice_cat: bool`
  - `convert_*_mm_to_fc: bool` (LHS/RHS const, single-batch bmm)

- **Principles**
  - All pass behaviors affecting numerics must be **explicitly toggleable**.
  - Default config aims for **max compatibility** (legalization on) with conservative fusions.

### 3.4 Invariants

- **EP-Invariant (pre-legalization)**:  
  - No inline tensor constants; shapes/dtypes consistent with ATen canonical forms.

- **Relaxed-Invariant (post-partial-legalization)**:  
  - Inline constants allowed for optimization passes that require them.

- **Circle-Legal Invariant**:
  - All ops and attributes must be **representable** in Circle; layouts/dtypes legal; masks legalized (e.g., `-inf → -120` when enabled).


## 4. Behavior Design

### 4.1 End-to-End Conversion Flow

1. **Pre-flight**
   - Ensure `model.eval()` (fatal if training mode).
   - Export: `torch.export.export` → `ExportedProgram` (strict by default).
   - Log & debug print of EP (debug level).

2. **Pre-Edge Decompositions**
   - Run `DecomposeFakeQuantize*`
   - `traced_run_decompositions(exported_program)` under warning suppression (Intel GPU TF32, composite autograd note).

3. **Legalization/Optimization (EP-invariant preserved)**
   - Large pass bundle (`FillMetaVal`, `ExtractDtypeKwargs`, `RemoveNop`, casts, const-prop, fusions, etc.).
   - Config-gated steps (e.g., mask legalization, expand→slice+cat, matmul→linear toggles).

4. **Legalization/Optimization (EP-invariant relaxed)**
   - Tiny pass bundle to **re-cast mixed types** to unblock later steps.

5. **Quantization (conditional)**
   - If quantization ops are present, run **qparam folding/propagation**, **bias quantization**, **dtype mismatch inserts**.

6. **Validation & Circle Build**
   - `check_unsupported_target(exported_program)`, `check_training_ops(exported_program)`
   - `build_circle(...)` → `bytes` → `CircleModel`

7. **(Optional) Runtime Inference**
   - Test harness loads the circle and runs with `circle-interpreter` or `onert`.

### 4.2 Pass Scheduling & Priority

- **Contract**
  - Each pass declares: *reads/writes* (graph/attributes), *preconditions* (invariants), *postconditions*.

- **Ordering Rules**
  - **Decomposition → Canonicalization → Legalization → Optimization → Quantization → Final Legalization (if needed)**

- **Conflict Avoidance**
  - After each major bundle, run **lightweight invariant check** and **shape/dtype sanity**.

- **Failure Policy**
  - On invariant breach: raise **diagnostic exception** with pass name and node spans.

### 4.3 Dynamic Shapes Behavior

- If `dynamic_shapes` is supplied at export:
  - Post-conversion verify **symbolic** dims exist in Circle input spec (`-1` sentinel).
  - If absent → error with guidance ("Dynamic shapes expected but not applied").

### 4.4 Error Handling & Logging

- **Fatal**: training mode, missing dynamic shapes where promised, unsupported target op.
- **Warnings**: backend decomposition warnings (suppressed where noisy & known-benign).
- **Debug**: EP dump pre/post major phases; pass-by-pass timing (optional flag).


## 5. Data & Interface Design

### 5.1 Public APIs

```python
tico.convert(mod: nn.Module, args: Tuple[Any, ...], kwargs: Dict|None = None,
             dynamic_shapes: dict|None = None, strict: bool = True,
             config: CompileConfigBase = get_default_config()) -> CircleModel

tico.convert_from_exported_program(ep: ExportedProgram,
             config: CompileConfigBase = get_default_config()) -> CircleModel

tico.convert_from_pt2(pt2_path: str|Path,
             config: CompileConfigBase = get_default_config()) -> CircleModel
```

### 5.2 Pass Interface

```python
class GraphPass(Protocol):
    name: str
    def run(self, ep: ExportedProgram, ctx: PassContext) -> ExportedProgram: ...
```

- PassContext: access to config, logger, stats (timing), invariant flags.

### 5.3 Configuration Schema

```yaml
version: "1.0"
legalize_causal_mask_value: True|False
convert_expand_to_slice_cat: True|False
convert_lhs_const_mm_to_fc: True|False
convert_rhs_const_mm_to_fc: True|False
convert_single_batch_lhs_const_bmm_to_fc: True|False
convert_expand_to_slice_cat: True|False
```


## 6. Performance Design

- Graph Size & File Size
  - Run const-prop, dead-node elimination, reshape/permute cleanups, fusion to minimize nodes/attrs.
  - Strip non-essential metadata (debug-level toggle to keep).

- Throughput & Latency
  - Optimize pass ordering and minimize redundant traversals to reduce end-to-end conversion time.  
  - Ensure predictable and stable latency across different model sizes by controlling pass complexity.  

- Hotspots
  - Matmul→Linear conversion with const sides; expand→slice+cat lowering for NPU-friendly memory patterns.


## 7. Reliability & Accuracy

- Parity
  - Compare PyTorch vs Circle outputs with per-test rtol/atol.

- Golden Tests
  - Optional golden outputs for numerically sensitive models.

- Mask Legalization Trade-off
  - -inf → -120 improves quantization stability (document expected tiny accuracy delta; tie to config).


## 8. Security & Compliance

- Respect Circle schema and NPU compiler license constraints.
- Avoid bundling model weights in logs or crash dumps.
- Follow internal OSS (Open Source Software) policy for third-party code.
