# Wrapper-based Quantization (WrapQ)

## 1. Design Philosophy

**WrapQ** is a module for wrapper-based post-training quantization. The module is designed to provide 
 a **lightweight, composable, and algorithm-agnostic quantization framework** that sits between model
 definition and export. Rather than enforcing a monolithic quantization flow, it focuses on
**fine-grained control**, **explicit calibration**, and **pluggable wrappers** that allow different model 
families (e.g., `nn`, `LLaMA`) to be quantized with consistent semantics.

This design assumes the user understands their model’s computational structure and is willing to decide:

- _What to quantize_ (module or tensor granularity)
- _How to calibrate_ (data, order, and statistics)
- _When to finalize_ (convert/export timing)

The guiding principles are:

1. **Transparency** — Quantization logic is explicit and inspectable.
2. **Orthogonality** — Quantization config, calibration, and algorithm are independent layers.
3. **Composability** — Wrappers can be reused or mixed with different algorithms (e.g., GPTQ).
4. **Determinism** — Once calibrated, all quantization parameters are fixed and reproducible.

---

## 2. Quantization Flow

A typical workflow consists of three conceptual stages:

1. **Prepare**

Modules are replaced with quantized wrappers that simulate quantization behavior and collect calibration statistics.

2. **Calibrate**

Run representative data through the model to estimate dynamic ranges for activations.

`EMAObserver` owns independent min/max buffers, including for one-dimensional
per-channel inputs and parameters. Mutating an input after collection does not
change saved statistics; resetting the observer does not modify the observed tensor.

3. **Convert**

The common lifecycle coordinator enumerates active sites once through arbitrary
module containers and aliases. Observers derive validated candidates without
mutation; model hooks check semantic constraints against the complete plan. Only
then does one transaction install the snapshots and switch wrappers to QUANT.
Failure preserves calibration state and the public retry handle.

Required affine sites need finite collected statistics or compatible imported
qparams. Finite zero ranges remain valid. Imported locks take precedence for every
affine subtype, including Percentile. Identity and MX declare stateless policies.
Optional cache/multimodal capabilities remain unavailable as a whole when their
sites were not calibrated; requesting one fails before use. Required activations
are never skipped merely because they were not observed.

Observer state schema version 1 saves exact grids, policy, provenance, lock,
bypass state and observed channel extent. Wrapper schema version 1 saves mode and
capability readiness. Frozen state dicts and whole-module checkpoints preserve
exact grids; incompatible/corrupt state rejects even with `strict=False`.
Range-only legacy state dicts load as not frozen and require recalibration.
Old whole-module objects lacking the schema require reprepare/recalibrate; Gemma's
old cache-only schema cannot establish whole-model readiness. Percentile frozen
execution needs no reservoir, but mid-calibration reservoir resumption is not
promised.

New wrappers implement `_all_observers`/`local_observer_groups`, optional subtree
capabilities, `_prepare_calibration` for local parameters, and pure
`_validate_freeze` semantic checks. They must not override recursive lifecycle
entrypoints or implement generic grid serialization. Legacy mutating overrides
reject with migration guidance. Exporters read `require_frozen()` snapshots;
profile-generated constants use separate local observers. They never recalibrate
source-model sites.

The [frozen-state report](../../../docs/archive/wrapq-frozen-state-design-investigation-20261027.md)
records the ownership rationale, migration decisions and validation evidence.

Affine observers accept external grids through `load_qparams(scale, zp, lock=True)`.
Locked grids survive calibration and conversion without requiring local samples,
including for `PercentileObserver`. The affine base owns this dispatch; range-based
observers customize the pure `_statistics_bounds()` hook to select their calibration
interval only when unlocked. Candidate installation records clipping diagnostics
without changing the observed min/max. Loading with `lock=False` still requires
calibration and recomputes qparams from those statistics.

---

## 3. Wrapper Design

Wrappers define how quantization is applied to a specific submodule family.

Each wrapper encapsulates:

- **The module-specific quantization points** (e.g., weights, activations, residuals).
- The **observation** process (what to record during calibration).
- The **conversion** logic (how to replace or re-instantiate with quantized parameters).

All wrappers inherit from `QuantModuleBase`, which allows different model types to share the 
same calibration and conversion routines while maintaining per-family specialization.

Each wrapper is registered automatically and discovered dynamically through the PTQ registry, 
so extending PTQ for a new model family simply requires subclassing and registration.

For unregistered module types, `strict_wrap` controls whether unsupported wrapping
boundaries raise an error or allow the existing recursive traversal to continue.
Once a registered wrapper is selected, its constructor errors propagate unchanged
in either mode. A wrapper's `NotImplementedError` for an unsupported model variant
therefore fails preparation immediately, including with `strict_wrap=False`.

---

## 4. Simple API Usage

WrapQ provides a unified entry point for quantizing any model:

```python
from tico.quantization import prepare, convert
from tico.quantization.config import PTQConfig

# Step 1: Prepare the model
model = prepare(model, PTQConfig())

# Step 2: Calibrate
for batch in calibration_dataloader:
    model(batch)

# Step 3: Convert to quantized form
quantized_model = convert(model)
```

This minimal interface abstracts away the details of module wrapping, registry lookup, 
 and quantizer instantiation. Advanced users can override the configuration to control 
dtype, scheme, or per-module granularity.

**Ready-to-run examples** for PTQ and wrapper usage can be found in the `examples/` folder,
including quantization flows for nn.Linear, LLaMA MLPs, and full decoder layers.

For output debugging, `utils.introspection.compare_outputs(..., full_tensor_diff=True)`
returns signed float32 tensor differences at every depth, including tensors in lists,
tuples, dictionaries, and object fields. The default returns tensor difference
statistics. `compare_side_by_side()` uses full differences for `interesting_modules`.

LLaMA attention, decoder-layer, and model wrappers interpret boolean and integer
(`uint8`, `int8`, `int16`, `int32`, `int64`) attention masks as keep masks: zero
masks a position and nonzero keeps it, subject to causality. Token masks of shape
`(B, K)` and query/key masks of shape `(B, Q, K)` or `(B, 1, Q, K)` share this
behavior. Floating-point masks remain additive biases; callers supply their
causal and padding constraints explicitly.

LLaMA attention keeps persistent KV-cache storage in the `new_k`/`new_v`
quantization domains for prefill outputs, decode inputs, and both delta and full
cache outputs. Attention converts past and current KV tensors independently to
the `present_key`/`present_value` compute domains before concatenation. Current KV
producers branch directly to storage and compute, without routing compute through
storage first. There are no `past_*` observers or independent read grids. Prefill
and decode share the storage ABI, so cache deltas can be appended and reused
without host requantization. Supplied HF `DynamicCache` and `StaticCache`
objects append/update S deltas in both cache return modes: `delta` returns the
new tensors, while `present` returns the updated cache object. Reads use the
occupied prefix and never write compute-grid values into persistent storage.

Calibration must enable cache generation to observe `new_*` storage outputs.
`calibration.decode_steps` optionally collects decode activation statistics; zero
is supported and no decode is needed solely to determine past-cache qparams.
Only active functional observers participate in freezing. Old Llama full-module
recipe checkpoints and state dicts containing removed past-observer state are
rejected, including non-strict state-dict loads: reprepare the original float model and recalibrate with cache generation
enabled. Obsolete past statistics are never reused as storage or compute qparams.

---

## 5. Extensibility

WrapQ acts as the shared quantization backend for different algorithms.

Through shared base interfaces and quantizer abstractions, users can:

- Compose multiple algorithms for hybrid quantization.
- Reuse calibration artifacts between different methods.
- Introduce custom quantizers by extending a single registry entry.

This modular structure ensures that adding a new quantization algorithm or wrapper 
 never breaks existing flows.

---

## 6. Summary

WrapQ offers a balance between **explicit control** and **simple APIs**. It is meant for 
 developers who want deterministic, reproducible quantization without opaque automation.
By decoupling configuration, calibration, and algorithm logic, it allows seamless 
integration across diverse models and backends while maintaining transparency and 
reliability in quantized inference.
