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

Replace wrapped modules with their quantized counterparts using fixed quantization parameters.

This approach minimizes model perturbation and avoids retraining.

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
