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
3. **Composability** — Wrappers can be reused or mixed with different algorithms (e.g., PT2E, GPTQ).
4. **Determinism** — Once calibrated, all quantization parameters are fixed and reproducible.

---

## 2. Quantization Flow

A typical workflow consists of three conceptual stages:

1. **Prepare**

Modules are replaced with quantized wrappers that simulate quantization behavior and collect calibration statistics.

2. **Calibrate**

Run representative data through the model to estimate dynamic ranges for activations.

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