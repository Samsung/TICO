# Debug and Inspect Developer Guide

Debug utilities live in `tico.quantization.recipes.debug` and are exposed through
`tico/quantization/examples/inspector.py`.

The purpose of this package is to avoid accumulating one-off debug scripts under
`examples/`.

## Current pattern

```bash
python -m tico.quantization.examples.inspector \
  --config tico/quantization/examples/configs/qwen3_vl_quantize.yaml \
  --mode trace \
  --interesting-modules model.language_model model.visual
```

`inspector.py` should stay thin. It should parse arguments, load a config, create a
`RecipeContext`, and dispatch to a reusable debug function.

## Existing trace mode

It performs this flow:

1. Load the recipe config.
2. Load the model through the registered adapter.
3. Build calibration inputs through the adapter.
4. Deep-copy the FP model.
5. Build the PTQ config from the recipe's `ptq` stage through the adapter.
6. Apply the stage's `specs` / `override_policies` / `raw_overrides` with the same
   `apply_ptq_override_policies()` helper that `PTQStage` uses, resolving model
   structure from the copied model through the selected adapter's
   `build_quant_target_context()` (see "PTQ selector override structure" in
   `../adapters/README.md`). Invalid overrides raise here, before `prepare()`.
7. Prepare and calibrate the copied model.
8. Call `convert()` only when `--enable-quantization` is set.
9. Run the first calibration sample through the FP model and collect module
   outputs with forward hooks.
10. Run the same sample through the PTQ/converted model.
11. Print per-module `mean|diff|` and `max|diff|`.

Trace mode reflects the selected PTQ stage's selector/raw overrides, so its
prepared copy uses the same effective PTQ policy as a regular `ptq` stage run.
It does not re-run the full recipe pipeline: GPTQ, SpinQuant, SmoothQuant, and
GPTQ-to-PTQ qparam reuse are not executed. `--enable-quantization` only decides
whether `convert()` runs after calibration.

Typical command:

```bash
python -m tico.quantization.examples.inspector \
  --config tico/quantization/examples/configs/qwen3_vl_quantize.yaml \
  --mode trace \
  --interesting-modules model.language_model model.visual \
  --set calibration.n_samples=1
```

Use trace mode without `--enable-quantization` when debugging wrapper parity.
Use it with `--enable-quantization` when debugging fake-quantization error.

`--interesting-modules` accepts exact module names and parent module prefixes.
For example, `--interesting-modules model.visual` traces `model.visual` and its
descendant modules. When the option is omitted, all named modules are traced.

Trace mode intentionally reads model, calibration, and PTQ behavior from the
recipe config instead of hard-coding model-specific setup in the debug utility.
For Qwen3-VL, use fields such as:

```yaml
calibration:
  n_samples: 1
  seq_len: 128

pipeline:
  - name: ptq
    activation: int16
    linear_weight: uint8
```

## Multi-batch side-by-side comparison

`tico.quantization.recipes.debug.trace` also owns `trace_parity()` and
`merge_diff_stats()` for comparing two models that are already prepared:

```python
from tico.quantization.recipes.debug.trace import merge_diff_stats, trace_parity

trace_parity(fp_model=fp_model, q_model=q_model, ctx=fp_ctx)
```

How it differs from the inspector's trace mode:

- `trace_ptq_parity()` copies the FP model, prepares and calibrates the copy with
  the recipe's PTQ policy, and compares the first calibration sample only.
- `trace_parity()` does not load, quantize, or export anything. It takes two
  prepared models and iterates over every input in `ctx.calibration_inputs`,
  running the FP model and then the quantized model on each batch.

The printed `mean|diff|`, `max|diff|`, and `peir` values are aggregates of the
per-batch summaries: `min`/`max` are the extremes across batches, and `mean`,
`peir`, and squared `stddev` are running averages that give every batch equal
weight regardless of its element count. They are not pooled statistics over all
elements.

The existing CLI is unchanged. `examples/compare_side_by_side.py` still owns
config loading, extension loading, adapter selection, model preparation, and
calibration-input preparation, then calls the debug-owned `trace_parity()`:

```bash
python -m tico.quantization.examples.compare_side_by_side \
  --config tico/quantization/examples/configs/llama_universal_gptq_quantize.yaml \
  --set "pipeline.1.gptq.weight_bits=8"
```

## When to add a debug mode

Add a debug mode when the workflow is for developers rather than normal users:

- FP vs PTQ or FP vs converted output comparison
- module-level tensor tracing
- wrapper parity checks
- export input inspection
- static runtime path verification
- shape or mask debugging
- synthetic input smoke checks

Do not add public scripts such as:

```text
trace_qwen.py
quantize_vision_attention.py
quantize_decoder_layer_prefill.py
```

Instead add:

```text
recipes/debug/<mode>.py
```

and expose it through:

```text
examples/inspector.py --mode <mode>
```

## Step-by-step: adding a new inspect mode

Assume the new mode is `layer_parity`.

### 1. Add a debug module

```text
tico/quantization/recipes/debug/layer_parity.py
```

Skeleton:

```python
from __future__ import annotations

from tico.quantization.recipes.context import RecipeContext


def run_layer_parity(ctx: RecipeContext, *, module_name: str) -> None:
    # Build or locate the target module.
    # Run FP and quantized paths.
    # Print compact numeric summaries.
    pass
```

### 2. Add an inspect mode

Edit `examples/inspector.py`:

```python
from tico.quantization.recipes.debug.layer_parity import run_layer_parity

parser.add_argument("--mode", choices=["trace", "layer_parity"], default="trace")
parser.add_argument("--module-name", default=None)

...

if args.mode == "layer_parity":
    run_layer_parity(ctx, module_name=args.module_name)
```

### 3. Add config if needed

If the debug mode needs unusual model args or calibration shapes, add a config
preset under `examples/configs/`:

```text
qwen3_vl_layer_parity_debug.yaml
```

### 4. Add a smoke test

Debug modes should have a small test that verifies imports, argument parsing, and
basic execution with tiny inputs or mocks.

## Output conventions

Debug modes should print compact summaries instead of dumping large tensors.

Recommended tensor summary:

```text
Tensor(shape=(1, 128, 4096), dtype=torch.float32, mean=..., min=..., max=..., std=...)
```

Recommended parity summary:

```text
module.name: mean|diff|=0.00012345, max|diff|=0.01234567
```

If a mode writes artifacts, put them under `export.output_dir` or an explicit
`--output-dir` flag. Do not write into the repository tree by default.

## Breakpoints and interactive debugging

Do not leave unconditional breakpoints in committed debug utilities.

Acceptable:

```python
if enable_breakpoint:
    breakpoint()
```

Not acceptable:

```python
breakpoint()
```

## Debug checklist

Before merging a new debug utility:

- It lives under `recipes/debug/`.
- It is exposed through `examples/inspector.py` if it is useful to other
  developers.
- It does not require editing source code to choose modules or breakpoints.
- It prints compact summaries, not full large tensors.
- It respects config-provided device, dtype, and calibration settings.
- It does not import from another example script.
- It has at least a small smoke test or documented manual command.
