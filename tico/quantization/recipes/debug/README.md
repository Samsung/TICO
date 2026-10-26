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
descendant modules. Prefixes match dot-separated path components, so
`model.layers.1` does not select `model.layers.10`. When the option is omitted
or an empty prefix is supplied, all named modules are traced.

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

## Static runtime token helpers

`token_mask.py` owns the shared padded-text mask, padding-layout, position-ID,
and last-token-index calculations used by the LLaMA and Qwen3-VL debug runtimes.
The runtime modules retain their existing private helper names as import aliases.

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

## Download-free CPU recipe validation

The CPU runner creates deterministic random HF models, uses the real adapters and
PTQ stage, and executes saved Circle files with the reference runtime. No model
weights or datasets are downloaded. Run it from the installed development tree:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m tico.quantization.recipes.debug.cpu_validation \
  --output ./out/cpu-validation-NEW
```

The default runs static/comparator regressions, real generation, cache coverage,
small in-memory evaluation, and actual Circle conversion/execution sequentially.
Use `--stage regression`, `generate`, `coverage`, `evaluation`, or `circle` to rerun
one stage in a new directory. Unit-stage reports include the actual test count;
zero tests or any skip cannot establish a pass. `--baseline-repo /path/to/start`
compares the same frozen checkpoint with that checkout's exporter. It does not
recalibrate the baseline. `--fixtures /path/to/prior/fixtures` reuses trusted local
checkpoints only when their checkpoint and effective-config hashes match their
recorded calibration provenance.

The bounded Llama and Qwen fixtures use two decoder layers, GQA with two query
heads and one KV head, native context 128, calibration samples of eight tokens,
export capacities 16/32 and eager prefill 48 plus four connected decode steps.
W4A16 means actual frozen four-bit linear weights and sixteen-bit activations;
embedding/LM-head weights use the standard explicitly recorded unsigned eight-bit policy.
Qwen includes a local RGB image, a real vision block, mRoPE and fixed external
mask/RoPE/KV stage inputs. Extra focused tests exercise head dimension 128 and a
non-square image grid. Gemma's small evaluation smoke explicitly disables PLE
and shared KV and does not establish those features' coverage.

The Circle stage includes Llama bundle/full-model export, Qwen static capacities
16 and 32, export/eager isolation, and an image-conditioned Qwen chain. That chain
writes actual Circle prefill delta-K/V into separate per-layer host buffers and
passes them into four one-token decodes. The final decode uses S-1 valid past
positions, masks padded slots and uses positions from the actual HF mRoPE host
calculation. Every saved stage's local Torch/Circle outputs, including delta-K/V,
are compared with `atol=1e-5`, `rtol=1e-4`. Strict numerical failures retain their
nonzero exit code; the runner never treats a matching old/new failure as strict
parity success. Detailed classification requires the recorded baseline evidence.

Each invocation preserves command lines, working directory, dependency versions,
resource limits, source digest, seed, effective configs, checkpoint hashes,
stdout/stderr and original child exit codes. Checkpoints, Circle files and tensor
archives remain under the caller's output directory and must not be committed.
An existing directory is refused. Missing dependencies, failed stages and skipped
checks remain visible in `summary.json`.

These checks cover software contracts and small numerical boundaries. Random-model
PPL and generated IDs are execution evidence, not pretrained quality. Large
pretrained accuracy, full 2048-token model capacity and NPU integer behavior or
performance remain outside this runner's scope.

For a separate calibration-change diagnostic, create the negative checkpoint with
an absolute script invocation from the starting checkout. This imports that
checkout normally and verifies the actual TICO import path. The saved FP source,
calibration tensors and effective config are shared; the resulting frozen
checkpoint remains separate from the correctly calibrated candidate:

```bash
cd /path/to/starting-checkout
PYTHONPATH=/path/to/starting-checkout OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python /path/to/candidate/tico/quantization/recipes/debug/cpu_validation.py \
  --stage _baseline_calibration --repo /path/to/starting-checkout \
  --fixtures /path/to/cpu-validation-NEW/fixtures \
  --output /path/to/calibration-before-NEW

cd /path/to/candidate
python -m tico.quantization.recipes.debug.cpu_validation \
  --stage _legacy_rejection \
  --checkpoint /path/to/calibration-before-NEW/qwen3_vl.missing-cache-coverage.pt \
  --output /path/to/legacy-rejection-NEW
```

The first diagnostic passes when it reproduces missing past-cache coverage in
that starting revision; its checkpoint is explicitly invalid for static decode.
The second passes when the candidate rejects each layer's decode export preflight
with calibration guidance and preserves all frozen weights/observers/qparams.
This is a calibration-change negative test, not numerical exporter equivalence
for an invalid old checkpoint. Neither diagnostic repairs or overwrites it.

A connected-chain strict failure can be diagnosed with the exact saved host calls,
using the same valid checkpoint and the starting checkout's saved Circle stages:

```bash
cd /path/to/starting-checkout
PYTHONPATH=/path/to/starting-checkout OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
python /path/to/candidate/tico/quantization/recipes/debug/cpu_validation.py \
  --stage _baseline_connected --repo /path/to/starting-checkout \
  --checkpoint /path/to/cpu-validation-NEW/fixtures/qwen3_vl.pt \
  --evidence /path/to/cpu-validation-NEW/qwen3_vl-16-connected \
  --artifacts /path/to/cpu-validation-NEW/qwen3_vl-16/baseline/before/artifacts \
  --output /path/to/connected-baseline-replay-NEW
```

Replay checks the checkpoint hash, executes the real baseline adapters and runtime,
and saves every output. A successful reproduction means all saved host inputs and
both Torch/Circle output arrays match the original run exactly. The separate
`strict_torch_circle_status` retains the numerical parity verdict; an identical
baseline failure does not become strict parity success. Repeat with capacity 32's
evidence and artifacts when that profile needs diagnosis.
