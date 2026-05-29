# Examples

This directory contains thin command-line examples for running quantization,
evaluation, and debugging workflows.

The examples are intentionally small. Real implementation code lives in
`tico.quantization.recipes`.

## Directory layout

```text
tico/quantization/examples/
├── README.md
├── quantize.py          # Run a config-driven quantization pipeline
├── evaluate.py          # Evaluate an FP model or saved checkpoint
├── inspect.py           # Run trace/parity/debug tools
└── configs/             # Reusable recipe presets
```

## Supported top-level actions

### Which command should I use?

Use `quantize.py` when you want to run a quantization recipe. It loads the
model, builds calibration inputs, runs enabled `pipeline` stages such as GPTQ,
PTQ, SpinQuant, or CLE, and then optionally evaluates and exports the resulting
model.

Use `evaluate.py` when you only want to evaluate a floating-point model or an
already saved checkpoint. `evaluate.py` does **not** run `pipeline` stages from
the config. If the config contains enabled stages such as `gptq` or `ptq`, they
are ignored by `evaluate.py`.

Use `inspect.py` for debug-oriented workflows such as trace, parity, runtime
inspection, and wrapper-level smoke checks.

Summary:

| Command | Runs `pipeline` stages? | Builds calibration inputs? | Evaluates? | Exports? |
|---|---:|---:|---:|---:|
| `quantize.py` | Yes | Yes | If `evaluation.enabled=true` | If `export.enabled=true` |
| `evaluate.py` | No | No | Yes | No |
| `inspect.py` | Mode-dependent | Mode-dependent | Debug only | Debug only |

Common usage patterns:

```bash
# Run GPTQ/PTQ and evaluate the quantized model in the same process.
python -m tico.quantization.examples.quantize \
  --config tico/quantization/examples/configs/llama_gptq_ptq.yaml \
  --set evaluation.enabled=true \
  --set export.enabled=false
```

```bash
# Evaluate a saved checkpoint without running quantization again.
python -m tico.quantization.examples.evaluate \
  --config tico/quantization/examples/configs/llama_gptq_ptq.yaml \
  --checkpoint ./out/llama/quantized_model.pt
```

### Quantize

`quantize.py` is the command that executes the recipe pipeline. It runs the
enabled stages under `pipeline` in the order they appear in the config.

```bash
python -m tico.quantization.examples.quantize \
  --config tico/quantization/examples/configs/llama_gptq_ptq.yaml \
  --model Maykeye/TinyLLama-v0
```

```bash
python -m tico.quantization.examples.quantize \
  --config tico/quantization/examples/configs/qwen3_vl_gptq_ptq.yaml \
  --model Qwen/Qwen3-VL-2B-Instruct
```

```bash
python -m tico.quantization.examples.quantize \
  --config tico/quantization/examples/configs/qwen3_vl_eval_suite.yaml \
  --model Qwen/Qwen3-VL-2B-Instruct
```

Examples:

```bash
# Run quantization only. Skip evaluation and export.
python -m tico.quantization.examples.quantize \
  --config tico/quantization/examples/configs/llama_gptq_ptq.yaml \
  --set evaluation.enabled=false \
  --set export.enabled=false
```

```bash
# Run quantization and evaluate the quantized model. Skip export.
python -m tico.quantization.examples.quantize \
  --config tico/quantization/examples/configs/llama_gptq_ptq.yaml \
  --set evaluation.enabled=true \
  --set export.enabled=false
```

```bash
# Run quantization and export configured artifacts. Skip evaluation.
python -m tico.quantization.examples.quantize \
  --config tico/quantization/examples/configs/llama_gptq_ptq.yaml \
  --set evaluation.enabled=false \
  --set export.enabled=true \
  --output-dir ./out/llama
```

### Evaluate

`evaluate.py` evaluates the model loaded by the adapter, or the model loaded
from `--checkpoint` when one is provided. It does not prepare, calibrate,
convert, or export the model.

Running this command with a GPTQ/PTQ config evaluates the floating-point model,
because the `pipeline` section is not executed by `evaluate.py`:

```bash
python -m tico.quantization.examples.evaluate \
  --config tico/quantization/examples/configs/llama_gptq_ptq.yaml
```

To evaluate a quantized result, pass a saved checkpoint:

```bash
python -m tico.quantization.examples.evaluate \
  --config tico/quantization/examples/configs/llama_gptq_ptq.yaml \
  --checkpoint ./out/llama/quantized_model.pt
```

Task overrides are supported:

```bash
python -m tico.quantization.examples.evaluate \
  --config tico/quantization/examples/configs/llama_gptq_ptq.yaml \
  --checkpoint ./out/llama/quantized_model.pt \
  --tasks winogrande,arc_easy
```

### Inspect / debug

```bash
python -m tico.quantization.examples.inspect \
  --config tico/quantization/examples/configs/qwen3_vl_ptq_only.yaml \
  --mode trace \
  --interesting-modules model.language_model model.visual
```

```bash
python -m tico.quantization.examples.inspect \
  --config tico/quantization/examples/configs/static_llama_runtime.yaml \
  --mode static-llama-runtime
```

```bash
python -m tico.quantization.examples.inspect \
  --config tico/quantization/examples/configs/tied_embedding_smoke.yaml \
  --mode tied-embedding-smoke
```

#### Trace mode

Trace mode compares a floating-point model with a PTQ-prepared copy of the same
model. It is useful for finding the first module where PTQ wrapping,
calibration, or fake quantization changes the model output.

Basic Qwen3-VL trace:

```bash
python -m tico.quantization.examples.inspect \
  --config tico/quantization/examples/configs/qwen3_vl_ptq_only.yaml \
  --mode trace
```

Trace only selected module subtrees:

```bash
python -m tico.quantization.examples.inspect \
  --config tico/quantization/examples/configs/qwen3_vl_ptq_only.yaml \
  --mode trace \
  --interesting-modules model.language_model model.visual
```

By default, trace mode does **not** call `convert()` on the PTQ-prepared model.
This checks FP-vs-PTQ wrapper parity and should normally show very small
differences. To inspect the numerical error introduced by fake quantization,
enable conversion explicitly:

```bash
python -m tico.quantization.examples.inspect \
  --config tico/quantization/examples/configs/qwen3_vl_ptq_only.yaml \
  --mode trace \
  --enable-quantization \
  --interesting-modules model.language_model model.visual
```

Output sections:

```text
=== FP trace ===
[module.name] Tensor(shape=..., dtype=..., mean=..., min=..., max=..., std=...)

=== PTQ trace ===
[module.name] Tensor(shape=..., dtype=..., mean=..., min=..., max=..., std=...)

=== Side-by-side diff ===
module.name: mean|diff|=..., max|diff|=...
```

`--interesting-modules` accepts module names or parent module prefixes. When it
is omitted, trace mode records all named modules. Use a small calibration sample
count before tracing large Qwen3-VL models.

#### Wrapper smoke mode

Wrapper smoke mode runs module-level sanity checks for quantization wrappers.
It is useful when you want to quickly validate `prepare -> calibrate -> convert`,
numerical parity, plotting, and optional Circle export for a single wrapped
module.

List available wrapper smoke cases:

```bash
python -m tico.quantization.examples.inspect \
  --mode wrapper-smoke \
  --list-cases
```

Example output:

```text
Available wrapper smoke cases:

  nn_linear
  nn_conv3d
  nn_conv3d_special_case
  nn_layernorm
  nn_tied_embedding

  llama_attention_prefill
  llama_attention_decode
  llama_mlp
  llama_decoder_layer_prefill
  llama_decoder_layer_decode

  qwen3_vl_text_attention
  qwen3_vl_text_mlp
  qwen3_vl_text_decoder_layer
  qwen3_vl_text_model
  qwen3_vl_vision_attention
```

Run one case:

```bash
python -m tico.quantization.examples.inspect \
  --config tico/quantization/examples/configs/wrapper_smoke.yaml \
  --mode wrapper-smoke \
  --case llama_attention_prefill
```

Run one case with Circle export:

```bash
python -m tico.quantization.examples.inspect \
  --config tico/quantization/examples/configs/wrapper_smoke.yaml \
  --mode wrapper-smoke \
  --case llama_attention_prefill \
  --export circle \
  --output-dir ./out/wrapper_smoke \
  --strict
```

## CLI overrides

All example CLIs accept `--set KEY=VALUE` for simple dotted overrides:

```bash
python -m tico.quantization.examples.quantize \
  --config tico/quantization/examples/configs/llama_ptq_only.yaml \
  --set runtime.device=cpu \
  --set calibration.n_samples=1 \
  --set export.enabled=false
```

Values are parsed as YAML-like scalars when possible:

```bash
--set evaluation.enabled=true
--set calibration.n_samples=1
--set model.hf_token=null
--set runtime.dtype=float32
```

List indices are zero-based. This is most useful for toggling entries in the
`pipeline` list:

```yaml
pipeline:
  - name: spinquant       # pipeline.0
    enabled: false

  - name: cle             # pipeline.1
    enabled: false

  - name: gptq            # pipeline.2
    enabled: true

  - name: ptq             # pipeline.3
    enabled: true
```

For the example above, enable SpinQuant from the command line with:

```bash
python -m tico.quantization.examples.quantize \
  --config tico/quantization/examples/configs/llama_gptq_ptq.yaml \
  --set pipeline.0.enabled=true
```

You can also override fields inside later stages by index. For example, if the
PTQ stage is `pipeline.3`, change its SpinQuant rotation weight bit-width with:

```bash
python -m tico.quantization.examples.quantize \
  --config tico/quantization/examples/configs/llama_gptq_ptq.yaml \
  --set pipeline.0.enabled=true \
  --set pipeline.3.spin_rotation_weight_bits=8
```

Convenience aliases are also available:

```bash
--model       # overrides model.name_or_path
--device      # overrides runtime.device
--output-dir  # overrides export.output_dir, quantize.py only
```

Prefer adding a dedicated config preset when the change adds, removes, or
reorders list-valued sections such as `pipeline`. Command-line list overrides
are convenient for toggling existing stages or changing simple scalar fields,
but they are easy to misuse when the stage order differs between configs.

Good command-line overrides:

```bash
--set pipeline.0.enabled=true
--set pipeline.2.weight_bits=4
--set pipeline.3.activation_dtype=int16
```

Prefer a new config file for structural changes:

```yaml
pipeline:
  - name: spinquant
    enabled: true

  - name: gptq
    enabled: true
    weight_bits: 4

  - name: ptq
    enabled: true
    activation_dtype: int16
```

## Developer rule: do not add one script per model or algorithm

Most new examples should be config files, not Python files.

Add a config when:

- the model family already has an adapter
- the algorithm already has a stage
- the workflow is a new combination of existing stages
- the change is only calibration size, dtype, qscheme, evaluation, or export

Examples:

```text
configs/llama_gptq_ptq.yaml
configs/llama_ptq_only.yaml
configs/qwen3_vl_gptq_ptq.yaml
configs/qwen3_vl_ptq_only.yaml
```

Do not add scripts like:

```text
quantize_llama_with_gptq.py
quantize_qwen3_vl_with_gptq.py
quantize_qwen3_vl_ptq_only.py
quantize_vision_attention_debug.py
```

Use:

```bash
python -m tico.quantization.examples.quantize --config configs/<preset>.yaml
python -m tico.quantization.examples.inspect --config configs/<preset>.yaml --mode <mode>
```

## When a new Python example is allowed

A new file under `examples/` should be rare. It is acceptable only when it
introduces a genuinely new user-facing top-level action that does not fit any of
these commands:

```text
quantize.py
evaluate.py
inspect.py
```

Before adding a new example script, check whether the feature should instead be:

| Feature | Preferred location |
|---|---|
| New model family | `recipes/adapters/` |
| New algorithm | `recipes/stages/` |
| New config combination | `examples/configs/` |
| New benchmark | `recipes/evaluation/` |
| New export artifact | `recipes/export/` |
| New debug trace/parity tool | `recipes/debug/` + `inspect.py` mode |

## Adding a new model-family example

For a new family such as `gemma`, do this:

1. Add `recipes/adapters/gemma.py`.
2. Register it in `recipes/adapters/__init__.py`.
3. Add `configs/gemma_ptq_only.yaml`.
4. Optionally add `configs/gemma_gptq_ptq.yaml`.
5. Add a smoke test.
6. Document any model-specific config fields in `configs/README.md`.

Then run:

```bash
python -m tico.quantization.examples.quantize \
  --config tico/quantization/examples/configs/gemma_ptq_only.yaml
```

No new example script is needed.

## Adding a new algorithm example

For a new algorithm such as `awq`, do this:

1. Add `recipes/stages/awq.py`.
2. Register it in `recipes/stages/__init__.py`.
3. Add the stage to an existing or new config preset:

   ```yaml
   pipeline:
     - name: awq
       enabled: true
       weight_bits: 4
     - name: ptq
       enabled: true
   ```

4. Add a smoke test.

No new example script is needed.

## Minimum smoke commands

Use these commands for quick sanity checks after changing examples or recipes:

```bash
python -m tico.quantization.examples.quantize \
  --config tico/quantization/examples/configs/llama_ptq_only.yaml \
  --set calibration.n_samples=1 \
  --set evaluation.enabled=false \
  --set export.enabled=false
```

```bash
python -m tico.quantization.examples.inspect \
  --config tico/quantization/examples/configs/qwen3_vl_ptq_only.yaml \
  --mode trace \
  --set calibration.n_samples=1
```

```bash
python -m tico.quantization.examples.inspect \
  --config tico/quantization/examples/configs/wrapper_smoke.yaml \
  --mode wrapper-smoke \
  --case nn_linear \
  --no-plot
```
