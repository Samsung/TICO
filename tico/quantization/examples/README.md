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
├── evaluate.py          # Evaluate an FP model or saved fake-quant checkpoint
├── inspect.py           # Run trace/parity/debug tools
└── configs/             # Reusable recipe presets
```

## Supported top-level actions

### Quantize

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

### Evaluate

```bash
python -m tico.quantization.examples.evaluate \
  --config tico/quantization/examples/configs/llama_gptq_ptq.yaml \
  --checkpoint ./out/llama/quantized_model.pt
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


## CLI overrides

All example CLIs accept `--set KEY=VALUE` for simple dotted overrides:

```bash
python -m tico.quantization.examples.quantize \
  --config tico/quantization/examples/configs/llama_ptq_only.yaml \
  --set runtime.device=cpu \
  --set calibration.n_samples=1 \
  --set export.enabled=false
```

Convenience aliases are also available:

```bash
--model       # overrides model.name_or_path
--device      # overrides runtime.device
--output-dir  # overrides export.output_dir, quantize.py only
```

Prefer adding a dedicated config preset when the change modifies list-valued
sections such as `pipeline`.

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
