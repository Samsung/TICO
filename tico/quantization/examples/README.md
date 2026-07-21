# Examples

This directory contains thin command-line examples for running quantization,
evaluation, export, and debugging workflows.

The examples are intentionally small. Real implementation code lives in
`tico.quantization.recipes`.

## Directory layout

```text
tico/quantization/examples/
├── README.md
├── quantize.py          # Run a config-driven quantization pipeline
├── evaluate.py          # Evaluate an FP model or saved checkpoint
├── export.py            # Export artifacts from a model or saved checkpoint
├── inspector.py         # Run trace/parity/debug tools
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

Use `export.py` to export either a floating-point model loaded through its
adapter or an already saved checkpoint. It writes configured artifacts such as
LLaMA per-layer Circle files and does **not** run `pipeline` stages.

Use `inspector.py` for debug-oriented workflows such as trace, parity, runtime
inspection, and wrapper-level smoke checks.

Summary:

| Command | Runs `pipeline` stages? | Builds calibration inputs? | Evaluates? | Exports? |
|---|---:|---:|---:|---:|
| `quantize.py` | Yes | Yes | If `evaluation.enabled=true` | If `export.enabled=true` |
| `evaluate.py` | No | No | Yes | No |
| `export.py` | No | No | No | Yes |
| `inspector.py` | Mode-dependent | Mode-dependent | Debug only | Debug only |

Common usage patterns:

```bash
# Quantize LLaMA and save the configured checkpoint.
python -m tico.quantization.examples.quantize \
  --config tico/quantization/examples/configs/llama_quantize.yaml
```

```bash
# Evaluate a saved LLaMA checkpoint without running quantization again.
python -m tico.quantization.examples.evaluate \
  --config tico/quantization/examples/configs/llama_eval_suite.yaml \
  --checkpoint ./out/llama_quantized/quantized_model.pt
```

```bash
# Export per-layer Circle artifacts from a saved LLaMA checkpoint.
python -m tico.quantization.examples.export \
  --config tico/quantization/examples/configs/llama_export.yaml \
  --checkpoint ./out/llama_quantized/quantized_model.pt
```

### LLaMA benchmark flow

Use this three-step loop for the common LLaMA post-quantization benchmark flow.

1. Quantize the model and save the configured checkpoint.

   ```bash
   python -m tico.quantization.examples.quantize \
     --config tico/quantization/examples/configs/llama_quantize.yaml
   ```

2. Evaluate the saved checkpoint with the configured LLM evaluation suite.

   ```bash
   python -m tico.quantization.examples.evaluate \
     --config tico/quantization/examples/configs/llama_eval_suite.yaml \
     --checkpoint ./out/llama_quantized/quantized_model.pt
   ```

3. Export per-layer Circle artifacts from the saved checkpoint.

   ```bash
   python -m tico.quantization.examples.export \
     --config tico/quantization/examples/configs/llama_export.yaml \
     --checkpoint ./out/llama_quantized/quantized_model.pt
   ```

### Quantize

`quantize.py` is the command that executes the recipe pipeline. It runs the
enabled stages under `pipeline` in the order they appear in the config.

```bash
python -m tico.quantization.examples.quantize \
  --config tico/quantization/examples/configs/llama_quantize.yaml
```

```bash
python -m tico.quantization.examples.quantize \
  --config tico/quantization/examples/configs/qwen3_vl_quantize.yaml \
  --model Qwen/Qwen3-VL-2B-Instruct
```

Examples:

```bash
# Run the default LLaMA benchmark quantization and save a checkpoint.
python -m tico.quantization.examples.quantize \
  --config tico/quantization/examples/configs/llama_quantize.yaml
```

```bash
# Run a smaller calibration smoke without exporting artifacts.
python -m tico.quantization.examples.quantize \
  --config tico/quantization/examples/configs/llama_quantize.yaml \
  --set calibration.n_samples=1 \
  --set export.enabled=false
```

```bash
# Run quantization and write the checkpoint to another output directory.
python -m tico.quantization.examples.quantize \
  --config tico/quantization/examples/configs/llama_quantize.yaml \
  --output-dir ./out/llama
```

#### Quantize output and diagnostics

`quantize.py` prints a high-level config summary before loading the model. This
helps verify which stages and key quantization specs are active.

Example:

```text
=== Config ===
Model                 : Maykeye/TinyLLama-v0
Device                : cuda
DType                 : float32
Seed                  : 42
GPTQ enabled          : True
GPTQ lm_head enabled  : False
PTQ enabled           : True
SpinQuant enabled     : True
CLE enabled           : False
Activation            : int16
Linear weight         : uint4
Embedding weight      : uint8
LM head weight        : uint8
Spin rotation weight  : int16
Calibration samples   : 128
Calibration seq length: 2048
Max seq length        : 2048
Profile               : npu_export
```

Disable the summary when needed:

```bash
python -m tico.quantization.examples.quantize \
  --config tico/quantization/examples/configs/llama_quantize.yaml \
  --set runtime.print_config=false
```

Print the GPTQ-to-PTQ qparam injection summary with `runtime.verbose=true`:

```bash
python -m tico.quantization.examples.quantize \
  --config tico/quantization/examples/configs/llama_quantize.yaml \
  --set runtime.verbose=true
```

The injection summary reports how many GPTQ quantizers were injected into PTQ
weight observers:

```text
[GPTQ → PTQ injection summary]
  matched : 56
  missed  : 21
  unused  : 0

  missed modules:
    - model.embed_tokens
    - model.norm
    - model.layers.0.input_layernorm
    ...
```

### Evaluate

`evaluate.py` evaluates the model loaded by the adapter, or the model loaded
from `--checkpoint` when one is provided. It does not prepare, calibrate,
convert, or export the model.

Running this command without `--checkpoint` evaluates the floating-point model,
because the `pipeline` section is not executed by `evaluate.py`:

```bash
python -m tico.quantization.examples.evaluate \
  --config tico/quantization/examples/configs/llama_eval_suite.yaml
```

To evaluate a quantized result, pass a saved checkpoint:

```bash
python -m tico.quantization.examples.evaluate \
  --config tico/quantization/examples/configs/llama_eval_suite.yaml \
  --checkpoint ./out/llama_quantized/quantized_model.pt
```

Task overrides are supported:

```bash
python -m tico.quantization.examples.evaluate \
  --config tico/quantization/examples/configs/llama_eval_suite.yaml \
  --checkpoint ./out/llama_quantized/quantized_model.pt \
  --tasks winogrande,arc_easy
```

### Export

`export.py` exports configured artifacts from a floating-point model or an
already saved checkpoint. It does not load calibration data or run quantization
stages.

Export the floating-point model on CPU while preserving the NPU-facing
per-layer input/output contract:

```bash
python -m tico.quantization.examples.export \
  --config tico/quantization/examples/configs/llama_export.yaml \
  --source model \
  --device cpu \
  --output-dir ./out/llama_float/circle_layers
```

Floating-point artifacts use the `.f32.circle` suffix. To export an already
saved quantized checkpoint instead:

```bash
python -m tico.quantization.examples.export \
  --config tico/quantization/examples/configs/llama_export.yaml \
  --checkpoint ./out/llama_quantized/quantized_model.pt
```

`llama_export.yaml` exports both prefill and decode layer artifacts by default.
Use an override only when decode layer export is not needed:

```bash
python -m tico.quantization.examples.export \
  --config tico/quantization/examples/configs/llama_export.yaml \
  --checkpoint ./out/llama_quantized/quantized_model.pt \
  --set export.prefill_decode=false
```

Export to another directory:

```bash
python -m tico.quantization.examples.export \
  --config tico/quantization/examples/configs/llama_export.yaml \
  --checkpoint ./out/llama_quantized/quantized_model.pt \
  --output-dir ./out/llama_layers
```

### LLaVA-Bench judge evaluation

> [!NOTE]
> `qwen3_vl_eval_suite.yaml` and
> `qwen3_vl_eval_suite_mx_override_polices.yaml` enable the other evaluation
> tasks by default. Disable them with `--set` overrides when running only
> LLaVA-Bench.

LLaVA-Bench-in-the-Wild is evaluated through the regular `evaluate.py` entry
point when the Qwen3-VL config enables `evaluation.llava_bench.mode=judge`.
This path treats LLaVA-Bench as open-ended natural QA. The generation prompt is
only:

```text
<image>
{question}
```

For static-runtime parity with `max_seq_len=2048`, cap the image resolution so
the visual placeholder tokens fit before generation. The example config uses
`image_max_pixels: 602112`, which is `768 * 28 * 28`, and the evaluator
pre-resizes PIL-like images before processor tokenization.

The adapter writes three artifacts:

- answer JSONL containing generated candidate answers
- review JSONL containing per-question judge scores
- summary JSON containing average score, relative score, and win/tie counts

Use the following three-step workflow to measure regression from the floating-
point model to a quantized checkpoint.

#### 1. Evaluate the floating-point model

Run `evaluate.py` first. This does not run the quantization pipeline; it only
loads the FP model and produces the FP LLaVA-Bench answers and judge summary.
Keep the FP answer JSONL because it is used as the regression baseline in step
3.

```bash
python -m tico.quantization.examples.evaluate \
  --set evaluation.llava_bench.output.dir=./out/llava_bench_fp \
  --set evaluation.llava_bench.output.answers=./out/llava_bench_fp/answers.jsonl \
  --set evaluation.llava_bench.output.reviews=./out/llava_bench_fp/reviews.jsonl \
  --set evaluation.llava_bench.output.summary=./out/llava_bench_fp/summary.json  \
  --config tico/quantization/examples/configs/qwen3_vl_eval_suite.yaml
```

#### 2. Quantize the model and save a checkpoint

Run `quantize.py` with a config that enables the desired pipeline stages. The
example below exports the quantized checkpoint.

```bash
python -m tico.quantization.examples.quantize \
  --config tico/quantization/examples/configs/qwen3_vl_quantize.yaml \
  --set export.output_dir=./out/qwen3_vl_quantized
```

This writes the default checkpoint:

```text
./out/qwen3_vl_quantized/quantized_model.pt
```

#### 3. Evaluate the quantized checkpoint against the FP answers

Run `evaluate.py` again with `--checkpoint`. Leave `candidate_answers` as
`null` so the checkpoint generates new candidate answers. Set
`baseline_answers` to the FP answer JSONL from step 1. The judge then compares:

```text
baseline = FP answers
candidate = quantized-checkpoint answers
```

```bash
python -m tico.quantization.examples.evaluate \
  --config tico/quantization/examples/configs/qwen3_vl_eval_suite.yaml \
  --checkpoint ./out/qwen3_vl_quantized/quantized_model.pt \
  --set evaluation.llava_bench.baseline_answers=./out/llava_bench_fp/answers.jsonl \
  --set evaluation.llava_bench.candidate_answers=null \
  --set evaluation.llava_bench.baseline_label=qwen3_vl_4b_fp \
  --set evaluation.llava_bench.candidate_label=qwen3_vl_4b_ptq \
  --set evaluation.llava_bench.output.dir=./out/llava_bench_quant \
  --set evaluation.llava_bench.output.answers=./out/llava_bench_quant/answers.jsonl \
  --set evaluation.llava_bench.output.reviews=./out/llava_bench_quant/quant_vs_fp.reviews.jsonl \
  --set evaluation.llava_bench.output.summary=./out/llava_bench_quant/quant_vs_fp.summary.json
```

The regression signal is the quantized candidate score relative to the FP
baseline score in the summary:

```text
candidate_relative_score = mean_candidate_score / mean_baseline_score * 100
```

A value near 100 means the quantized checkpoint is close to the FP model under
the same prompt, image cap, judge model, and sample set.

#### Reusing existing answer files

If `candidate_answers` points to an existing JSONL file, generation is skipped
and that file is judged. If `candidate_answers` is `null`, the currently loaded
model generates candidate answers and writes them to `evaluation.llava_bench.output.answers`.

Per-sample answer and judge progress is controlled by `runtime.show_progress`;
set it to `false` when you want quieter output:

```bash
--set runtime.show_progress=false
```

When the judge is not the official GPT-4-style judge, report the judge model ID
with the results. The default Llama 3.2 3B judge is intended for inexpensive
regression checks, not for leaderboard-compatible LLaVA-Bench reporting.

### Inspect / debug

```bash
python -m tico.quantization.examples.inspector \
  --config tico/quantization/examples/configs/qwen3_vl_quantize.yaml \
  --mode trace \
  --interesting-modules model.language_model model.visual
```

```bash
python -m tico.quantization.examples.inspector \
  --config tico/quantization/examples/configs/static_llama_runtime.yaml \
  --mode static-llama-runtime
```

```bash
python -m tico.quantization.examples.inspector \
  --config tico/quantization/examples/configs/tied_embedding_smoke.yaml \
  --mode tied-embedding-smoke
```

#### Trace mode

Trace mode compares a floating-point model with a PTQ-prepared copy of the same
model. It is useful for finding the first module where PTQ wrapping,
calibration, or fake quantization changes the model output.

Basic Qwen3-VL trace:

```bash
python -m tico.quantization.examples.inspector \
  --config tico/quantization/examples/configs/qwen3_vl_quantize.yaml \
  --mode trace
```

Trace only selected module subtrees:

```bash
python -m tico.quantization.examples.inspector \
  --config tico/quantization/examples/configs/qwen3_vl_quantize.yaml \
  --mode trace \
  --interesting-modules model.language_model model.visual
```

By default, trace mode does **not** call `convert()` on the PTQ-prepared model.
This checks FP-vs-PTQ wrapper parity and should normally show very small
differences. To inspect the numerical error introduced by fake quantization,
enable conversion explicitly:

```bash
python -m tico.quantization.examples.inspector \
  --config tico/quantization/examples/configs/qwen3_vl_quantize.yaml \
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
python -m tico.quantization.examples.inspector \
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
python -m tico.quantization.examples.inspector \
  --config tico/quantization/examples/configs/wrapper_smoke.yaml \
  --mode wrapper-smoke \
  --case llama_attention_prefill
```

Run one case with Circle export:

```bash
python -m tico.quantization.examples.inspector \
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
  --config tico/quantization/examples/configs/llama_quantize.yaml \
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
    enabled: true

  - name: cle             # pipeline.1
    enabled: false

  - name: gptq            # pipeline.2
    enabled: true

  - name: ptq             # pipeline.3
    enabled: true
```

`llama_quantize.yaml` already enables SpinQuant. Disable it from the command
line with:

```bash
python -m tico.quantization.examples.quantize \
  --config tico/quantization/examples/configs/llama_quantize.yaml \
  --set pipeline.0.enabled=false
```

You can also override fields inside later stages by index. For example, if the
PTQ stage is `pipeline.3`, change its SpinQuant rotation weight spec with:

```bash
python -m tico.quantization.examples.quantize \
  --config tico/quantization/examples/configs/llama_quantize.yaml \
  --set pipeline.3.spin_rotation_weight=int8
```

Convenience aliases are also available:

```bash
--model       # overrides model.name_or_path
--device      # overrides runtime.device
--output-dir  # overrides export.output_dir, quantize.py/export.py only
```

## GPTQ SMSE sensitivity

When GPTQ uses sensitivity-aware MSE, set `pipeline.<index>.mse=smse` and
configure sensitivity behavior with a nested `sensitivity` mapping under the
GPTQ stage.

Schema:

```yaml
pipeline:
  - name: gptq
    enabled: true
    mse: smse
    sensitivity:
      mode: compute
      path: null
```

Supported modes:

| Mode | Behavior |
|---|---|
| `compute` | Compute sensitivity for the current run only. `path` must be `null`. |
| `save` | Compute sensitivity, use it for this run, and save it to `path`. |
| `load` | Load sensitivity from `path` and use it for this run. |
| `cache` | Load from `path` if it exists; otherwise compute and save it. |

Compute sensitivity without saving it:

```bash
python -m tico.quantization.examples.quantize \
  --config tico/quantization/examples/configs/llama_quantize.yaml \
  --set pipeline.2.mse=smse \
  --set pipeline.2.sensitivity.mode=compute
```

Compute and save sensitivity:

```bash
python -m tico.quantization.examples.quantize \
  --config tico/quantization/examples/configs/llama_quantize.yaml \
  --set pipeline.2.mse=smse \
  --set pipeline.2.sensitivity.mode=save \
  --set pipeline.2.sensitivity.path=./out/llama/gptq_sensitivity.pt
```

Load a saved sensitivity file:

```bash
python -m tico.quantization.examples.quantize \
  --config tico/quantization/examples/configs/llama_quantize.yaml \
  --set pipeline.2.mse=smse \
  --set pipeline.2.sensitivity.mode=load \
  --set pipeline.2.sensitivity.path=./out/llama/gptq_sensitivity.pt
```

Use a cache file when available, otherwise compute and save it:

```bash
python -m tico.quantization.examples.quantize \
  --config tico/quantization/examples/configs/llama_quantize.yaml \
  --set pipeline.2.mse=smse \
  --set pipeline.2.sensitivity.mode=cache \
  --set pipeline.2.sensitivity.path=./out/llama/gptq_sensitivity.pt
```

For `qwen3_vl_quantize.yaml`, the GPTQ stage is also `pipeline.2`, so the same
override pattern applies.

## Developer rule: do not add one script per model or algorithm

Most new examples should be config files, not Python files.

Add a config when:

- the model family already has an adapter
- the algorithm already has a stage
- the workflow is a new combination of existing stages
- the change is only calibration size, dtype, qscheme, evaluation, or export

Examples:

```text
configs/llama_quantize.yaml
configs/llama_eval_suite.yaml
configs/llama_export.yaml
configs/qwen3_vl_quantize.yaml
configs/qwen3_vl_eval_suite.yaml
configs/qwen3_vl_eval_suite_mx_override_polices.yaml
configs/qwen3_vl_export.yaml
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
python -m tico.quantization.examples.inspector --config configs/<preset>.yaml --mode <mode>
```

## When a new Python example is allowed

A new file under `examples/` should be rare. It is acceptable only when it
introduces a genuinely new user-facing top-level action that does not fit any of
these commands:

```text
quantize.py
evaluate.py
inspector.py
```

Before adding a new example script, check whether the feature should instead be:

| Feature | Preferred location |
|---|---|
| New model family | `recipes/adapters/` |
| New algorithm | `recipes/stages/` |
| New config combination | `examples/configs/` |
| New benchmark | `recipes/evaluation/` |
| New export artifact | `recipes/export/` |
| New debug trace/parity tool | `recipes/debug/` + `inspector.py` mode |

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
  --config tico/quantization/examples/configs/llama_quantize.yaml \
  --set calibration.n_samples=1 \
  --set calibration.seq_len=64 \
  --set calibration.decode_steps=0 \
  --set pipeline.spinquant.enabled=false \
  --set pipeline.gptq.enabled=false \
  --set pipeline.ptq.decode_calibration_steps=0 \
  --set evaluation.enabled=false \
  --set export.enabled=false
```

```bash
python -m tico.quantization.examples.inspector \
  --config tico/quantization/examples/configs/qwen3_vl_quantize.yaml \
  --mode trace \
  --set calibration.n_samples=1
```

```bash
python -m tico.quantization.examples.inspector \
  --config tico/quantization/examples/configs/wrapper_smoke.yaml \
  --mode wrapper-smoke \
  --case nn_linear \
  --no-plot
```
