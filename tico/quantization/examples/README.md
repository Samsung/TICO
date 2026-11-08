# Examples

This directory contains thin command-line examples for running quantization,
evaluation, export, Circle-bundle generation, and debugging workflows.

The examples are intentionally small. Real implementation code lives in
`tico.quantization.recipes`.

See the [quantization overview](../README.md) for the Python API, the
[configuration reference](configs/README.md) for YAML fields, and the
[recipe developer guide](../recipes/README.md) for reusable implementation code.

## Contents

- [Choose a command](#which-command-should-i-use)
- [LLaMA benchmark flow](#llama-benchmark-flow)
- [Gemma4 E2B flow](#gemma4-e2b-flow)
- [Quantize](#quantize)
- [Evaluate](#evaluate)
- [Export](#export)
- [Generate from a LLaMA Circle bundle](#generate-from-a-llama-circle-bundle)
- [LLaVA-Bench judge evaluation](#llava-bench-judge-evaluation)
- [Inspect and debug](#inspect--debug)
- [CLI overrides](#cli-overrides)
- [GPTQ SMSE sensitivity](#gptq-smse-sensitivity)
- [Contributing examples](#developer-rule-do-not-add-one-script-per-model-or-algorithm)
- [Minimum smoke commands](#minimum-smoke-commands)

## Directory layout

```text
tico/quantization/examples/
├── README.md
├── quantize.py          # Run a config-driven quantization pipeline
├── evaluate.py          # Evaluate an FP model or saved checkpoint
├── export.py            # Export artifacts from a model or saved checkpoint
├── generate.py          # Generate from an exported LLaMA Circle bundle
├── inspector.py         # Run trace/parity/debug tools
├── compare_side_by_side.py
└── configs/             # Reusable recipe presets
```

## Supported top-level actions

### Which command should I use?

Use `quantize.py` when you want to run a quantization recipe. It loads the
model, builds calibration inputs when an enabled stage or the adapter requires
them, runs enabled `pipeline` stages such as GPTQ, PTQ, SpinQuant, or CLE, and
then optionally evaluates and exports the resulting model.

Use `evaluate.py` when you only want to evaluate a floating-point model or an
already saved checkpoint. `evaluate.py` does **not** run `pipeline` stages from
the config. If the config contains enabled stages such as `gptq` or `ptq`, they
are ignored by `evaluate.py`.

For LLaMA, Qwen3-VL, and Gemma4 the standard quality protocol is **FP eager vs
Quant eager**. Calibration length, evaluation context, and static export capacity
are independent. Run both models with the same evaluation config and use
`--output-dir` to save separate `evaluation_protocol.json` records. See the
[protocol and paired commands](../recipes/evaluation/README.md#standard-quality-protocol-fp-eager-vs-quant-eager).

Use `export.py` to export either a floating-point model loaded through its
adapter or an already saved checkpoint. It writes configured artifacts such as
LLaMA and Qwen3-VL staged Circle files and does **not** run `pipeline` stages.

Use `inspector.py` for debug-oriented workflows such as trace, parity, runtime
inspection, and wrapper-level smoke checks.

Use `generate.py` to run greedy text generation from a saved LLaMA Circle
bundle. It consumes bundle artifacts directly and has no recipe config or
checkpoint-loading step.

Summary:

| Command | Runs `pipeline` stages? | Builds calibration inputs? | Evaluates? | Exports? |
|---|---:|---:|---:|---:|
| `quantize.py` | Yes | When required by enabled stages or the adapter | If `evaluation.enabled=true` | If `export.enabled=true` |
| `evaluate.py` | No | No | Yes | No |
| `export.py` | No | No | No | Yes |
| `generate.py` | No | No | No | No |
| `inspector.py` | Mode-dependent | Mode-dependent | Debug only | Debug only |

The four config-driven commands load the optional top-level `extensions` list
from the config before resolving the adapter, and honor `model.adapter` for
selecting a registered adapter variant. See the
[config guide](./configs/README.md#extensions-optional) for the schema and the
trust implications.

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

### Gemma4 E2B flow

Gemma4 E2B uses the same config-driven three-step interface. The export preset
writes static per-stage Circle files for the vision and text runtime.

1. Quantize Gemma4 E2B and save the checkpoint.

   ```bash
   python -m tico.quantization.examples.quantize \
     --config tico/quantization/examples/configs/gemma4_quantize.yaml
   ```

2. Evaluate either the floating-point model or the saved checkpoint with the
   configured Gemma4 evaluation suite.

   ```bash
   python -m tico.quantization.examples.evaluate \
     --config tico/quantization/examples/configs/gemma4_eval_suite.yaml \
     --checkpoint ./out/gemma4_quantized/quantized_model.pt
   ```

3. Export quantized per-stage Circle artifacts.

   ```bash
   python -m tico.quantization.examples.export \
     --config tico/quantization/examples/configs/gemma4_export.yaml \
     --checkpoint ./out/gemma4_quantized/quantized_model.pt
   ```

   Export the floating-point model with the same static runtime contract:

   ```bash
   python -m tico.quantization.examples.export \
     --config tico/quantization/examples/configs/gemma4_export.yaml \
     --source model \
     --device cpu \
     --output-dir ./out/gemma4_float/circle_layers
   ```

   Floating-point files use the `.f32.circle` suffix. Quantized checkpoint
   files use `.q.circle`.

### Quantize

`quantize.py` executes enabled `pipeline` stages in config order. The
[LLaMA](#llama-benchmark-flow) and [Gemma4](#gemma4-e2b-flow) quickstarts show
complete quantize/evaluate/export flows. To select a different checkpoint of an
existing model family, use `--model`:

```bash
python -m tico.quantization.examples.quantize \
  --config tico/quantization/examples/configs/qwen3_vl_quantize.yaml \
  --model Qwen/Qwen3-VL-2B-Instruct
```

Run a smaller calibration smoke without exporting artifacts:

```bash
python -m tico.quantization.examples.quantize \
  --config tico/quantization/examples/configs/llama_quantize.yaml \
  --set calibration.n_samples=1 \
  --set export.enabled=false
```

Write the checkpoint to another output directory:

```bash
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
Evaluation context    : model context
Export capacity       : 2048
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

Top-level evaluation target selection is supported through `--tasks`.

`--tasks` is an exclusive allow-list. When it is present, only the listed
top-level evaluation targets run, even when other targets are enabled in the
YAML file. The option does not replace benchmark details such as LM-Eval task
names, VQA datasets, MMMU subjects, or sample limits. Configure those details
in YAML or with `--set`.

Target names are canonical and model-family specific. See
[Selecting top-level evaluation targets](configs/README.md#selecting-top-level-evaluation-targets)
for the supported names, required benchmark details, and validation rules.

Run only the configured LM-Eval suite:

```bash
python -m tico.quantization.examples.evaluate \
  --config tico/quantization/examples/configs/llama_eval_suite.yaml \
  --checkpoint ./out/llama_quantized/quantized_model.pt \
  --tasks lm_eval
```

Override the LM-Eval benchmark details separately:

```bash
python -m tico.quantization.examples.evaluate \
  --config tico/quantization/examples/configs/llama_eval_suite.yaml \
  --checkpoint ./out/llama_quantized/quantized_model.pt \
  --tasks lm_eval \
  --set evaluation.lm_eval_tasks=mmlu,hellaswag
```

Run only MMMU and text perplexity from a VLM suite:

```bash
python -m tico.quantization.examples.evaluate \
  --config tico/quantization/examples/configs/qwen3_vl_eval_suite.yaml \
  --tasks mmmu,ppl
```

Run only the VQA evaluator while selecting its datasets separately:

```bash
python -m tico.quantization.examples.evaluate \
  --config tico/quantization/examples/configs/qwen3_vl_eval_suite.yaml \
  --tasks vqa \
  --set 'evaluation.vlm_tasks=[textvqa]'
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

Supported explicit embedding export is opt-in for Llama, Qwen3-VL and Gemma4.
Add `--preserve-embedding-quantization` to the export command to enable it; the
flag defaults to false. See the [export configuration](configs/README.md#export)
for the equivalent YAML setting and
[CPU embedding contracts](../../../docs/cpu_token_embedding.md) for supported
paths, compilation behavior, and limitations.

Qwen3-VL uses the same generic `circle_per_layer` artifact key. The exporter
keeps the fixed-grid vision model as one prefill stage and emits each text
decoder layer separately for prefill and decode:

```bash
python -m tico.quantization.examples.export \
  --config tico/quantization/examples/configs/qwen3_vl_export.yaml \
  --checkpoint ./out/qwen3_vl_quantized/quantized_model.pt
```

The fixed values under `model_args.vision` must match those used when the
checkpoint was wrapped and quantized. See the
[artifact naming contract](../recipes/export/README.md#artifact-naming) for the
Qwen3-VL filenames, vision profiles, and host/runtime input responsibilities.

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

### Generate from a LLaMA Circle bundle

Export a pretrained floating-point model with the existing LLaMA export preset
and opt in to the `circle_bundle` artifact:

```bash
python -m tico.quantization.examples.export \
  --config tico/quantization/examples/configs/llama_export.yaml \
  --source model \
  --artifacts circle_bundle \
  --output-dir ./out/llama_bundle \
  --set export.max_seq_len=128
```

The directory includes the staged Circle files, a versioned manifest, frozen
host RoPE/mask context, and local tokenizer files when the exporter has a
tokenizer. See the [bundle contract](../recipes/export/README.md#llama-circle-bundles)
for the layout and runtime API. Existing `circle_per_layer` exports remain
available; select `circle_bundle` to add the metadata needed for generation.

Generate from prompt text:

```bash
python -m tico.quantization.examples.generate \
  --bundle ./out/llama_bundle \
  --prompt "The capital of France is" \
  --max-new-tokens 16 \
  --output-json ./out/llama_generation.json
```

Prompt mode uses `transformers.AutoTokenizer` with `local_files_only=True` and
`trust_remote_code=False` on the bundle's `tokenizer/` directory. It adds the
tokenizer's special tokens to the supplied text; it does not apply a chat
template. Generation does not load the original Transformers model or download
model/tokenizer files. Install `transformers` separately when using prompt mode.

For bundles without tokenizer files, or to replay exact token inputs without
loading `transformers`, use numeric mode:

```bash
python -m tico.quantization.examples.generate \
  --bundle ./out/llama_bundle \
  --input-ids 1,2,3 \
  --max-new-tokens 4
```

Use IDs from the exported model's vocabulary. The two input options are mutually
exclusive. The runtime accepts one unpadded prompt, generates greedily, and
stops on a bundle EOS token or after `--max-new-tokens` steps (default: 16).
The prompt length plus the requested number of new tokens must fit
`export.max_seq_len`; an oversized request is rejected without truncation.

Both modes print JSON. `token_ids` contains the prompt and continuation as a
flat list; `generated_token_ids` contains only the continuation. `text` is the
decoded continuation in prompt mode and `null` in numeric mode.
`--output-json` saves the same JSON, including `execution_mode`, to an existing
parent directory.

Floating-point bundles run in `native` mode. Quantized bundles run in explicit
`fake_quantize` reference mode. These CPU reference runs validate artifact
composition under reference semantics; they do not establish NPU speed or
bit-exact integer-backend behavior. Tiny random-model tests cover floating-point
numerical parity, quantized tensor domains, and deterministic generation.
Quantized PyTorch parity is limited by the reference runtime's value-preserving
GATHER behavior; see the
[embedding domain contract](../../../docs/design.md#102-conversion-time-quantized-graph-legalization).
Use a pretrained bundle and a separate quality evaluation for meaningful text,
and validate correctness and performance on the intended backend.

### LLaVA-Bench judge evaluation

> [!NOTE]
> Pass `--tasks llava_bench` to run only LLaVA-Bench, even when the selected
> YAML preset enables other evaluation targets. Benchmark details under
> `evaluation.llava_bench` remain unchanged.

LLaVA-Bench-in-the-Wild is evaluated through the regular `evaluate.py` entry
point when a supported VLM config enables `evaluation.llava_bench.mode=judge`.
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
  --tasks llava_bench \
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
  --tasks llava_bench \
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

Use `inspector.py` for PTQ trace comparisons, static runtime checks, and synthetic
wrapper checks. The [debug developer guide](../recipes/debug/README.md) owns trace
internals, CPU validation, and new inspect modes. For `compare_side_by_side.py`,
use the [multi-batch comparison guide](../recipes/debug/README.md#multi-batch-side-by-side-comparison).

Run a static LLaMA runtime check or a tied-embedding smoke check with the matching
preset:

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

Trace compares an FP model with a PTQ-prepared copy. By default it checks wrapper
parity without conversion; add `--enable-quantization` to inspect fake-quantization
error. Start with one calibration sample and selected module subtrees:

```bash
python -m tico.quantization.examples.inspector \
  --config tico/quantization/examples/configs/qwen3_vl_quantize.yaml \
  --mode trace \
  --enable-quantization \
  --interesting-modules model.language_model model.visual \
  --set calibration.n_samples=1
```

See [Existing trace mode](../recipes/debug/README.md#existing-trace-mode) for the
PTQ-only preparation flow, module-prefix matching, and output interpretation.

#### Wrapper smoke mode

Wrapper smoke validates `prepare -> calibrate -> convert`, numerical parity,
and optional Circle export for synthetic modules. List the currently registered
cases, then select a case:

```bash
python -m tico.quantization.examples.inspector \
  --mode wrapper-smoke \
  --list-cases
```

```bash
python -m tico.quantization.examples.inspector \
  --config tico/quantization/examples/configs/wrapper_smoke.yaml \
  --mode wrapper-smoke \
  --case llama_attention_prefill \
  --export circle \
  --output-dir ./out/wrapper_smoke \
  --strict
```

The [wrapper smoke reference](../recipes/debug/wrapper_smoke/README.md) owns case
profiles, static decoder input contracts, plotting controls, and output/failure
formats. Omit `--export circle` for a numerical check without export.

## CLI overrides

The config-driven example CLIs accept `--set KEY=VALUE` for simple dotted
overrides. `generate.py` consumes a bundle and uses the options described above.

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

Lists of mappings with a unique `name` also accept named selectors. For
example, these overrides select the stages without depending on their indices:

```bash
--set pipeline.spinquant.enabled=false
--set pipeline.ptq.spin_rotation_weight=int8
```

The selector must exactly match the configured `name`. Missing names are
rejected; if a name appears more than once, use a numeric index.

Convenience aliases are also available:

| Option | Effect |
|---|---|
| `--model` | Overrides `model.name_or_path`. |
| `--device` | Overrides `runtime.device`. |
| `--output-dir` with `quantize.py` or `export.py` | Overrides `export.output_dir`. |
| `--output-dir` with `evaluate.py` | Overrides `evaluation.output_dir` for `evaluation_protocol.json`; benchmark-specific output paths keep their own settings. |
| `--output-dir` with `inspector.py --mode wrapper-smoke` | Sets the directory for wrapper-smoke plots, reports, and Circle files. |

## GPTQ SMSE sensitivity

The [GPTQ sensitivity configuration](configs/README.md#gptq-smse-sensitivity)
defines the schema and the `compute`, `save`, `load`, and `cache` modes. To reuse
a cache when it exists and compute/save it otherwise:

```bash
python -m tico.quantization.examples.quantize \
  --config tico/quantization/examples/configs/llama_quantize.yaml \
  --set pipeline.2.mse=smse \
  --set pipeline.2.sensitivity.mode=cache \
  --set pipeline.2.sensitivity.path=./out/llama/gptq_sensitivity.pt
```

The GPTQ stage is `pipeline.2` in both `llama_quantize.yaml` and
`qwen3_vl_quantize.yaml`.

## Developer rule: do not add one script per model or algorithm

Add a YAML preset for a new combination of existing model, stage, calibration,
evaluation, or export behavior. The recipe guide owns
[preset-versus-entrypoint decisions](../recipes/README.md#choosing-a-preset-or-entrypoint)
and the configuration reference owns [preset naming](configs/README.md#file-naming).

## When a new Python example is allowed

A new script requires a new top-level user action. Check the
[recipe responsibility map](../recipes/README.md#responsibility-boundaries) before
choosing an implementation location; keep public entrypoints thin.

## Adding a new model-family example

Follow the [adapter tutorial](../recipes/adapters/README.md#step-by-step-adding-a-new-model-family)
for implementation, registration, presets, and tests. Document model-specific YAML
fields in the [configuration reference](configs/README.md) and run the new preset
through the existing commands above.

## Adding a new algorithm example

Follow the [quantization contribution guide](../README.md#contributing-adding-a-new-algorithm)
for the quantizer and its configuration, then the
[stage tutorial](../recipes/stages/README.md#step-by-step-adding-a-new-algorithm-stage)
for recipe orchestration, registration, YAML usage, and tests.

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
