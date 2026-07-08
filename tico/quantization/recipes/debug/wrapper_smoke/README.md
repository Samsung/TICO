# Wrapper Smoke Check

Wrapper smoke check is a lightweight developer workflow for validating
quantization wrappers, export paths, and numerical parity at module level.

The goal is to make it easy to:

- validate `prepare -> calibrate -> convert` flows
- compare floating-point and quantized outputs
- visualize numerical drift quickly
- export individual wrapped modules to Circle
- debug wrapper-specific regressions without running a full model benchmark

The workflow is intentionally synthetic and fast. Each case builds a small
deterministic module together with representative inputs that approximate the
real runtime contract of the wrapper.

## CLI Usage

List available cases:

```bash
python -m tico.quantization.examples.inspector \
  --mode wrapper-smoke \
  --list-cases
```

Run one wrapper smoke check:

```bash
python -m tico.quantization.examples.inspector \
  --config tico/quantization/examples/configs/wrapper_smoke.yaml \
  --mode wrapper-smoke \
  --case llama_attention_prefill
```

Run with Circle export:

```bash
python -m tico.quantization.examples.inspector \
  --config tico/quantization/examples/configs/wrapper_smoke.yaml \
  --mode wrapper-smoke \
  --case qwen3_vl_vision_attention \
  --export circle   \
  --output-dir ./out/wrapper_smoke
```

Run all registered cases:

```bash
python -m tico.quantization.examples.inspector \
  --config tico/quantization/examples/configs/wrapper_smoke.yaml \
  --mode wrapper-smoke \
  --case all
```

Fail immediately if parity thresholds are exceeded:

```bash
python -m tico.quantization.examples.inspector \
  --config tico/quantization/examples/configs/wrapper_smoke.yaml \
  --mode wrapper-smoke \
  --case llama_decoder_layer_prefill \
  --strict
```

Disable scatter-plot visualization:

```bash
python -m tico.quantization.examples.inspector \
--config tico/quantization/examples/configs/wrapper_smoke.yaml \
--mode wrapper-smoke \
--case nn_linear \
--no-plot
```

## Output Example

```text
============================================================
Wrapper Smoke Result
============================================================

Case:
  llama_attention_prefill

Mode:
  quantized parity

Metrics:
  mean_abs_diff : 0.042131
  max_abs_diff  : 0.319825
  peir_percent  : 0.812314

Output:
  shape   : (2, 6, 16)
  finite  : True

Artifacts:
  circle_model : ./out/wrapper_smoke/llama_attention_prefill/model.circle

Status:
  PASS
```

## Adding a New Case

Add a new file or registration entry under:

```text
tico/quantization/recipes/debug/wrapper_smoke/cases/
```

A case should define:

- module construction
- calibration samples
- evaluation samples
- export example inputs
- parity thresholds

Then register the case in:

```text
registry.py
```

After registration, the case automatically becomes available through:

```bash
python -m tico.quantization.examples.inspector   --mode wrapper-smoke   --case <new_case>
```

## Main Features

Wrapper smoke check supports:

- module-level quantization sanity checks
- floating-point vs quantized parity metrics
- PEIR and mean absolute error reporting
- `plot_two_outputs()` visualization
- Circle export for wrapped modules
- deterministic synthetic calibration data
- reusable shared runner infrastructure
- CLI integration through `examples/inspector.py`

## Architecture

```text
examples/inspector.py
    └── wrapper_smoke runner
            ├── registry
            ├── shared utilities
            ├── export helpers
            └── per-wrapper cases
```

Each case defines:

- how to build the module
- calibration inputs
- evaluation inputs
- export behavior
- parity thresholds
- optional visualization behavior

The shared runner owns:

- calibration loops
- conversion flow
- parity metrics
- plotting
- Circle export
- result formatting
- failure handling

## Design Principles

### Small deterministic modules

Wrapper smoke checks should run quickly on CPU whenever possible.

### Synthetic but representative inputs

Inputs should approximate realistic wrapper contracts without requiring
large datasets or external preprocessing pipelines.

### Shared infrastructure

Cases should focus only on wrapper-specific logic while the runner handles
common quantization and export behavior.

### Developer-focused debugging

Wrapper smoke check is designed for rapid iteration during wrapper
development, export debugging, and quantization regression analysis.

## Intended Usage

Wrapper smoke check is intended for:

- wrapper development
- quantization debugging
- export debugging
- parity inspection
- CI smoke validation
- fast local sanity checks

It is not intended to replace:

- end-to-end model evaluation
- benchmark suites
- dataset-driven accuracy validation
- latency benchmarking
