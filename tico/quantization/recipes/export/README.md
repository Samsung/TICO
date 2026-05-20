# Export Helpers Developer Guide

`recipes/export/` contains reusable artifact writers for quantized models.

Export helpers are kept outside examples so that checkpoint saving, Circle
export, per-layer export, and future artifact formats can be shared by multiple
model families and workflows.

## Responsibilities

Export helpers should:

- save model artifacts;
- encapsulate target-format-specific export calls;
- normalize example inputs for export;
- return artifact paths for logging and tests.

Export helpers should not:

- parse CLI arguments;
- build calibration datasets;
- choose which model family is being exported;
- run quantization stages;
- evaluate benchmark metrics.

The model adapter decides **which** artifacts to export. The export helper
decides **how** to write an artifact.

## Current layout

```text
recipes/export/
├── README.md
├── checkpoint.py     # torch checkpoint saving
└── circle.py         # full-model Circle export helpers
```

Additional files can be added for per-layer export, ONNX-like formats, debug
dump formats, or hardware-specific artifacts.

## Artifact naming

Use stable, predictable artifact names. Avoid embedding timestamps in filenames
unless the config explicitly requests it.

Recommended names:

```text
quantized_model.pt
model.q.circle
layer_<index>.q.circle
effective_config.yaml
export_inputs.pt
```

The output directory should come from `export.output_dir`.

## Function shape

Use explicit keyword-only arguments for non-trivial exporters:

```python
def export_full_circle(
    *,
    model: Any,
    example_input: Any,
    output_dir: str | Path,
    name: str = "model.q.circle",
    strict: bool = False,
) -> Path:
    ...
```

Return a `Path` so callers can store it in `ctx.artifacts` or assert it in tests.

For very simple helpers, positional arguments are acceptable:

```python
def save_checkpoint(model: Any, output_dir: str | Path, name: str = "quantized_model.pt") -> Path:
    ...
```

## Adding a new export target

1. Add a helper under `recipes/export/`.
2. Keep target-specific imports local if they are optional or heavy.
3. Return the created artifact path.
4. Add an artifact name under `export.artifacts`.
5. Call the helper from the relevant adapter’s `export()` method.
6. Add a tiny smoke export test if the target supports small models.
7. Document any required input shape or static-shape constraints.

Example config:

```yaml
export:
  enabled: true
  output_dir: ./out/llama
  artifacts:
    - ptq_checkpoint
    - circle_full
    - circle_per_layer
  strict: false
```

Example adapter call:

```python
artifacts = set(export_cfg.get("artifacts", []))

if "circle_per_layer" in artifacts:
    export_llama_layers_to_circle(
        model=ctx.model,
        example_inputs=ctx.calibration_inputs[:1],
        output_dir=output_dir / "layers",
        strict=bool(export_cfg.get("strict", False)),
    )
```

## Artifact key conventions

Use short, format-oriented names:

```text
ptq_checkpoint
checkpoint
circle_full
circle_per_layer
export_inputs
trace_dump
```

Avoid model-family-specific artifact names unless the artifact is truly unique to
that family.

Good:

```yaml
artifacts:
  - circle_per_layer
```

Bad:

```yaml
artifacts:
  - llama_decoder_layer_circle
```

If the implementation is model-specific, keep the generic artifact key and route
to the correct adapter-specific exporter.

## Example inputs

Many export formats require example inputs. The adapter should provide inputs in
the correct shape because input shape is model-family-specific.

Export helpers may normalize tensors to CPU:

```python
if isinstance(example_input, torch.Tensor):
    args = (example_input.cpu(),)
```

Do not silently create random export inputs inside a generic exporter. Random
input generation belongs in `recipes/data/` or an adapter-specific helper.

## Effective config

The runner should save `effective_config.yaml` when `export.output_dir` is set.
Export helpers should not mutate the config to record paths. Instead, return
paths and let the runner or adapter attach them to `ctx.artifacts` if needed.

## Error handling

Fail early when export prerequisites are missing:

```python
if not ctx.calibration_inputs:
    raise RuntimeError("Circle export requires at least one calibration input.")
```

Prefer clear errors over partially written artifacts.

## Testing checklist

For a new exporter, test:

- output directory creation;
- returned `Path`;
- existing output directory behavior;
- CPU-only execution when possible;
- missing example input behavior;
- invalid artifact key behavior at the adapter/config level;
- that the exporter does not run evaluation or quantization implicitly.
