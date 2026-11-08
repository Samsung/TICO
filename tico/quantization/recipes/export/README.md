# Export Helpers Developer Guide

`recipes/export/` contains reusable artifact writers for quantized models.

Export helpers are kept outside examples so that checkpoint saving, Circle
export, per-layer export, and future artifact formats can be shared by multiple
model families and workflows.

See the [export CLI](../../examples/README.md#export) for commands and the
[recipe developer guide](../README.md) for adapter/helper responsibilities.

## Export guides

| Guide | Scope |
|---|---|
| [Artifact naming](#artifact-naming) | Shared naming conventions and Qwen3-VL profile-qualified stage files. |
| [LLaMA Circle bundles](#llama-circle-bundles) | Bundle manifests, frozen host context, and runtime consumption. |
| [LLaMA RoPE and checkpoint migration](LLAMA_ROPE_UNIFICATION.md) | Rotary-source contracts and migration of legacy checkpoints. |
| [Gemma4 artifact reference](gemma4_artifacts.md) | Text, PLE, vision, and cache-boundary contracts. |
| [Complete fixed-image Qwen3-VL prefill](#complete-fixed-image-qwen3-vl-prefill) | Composition of fixed-grid vision and text stages into a complete graph. |
| [Static export regression verification](STATIC_EXPORT_REGRESSION.md) | Export isolation and comparison with a selected baseline checkout. |

## Responsibilities

Export helpers should:

- save model artifacts;
- encapsulate target-format-specific export calls;
- normalize example inputs for export;
- expose output locations and document whether they return artifact paths or `None`.

Export helpers should not:

- parse CLI arguments;
- build calibration datasets;
- choose which model family is being exported;
- run quantization stages;
- evaluate benchmark metrics.

The model adapter decides **which** artifacts to export. The export helper
decides **how** to write an artifact.

Quantized export consumes validated WrapQ frozen snapshots. It does not derive
activation grids, resume calibration, or repair incomplete checkpoints. Recipe
checkpoint entrypoints validate the common frozen-state contract before export;
legacy state that cannot preserve every admitted site's exact grid requires
repreparing and recalibrating the original float model. Frozen state-dict loads
restore wrapper mode and capability readiness without another freeze.

Constants generated for a static export profile have separate local observers
finalized through the common validator. Model-specific ownership rules belong
with the artifact contract; see [Gemma4 static vision profiles](gemma4_artifacts.md#static-vision-profile-ownership).

For LLaMA, Qwen3-VL, and Gemma4, `export.max_seq_len` owns static text capacity.
It defaults independently to `min(2048, model context)` and never reads
`calibration.seq_len` or `evaluation.max_seq_len`. LLaMA full-model export also
builds its shape example at this export length without consuming calibration
samples. Existing static vision/profile validation and staged prefill/decode
contracts still apply. See the
[eager quality protocol](../evaluation/README.md#standard-quality-protocol-fp-eager-vs-quant-eager)
for the separate evaluation policy.

## Current layout

```text
recipes/export/
├── README.md
├── checkpoint.py     # torch checkpoint saving
├── circle.py         # full-model and shared single-stage Circle export helpers
├── gemma4.py         # static Gemma4 vision/text staged Circle export
├── llama.py          # staged LLaMA Circle export
├── llama_bundle.py   # versioned LLaMA bundle manifest and frozen host context
└── qwen3_vl.py       # fixed-grid Qwen3-VL staged Circle export
```

Additional files can be added for per-layer export, ONNX-like formats, debug
dump formats, or hardware-specific artifacts.

`convert_and_save_circle()` is the shared single-stage writer used by the
LLaMA, Qwen3-VL, and Gemma4 exporters. It puts the supplied module in eval mode,
converts and saves it, and returns `None`. Callers own input preparation,
device placement, output-directory creation, and staged I/O contracts. Its optional
`config=` argument forwards the public compile configuration. The LLaMA, Qwen3-VL,
and Gemma4 staged exporters accept `compile_config=` and forward it to each stage;
`export_full_circle()` accepts `config=`. Recipe adapters resolve the boolean
`export.preserve_embedding_quantization` (default `false`) into that configuration.
See [CPU token embedding](../../../../docs/cpu_token_embedding.md) for the supported
explicit embedding policy and its limits.

Frozen-state validation and fake-quant execution compatibility are separate.
CPU-resident BF16 per-channel scales remain valid checkpoint state, but enabled
per-channel fake quantization with those scales cannot run during CPU preparation
or tracing. BF16 inputs with FP32 scales are a different, supported CPU case. An
NPU deployment target does not change where the preparation/tracing operation
runs. Callers must explicitly prepare FP32 qparams when that execution route needs
them; export never repairs or converts a frozen grid silently.

The shared single-stage writer and `export_full_circle()` run metadata-only
execution preflight before conversion. Stages declare participating observer
sites through `_export_execution_sites(example_inputs, kwargs)` and transparent
wrappers delegate that declaration. Leaf modules identify the actual operation
device: a weight observer executes on its weight tensor's device, while a directly
bound input observer uses the corresponding example input's device. The check
never infers execution from qparam storage, the first model parameter or global
CUDA availability.

Current declarations cover `QuantLinear`, `QuantEmbedding`, Gemma4 scaled-word
embeddings, and the LLaMA/Gemma4/Qwen3-VL token lookup adapters. They also cover
Gemma4 PLE lookup, its producer-owned output observer, and Qwen3-VL's selected
multimodal token lookup. Rotation children delegate their declared operand sites;
an input-dependent rotation site without a bound example input retains its
runtime check.

This declaration is explicit rather than a recursive observer inventory. A live
capability owner retained outside the stage tree, an unused conditional branch,
or the source pooler behind a specialized profile is not stage computation.
Capability and captured-mode guards still apply before checking declared sites.
Stages without an exact declaration, including decoder, whole-vision, assistant
core and arbitrary full-model/custom forward routes, retain the per-observer
runtime check. Direct `tico.convert()` or
`torch.export.export()` calls also retain that check without recipe preflight.
Unknown routes do not gain an executability guarantee from a successful
preflight; moving the model or changing execution inputs requires checking again.

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

Qwen3-VL `circle_per_layer` export profile-qualifies every graph whose tensor
shape depends on the vision grid. Text-only stages keep stable shared names:

```text
vision_prefill_t<T>_h<H>_w<W>.q.circle
token_embedding.q.circle
multimodal_embedding_prefill_t<T>_h<H>_w<W>.q.circle
decoder_layer_prefill_<index>.q.circle
deepstack_fusion_<index>_t<T>_h<H>_w<W>.q.circle
decoder_layer_decode_<index>.q.circle
lm_head.q.circle
```

For example, `grid_thw: [1, 54, 72]` produces
`vision_prefill_t1_h54_w72.q.circle`. The vision profile is baked into the
vision, multimodal-embedding, and DeepStack graph shapes. The vision NPU graph
accepts only `pixel_values`; `image_grid_thw` remains a CPU-runtime value used
to select the matching profile-specific artifacts. The fixed
`model_args.vision` settings must match the wrapped/quantized checkpoint.

`token_embedding` has a dynamic sequence dimension and is shared by prefill
and decode. The runtime owns mRoPE lookup, additive attention masks, and KV
storage/update, and passes the required tensors to the decoder graphs. Exporting
with `--source model` produces the same artifact set with `.f32.circle` tags.

The output directory should come from `export.output_dir`.

### LLaMA Circle bundles

Opt in to `export.artifacts: [circle_bundle]` for a self-contained LLaMA
generation bundle. The LLaMA adapter writes it directly under
`export.output_dir`; the existing `circle_per_layer` artifact keeps its current
behavior. A bundle requires paired prefill/decode export and batch size 1.

```text
llama_manifest.json
llama_host.npz
token_embedding.<tag>.circle
decoder_layer_prefill_<index>.<tag>.circle
decoder_layer_decode_<index>.<tag>.circle
lm_head.<tag>.circle
tokenizer/                                  # when a tokenizer is available
```

`<tag>` is `f32` for float32 model export and `q` for a quantized checkpoint.
The versioned manifest records model geometry, sequence capacity, pad/EOS IDs,
stage input/output roles, the host-context file, and the optional tokenizer
directory. The host context preserves the prepared decoder layers' RoPE tables
and attention-mask fill values. The consumer uses these frozen values instead
of reconstructing them from a model name or a Transformers config.

`LlamaCircleRuntime` loads the saved Circle stages through the built-in reference
runtime. It owns causal masks, right-padding for static prefill, logical token
positions, and per-layer KV cache updates:

```python
import numpy as np

from tico.quantization.recipes.runtime.llama import LlamaCircleRuntime

runtime = LlamaCircleRuntime("./out/llama_bundle")
tokens = runtime.generate(
    np.asarray([[1, 2, 3]], dtype=np.int64),
    max_new_tokens=4,
)
```

Input IDs are one unpadded sequence with shape `[1, prompt_length]`. The result
has shape `[1, prompt_length + generated_length]`, including EOS when emitted.
Generation is greedy and uses the manifest's EOS ID or IDs by default. The
requested prompt length plus `max_new_tokens` must not exceed `max_seq_len`;
capacity errors do not silently truncate the prompt or reduce the requested
steps. Generation starts a fresh sequence when at least one token is requested;
a zero-token request returns the prompt without changing the current cache.

`runtime.execution_mode` is `native` for floating-point artifacts or
`fake_quantize` for quantized artifacts. Fake-quantize reference execution is a
semantic check, not bit-exact validation of an integer target backend. The
runtime itself does not import Transformers, load a PyTorch model, or fetch
files. The optional [generation CLI](../../examples/README.md#generate-from-a-llama-circle-bundle)
uses only local tokenizer files for prompt encoding/continuation decoding.

Tiny deterministic random-model tests exercise floating-point saved-artifact
parity, quantized tensor domains, and cache composition. Quantized reference
execution does not promise PyTorch fake-quantization parity: a directly quantized
embedding carries independent table/output qparams, while reference GATHER
forwards table values without applying the output grid. See the
[quantized Circle contract and reference limitation](../../../../docs/design.md#102-conversion-time-quantized-graph-legalization).
Meaningful language output requires pretrained weights; model quality and
target-device correctness and performance need their own evaluation.

A file-by-file description of a complete Gemma4 E2B quantized export is kept in
[`gemma4_artifacts.md`](gemma4_artifacts.md).

### Gemma4 external-embedding text logits

The composite text-logits adapter consumes external embeddings and keeps
vision, fusion, and PLE preparation separate. See the
[external-embedding contract](gemma4_artifacts.md#external-embedding-text-logits)
for its inputs, calibration requirements, and shared-KV limitation.

### Gemma4 Per-Layer Embedding (PLE) artifacts

Gemma4 models with PLE export a token lookup plus projection/combine stages.
The [PLE artifact contract](gemma4_artifacts.md#per-layer-embedding-ple-contract)
owns the format policy, tensor shapes, frozen observer boundaries, and the
prefill-only variant.

### Gemma4 split vision artifacts

`export.vision.granularity` selects monolithic, per-layer, or both vision forms.
See the [vision artifact contract](gemma4_artifacts.md#circle_layers-vision-pipeline)
for the stage files, runtime context, manifest, and profile ownership rules.

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
    max_seq_len: int | None = None,
    config: CompileConfigBase | None = None,
) -> Path:
    ...
```

`export_full_circle()` returns a `Path` so callers can store it in
`ctx.artifacts` or assert it in tests. Multi-artifact writers such as
`export_llama_per_layer()` return `None` and write into the supplied output
directory; callers should inspect their documented artifacts there.

For very simple helpers, positional arguments are acceptable:

```python
def save_checkpoint(
    model: Any,
    output_dir: str | Path,
    name: str = "quantized_model.pt",
) -> Path:
    ...
```

## Adding a new export target

1. Add a helper under `recipes/export/`.
2. Keep target-specific imports local if they are optional or heavy.
3. Document the output paths and return value; return a `Path` for a single-artifact helper.
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

Example inside the LLaMA adapter, after validating the export profile:

```python
from pathlib import Path

from tico.quantization.recipes.export.circle import resolve_compile_config
from tico.quantization.recipes.export.llama import export_llama_per_layer
from tico.quantization.recipes.sequence_length import resolve_export_max_seq_len

export_cfg = ctx.cfg.get("export", {})
output_dir = Path(export_cfg.get("output_dir", "./out/llama"))
artifacts = set(export_cfg.get("artifacts", []))

if "circle_per_layer" in artifacts:
    export_llama_per_layer(
        q_model=ctx.model,
        max_seq_len=resolve_export_max_seq_len(ctx),
        output_dir=output_dir,
        prefill_decode=bool(export_cfg.get("prefill_decode", False)),
        append_prefill_buckets=export_cfg.get("append_prefill_buckets"),
        compile_config=resolve_compile_config(export_cfg),
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
paths when the helper's API provides them, or let the adapter record the known
output locations in `ctx.artifacts` if needed.

## Error handling

Fail early when the requested artifact lacks a valid static profile, compatible
example inputs, or the required frozen quantization state. Use
`resolve_export_max_seq_len(ctx)` for text capacity and the model-specific
profile validators for vision geometry. Export builds shape examples from
these profiles; it does not require `ctx.calibration_inputs`. A checkpoint
missing required frozen qparams must be recalibrated through the quantization
workflow before export. Prefer clear errors over partially written artifacts.

## Testing checklist

For a new exporter, test:

- output directory creation;
- the documented return value and presence of the written artifacts;
- existing output directory behavior;
- CPU-only execution when possible;
- missing example input behavior;
- invalid artifact key behavior at the adapter/config level;
- that the exporter does not run evaluation or quantization implicitly.

## Complete fixed-image Qwen3-VL prefill

`Qwen3VLModelPrefillExportAdapter` composes the existing fixed-grid vision,
fixed-span fusion, decoder and DeepStack adapters into one complete graph. It
accepts `[1, S]` token IDs and patchified pixels for one fixed image profile,
and returns normalized hidden states for `QuantQwen3VLModel` or vocabulary logits
for `QuantQwen3VLForConditionalGeneration`. The prompt must contain the profile's
contiguous image-token span at the configured `visual_start_idx`. This path is
cache-free prefill; external cache generation continues to use the staged exporter.

Construct and capture this adapter inside `prepared_static_export(model, S)`.
That context supplies the configured fixed-grid mRoPE positions, causal mask and
RoPE tables, then removes its temporary source buffers on exit, including errors.
The adapter reuses the actual vision, text and optional LM-head weights and
preserves the full wrapper's multimodal, external-embedding and DeepStack frozen
observer boundaries. Calibrate those real paths before conversion. No source
observer is recalibrated during preparation or export.

The adapter aligns temporary mask and RoPE operands to the actual hidden tensor's
dtype and device before fake quantization. Prepared static buffers keep their
original dtype and values, and position IDs remain integer tensors. CPU BF16
activations still require FP32 per-channel scales for enabled fake quantization;
see the [execution compatibility contract](../../wrapq/README.md). Successful BF16
adapter execution does not establish BF16 Circle serialization/runtime support.
