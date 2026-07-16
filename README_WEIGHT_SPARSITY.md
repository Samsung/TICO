# TICO post-convert weight sparsity

This patch adds a `weight_sparsity` recipe stage for measuring affine-quantized
weights immediately after the PTQ `convert()` stage. It does not load or inspect
Circle artifacts.

The implementation targets:

- `meta-llama/Llama-3.2-3B-Instruct`
- `Qwen/Qwen3-VL-4B-Instruct`

Two reports are produced from a single scan of the quantized weights.

The summary report contains exactly three columns:

```text
Scope | Qdtype | Sparsity (%)
```

The layer-level report adds the normalized layer path:

```text
Layer | Scope | Qdtype | Sparsity (%)
```

`Qscheme` is used internally to interpret per-tensor and per-channel qparams, but
it is intentionally not emitted in either report.

## Base revision

The patch was prepared against Samsung/TICO `main` at:

```text
ef42e2d2323187fc41406e1787b1d840aa316f9a
```

## Added functionality

The analyzer:

- visits direct TICO weight observers on leaf quantization wrappers;
- verifies that each measured wrapper is in post-convert `QUANT` mode;
- skips `IdentityObserver` weights because they remain floating point;
- reconstructs affine integer codes in bounded chunks;
- counts semantic zeros using `qcode == zero_point`;
- aggregates by total zero count and total element count, not by an unweighted
  average of tensor-level percentages;
- deduplicates shared weights only when storage, qdtype, and finalized qparams
  are identical;
- emits projection-level scopes for Llama and Qwen3-VL;
- emits an overall row and projection rows for each normalized model layer;
- produces summary and layer reports from the same collected tensor statistics;
- writes CSV and Markdown reports without a Qscheme column.

## Apply the patch

From a TICO checkout at the base revision:

```bash
git checkout ef42e2d2323187fc41406e1787b1d840aa316f9a
git apply /path/to/tico_weight_sparsity.patch
```

Alternatively, copy the `tico/` and `test/` directories from this archive over a
TICO checkout.

## Run the tests

```bash
python -m unittest \
  test.quantization.analysis.test_weight_sparsity \
  test.quantization.recipes.stages.test_weight_sparsity
```

## Run Llama 3.2 3B

The Meta model is gated, so configure a Hugging Face token through the normal
TICO/Hugging Face authentication path before running.

```bash
python -m tico.quantization.examples.quantize \
  --config tico/quantization/examples/configs/llama32_3b_weight_sparsity.yaml
```

Generated files:

```text
./out/llama32_3b_weight_sparsity/weight_sparsity.csv
./out/llama32_3b_weight_sparsity/weight_sparsity.md
./out/llama32_3b_weight_sparsity/weight_sparsity_by_layer.csv
./out/llama32_3b_weight_sparsity/weight_sparsity_by_layer.md
```

Llama summary scopes are ordered as follows when present:

```text
All quantized weights
Token embedding
Attention / q_proj
Attention / k_proj
Attention / v_proj
Attention / o_proj
MLP / gate_proj
MLP / up_proj
MLP / down_proj
LM head
Norm weights
SpinQuant rotation weights
Unclassified quantized weights
```

The layer report uses normalized paths such as:

```text
model.embed_tokens
model.rotate_embedding
model.layers.0
model.layers.1
...
model.norm
rotate_lm_head
lm_head
```

For each decoder layer, the report contains an `All quantized weights` row and
projection-level rows for Q/K/V/O and gate/up/down projections. Norm weights are
reported in the same decoder-layer group.

## Run Qwen3-VL 4B

```bash
python -m tico.quantization.examples.quantize \
  --config tico/quantization/examples/configs/qwen3_vl_4b_weight_sparsity.yaml
```

Generated files:

```text
./out/qwen3_vl_4b_weight_sparsity/weight_sparsity.csv
./out/qwen3_vl_4b_weight_sparsity/weight_sparsity.md
./out/qwen3_vl_4b_weight_sparsity/weight_sparsity_by_layer.csv
./out/qwen3_vl_4b_weight_sparsity/weight_sparsity_by_layer.md
```

Qwen3-VL summary scopes are ordered as follows when present:

```text
All quantized weights
Vision / patch_embed.proj
Vision attention / qkv
Vision attention / proj
Vision MLP / linear_fc1
Vision MLP / linear_fc2
Vision merger / linear_fc1
Vision merger / linear_fc2
Deepstack merger / linear_fc1
Deepstack merger / linear_fc2
Text / token embedding
Text attention / q_proj
Text attention / k_proj
Text attention / v_proj
Text attention / o_proj
Text MLP / gate_proj
Text MLP / up_proj
Text MLP / down_proj
LM head
Norm weights
SpinQuant rotation weights
Unclassified quantized weights
```

The Qwen3-VL layer report separates vision and text layers with normalized paths:

```text
model.visual.patch_embed
model.visual.blocks.0
model.visual.blocks.1
...
model.visual.merger
model.visual.deepstack_merger_list.0
...
model.language_model.embed_tokens
model.language_model.rotate_embedding
model.language_model.layers.0
model.language_model.layers.1
...
model.language_model.norm
rotate_lm_head
lm_head
```

## Pipeline placement

The stage must immediately follow `ptq`:

```yaml
pipeline:
  - name: gptq
    enabled: true
    # ...

  - name: ptq
    enabled: true
    # ...

  - name: weight_sparsity
    enabled: true
    output_dir: ./out/model_weight_sparsity
    filename_stem: weight_sparsity
    layer_filename_stem: weight_sparsity_by_layer
    formats: [csv, markdown]
    precision: 6
    max_chunk_elements: 4194304
    deduplicate_shared_weights: true
    include_empty_scopes: false
    include_layer_report: true
    include_layer_totals: true
    print_layer_report: false
```

Configuration fields for the layer report:

- `include_layer_report`: Generate the layer CSV and Markdown files. Default: `true`.
- `include_layer_totals`: Add `All quantized weights` within every layer. Default: `true`.
- `layer_filename_stem`: Set the layer report filename stem. The default is
  `<filename_stem>_by_layer`.
- `print_layer_report`: Print the full layer table to stdout. When an output
  directory exists, the default is `false` to avoid large console output.

If the stage runs before PTQ conversion, it raises an error instead of reporting
raw floating-point zero counts.

## Programmatic use

```python
from tico.quantization import convert
from tico.quantization.analysis import (
    format_layer_weight_sparsity_table,
    format_weight_sparsity_table,
    measure_weight_sparsity_report,
)

quantized_model = convert(prepared_model, ptq_config)
report = measure_weight_sparsity_report(quantized_model, family="llama")

print(format_weight_sparsity_table(report.summary_rows))
print(format_layer_weight_sparsity_table(report.layer_rows))
```

Use `family="qwen3_vl"` for Qwen3-VL.

## Sparsity definition

For an affine-quantized weight element, the integer code is reconstructed as:

```text
qcode = clamp(round(weight / scale) + zero_point, qmin, qmax)
```

The element is counted as zero when:

```text
qcode == zero_point
```

A scope percentage is calculated as:

```text
sum(scope semantic-zero counts) / sum(scope element counts) * 100
```

The same weighted calculation is used for each layer total and each
layer-projection row.

Shared weights are deduplicated inside each individual row. Because the same
shared weight may be intentionally visible under more than one logical layer,
layer rows should not be summed to reconstruct the global total. Use the summary
`All quantized weights` row for the globally deduplicated value.

This is the effective quantized-value sparsity at Torch level after conversion.
