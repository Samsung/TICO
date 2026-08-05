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

### Llama 3.2-3B width and static-runtime Circle export

Llama cases default to `tiny`. The following profiles are available for every
registered Llama module case:

- `llama3_2_3b_dims` uses the original 3B MLP, hidden, and attention dimensions
  with smoke-sized inputs.
- `llama3_2_3b_static_runtime` uses the same dimensions with batch one and a
  fixed 2,048-token prefill/decode capacity.

Both profiles keep one synthetic layer and deterministic random weights. The
checkpoint's 131,072-token context limit is not copied into
`max_position_embeddings`, because the attention wrapper allocates its static
causal-mask template from that value. The dimensions profile uses a bounded
16-token capacity, while the static profile uses the configured runtime value.

Supported cases:

```text
llama_mlp
llama_attention_prefill
llama_attention_decode
llama_decoder_layer_prefill
llama_decoder_layer_decode
```

Run one fixed-shape Llama decoder layer:

```bash
python -m tico.quantization.examples.inspector \
  --config tico/quantization/examples/configs/wrapper_smoke.yaml \
  --mode wrapper-smoke \
  --case llama_decoder_layer_prefill \
  --export circle \
  --output-dir ./out/wrapper_smoke/llama3_2_3b_static_runtime \
  --calibration-iters 1 \
  --no-plot \
  --set debug.wrapper_smoke.llama.size_profile=llama3_2_3b_static_runtime
```

The decode profile uses hidden shape `(1, 1, 3072)`, an attention mask with
2,048 key slots, and K/V inputs with shape `(1, 8, 2047, 128)` by default.
Override the capacity with:

```bash
--set debug.wrapper_smoke.llama.static_runtime.max_seq=1024
```

### Qwen3-VL-4B width and static-runtime Circle export

Qwen3-VL cases also default to `tiny`:

- `qwen3_vl_4b_dims` uses the original 4B text and vision channel dimensions
  with smoke-sized inputs.
- `qwen3_vl_4b_static_runtime` uses those dimensions with the fixed TICO text
  and image runtime contract.

The bounded text cases keep one decoder layer. Bounded vision cases keep one
vision block while preserving the original patch, merger, attention, and MLP
widths. No checkpoint is downloaded.

Supported cases:

```text
qwen3_vl_text_attention_prefill
qwen3_vl_text_attention_decode
qwen3_vl_text_mlp
qwen3_vl_text_decoder_layer_prefill
qwen3_vl_text_decoder_layer_decode
qwen3_vl_vision_attention
qwen3_vl_vision_mlp
qwen3_vl_vision_block
qwen3_vl_vision_patch_embed
qwen3_vl_vision_patch_merger
qwen3_vl_vision_model
```

The default static contract is:

```text
Text prefill hidden           : (1, 2048, 2560)
Text decode hidden            : (1, 1, 2560)
Text decode K/V               : (1, 8, 2047, 128)
Vision grid_thw               : (1, 54, 72)
Vision patch tokens           : 3888
Merged visual tokens          : 972
Non-visual tokens             : 14
Logical valid sequence        : 986
Reserved visual capacity      : 1000
Physical visual arena start   : 1048
Visual segment start          : 4
```

Run one static Qwen text decoder layer:

```bash
python -m tico.quantization.examples.inspector \
  --config tico/quantization/examples/configs/wrapper_smoke.yaml \
  --mode wrapper-smoke \
  --case qwen3_vl_text_decoder_layer_prefill \
  --export circle \
  --output-dir ./out/wrapper_smoke/qwen3_vl_4b_static_runtime \
  --calibration-iters 1 \
  --no-plot \
  --set debug.wrapper_smoke.qwen3_vl.size_profile=qwen3_vl_4b_static_runtime
```

Run the one-layer vision model with the same fixed image grid:

```bash
python -m tico.quantization.examples.inspector \
  --config tico/quantization/examples/configs/wrapper_smoke.yaml \
  --mode wrapper-smoke \
  --case qwen3_vl_vision_model \
  --export circle \
  --output-dir ./out/wrapper_smoke/qwen3_vl_4b_static_runtime \
  --calibration-iters 1 \
  --no-plot \
  --set debug.wrapper_smoke.qwen3_vl.size_profile=qwen3_vl_4b_static_runtime
```

Static inputs can be overridden together:

```bash
--set debug.wrapper_smoke.qwen3_vl.static_runtime.max_seq=1024
--set debug.wrapper_smoke.qwen3_vl.static_runtime.grid_thw=[1,32,32]
--set debug.wrapper_smoke.qwen3_vl.static_runtime.visual_capacity=256
--set debug.wrapper_smoke.qwen3_vl.static_runtime.non_visual_tokens=14
--set debug.wrapper_smoke.qwen3_vl.static_runtime.visual_start_idx=4
```

The grid height and width must be divisible by the spatial merge size. The
merged visual tokens must fit both `visual_capacity` and `max_seq`.

The following embedding/full-model cases reject both 4B profiles before model
allocation:

```text
qwen3_vl_text_model
qwen3_vl_model
qwen3_vl_for_conditional_generation
```

Use the regular Qwen3-VL quantize/export recipe when full text depth, vocabulary
embeddings, LM head, or end-to-end multimodal execution is required.

### Gemma4 E2B-width and static-runtime Circle export

Gemma4 cases default to the existing `tiny` profile. Two larger profiles are
available for bounded wrapper cases:

- `e2b_dims` uses the original E2B channel, projection, patch, and position-table
  dimensions while retaining smoke-sized input tensors.
- `e2b_static_runtime` uses the same E2B dimensions and replaces the smoke input
  shapes with the fixed runtime contract used by the static Gemma4 path.

Both profiles retain a one- or two-layer case topology and deterministically
random weights. They do not download a Hugging Face checkpoint or construct the
full text, vocabulary, PLE-table, or multimodal model.

Supported `e2b_dims` cases:

```text
gemma4_text_mlp
gemma4_text_attention
gemma4_text_attention_sliding
gemma4_text_attention_k_eq_v
gemma4_text_attention_shared_kv
gemma4_text_decoder_layer_prefill
gemma4_text_decoder_layer_sliding_prefill
gemma4_text_decoder_layer_decode
gemma4_text_decoder_layer_shared_kv
gemma4_vision_attention
gemma4_vision_encoder_layer
gemma4_vision_encoder
gemma4_vision_patch_embedder
gemma4_vision_pooler
gemma4_vision_model
gemma4_multimodal_embedder
```

`e2b_static_runtime` supports the same bounded cases except
`gemma4_text_attention_k_eq_v`, which is a synthetic alternative-attention
branch rather than the E2B runtime configuration.

The default static-runtime shape is:

```text
Text prefill                  : (1, 2048, 1536)
Text decode hidden            : (1, 1, 1536)
Text decode K/V capacity      : 2048 tokens
Sliding window                : 512 tokens
Per-layer input (PLE)         : (1, S, 256)
Vision patch slots            : 2520
Vision valid patches          : 2304 = 48 x 48
Vision padding patches        : 216
Pooler output slots           : 280
Valid visual tokens           : 256 = 16 x 16
Vision hidden size            : 768
```

The vision layout matches Gemma4 processor semantics for the default
`max_soft_tokens=280` and `pooling_kernel_size=3`: valid position IDs are
followed by `(-1, -1)` padding slots, and the pooler removes 24 padded output
slots after producing 280 fixed slots.

Run one original-width module with smoke-sized inputs:

```bash
python -m tico.quantization.examples.inspector \
  --config tico/quantization/examples/configs/wrapper_smoke.yaml \
  --mode wrapper-smoke \
  --case gemma4_text_decoder_layer_prefill \
  --export circle \
  --output-dir ./out/wrapper_smoke/gemma4_e2b_dims \
  --calibration-iters 1 \
  --no-plot \
  --set debug.wrapper_smoke.gemma4.size_profile=e2b_dims
```

Run the same case with the fixed E2B runtime shape:

```bash
python -m tico.quantization.examples.inspector \
  --config tico/quantization/examples/configs/wrapper_smoke.yaml \
  --mode wrapper-smoke \
  --case gemma4_text_decoder_layer_prefill \
  --export circle \
  --output-dir ./out/wrapper_smoke/gemma4_e2b_static_runtime \
  --calibration-iters 1 \
  --no-plot \
  --set debug.wrapper_smoke.gemma4.size_profile=e2b_static_runtime
```

The resulting filenames include the profile:

```text
gemma4_text_decoder_layer_prefill.e2b_dims.q.circle
gemma4_text_decoder_layer_prefill.e2b_static_runtime.q.circle
```

Static dimensions can be overridden for targeted debugging:

```bash
--set debug.wrapper_smoke.gemma4.static_runtime.max_seq=1024
--set debug.wrapper_smoke.gemma4.static_runtime.num_visual_tokens=256
--set debug.wrapper_smoke.gemma4.static_runtime.max_soft_tokens=280
```

`num_visual_tokens` must form a square grid, and `max_soft_tokens` must use a
Gemma4 processor-supported budget. The full default shapes intentionally create
large attention tensors, so use one calibration iteration and disable plotting
when the goal is Circle/compiler validation.

The following composite or vocabulary-sized cases reject both E2B profiles
before allocating a model:

```text
gemma4_text_scaled_word_embedding
gemma4_text_model
gemma4_model
gemma4_for_conditional_generation
gemma4_for_causal_lm
```

Use the regular Gemma4 quantize/export recipe when full layer depth, the full
vocabulary/PLE tables, pretrained weights, or end-to-end model evaluation is
required.

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
