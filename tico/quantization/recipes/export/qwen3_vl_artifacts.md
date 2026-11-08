# Qwen3-VL artifact and static-profile contracts

This guide owns the Qwen3-VL stage filenames, fixed vision profiles, host
responsibilities, and complete fixed-image prefill contract. For the shared
exporter lifecycle and execution preflight, see the
[export developer guide](README.md). Use the
[export CLI guide](../../examples/README.md#export) for runnable commands.

## Contents

- [Artifact naming](#artifact-naming)
- [Complete fixed-image Qwen3-VL prefill](#complete-fixed-image-qwen3-vl-prefill)

## Artifact naming

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
