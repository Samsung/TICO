# PTQ override policies

PTQ YAML stages support selector-based `override_policies` for mixed activation
and parameter quantization without editing model-specific Python code. The base
PTQ fields define the default policy:

```yaml
pipeline:
  - name: ptq
    enabled: true
    activation: int16
    linear_weight: uint8
```

Use `specs` to define reusable quantization specs, then reference them from
policies:

```yaml
pipeline:
  - name: ptq
    enabled: true
    activation: int16
    linear_weight: uint8

    specs:
      mx_fp8_act:
        kind: mx
        elem_format: fp8_e4m3
        axis: -1
      int16_act:
        kind: affine
        dtype: int16

    override_policies:
      - name: all_text_linear_activations_mx
        target:
          component: text
          layers: all
          op_type: linear
          observer_role: activation
        spec: mx_fp8_act

      - name: layer_7_down_proj_output_int16
        target:
          component: text
          layers: [7]
          module: mlp.down_proj
          observers: [act_out]
        spec: int16_act
```

For advanced cases, use `raw_overrides` with exact internal PTQ paths. Raw
overrides are applied after selector policies.

```yaml
raw_overrides:
  model.layers.7.mlp.down_proj.act_out: int16
```

## Target fields

Supported target fields are:

```yaml
target:
  component: text | vision | all
  layers: all | 7 | [0, 1, 7] | "0:8" | "0:32:2"
  module: mlp.down_proj
  modules: [self_attn.q_proj, mlp.down_proj]
  op_type: linear
  observer_role: activation | parameter | input_activation | output_activation
  observers: [act_in, act_out]
```

Use only one of `module`, `modules`, or `op_type` in each policy. Use only one
of `observers` or `observer_role` in each policy.

`component: text` is available for text-only LLMs and multimodal models.
`component: vision` is available for supported multimodal models. Use
`component: all` to apply the same selector to every available component.

The components, layer paths, layer counts, and `op_type` aliases of a model
family are declared by its recipe adapter through
`build_quant_target_context()`; see "PTQ selector override structure" in
[`../../recipes/adapters/README.md`](../../recipes/adapters/README.md). Built-in
declarations exist for `llama`, `qwen3_vl`, and `gemma4`; a family without a
declaration rejects `override_policies` and `raw_overrides` with an error.

## Precedence

Policies are resolved deterministically:

1. The base PTQ policy applies first.
2. Model-family default override trees apply next.
3. Selector-based `override_policies` apply after defaults.
4. More specific selector policies override broader selector policies.
5. Policies with the same specificity are resolved in YAML order, where later
   policies win.
6. `raw_overrides` are applied last.

`allow_empty` controls whether a selector policy is allowed to match no
targets. It defaults to `false`.

This is mainly useful for shared configs that may target optional model
components. For example, a config shared by text-only LLMs and VLMs may include
a `component: vision` policy with `allow_empty: true`.

Do not use `allow_empty: true` for required overrides, because it can hide
configuration mistakes.

## Gemma4 INT16 cache with MX attention

Keep `activation: int16` for the residual stream and producer `new_k`/`new_v`
storage boundaries, then override the attention operands:

```yaml
override_policies:
  - target:
      component: text
      layers: all
      module: self_attn
      observers: [softmax, attn_weights]
    spec: {kind: mx, elem_format: int8, axis: -1}
  - target:
      component: text
      layers: all
      module: self_attn
      observers: [present_value]
    spec: {kind: mx, elem_format: int8, axis: -2}
```

The probability key axis (`-1`) and full-V sequence axis (`-2`) are the
`P @ V` reduction dimensions in both attention layouts. The result and residual
stream retain the base INT16 policy. `present_value` already owns the full
attention V boundary; no additional `attn_value` observer is needed.

`past_value` remains an independent per-tensor affine read observer. During
cached decode, only existing tokens pass through that observer. The compute
branch converts current V to the read grid, concatenates it with read V, and
applies MX once to the complete sequence. Quantizing partial past/current MX
blocks before concatenation would change their shared exponents. Producer
storage, returned deltas, full cache returns, and shared-KV storage keep the
frozen `new_*` grid.

This MX compute path supports INT8 elements on the sequence axis with the
default max shared-exponent and nearest-rounding policy, and per-tensor
UINT8/INT16 affine storage/read grids. MX persistent storage and MX `past_*`
read policies remain unsupported. Use positive cached decode calibration to
prepare affine read statistics; an empty-cache prefill never supplies them.
See the [Gemma4 KV cache contract](../../recipes/export/gemma4_artifacts.md#kv-cache-boundaries)
for binding, cache returns, and state loading.

## Diagnostics

Set `print_overrides: true` on the PTQ stage to print the normalized final
values only for observer paths targeted by enabled `override_policies` or
`raw_overrides`. Values are read after all precedence rules have been applied,
so each targeted path is printed once with the final value passed to
`prepare()`. Model-family defaults and unrelated adapter override paths are not
printed. The internal `__quant_spec_replace_role__` merge marker is omitted.

Set `print_model: true` to print the converted model immediately after the PTQ
`convert()` call.

```yaml
pipeline:
  - name: ptq
    enabled: true
    print_overrides: true
    print_model: true
```

Both options default to `false`.
