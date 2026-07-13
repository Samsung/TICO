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
