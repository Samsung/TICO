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

## MXINT8 linear inputs with INT16 outputs

A policy that selects only `observer_role: input_activation` (`act_in`) of
linear modules with an MX spec keeps their `act_out` in the base affine policy:

```yaml
override_policies:
  - name: text_linear_inputs_mx
    target:
      component: text
      layers: all
      op_type: linear
      observer_role: input_activation
    spec: mx_int8_act
```

With an INT16 base activation, Circle export keeps both calibrated domains.
Each such linear becomes one `FULLY_CONNECTED` that reads the MXINT8
activation and produces the INT16 output, preceded by a `QUANTIZE` from the
INT16 producer. No conversion is inserted after the linear, and this default
path does not need `legalize_mx_affine_domains`. Other mixed linear
signatures, such as an affine input with an MX output or an MXINT8 input with
a UINT8 output, remain rejected on this default path. Whether a backend
accepts this `FULLY_CONNECTED` signature is a separate target-backend check.

## Qwen3-VL MXINT8 deployment policy

The committed
[`qwen3_vl_eval_suite_mx_override_polices.yaml`](qwen3_vl_eval_suite_mx_override_polices.yaml)
preset calibrates a mixed MXINT8/INT16 policy for the `npu_export` profile.
Its broad block-linear MX activation override is followed by explicit affine
output overrides at these existing observation sites:

| Existing observer | Policy and purpose |
|---|---|
| Text `self_attn.{q,k,v}_proj.act_out` | INT16 before head splitting, Q/K RMSNorm, and the shared V storage/compute branch |
| Vision `attn.qkv.act_out` | INT16 before reshape/permute/unbind and RoPE half-splits, including head dimension 32 |
| Text/vision `attn_out` | Base INT16 policy before head and chunk merge |
| Patch merger `norm` output | Base/norm INT16 policy before the view and first FC |
| Text `self_attn.o_proj.act_in`, vision `attn.proj.act_in` | MXINT8 reentry after affine head merge |
| Text `self_attn.mask_add`, vision `attn.logits` | MXINT8 immediate inputs to native MX Softmax |
| Text/vision `softmax` | MXINT8, including the PV operand boundary |

Softmax retains matching MXINT8 input and output domains; configuring only
its output as MX would leave an unsupported affine-to-MX Softmax signature.
Other block-linear activation sites retain MXINT8, so this policy preserves
native MX arithmetic. The `axis: -1` policy continues to mean the logical
feature axis for Q/K/V and the key-token axis for Softmax. The attention
sequence length and head dimension may differ; they must not share a grid
merely because their tensors have the same rank. Vision uses rank-3 per-head
BMM with this trailing-axis MX and scalar-affine attention policy. Rank-sensitive
legacy per-channel or explicitly non-trailing MX policies retain their original
rank-4 attention path; adopting the rank-3 policy requires fresh calibration
rather than reinterpreting frozen axes. KV `new_k`/`new_v` storage
and `present_key`/`present_value` compute observers remain independent.

Circle export of this policy requires both explicit options:

```yaml
export:
  move_affine_quantize_before_layout: true
  legalize_mx_affine_domains: true
```

The first moves eligible existing private affine grids; the second retains
source quantization and adds the native MX operation rounding plus conversion
to the requested affine output. Neither option calibrates an observer. Their
global defaults remain `true` and `false`, respectively.

These observer policy changes apply when preparing and calibrating a new PTQ
model. Reprepare the original float model and recalibrate to adopt them. Loading
an older frozen checkpoint and changing its export YAML does not change its
frozen grids or create the required affine projection outputs. Such a checkpoint
can retain unsupported MX layouts; successful generic Circle verification does
not establish this NPU layout contract. Export never retags its MX quantizers
as INT16 or fabricates an affine scale. No observer sites or checkpoint schema
were added for this policy: existing frozen affine policies continue to load
and use their exact grids, and source quantizers are preserved during export.

The preset retains its evaluation and artifact selection settings. To request
Circle stages after calibration, enable export and select `circle_per_layer`;
for a separate export-only invocation use the saved checkpoint and the same
compile options. See the [export reference](README.md#export) for static profiles
and staged artifact settings. CPU regression tests inspect final Circle layout
and conversion domains with O1 off/on; they do not validate an NPU toolchain or
full-model accuracy.
