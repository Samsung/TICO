# quantization

[Documentation index](../../docs/README.md) · [Guide map](#documentation-map)

## Overview

The _quantization_ module provides the public `prepare` / `convert` API for neural
networks, including LLMs and VLMs. This guide owns the shared API, Python PTQ
configuration, and quantizer registration contract. Algorithm-specific options
and numerical behavior are documented in the
[algorithm guides](./algorithm/README.md); wrapper lifecycle and frozen-state
contracts are documented in [WrapQ](./wrapq/README.md).

- Public Interface
  - `prepare(model, quant_config, args, kwargs, inplace)`
    - Prepares the given model for quantization based on the provided algorithm-specific
      configuration. This involves setting up necessary observers or hooks, and may
      optionally use example inputs (provided as separate positional and keyword
      arguments), which is particularly useful for activation quantization.
  - `convert(model, inplace=True)`
    - Converts the model that has been prepared and calibrated into its quantized form.
      It retrieves the quantizer attached by `prepare()` and uses the existing
      configuration and calibration state. It does not accept `quant_config` or
      select or instantiate a new quantizer. PTQ wrappers retain their resolved
      quantization policies.

- Algorithm-Specific Configuration
  - Users supply algorithm-specific configuration objects (e.g., `PTQConfig` for
    activation quantization or `GPTQConfig` for weight quantization) to `prepare()`.
    These configuration objects encapsulate the parameters for each
    quantization method.

- Internal Dispatch to Quantizer Implementations
  - `prepare()` selects the algorithm-specific quantizer through
    `get_quantizer(quant_config)` and attaches it to the returned prepared model as
    `tico_quantizer`. That quantizer owns its configuration; `convert()` reuses the
    attached instance.

After successful conversion, the returned model no longer carries `tico_quantizer`.
If conversion fails, the original prepared model retains the handle for retry.
Retaining the handle does not roll back arbitrary in-place algorithm mutations.
For algorithms supporting `convert(prepared_model, inplace=False)`, the converted
copy has no handle and the original prepared model keeps its handle even after
success. The classic `GPTQQuantizer` requires `inplace=True`.

## Basic Usage

This small CPU example runs wrapper-based PTQ without a model download or an
external calibration dataset:

```python
import torch
from torch import nn

from tico.quantization import prepare, convert
from tico.quantization.config import PTQConfig

torch.manual_seed(0)
model = nn.Linear(8, 4, bias=False).eval()
prepared_model = prepare(model, PTQConfig(), inplace=False).eval()

with torch.no_grad():
    # Replace these synthetic batches with representative inputs for a real model.
    for _ in range(4):
        prepared_model(torch.randn(2, 8))
    quantized_model = convert(prepared_model)
    example_input = torch.randn(2, 8)
    output = quantized_model(example_input)

assert output.shape == (2, 4)
assert torch.isfinite(output).all()
assert not hasattr(quantized_model, "tico_quantizer")
```

PTQ `prepare()` enables calibration; forward calls collect activation statistics.
`convert()` freezes the required observers and switches the wrappers to fake
quantization. Calibration must cover the execution paths used after conversion;
see [WrapQ freezing and checkpoints](./wrapq/README.md#freezing-and-checkpoints)
for readiness and failure behavior. For weight reconstruction with GPTQ, use the
[GPTQ guide](./algorithm/gptq/README.md).

### Quantization conversion and Circle export

The two `convert` functions have different inputs and results:

| API | Result |
|---|---|
| `tico.quantization.convert(prepared_model)` | A PyTorch module. For PTQ, its wrappers execute fake quantization with frozen qparams. This does not serialize a Circle file. |
| `tico.convert(quantized_model, example_args)` | A `CircleModel` containing the exported and lowered Circle binary. It requires example inputs and supports the operators and quantization policies handled by the converter. |

Continue the example above to export and save its Circle artifact:

```python
import tico

circle_model = tico.convert(quantized_model, (example_input,))
circle_model.save("linear.circle")
```

Exported input/output dtypes can differ from the PyTorch module's floating-point
interface. The [numerical evaluation API](./evaluation/README.md) handles
comparisons with Circle; the [export guide](./recipes/export/README.md) covers
model-specific export stages and artifacts.

## Python PTQ configuration

Import the public configuration types and helpers from
[`tico.quantization.config`](./config/__init__.py). They describe the policies
resolved when `prepare(model, config)` constructs wrappers. Finish configuring
the object before preparation; changing it later does not rebuild existing
observers. YAML recipes and selector-based policies are documented separately in
the [configuration reference](./examples/configs/README.md) and
[PTQ override policies](./examples/configs/ptq_override_policies.md).

### Quantization specifications and roles

`QuantSpec(observer, dtype=None, qscheme=None, kwargs={})` is the policy unit for a
logical observer. Prefer its helpers for common policies:

| Helper | Policy |
|---|---|
| `affine(dtype, *, qscheme=None, observer=MinMaxObserver, **kwargs)` | Affine integer quantization. Pass a `DType` such as `DType.uint(8)` or `DType.int(16)`, an optional `QScheme`, and observer-specific constructor kwargs. In Python, `observer` is a class, not a YAML registry string. |
| `mx(elem_format="int8", *, axis=-1, shared_exp_method="max", round="nearest")` | An `MXObserver` policy with a shared exponent along the selected axis. The helper does not use affine `dtype`/`qscheme` fields. |
| `no_quant()` | An `IdentityObserver` policy that participates in the wrapper lifecycle but collects no statistics and leaves values unchanged. |

`DType` and `QScheme` are imported from
[`wrapq.dtypes`](./wrapq/dtypes.py) and [`wrapq.qscheme`](./wrapq/qscheme.py).
For example, a weight-only PTQ configuration is:

```python
from tico.quantization.config import PTQConfig, affine, no_quant
from tico.quantization.wrapq.dtypes import DType

weight_only = PTQConfig(
    activation=no_quant(),
    weight=affine(DType.uint(8)),
)
```

`PTQConfig.activation` supplies the role policy for activation-like observer names,
including `act_in` and `act_out`; `weight` supplies it for the exact local names
`weight` and `bias`. A role policy affects only observers implemented by a
wrapper: it does not add a bias observer where none exists. Bare `PTQConfig()`
uses `affine(DType.uint(8))` for both roles, with `MinMaxObserver`.

When no higher-priority scheme is supplied, affine inference uses per-tensor
symmetric quantization for signed dtypes, per-channel asymmetric quantization
for unsigned `weight`, and per-tensor asymmetric quantization for other unsigned
observers. Wrappers may supply their own scheme and channel axis. For example,
`QuantLinear` supplies per-channel asymmetric weights with `channel_axis=0`.
An unsigned dtype with a symmetric scheme is rejected. Use an explicit
observer-level override when a particular site's scheme must win over wrapper
defaults.

MX eager execution, Circle serialization, and the chosen Circle runtime have
separate format and operator support. Selecting an MX policy does not by itself
establish that the whole model can be exported or executed on a target backend.

### Observer overrides and precedence

Override paths follow the wrapped model's child scopes and end in a logical
observer name. They do not use internal `wrapped`/`module` attribute prefixes.
For an `nn.Sequential` whose first child is an `nn.Linear`, these two forms
select the same weight observer:

```python
from tico.quantization.config import PTQConfig, affine, no_quant
from tico.quantization.wrapq.dtypes import DType

by_path = PTQConfig(overrides={"0.weight": affine(DType.uint(4))})
by_tree = PTQConfig(overrides={"0": {"weight": affine(DType.uint(4))}})

# Add another site before prepare(); a sequence of path components also works.
by_path.set_override("0.act_out", no_quant())
```

Use the observer names declared by the relevant
[wrapper](./wrapq/README.md#3-wrapper-design), such as `act_in`, `act_out`, and
`weight`. A path for one wrapper family is not automatically valid for another.
These are exact hierarchical paths; Python `PTQConfig.overrides` does not expand
the recipe layer's glob, regex, or semantic selectors.

For each observer field, the precedence is **explicit observer override > wrapper
defaults > role policy**. The override leaf determines how the role participates:

- A `QuantSpec` leaf replaces the role policy entirely. For example,
  `overrides={"0.act_out": mx("int8")}` discards the activation role's affine
  dtype, scheme, and kwargs. Wrapper defaults may still fill omitted fields.
  Use `affine()`, `mx()`, or `no_quant()` when replacing an observer policy.
- A raw kwargs mapping merges with the role and wrapper defaults. For example,
  if the role uses `affine(..., observer=EMAObserver, momentum=0.1)`, then
  `overrides={"0.act_out": {"momentum": 0.2}}` retains the EMA observer and dtype
  while changing its momentum. Merely changing `observer` in a raw mapping does
  not remove incompatible inherited kwargs.

Affine override leaves that specify a dtype also resolve an omitted qscheme
during configuration normalization. Such a resolved scheme is part of the
explicit override; role-level omitted schemes instead allow wrapper defaults
to participate. See [`PTQConfig`](./config/ptq.py) and
[`QuantModuleBase`](./wrapq/wrappers/quant_module_base.py) for the resolution code.

### Wrapping and model metadata

`strict_wrap=True` is the default. A missing wrapper at a boundary selected by
the wrapping helper raises `NotImplementedError`. With `strict_wrap=False`, the
helper can leave unsupported modules in place and recurse into their children.
Errors constructing a registered wrapper still propagate. Strict wrapping
checks wrapper availability at those boundaries; it does not mean every
operation inside a model is quantized or exportable.

`model_args` is a mapping of model-specific metadata read by wrappers, such as an
execution profile or attention geometry. It is separate from both the example
`args`/`kwargs` accepted by the public `prepare()` API and the arguments passed
to the model's `forward()`. The PTQ implementation does not run a forward pass
from `prepare(args=..., kwargs=...)`; call the prepared model with calibration
inputs, as in [Basic Usage](#basic-usage). `attention_mask_fill_value` defaults to
`-120.0` for wrappers that use it to fill masked attention scores.

### Model-specific configuration builders

The public builders return `PTQConfig` objects with generated per-module
overrides. They do not load a model or perform calibration:

| Builder | Required model information | Additional policies |
|---|---|---|
| `build_llm_ptq_config` | `model_type="llama"` and `num_hidden_layers`. Other model types currently raise `NotImplementedError`. | Shared `linear_weight`, `embedding_weight`, `lm_head_weight`, `spin_rotation_weight`, `norm`, and `norm_weight`; `profile` is stored in `model_args`. |
| `build_qwen3_vl_ptq_config` | `num_vision_blocks`, `num_text_layers`, `num_deepstack_mergers`, and `model_args`. | The shared policies plus `text_linear_weight`, `vision_linear_weight`, and `vision_patch_embed_weight`. Text/vision linear policies fall back to `linear_weight` when omitted. |

Both builders default **activation and weight to `affine(DType.int(16))`**,
unlike the unsigned 8-bit defaults of `PTQConfig()` itself. Pass role specs
explicitly when that distinction matters. `norm` selects norm activation
observers; `norm_weight` selects norm parameters. Optional policy arguments set
to `None` leave the inherited/fallback policy in effect; use `no_quant()` for an
explicit bypass. Both builders accept `strict_wrap`.

For example, this constructs the policy for a two-layer Llama model:

```python
from tico.quantization.config import affine, build_llm_ptq_config
from tico.quantization.wrapq.dtypes import DType

llama_config = build_llm_ptq_config(
    model_type="llama",
    num_hidden_layers=2,
    activation=affine(DType.int(16)),
    weight=affine(DType.uint(8)),
    linear_weight=affine(DType.uint(4)),
)
```

Supply counts and metadata from the actual model being prepared. The generated
paths target the supported wrapper hierarchy; a builder is not an architecture
discovery step or a guarantee that conflicting policies for tied parameters are
compatible. See the [builder implementation](./config/builders.py) for the exact
generated sites and the [WrapQ guide](./wrapq/README.md) for model and frozen-state
contracts.

## Directory Structure

```bash
quantization/
├── algorithm/        # Algorithm-specific quantizers
│   ├── adaround/     #   Conv2d weight rounding
│   ├── block_reconstruction/ # Activation-qparam reconstruction
│   ├── cle/          #   Cross-Layer Equalization
│   ├── fpi_gptq/     #   Fixed-point-iteration GPTQ variant
│   ├── gemma4_gptq/  #   GPTQ specialization for Gemma4
│   ├── gptq/         #   GPTQ weight quantization
│   ├── qwen3_vl_gptq/#   GPTQ specialization for Qwen3-VL
│   ├── smoothquant/  #   SmoothQuant
│   ├── spinquant/    #   SpinQuant (rotation-based)
│   └── universal_gptq/ # Replay-based GPTQ with configurable caching
├── analysis/         # Numerical ablation, clipping, and sensitivity
├── config/           # Configuration definitions (PTQConfig, GPTQConfig, ...)
├── evaluation/       # Evaluation utilities
├── examples/         # Config-driven CLI examples (quantize / evaluate / inspect)
├── passes/           # Graph-level passes and transformations
├── recipes/          # Reusable pipeline layer behind the example CLIs
└── wrapq/            # Wrapper-based PTQ infrastructure (observers, wrappers)
```

The module is composed of two main implementation layers:

- Algorithm layer ([`quantization/algorithm/`](./algorithm/README.md))
  - Implements algorithm-specific quantization logic (e.g., GPTQ, SmoothQuant,
    SpinQuant, CLE).
- Infrastructure layer ([`quantization/wrapq/`](./wrapq/README.md))
  - Provides a generic, wrapper-based quantization backend shared across algorithms.

On top of these, two layers make the toolkit usable end-to-end from the command line:

- Recipes ([`quantization/recipes/`](./recipes/README.md))
  - The reusable pipeline layer: model-family adapters, algorithm stages, calibration
    data builders, evaluation, export, and debug helpers.
- Examples ([`quantization/examples/`](./examples/README.md))
  - Thin CLI entry points (`quantize.py`, `evaluate.py`, `inspector.py`) driven by YAML
    presets in [`examples/configs/`](./examples/configs/README.md).

## Documentation Map

| Task | Guide |
|---|---|
| Use the Python `prepare` / `convert` API | [Basic usage](#basic-usage) |
| Configure Python PTQ roles, observers, overrides, and model builders | [Python PTQ configuration](#python-ptq-configuration) |
| Choose a quantization algorithm or model-specific implementation | [Algorithm catalog](./algorithm/README.md#available-algorithms) |
| Quantize weights with GPTQ | [GPTQ guide](./algorithm/gptq/README.md) |
| Apply smoothing, rotations, or equalization | [SmoothQuant](./algorithm/smoothquant/README.md), [SpinQuant](./algorithm/spinquant/README.md), and [CLE](./algorithm/cle/README.md) |
| Understand wrappers, observers, calibration, and frozen state | [WrapQ guide](./wrapq/README.md) |
| Compare PyTorch and exported Circle outputs | [Numerical evaluation API](./evaluation/README.md) |
| Run model benchmarks through recipes | [Recipe evaluation guide](./recipes/evaluation/README.md) |
| Analyze PTQ error, clipping, and sensitivity | [Quantization analysis](./analysis/README.md) |
| Run quantization, evaluation, export, generation, or inspection from the CLI | [CLI examples](./examples/README.md) |
| Configure pipelines and override policies | [YAML configuration reference](./examples/configs/README.md) |
| Export checkpoints and staged Circle artifacts | [Export guide](./recipes/export/README.md) |
| Trace outputs and inspect wrapper or export regressions | [Debug guide](./recipes/debug/README.md) |
| Add adapters, stages, or reusable workflows | [Recipes developer guide](./recipes/README.md) |
| Understand fake-quant lowering, qparam folding, and requantization | [Quantized Circle design](../../docs/design.md#102-conversion-time-quantized-graph-legalization) |

## Contributing: Adding a New Algorithm

We welcome contributions that enhance the module by adding support for new quantization
algorithms. If you are interested in contributing a new algorithm, please follow these
steps:

1. Implement New Configuration and Quantizer Classes

- Configuration
  - Create a new configuration class (e.g., `NewAlgoConfig`) that inherits from
    `BaseConfig`. Include all necessary parameters and default values specific to your
    algorithm.

- Quantizer
  - Implement a new quantizer class (e.g., `NewAlgoQuantizer`) that inherits from
    `BaseQuantizer`. Implement the _prepare_ and _convert_ methods.

2. Register the Quantizer

Decorate the quantizer with `register_quantizer(NewAlgoConfig)` from
`tico/quantization/quantizer_registry.py`. `prepare()` resolves the quantizer by exact
config type; configs whose `name` matches `tico.quantization.algorithm.<name>.quantizer`
are also imported lazily on first use. Registration rules:

- One config type maps to exactly one quantizer class. Re-registering the same class is
  a no-op, so repeated activation of an extension is safe.
- Registering a different quantizer for an already mapped config type raises
  `ValueError`. Do not reuse a built-in config type for a new algorithm; define your own
  config class, including for quantizers shipped in separately installed packages.
- `registered_quantizer(config_cls)` returns the current mapping without triggering the
  lazy import.

3. Write Tests

Ensure that you add unit tests and integration tests for your new algorithm. Tests
should cover both the _prepare_ and _convert_ phases and validate that the quantized
model meets performance expectations.

4. Document Your Changes

Update the README and any related documentation to include details about the new
algorithm, its configuration parameters, and usage examples.

To expose the new algorithm through the CLI examples as a pipeline stage, follow the
[recipes developer guide](./recipes/README.md) — most new workflows should be added as
YAML presets, not new scripts.
