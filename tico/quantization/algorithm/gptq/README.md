## gptq

GPTQ is a post-training, weight-only quantization algorithm. It quantizes weights
column-by-column using second-order (Hessian) information collected from calibration
data, compensating the not-yet-quantized columns for the error introduced at each
step. Activations are left untouched. The updated weights remain floating-point
tensors; see [Precautions](#precautions) for the subsequent quantization step.

The classic `GPTQQuantizer` supports `nn.Linear`, `nn.Conv1d`, `nn.Conv2d`, `nn.Conv3d`,
and `nn.ConvTranspose2d` modules, and targets LLaMA-like decoder architectures
(models exposing `model.layers` as an `nn.ModuleList`). Models without that
structure are treated as a single layer (fallback). For replay and caching without
that decoder layout, see [Universal GPTQ](#universal-gptq).

### How it works

The quantizer follows the standard `prepare()` → calibration → `convert()` flow of
the public interface:

1. **`prepare(model, config)`** — patches the forward of the first decoder layer
   with a "catcher" that records the layer's inputs (args/kwargs) and stops the
   forward pass immediately. No quantization happens here.
2. **Calibration** — you run the model on any number of batches. Each call caches
   one batch of first-layer inputs (moved to CPU); the rest of the model is never
   executed, so calibration is cheap.
3. **`convert(model)`** — restores the original forwards, then walks the decoder
   layers sequentially. For each layer it:
   - collects input/output statistics for every quantizable submodule via hooks,
   - builds the Hessian approximation and runs `fasterquant()` to update the weights,
   - re-runs the layer on the cached inputs to produce inputs for the next layer.

   If `quantize_lm_head=True`, the output head is quantized last using the final
   hidden states (after `model.norm`).

Statistics hooks installed by GPTQ are removed even if registration, device movement,
or replay fails. Existing caller-owned hooks remain installed and the original error
propagates. This cleanup does not roll back weights or device placement.

After `convert()`, the per-module `Quantizer` objects (holding scale/zero-point)
are attached to the model as `model.quantizers` (a `dict[str, Quantizer]` keyed by
full module name), so downstream tooling can reuse the exact quantization grid.

### Configuration (`GPTQConfig`)

Pass a `GPTQConfig` instance to `prepare()` so the framework dispatches to the
GPTQ quantizer.

| Field | Default | Description |
|---|---|---|
| `weight_bits` | `8` | Default weight bit-width. |
| `weight_bits_overrides` | `{}` | Per-module bit-width overrides. Keys are matched in order: (1) full module name (`model.layers.0.self_attn.o_proj`), (2) layer-local name (`self_attn.o_proj`), (3) full-name `.`-delimited suffix (`down_proj`); when several suffix keys match, the last one in mapping order wins. |
| `perchannel` | `True` | Per-channel (vs. per-tensor) weight quantization. |
| `symmetric` | `False` | Symmetric quantization grid. |
| `mse` | `None` | MSE-based quantizer tuning; one of `"mse"`, `"smse"`, `"mse_for_gptq"`, `"smse_for_gptq"` (see [MSE](#mse) below). |
| `sensitivity` | `None` | `{module_name: tensor}` sensitivities for the `smse*` modes. |
| `percdamp` | `0.01` | Hessian damping, relative to the mean diagonal. |
| `groupsize` | `-1` | Group size for grouped quantization; `-1` disables grouping. |
| `actorder` | `True` | Process columns in order of decreasing Hessian diagonal (activation order). |
| `static_groups` | `False` | Precompute group quantizers before reordering. |
| `quantize_lm_head` | `False` | Also apply GPTQ to `lm_head`. Off by default: many models tie `lm_head.weight` to the input embedding, and quantizing the head would modify the shared weights. |
| `use_orig_model_inference` | `False` | Keep a float copy of the model and use *its* layers to produce inputs for the next layer during `convert()`. Stabilizes GPTQ for deep models at the cost of one extra model copy in memory. |
| `verbose` | `False` | Print per-module timing/error logs. |
| `show_progress` | `True` | Show tqdm progress bars. |

### How to use GPTQQuantizer

```python
from tico.quantization import prepare, convert
from tico.quantization.config.gptq import GPTQConfig

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Maykeye/TinyLLama-v0")
model = AutoModelForCausalLM.from_pretrained("Maykeye/TinyLLama-v0")
model.eval()

# 1. Prepare: attaches the input catcher; nothing is executed yet.
gptq_config = GPTQConfig(weight_bits=4)
prepare(model, gptq_config, inplace=True)

# 2. Calibration: run as many batches as you like.
#    Each call caches one batch of first-layer inputs and stops early.
dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
for i in range(16):
    input_ids = tokenizer(dataset[i]["text"], return_tensors="pt").input_ids
    if input_ids.numel() == 0:
        continue
    model(input_ids)

# 3. Convert: consumes the cached batches and applies GPTQ layer by layer.
convert(model, inplace=True)

# Per-module quantizers (scale / zero-point) are available afterwards:
print(model.quantizers.keys())
```

> **Note** The classic `GPTQQuantizer` only supports `inplace=True` in `convert()`.
> A rejected `convert(model, inplace=False)` call preserves the prepared quantizer and cached
> calibration inputs, so you can retry with `inplace=True` without recalibrating.

Mixed precision via overrides:

```python
gptq_config = GPTQConfig(
    weight_bits=4,
    weight_bits_overrides={
        "down_proj": 8,                          # suffix match
        "model.layers.0.self_attn.o_proj": 8,    # exact match
    },
)
```

### Universal GPTQ

`UniversalGPTQConfig` selects the replay and caching implementation in
[`universal_gptq/quantizer.py`](../universal_gptq/quantizer.py). It discovers module
execution order from calibration inputs instead of requiring a `model.layers`
decoder layout. **Universal describes this orchestration, not a different GPTQ
numerical core:** the default still uses classic GPTQ v1; a compatible factory can
select the GPTQv2 core.

#### Lifecycle and a small CPU example

1. `prepare(model, config)` returns the prepared model and replaces its root
   `forward` with an input recorder. No model execution or Hessian accumulation
   happens at this point.
2. Call the prepared model with representative batches. Each call records its
   positional and keyword arguments on CPU and **returns `None`**. The optional
   `args` / `kwargs` passed to `prepare()` do not populate this cache; collect at
   least one batch by calling the prepared model.
3. `convert(prepared)` restores the root forward, analyzes calls on the cached
   inputs, and repeatedly replays the model. It collects inputs for modules whose
   predecessors are ready, accumulates their Hessians, quantizes their weights,
   and optionally caches outputs for later replay. Statistics are therefore
   collected during conversion, not during the input-recording calls.

This example uses only synthetic data and runs on CPU:

```python
import torch
from torch import nn

from tico.quantization import convert, prepare
from tico.quantization.config.gptq import UniversalGPTQConfig


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))

    def forward(self, x):
        return self.block(x)


torch.manual_seed(0)
model = TinyMLP().eval()
config = UniversalGPTQConfig(
    weight_bits=4,
    cacheable_modules=[r"block"],
    show_progress=False,
)

with torch.no_grad():
    prepared = prepare(model, config, inplace=True)
    for _ in range(4):
        assert prepared(torch.randn(16, 8)) is None
    quantized = convert(prepared, inplace=True)
    output = quantized(torch.randn(2, 8))

assert output.shape == (2, 4) and torch.isfinite(output).all()
assert set(quantized.quantizers) == {"block.0", "block.2"}
assert quantized.block[0].weight.is_floating_point()
assert not hasattr(quantized, "tico_quantizer")
```

Successful conversion restores module forwards and attaches `model.quantizers`,
keyed by full module name. With `debug_mode=False`, it also removes the internal
module replay state and its buffers. As with classic GPTQ, weights remain float
values on a quantization grid; a later PTQ stage can reuse the qparams. See the
[GPTQ-to-PTQ handoff](../../examples/configs/README.md#gptq-to-ptq-weight-qparam-reuse).

#### Replay and caching options

These fields extend the [shared GPTQ options](#configuration-gptqconfig):

| Field | Default | Universal behavior |
|---|---|---|
| `cacheable_modules` | `[]` | Regexes matched with `fullmatch` against non-root names from `model.named_modules()`. For example, `r"model\.layers\.[0-9]+"` selects whole decoder blocks. Cached outputs are kept on CPU to avoid recomputing those modules during replay. |
| `allow_calls_between_cacheable_modules` | `True` | Allows a cacheable parent to call cacheable descendants; completed parent caches can release descendant caches. Calls to a cacheable module outside the caller's subtree are still rejected. With `False`, all cacheable-to-cacheable calls are rejected, including nested parent/child calls. |
| `ignore_multi_call_modules` | `True` | Excludes a module from GPTQ when calibration analysis observes it being called more than once within any single batch. With `False`, shared modules participate, but dependency cycles can force quantization from only the inputs collected so far; verify the result for the model. A later PTQ stage may quantize skipped modules. |
| `debug_mode` | `False` | Retains `gptq_data`, including module replay buffers, after conversion when enabled. This is for inspection and can retain substantial memory. |
| `collect_native_inputs` | `False` | Adds paired native and quantized replay for a GPTQv2-compatible core. It requires a compatible explicit factory and `allow_calls_between_cacheable_modules=True`, and forces caching of quantizable modules. |

Inherited options have a few Universal-specific details:

- `use_orig_model_inference=True` is rejected. Native replay for GPTQv2 is
  controlled by `collect_native_inputs`, not by this classic GPTQ option.
- `weight_bits_overrides` first checks the exact full name, then its final path
  component (for example `o_proj`), then dot-delimited suffixes. The last matching
  suffix in mapping order wins. Sensitivity dictionaries use exact full names.
- `quantize_lm_head=False` excludes full paths matching `lm_head.*`, such as
  `lm_head` and `lm_head.proj`; this is not a filter for every nested head name.
  Check shared embedding/head weights before enabling head quantization.

#### Choosing the GPTQ core and precision

`gptq_factory` accepts three forms:

| Value | Core construction |
|---|---|
| `None` (default) | Constructs classic `gptq.gptq.GPTQ(layer)`. It does not consume the optional precision, normalization, or GPTQv2 constructor settings below. |
| Dotted class-path string | Imports the class and passes `layer`, plus any explicitly accepted constructor parameters among `normalize_H`, `hessian_dtype`, `inp_dtype`, and `gptq_v2_alpha`. |
| Callable | Calls the factory with `layer`. The callable is used unchanged, so it must pass any desired constructor options itself. |

For the existing GPTQv2 core, use the following configuration in place of `config`
in the CPU example, before `prepare()`:

```python
config = UniversalGPTQConfig(
    weight_bits=4,
    gptq_factory="tico.quantization.algorithm.qwen3_vl_gptq.gptq.GPTQ",
    collect_native_inputs=True,
    allow_calls_between_cacheable_modules=True,
    cacheable_modules=[r"block"],
    hessian_dtype=torch.float64,
    inp_dtype=torch.float32,
    show_progress=False,
)
```

The class path selects the per-layer numerical core; it does not select the
Qwen3-VL model-specific quantizer. Selecting that class without
`collect_native_inputs=True` does not supply the native-input difference needed
for its GPTQv2 correction. Conversely, enabling native inputs with the default or
classic GPTQ v1 factory fails validation. A custom compatible factory must expose
`native_inp` on the core it returns and support the tiny CPU `nn.Linear` probe
used by configuration validation.

For cores accepting these constructor settings, `inp_dtype` controls the input
Gram products and GPTQv2 cross-term products, while `hessian_dtype` controls Hessian
and cross-term storage/accumulation and Hessian factorization. Both default to
`torch.float32` and accept only `torch.float32` or `torch.float64`; they do not set
model weight dtype. The existing GPTQv2 core defaults to `normalize_H=True`
(running normalized statistics) and `gptq_v2_alpha=0.25` (correction strength);
`normalize_H=False` uses sums and an alpha of zero disables the correction.

#### Scope and recipe presets

The current dispatcher selects exact `nn.Linear`, `nn.Conv1d`, `nn.Conv2d`,
`nn.Conv3d`, and `nn.ConvTranspose2d` types. Custom subclasses and functional
weight operations are not automatically quantized. Quantizable modules must
receive their tensor input as the first positional argument, even though the
root model's input recorder accepts keyword arguments. Use evaluation mode and
inputs whose repeated execution is stable; caches, shared modules, and
data-dependent execution must be checked against the model's actual call graph.
The name Universal does not guarantee support for every PyTorch model.

For complete CLI workflows, reuse the
[LLaMA Universal GPTQ preset](../../examples/configs/llama_universal_gptq_quantize.yaml)
or [Qwen3-VL Universal GPTQ preset](../../examples/configs/qwen3_vl_universal_gptq_quantize.yaml)
with the [configuration reference](../../examples/configs/README.md). These presets
select recipe `variant: universal` and include a subsequent PTQ stage. Their
model paths and cache patterns are examples for those model families, not generic
defaults; `variant: universal` alone does not enable GPTQv2.

### `MSE`

The `mse` parameter of `GPTQConfig` tunes the quantizer used inside GPTQ.
There are four options:

#### 1. `mse` — vanilla MSE

Produce quantization parameters for the GPTQ quantizer (`min`/`max`) which
minimize the mean squared error of quantization:

$$
MSE_{MIN, MAX}(W) = \underset{min, max}{argmin} \, ||W-Q_{min, max}(W)||^2
$$

#### 2. `smse` — sensitivity-based MSE

Use sensitivity of some global feature (e.g. float model logits) to parameter
changes to minimize the global effect of quantization:

$$
SMSE_{MIN, MAX}(W) = \underset{min, max}{argmin} \, |(W-Q_{min, max}(W))^2 \cdot Sensitivity(W)|
$$

We try to keep `important` parameters unchanged, while quantizing `unimportant`
parameters more aggressively.

#### 3. `smse_for_gptq` — `smse` adjusted for GPTQ

GPTQ modifies the matrix during the quantization process, so the most accurate
method would consist in finding a quantizer that yields the smallest
quantization error *after* the GPTQ method has been applied:

$$
SMSE\_FOR\_GPTQ_{MIN, MAX}(W) = \underset{min, max}{argmin} \, |(W-Q_{min, max}(W_{GPTQ}))^2 \cdot Sensitivity(W)|
$$

Since this would be quite computationally expensive, we use an accelerated
approximate GPTQ method — FPI_GPTQ:

$$
SMSE\_FOR\_GPTQ_{MIN, MAX}(W) = \underset{min, max}{argmin} \, |(W-Q_{min, max}(W_{FPI\_GPTQ}))^2 \cdot Sensitivity(W)|
$$

This is slower than `mse`/`smse` but can provide better accuracy.

#### 4. `mse_for_gptq` — `mse` adjusted for GPTQ

Minimize the GPTQ objective after quantization:

$$
MSE\_FOR\_GPTQ_{MIN, MAX}(W) = \underset{min, max}{argmin} \, \left|\left|\frac{W_{GPTQ}-Q_{min, max}(W_{GPTQ})}{diag(Hinv)}\right|\right|^2
$$

again approximated with FPI_GPTQ:

$$
MSE\_FOR\_GPTQ_{MIN, MAX}(W) = \underset{min, max}{argmin} \, \left|\left|\frac{W_{FPI\_GPTQ}-Q_{min, max}(W_{FPI\_GPTQ})}{diag(Hinv)}\right|\right|^2
$$

This is slower than `mse`/`smse` but can provide better accuracy. To stabilize
computations you may need to increase `percdamp` to >= 0.1, because
`mse_for_gptq` overfits pretty fast.

Examples:

```python
cfg = GPTQConfig(..., mse="mse")                                     # vanilla mse
cfg = GPTQConfig(..., mse="smse", sensitivity=some_sensitivity)      # sensitivity-weighted
cfg = GPTQConfig(..., mse="mse_for_gptq")                            # GPTQ-adjusted
cfg = GPTQConfig(..., mse="smse_for_gptq", sensitivity=some_sensitivity)
```

`some_sensitivity` is a dictionary of module sensitivities
(`{module_name: module_sensitivity}`). Sensitivities can be computed using
empirical Fisher information (see the `SensitivityCalibrator` util class).

### Precautions

- **Fake quantization.** After `convert()`, weights are updated but remain float
  tensors (they are snapped to the quantization grid). A real quantization step —
  e.g. using `wrapq` or the PTQ stage — must be applied afterwards. If the weight
  quantization applied later differs from GPTQ's internal grid, the benefit of
  GPTQ may be diminished; use `model.quantizers` to keep them consistent.
- **Memory.** Classic GPTQ keeps cached first-layer inputs on CPU, but `convert()` moves
  each batch to the model device per layer. For deep/large models consider
  `use_orig_model_inference=True` for stability, keeping the extra memory cost
  in mind.

### See also

- [`../qwen3_vl_gptq/`](../qwen3_vl_gptq/README.md) — GPTQ specialization for
  Qwen3-VL (stagewise vision + text quantization).
- [`../fpi_gptq/`](../fpi_gptq/README.md) — accelerated approximate GPTQ used by the
  `*_for_gptq` MSE modes.
