## gptq

GPTQ is a post-training, weight-only quantization algorithm. It quantizes weights
column-by-column using second-order (Hessian) information collected from calibration
data, compensating the not-yet-quantized columns for the error introduced at each
step. Activations are left untouched, so the model's memory footprint shrinks while
accuracy is largely preserved.

This implementation supports `nn.Linear`, `nn.Conv1d`, `nn.Conv2d`, `nn.Conv3d`,
and `nn.ConvTranspose2d` modules, and targets LLaMA-like decoder architectures
(models exposing `model.layers` as an `nn.ModuleList`). Models without that
structure are treated as a single layer (fallback).

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

After `convert()`, the per-module `Quantizer` objects (holding scale/zero-point)
are attached to the model as `model.quantizers` (a `dict[str, Quantizer]` keyed by
full module name), so downstream tooling can reuse the exact quantization grid.

### Configuration (`GPTQConfig`)

Pass a `GPTQConfig` instance to `prepare()` so the framework dispatches to the
GPTQ quantizer.

| Field | Default | Description |
|---|---|---|
| `weight_bits` | `8` | Default weight bit-width. |
| `weight_bits_overrides` | `{}` | Per-module bit-width overrides. Keys are matched in order: (1) full module name (`model.layers.0.self_attn.o_proj`), (2) layer-local name (`self_attn.o_proj`), (3) full-name suffix (`down_proj`). |
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
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
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

> **Note** GPTQ only supports `inplace=True` in `convert()`.

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
- **Memory.** Cached first-layer inputs are kept on CPU, but `convert()` moves
  each batch to the model device per layer. For deep/large models consider
  `use_orig_model_inference=True` for stability, keeping the extra memory cost
  in mind.

### See also

- [`../qwen3_vl_gptq/`](../qwen3_vl_gptq/README.md) — GPTQ specialization for
  Qwen3-VL (stagewise vision + text quantization).
- [`../fpi_gptq/`](../fpi_gptq/README.md) — accelerated approximate GPTQ used by the
  `*_for_gptq` MSE modes.
