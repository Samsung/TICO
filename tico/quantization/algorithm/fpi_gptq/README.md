## fpi_gptq

FPI-GPTQ (**F**ixed-**P**oint **I**teration GPTQ) is an accelerated approximation of
the GPTQ algorithm. Instead of quantizing weight columns one at a time with
per-column error compensation (as in [`../gptq/`](../gptq/README.md)), it reformulates
the GPTQ update as a fixed-point iteration over the *whole* weight matrix and solves
it with a small number of dense matrix multiplications.

It produces results very close to reference GPTQ but runs much faster on CUDA,
because the sequential column-by-column loop is replaced by batched matrix ops.

### How it works

Reference GPTQ processes columns sequentially: quantize column *i*, then propagate
the scaled error to all remaining columns via the inverse-Hessian factor. FPI-GPTQ
observes that the fully-updated weight matrix is a fixed point of the mapping

```
W_{k+1} = W_0 − ((W_k − Q(W_k)) / diag(Hinv)) · triu(Hinv)
```

where `Q(·)` snaps to the quantization grid and `Hinv` is the upper Cholesky factor
of the damped inverse Hessian (columns pre-sorted in activation order). Iterating
this map (up to `min(50, columns)` iterations) converges to approximately the same
weights that sequential GPTQ would produce, and the final `Q(W)` is written back to
the layer.

Everything else mirrors the GPTQ implementation:

- **Same calibration flow.** `FPIGPTQQuantizer` subclasses `GPTQQuantizer`, so
  `prepare()` / calibration behave identically: the first decoder layer's inputs
  are captured and the forward pass is stopped early.
- **Same Hessian accumulation** from hooked inputs, layer by layer, with a
  re-forward after quantization to feed the next layer.
- **Supported modules:** `nn.Linear`, `nn.Conv1d`, `nn.Conv2d`,
  `nn.ConvTranspose2d` (grouped convolutions are approximated by averaging
  groupwise Hessians).
- Per-module `Quantizer` objects are attached as `model.quantizers` after
  `convert()`.

### Role inside this library

Besides being usable as a standalone quantizer, FPI-GPTQ serves as the fast inner
loop for the GPTQ-adjusted MSE modes (`mse_for_gptq`, `smse_for_gptq`) of
`GPTQConfig` — finding quantization parameters that minimize the *post-GPTQ* error
requires running (an approximation of) GPTQ inside the parameter search, which is
only affordable with this accelerated variant. See the
[MSE section of the GPTQ README](../gptq/README.md#mse).

### Configuration (`FPIGPTQConfig`)

`FPIGPTQConfig` currently exposes only:

| Field | Default | Description |
|---|---|---|
| `verbose` | `False` | Print per-module timing/error logs. |
| `show_progress` | `True` | Show tqdm progress bars. |

The remaining knobs are fixed in this implementation: 8-bit, per-channel,
asymmetric weight quantization with `percdamp=0.01` and activation ordering always
on. Grouped quantization (`groupsize`), `weight_bits` overrides, MSE tuning, and
`lm_head` quantization are not supported here — use `gptq/` if you need them.

### How to use FPIGPTQQuantizer

```python
from tico.quantization import prepare, convert
from tico.quantization.config.fpi_gptq import FPIGPTQConfig

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Maykeye/TinyLLama-v0")
model = AutoModelForCausalLM.from_pretrained("Maykeye/TinyLLama-v0")
model.eval()

# 1. Prepare: attaches the first-layer input catcher.
prepare(model, FPIGPTQConfig(), inplace=True)

# 2. Calibration: run any number of batches; each stops after the first layer.
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
for i in range(16):
    input_ids = tokenizer(dataset[i]["text"], return_tensors="pt").input_ids
    if input_ids.numel() == 0:
        continue
    model(input_ids)

# 3. Convert: fixed-point-iteration GPTQ, layer by layer.
convert(model, inplace=True)

print(model.quantizers.keys())
```

As with GPTQ, the result is a **fake-quantized** model: weights are snapped to the
grid but remain float, and a real quantization step (e.g. `wrapq` / PTQ) must
follow.

### When to choose FPI-GPTQ vs. GPTQ

- **FPI-GPTQ** — you want near-GPTQ accuracy with much shorter wall-clock time on
  GPU, and the fixed 8-bit / per-channel / asymmetric setting suits your target.
- **GPTQ** — you need configurable bit-widths (incl. per-module overrides), grouped
  quantization, symmetric grids, MSE-tuned quantizers, or `lm_head` quantization.

### Attribution

The fixed-point iteration and quantization kernels are derived from
[IST-DASLab/gptq](https://github.com/IST-DASLab/gptq) (Apache License 2.0).
