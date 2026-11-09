## smoothquant

SmoothQuant redistributes activation-channel magnitudes into weights before
quantization. TICO collects channel-wise activation maxima and applies compensating
scales to supported normalization/projection or ReLU feed-forward pairs.

`SmoothQuantQuantizer.convert()` returns a **smoothed floating-point model**. It
does not freeze WrapQ observers, quantize to W8A8, or create a Circle artifact.
Prepare and calibrate a subsequent PTQ stage on that smoothed model; see the
[Python PTQ guide](../../README.md#python-ptq-configuration) and
[recipe configuration guide](../../examples/configs/README.md#pipeline).

### Configuration

Pass [`SmoothQuantConfig`](../../config/smoothquant.py) to the public `prepare()`
API to select the SmoothQuant implementation.

| Field | Default | Meaning |
|---|---|---|
| `alpha` | `0.5` | Smoothing factor used by the selected module handlers. |
| `custom_alpha_map` | `None` | Exact handled-module names mapped to smoothing factors, overriding `alpha`; for example, `{"model.layers.0": 0.6}`. |
| `acts_from` | `"input"` | `"input"` collects Linear input statistics with pre-hooks; `"output"` collects Linear output statistics with forward hooks. The selected mode applies to the whole prepared model. |

Calibration collects one absolute maximum per channel in the last activation
dimension for both contiguous and non-contiguous Linear inputs or outputs. It
preserves activation values and strides; flattening the statistics may allocate a
temporary copy for a non-contiguous tensor.

### Supported transformations and statistics

The handlers in [`smooth_quant.py`](./smooth_quant.py) select the following
module families. Merely containing Linear layers does not make an arbitrary model
eligible for smoothing.

| Handler | Transformation | Required statistics |
|---|---|---|
| Hugging Face LLaMA decoder | RMSNorm to Q/K/V and gate/up projections | `acts_from="input"` |
| Qwen3-VL text decoder | RMSNorm to Q/K/V and gate/up projections | `acts_from="input"` |
| Qwen3-VL vision block | LayerNorm to attention and MLP input projections | `acts_from="input"` |
| Qwen3-VL vision patch merger | Norm to `linear_fc1`, only when the norm extent equals its input width; reshape-expanded mergers are skipped | `acts_from="input"` |
| fairseq Transformer encoder/decoder with a ReLU FFN | Compensating scales across `fc1 -> ReLU -> fc2` | `acts_from="output"` for `fc1` output channels |

Choose the mode required by the handler. The collector can record either side of
a Linear, but the normalization/projection handlers consume input-channel
statistics, while the fairseq bridge consumes the hidden channels after `fc1`.
Changing the hook side does not adapt those transformations automatically and can
produce incompatible channel counts. Non-ReLU fairseq FFNs and modules without a
matching handler are left unchanged. The corresponding Transformers or fairseq
classes must be available for their handlers to match.

### How to use SmoothQuantQuantizer

This tiny randomly initialized LLaMA example runs on CPU without downloading a
model or dataset. For a trained model, replace the synthetic token batches with
representative calibration inputs.

```python
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from tico.quantization import convert, prepare
from tico.quantization.config.smoothquant import SmoothQuantConfig

torch.manual_seed(0)
model = LlamaForCausalLM(LlamaConfig(
    vocab_size=32,
    hidden_size=16,
    intermediate_size=32,
    num_hidden_layers=1,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=32,
)).eval()
calibration_inputs = [torch.randint(0, 32, (1, 8)) for _ in range(4)]

with torch.no_grad():
    expected = model(calibration_inputs[0], use_cache=False).logits
    prepared = prepare(model, SmoothQuantConfig(alpha=0.5, acts_from="input"))
    for input_ids in calibration_inputs:
        prepared(input_ids, use_cache=False)
    smoothed_model = convert(prepared)
    actual = smoothed_model(calibration_inputs[0], use_cache=False).logits

torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
assert not hasattr(smoothed_model, "tico_quantizer")
```
