## gptq

The GPTQ algorithm is a weight-only quantization method designed specifically for large
language models. It focuses solely on compressing model weights without altering activations, 
which helps reduce the model’s memory footprint and inference latency while preserving 
performance.

### Configuration

The _GPTQConfig_ object holds all necessary parameters for the GPTQ quantization process. 
When using the public interface functions, pass an instance of GPTQConfig to ensure that 
the framework dispatches the request to the GPTQ-specific implementation.

**TODO**: Add configuration contents.

### How to use GPTQQuantizer

Below is an example that demonstrates how to use the GPTQ algorithm via the public interface:

```python
from tico.quantization. import prepare, convert
from tico.quantization.config.gptq import GPTQConfig
import tico

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Maykeye/TinyLLama-v0")
model = AutoModelForCausalLM.from_pretrained("Maykeye/TinyLLama-v0")

# Load data
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
sample_input = tokenizer(dataset[0]["text"], return_tensors="pt").input_ids

gptq_config = GPTQConfig()

prepare(model, gptq_config, args=(sample_input,), inplace=True)
# No calibration is needed
convert(q_m, inplace=True)
```

### Precautions and Future Enhancements

After _convert()_ is executed, the weights are updated; however, the algorithm performs
a fake quantization internally, so the weights remain as float types. Therefore, an additional
quantization step—such as using `wrapq`—must be applied afterwards. One potential issue is
that if the internal fake quantization method for weights differs from the weight quantization
applied after _convert()_, the effectiveness of GPTQ may be diminished.

### `MSE`
`mse` parameter of `GPTQConfig` is supposed to tune quantizer for using in GPTQ.
There are four options :
1. `mse`- vanilla `mse`. Produce quantization parameters for GPTQ quantizer (`min`\`max`) which minimize mean squared error of quantization. $MSE_{MIN, MAX}(W) = argmin_{min, max}||W-Q_{min, max}(W)||^2$.
2. `smse` - sensitivity-based `mse`. Use sensitivity of some global feature (e.g. float model logits) to parameters change to minimize global effect of quantization. $SMSE_{MIN, MAX}(W) = argmin_{min, max}|(W-Q_{min, max}(W))^2*Sensitivity(W)|$. So we try to keep `important` parameters unchanged, while quantizing `unimportant` parameters more aggressively.
3. `smse_for_gptq` - `smse` adjusted for GPTQ. GPTQ modifies the matrix during the quantization process, so the most accurate method would consist in finding a quantizer that yields the smallest quantization error after the GPTQ method has been applied $SMSE\_FOR\_GPTQ_{MIN, MAX}(W) = argmin_{min, max}|(W-Q_{min, max}(W_{GPTQ}))^2*Sensitivity(W)|$. Since this would be quite computationally expensive, we can use an accelerated approximate GPTQ method — FPI_GPTQ $SMSE\_FOR\_GPTQ_{MIN, MAX}(W) = argmin_{min, max}|(W-Q_{min, max}(W_{FPI\_GPTQ}))^2*Sensitivity(W)|$. This is slower than `mse`/`smse` but can provide better accuracy.
4. `mse_for_gptq` - `mse` adjusted for GPTQ. GPTQ modifies the matrix during the quantization process, so the most accurate method would consist in finding a quantizer that yields the smallest error (minimized by GPTQ) after the GPTQ method has been applied $MSE\_FOR\_GPTQ_{MIN, MAX}(W) = argmin_{min, max}||\frac{(W_{GPTQ}-Q_{min, max}(W_{GPTQ}))}{diag(Hinv)} ||^2$. Since this would be quite computationally expensive, we can use an accelerated approximate GPTQ method — FPI_GPTQ $MSE\_FOR\_GPTQ_{MIN, MAX}(W) = argmin_{min, max}||\frac{(W_{FPI\_GPTQ}-Q_{min, max}(W_{FPI\_GPTQ}))}{diag(Hinv)} ||^2$. This is slower than `mse`/`smse` but can provide better accuracy. To stabilize computations you may need to increase percdamp to >= 0.1 , because `mse_for_gptq` overfits pretty fast.



You can turn this feature `on`/`off` by using `mse` parameter of `GPTQConfig`:
```
cfg = GPTQConfig(..., mse="mse", ...)
```
for vanilla `mse` or
```
cfg = GPTQConfig(..., mse="smse", sensitivity=some_sensitivity)
```
for `smse` with `some_sensitivity` being dictionary of modules sensitivities {`module_name:module_sensitivity`}.
Sensitivities can be computed using empirical Fisher information e.g. (see `SensitivityCalibrator` util class).

For GPTQ-adjusted methods:
```
cfg = GPTQConfig(..., mse="mse_for_gptq", ...)
```
for `mse_for_gptq` or
```
cfg = GPTQConfig(..., mse="smse_for_gptq", sensitivity=some_sensitivity)
```
for `smse_for_gptq` with sensitivity weighting.


**TODO**: Modify the GPTQ algorithm to directly perform quantization using the scale and zero
point computed internally. Or, pass the computed scale and zero point so that other quantizers
can use these values for consistent weight quantization.
