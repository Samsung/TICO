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

**TODO**: Modify the GPTQ algorithm to directly perform quantization using the scale and zero
point computed internally. Or, pass the computed scale and zero point so that other quantizers
can use these values for consistent weight quantization.
