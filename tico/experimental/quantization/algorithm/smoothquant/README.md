## smoothquant

SmoothQuant is a training-free, accuracy-preserving, and general-purpose post-training quantization (PTQ) solution to enable 8-bit weight, 8-bit activation (W8A8) quantization for LLMs. Based on the fact that weights are easy to quantize while activations are not, SmoothQuant smooths the activation outliers by offline migrating the quantization difficulty from activations to weights with a mathematically equivalent transformation. 

### Configuration

The _SmoothQuantConfig_ object holds all necessary parameters for the SmoothQuant quantization process. 
When using the public interface functions, pass an instance of SmoothQuantConfig to ensure that 
the framework dispatches the request to the SmoothQuant-specific implementation.

- alpha
    - The default smoothing factor to apply across all modules.
- custom_alpha_map
    - A dictionary mapping layer/module names to custom alpha values.
      Layers specified in this dictionary will use the corresponding alpha
      value instead of the default.

### How to use SmoothQuantQuantizer

Below is an example that demonstrates how to use the SmoothQuant algorithm via the public interface:

```python
from tico.experimental.quantization import convert, prepare
from tico.experimental.quantization.config.smoothquant import SmoothQuantConfig

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Maykeye/TinyLLama-v0")
model = AutoModelForCausalLM.from_pretrained("Maykeye/TinyLLama-v0")

# Load data
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
sample_input = tokenizer(dataset[0]["text"], return_tensors="pt").input_ids

device = next(model.parameters()).device
num_samples = 10

# attach observers
model = prepare(model, SmoothQuantConfig())

# run calibration
for i in range(num_samples):
    input_ids = tokenizer(dataset[i]["text"], return_tensors="pt").input_ids.to(
        device
    )
    model(input_ids)

# apply smoothing
q_m = convert(model)
```