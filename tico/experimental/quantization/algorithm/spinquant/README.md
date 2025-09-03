## SpinQuant

SpinQuant is a ~~~. 

### Configuration

The _SpionQuantConfig_ object holds all necessary parameters for the SpinQuant quantization process. 
When using the public interface functions, pass an instance of SpinQuantConfig to ensure that 
the framework dispatches the request to the SpinQuant-specific implementation.

- mode
    - The mode for generating the rotation matrices which rotates the model's weights. There are two options, which are `hadamard` and `random`, denoting `Randomized hadamard matrix` and `Random orthogonal matrix` respectively. By default, `hadamard` is set.

### How to use SpinQuantQuantizer

Below is an example that demonstrates how to use the SpinQuant algorithm via the public interface:

```python
from tico.experimental.quantization import convert, prepare
from tico.experimental.quantization.config import SpinQuantConfig

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Maykeye/TinyLLama-v0")
model = AutoModelForCausalLM.from_pretrained("Maykeye/TinyLLama-v0")

# Load data
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
sample_input = tokenizer(dataset[0]["text"], return_tensors="pt").input_ids

# Set spin quant quantizer
model = prepare(model, SpinQuantConfig())

# Apply spin
q_m = convert(model)
```
