## pt2e

This module is for post training static quantization in fx graph mode based on 'torch.export.export'.

The high level architecture of the quantization process looks like below.

```
float_model(Python)                          Example Input
    \                                              /
     \                                            /
—-------------------------------------------------------
|                        export                        |
—-------------------------------------------------------
                            |
                    FX Graph in ATen       *PT2EAnnotator*
                            |                    /
—--------------------------------------------------------
|                     prepare_pt2e                      |
—--------------------------------------------------------
                            |
                       Calibration
                            |
—--------------------------------------------------------
|                    convert_pt2e                       |
—--------------------------------------------------------
                            |
                    Quantized Model
                            |
—--------------------------------------------------------
|                        convert                        |
—--------------------------------------------------------
                            |
                      Circle Model
```

[Reference]
- https://docs.pytorch.org/ao/main/tutorials_source/pt2e_quant_ptq.html

## How to use PT2EQuantizer

```python
import torch

from tico.quantization. import prepare, convert
from tico.quantization.config import PT2EConfig
import tico

class TwoLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 10)
        self.linear2 = torch.nn.Linear(10, 20)

    def forward(self, x):
        z = self.linear(x)
        z = self.linear2(z)
        return z

    def get_example_inputs(self):
        return (torch.randn(1, 5),)

m = TwoLinear().eval()
example_inputs = m.get_example_inputs()

q_m = prepare(m, PT2EConfig(), inplace=False)

# Calibration
for i in range(100):
    cal_in = m.get_example_inputs()
    m(*cal_in)

convert(q_m)


# Export circle
cm = tico.convert(q_m, example_inputs)
cm.save('linear.q.circle')
```

## Export Post-Training Quantization Pipeline (PyTorch Aten IR to Circle IR)

This section documents an export post-training quantization (PTQ) pipeline for converting PyTorch (Aten IR) 
 models to a Circle IR format. The main goal is to optimize models for inference by reducing 
precision (e.g., from float32 to uint8) while maintaining acceptable accuracy.

### Overview

Post-training quantization helps reduce the model size and speed up inference by converting floating-point 
 operations to lower-precision operations (e.g., uint8). The pipeline below outlines each step, 
from the initial PyTorch (Aten IR) model to the final Circle IR model with quantized operations.

### Pipeline Steps

#### Step 1: Decompose and Fuse (Aten Graph)

**Goal**: Simplify the Aten graph by decomposing complex ops into smaller ones and fusing compatible ops 
(e.g., Conv → BatchNorm → ReLU).

**Why**:
- Some ops are easier to quantize when represented as basic arithmetic or simpler building blocks.
- Certain sequences of ops (e.g., Conv+BN+ReLU) can be fused to reduce overhead and improve performance.
- Performing decomposition before calibration ensures all ops (including newly generated ones) can be
calibrated.
  - If you decompose after calibration, newly generated ops would not have calibration statistics 
(e.g. min/max histograms), making it impossible to compute the correct quantization parameters.

#### Step 2: Calibration

**Goal**: Collect min/max (or other statistical) information for all relevant tensors in the graph 
to determine appropriate quantization parameters (scale and zero-point).

**Process**:
1.	Run inference on a calibration dataset.
2.	Track the distribution of activations/weights (e.g., min/max, histogram, etc.).
3.	Decide on quantization strategy (e.g., per-tensor vs. per-channel, etc.).

#### Step 3: Q-DQ Insertion

**Goal**:
1. For weights: perform real quantization (e.g. store them as uint8) and insert only a Dequantize (DQ)
 so that the rest of the float graph can still process them in float during subsequent steps.
2. For activations: Insert both “Quantize” and “Dequantize” (Q, DQ) ops around tensors or subgraphs
 scheduled for quantization.

**Why**:
- Placing Q/DQ ops makes the graph “quantization-aware,” allowing you to emulate uint8 behavior while still in 
floating-point precision.
- This step ensures the rest of the pipeline can accurately track which ops/tensors will be quantized.

#### Step 4: Remove Q-DQ for Activations and Store Quantization Parameters in Metadata

**Goal**: Remove the Q-DQ pairs for activations and store the relevant quantization parameters 
 (scale/zero-point) in each op's metadata. Leave weight-related DQ as-is.

**Process**:
1. Identify all Q-DQ pairs associated with activation tensors.
2. Record the scale/zero-point in the operator's metadata.
3. Remove both Q and DQ ops for activations, returning those paths to pure float.
4. Do **not** remove the weight DQ, since we still need it to keep the graph float-compatible during 
    later passes.

**Reasoning**:
- Having Q-DQ for activations can complicate subsequent graph optimizations.
- We still capture the necessary quantization parameters in metadata for eventual final quantization.
- Weight remains quantized, and its DQ remains so the rest of the graph can operate in float.

#### Step 5: Replacement and No-Op Removal Optimization

**Goal**: Perform additional graph-level optimizations such as:
- Removing no-ops (e.g., unnecessary reshapes, transposes, identity operations).
- Replacing certain ops with more efficient variants.
- Constraint:
  - No further decomposition that introduces new ops which cannot store quantization parameters.
  - We do not want to break the flow of quantization by adding ops that lack quantized kernels.
  - During these optimizations, the graph is still functionally in float mode (except for statically quantized
  weights) due to the remaining DQ ops.

#### Step 6: Legalization for Circle IR

**Goal**: Prepare the graph so it conforms to the Circle IR constraints.

**Changes**:
- Insert Transpose or other layout transformation ops if needed (e.g., NCHW → NHWC).
- Adjust operator inputs/outputs or attributes to match Circle IR specifications.

#### Step 7: Quantization Parameter Propagation to Transpose Ops

**Goal**: Propagate scale/zero-point information through any newly inserted Transpose ops.

**Reasoning**:
- If the channel layout changes (e.g., channel dimension moves), per-channel parameters 
 must be updated to the correct channel indices.
- Ensures consistent quantization across the entire graph.

#### Step 8: Final Quantization and Removal of DQ for Weights

**Goal**: Apply the final quantization transformations to activations and remove the remaining DQ ops.

**Process**:
1. For each operator with stored scale/zero-point in metadata, change the op’s input/output data type 
 to uint8 (or the chosen quant format).
2. Remove the DQ ops that are no longer needed, as the model is now fully quantized end-to-end.
3. Validate that every op has been successfully quantized and no float-only operators remain.

**Reasoning**:
- After all float-based optimizations are complete, we can safely finalize quantization without risking
 the addition of new float-only ops.
- Removing DQ ops at this stage yields a fully uint8 (or other quant format) graph.

#### Step 9: Convert to Circle IR

**Goal**: Output the final quantized model as Circle IR.

**Process**:
- Generate the Circle IR by traversing the updated, quantized graph.
- Verify that each op, tensor, and parameter has the correct shape, layout, and quantization attributes.

### Considerations & Best Practices

#### Calibration Dataset

- Use a representative dataset to minimize accuracy drop.
- Ensure that the distribution of data in calibration is similar to real-world data.

#### Bias Quantization

- Remember that bias typically remains in int32 when the inputs are uint8.
- Ensure consistent scale factors for both weights and activations.

#### Validation and Testing

- Always measure final model accuracy on both calibration and test datasets to ensure the quantization 
 does not degrade performance too much.
- Profile runtime speed and memory to confirm that quantization goals are met.
