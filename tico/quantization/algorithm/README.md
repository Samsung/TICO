# algorithm

[Quantization overview](../README.md) · [Documentation index](../../../docs/README.md)

The _algorithm_ module implements quantization algorithms and preprocessing methods
for deep learning. Some methods transform model weights before a later quantization
stage, while others quantize weights or reconstruct activation quantization parameters.
They are not all graph rewrites or preprocessing steps.

## Available Algorithms

| Algorithm | Reference | Description |
|---|---|---|
| GPTQ | [`gptq/`](./gptq/README.md) | Accurate post-training weight quantization |
| Universal GPTQ | [Configuration source](../config/gptq.py), [implementation](./universal_gptq/quantizer.py) | Replay-based GPTQ with configurable module caching and GPTQ core |
| FPI-GPTQ | [`fpi_gptq/`](./fpi_gptq/README.md) | Fixed-point-iteration GPTQ variant |
| Qwen3-VL GPTQ | [`qwen3_vl_gptq/`](./qwen3_vl_gptq/README.md) | GPTQ specialization for Qwen3-VL models |
| Gemma4 GPTQ | [`gemma4_gptq/`](./gemma4_gptq/README.md) | Stage-wise GPTQ for Gemma4 vision and text components |
| Block reconstruction | [`block_reconstruction/`](./block_reconstruction/README.md) | Fixed-weight activation-qparam reconstruction |
| AdaRound | [`adaround/`](./adaround/README.md) | Validation-aware hard Conv2d weight rounding |
| SmoothQuant | [`smoothquant/`](./smoothquant/README.md) | Migrates activation outliers into weights via per-channel smoothing |
| SpinQuant | [`spinquant/`](./spinquant/README.md) | Rotation-based outlier reduction |
| CLE | [`cle/`](./cle/README.md) | Cross-Layer Equalization |

For Universal GPTQ recipe examples, see the
[LLaMA preset](../examples/configs/llama_universal_gptq_quantize.yaml) and
[Qwen3-VL preset](../examples/configs/qwen3_vl_universal_gptq_quantize.yaml).
Both select the `universal` GPTQ stage variant; the configuration source documents
`UniversalGPTQConfig` and its `gptq_factory` selection.

## Design Philosophy

Each algorithm guide owns its numerical method, supported inputs, configuration,
and limitations. The [quantization overview](../README.md#directory-structure)
describes the shared API and implementation layers; the
[recipe responsibility map](../recipes/README.md#responsibility-boundaries)
defines where pipeline integration belongs.

## Usage Guidelines

Install the dependencies required by the selected algorithm or model family; for
example, SmoothQuant lists them in [`smooth_quant.txt`](./smoothquant/smooth_quant.txt).
Keep optional dependencies local to the paths that need them. Algorithms may use
shared quantization infrastructure but must not import recipe orchestration or
example entrypoints; see the [import rules](../recipes/README.md#import-rules).

## Example Use Case

Start with the [public Python quickstart](../README.md#basic-usage), then follow
the selected algorithm's guide above for its configuration and calibration
requirements. For composed CLI workflows, use the
[recipe examples](../examples/README.md).

## Contributing to the Module

Follow the [quantization contribution guide](../README.md#contributing-adding-a-new-algorithm)
for configuration classes, quantizer registration, and tests. Document the new
method and its optional dependencies in its algorithm guide, and add it to this
catalog. To expose it as a CLI stage, follow the
[recipes developer guide](../recipes/README.md).
