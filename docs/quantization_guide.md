# Quantization Guide: Op-Level Sensitivity (PTQ)

## Why this document?

This guide exists for practitioners who feel uncertain about which ops can be quantized to INT8/INT4 and which should stay FP16/FP32, and how to do it (weights vs activations, per-channel vs per-tensor, FP "islands," KV-cache policy, etc.). It provides op-level sensitivity guidance, safe defaults, and concrete Llama-3.2 presets as an example to reduce trial-and-error and align review discussions.

- **Audience**: Engineers implementing per-op PTQ for LLMs in runtimes/compilers.
- **Scope**: Post-Training Quantization (PTQ) on decoder-style LLMs with RMSNorm, RoPE, (Swi)GLU (e.g., Llama-3.2).
- **Out of scope**: QAT, kernel-specific tricks, detailed data curation beyond brief calibration tips.

## Summary (TL;DR)

When quantizing LLMs **per operation**, some spots are fragile (keep FP16/FP32), while others tolerate INT8/INT4 well.

- **Keep FP (or at least 16-bit)**: LayerNorm/RMSNorm, (QKᵀ)/√d, softmax, SwiGLU gate multiply, RoPE rotation, logits-adjacent ops, and (preferably) K cache.
- **Usually safe to quantize**: ReLU and simple elementwise ops (after scale alignment), linear weights per-channel (INT4/INT8), V path activations/outputs (with high-precision accumulation).
- **Case-by-case**: GELU/SiLU, residual adds (only if scales are aligned), KV cache (V safer than K).

## Op-Level Sensitivity Matrix

| Operation | Quantize Weights? | Quantize Activations? | Safe Precision | Key Considerations & Failure Modes |
|---|---|---|---|---|
| **Linear / Gemm (General)** | Yes | Yes (Optional) | W4/W8, A8/A16 | Per-channel weight quantization is standard. Dynamic per-token activation quantization avoids outlier issues. |
| **Attention QKᵀ MatMul** | N/A | No (Keep FP) | FP16 / FP32 | Highly sensitive to precision loss; quantization can cause attention collapse or overflow in logits. |
| **Attention Softmax** | N/A | No (Keep FP) | FP16 / FP32 | Requires exact exponentiation and normalization. Quantizing Softmax output degrades attention distribution. |
| **Attention Score x V MatMul** | N/A | Case-by-case | FP16 / A8 | V activations tolerate quantization better than Q/K, but high-precision accumulation is recommended. |
| **RMSNorm / LayerNorm** | N/A | No (Keep FP) | FP16 / FP32 | Small numerical shifts in norm parameters or variance calculation propagate errors across subsequent layers. |
| **RoPE (Rotary Embedding)** | N/A | No (Keep FP) | FP16 / FP32 | Phase angles and trigonometric operations require full floating-point accuracy. |
| **SwiGLU / Activation Fn** | N/A | Case-by-case | FP16 / A8 | Gate multiplication in SwiGLU can amplify quantization error if gate branch is quantized coarsely. |
| **Embedding Layer** | Yes | No | W4 / W8 | Weight-only quantization is effective; activation quantization is unnecessary as input tokens are discrete IDs. |
| **LM Head (Output)** | Case-by-case | No | FP16 / W8 | Final projection to vocabulary logits affects top-k selection. Keep FP16 or high-bit W8 to preserve token ranking. |

## Recommended Presets (e.g. Llama-3.2)

1. **Conservative (High Accuracy)**:
   - Weights: INT8 per-channel
   - Activations: FP16
   - KV Cache: FP16

2. **Balanced (Production Default)**:
   - Weights: INT4 per-group (group_size=128) for Linear layers
   - Activations: FP16
   - KV Cache: INT8 per-tensor (V only)

3. **Aggressive (Maximum Compression)**:
   - Weights: INT4 per-group (group_size=64)
   - Activations: INT8 dynamic per-token (Linear inputs only)
   - Keep Norm, Softmax, RoPE, and QKᵀ in FP16/FP32.
