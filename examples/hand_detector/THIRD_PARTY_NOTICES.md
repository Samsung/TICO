MediaPipe Hand Tracking Models
------------------------------

This project includes PyTorch conversions of the MediaPipe Hand Tracking
palm detection and hand landmark models.

The original models are licensed under the Apache License, Version 2.0.

Modifications:
- Converted the original TensorFlow Lite models to PyTorch modules.
- Adapted tensor layouts and operator implementations for PyTorch.
- Added quantization and model-debugging support.
- The converted and quantized models may produce results different from
  the original MediaPipe models.
