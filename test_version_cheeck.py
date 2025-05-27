# -*- coding: utf-8 -*-
import torch
print(f"PyTorch version: {torch.__version__}")

try:
    from tico import __version__ as tico_version
    print(f"TICO loaded successfully (version: {tico_version})")
except RuntimeError as e:
    print(f"[ERROR] {str(e)}")
