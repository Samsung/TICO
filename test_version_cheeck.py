# -*- coding: utf-8 -*-
# test_version_check.py
import torch
print(f"PyTorch version: {torch.__version__}")

try:
    from tico import __version__ as tico_version
    print(f"TICO version: {tico_version}")
except ImportError as e:
    print(f"[ERROR] Missing version export: {str(e)}")
except RuntimeError as e:
    print(f"[FEATURE ERROR] {str(e)}")