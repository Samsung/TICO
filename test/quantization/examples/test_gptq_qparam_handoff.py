# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Exercise the actual legacy example helpers without model/data dependencies.

Each probe runs in a child process so optional dependency placeholders and any
example imports holding them cannot affect other tests in the discovery process.
The GPTQ quantizer, PTQ wrapper, and qparam handoff implementations are real.
"""

import importlib
import importlib.machinery
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

_EXAMPLE_PACKAGE = "tico.quantization.wrapq.examples"


def _install_example_dependencies() -> None:
    """Stub only unused model loaders and optional data/evaluation packages."""
    # Initialize TICO and PyTorch before installing optional import placeholders.
    import tico  # noqa: F401

    from test.quantization.recipes.optional_dependency_stubs import (
        install_optional_dependency_stubs,
    )

    install_optional_dependency_stubs()

    def unavailable(*args, **kwargs):
        raise AssertionError("This qparam test must not load a pretrained model.")

    class Loader:
        """Fail if a helper unexpectedly reaches model/tokenizer loading."""

        from_pretrained = staticmethod(unavailable)

    transformers = types.ModuleType("transformers")
    transformers.__path__ = []
    transformers.__spec__ = importlib.machinery.ModuleSpec(
        "transformers", loader=None, is_package=True
    )
    for name in ("AutoModelForCausalLM", "AutoTokenizer", "AutoProcessor"):
        setattr(transformers, name, Loader)
    cache_utils = types.ModuleType("transformers.cache_utils")
    setattr(cache_utils, "Cache", type("Cache", (), {}))
    setattr(transformers, "cache_utils", cache_utils)
    utils = types.ModuleType("transformers.utils")
    utils.__path__ = []
    generic = types.ModuleType("transformers.utils.generic")
    setattr(generic, "ModelOutput", type("ModelOutput", (dict,), {}))
    setattr(utils, "generic", generic)
    setattr(transformers, "utils", utils)
    sys.modules["transformers"] = transformers
    sys.modules["transformers.cache_utils"] = cache_utils
    sys.modules["transformers.utils"] = utils
    sys.modules["transformers.utils.generic"] = generic


def _check_helper(example_name: str) -> None:
    """Verify both mismatch directions and the existing successful return API."""
    import torch

    from tico.quantization.algorithm.gptq.quant import quantize, Quantizer
    from tico.quantization.config.ptq import PTQConfig
    from tico.quantization.config.specs import affine
    from tico.quantization.wrapq.dtypes import DType
    from tico.quantization.wrapq.wrappers.nn.quant_linear import QuantLinear

    example = importlib.import_module(f"{_EXAMPLE_PACKAGE}.{example_name}")
    check = unittest.TestCase()

    for source_bits, target_bits in ((4, 8), (8, 4), (4, 4)):
        linear = torch.nn.Linear(4, 1, bias=False)
        with torch.no_grad():
            linear.weight.copy_(torch.tensor([[0.0, 3.0, 7.0, 15.0]]))
        quantizer = Quantizer()
        quantizer.configure(source_bits, perchannel=True, sym=False)
        quantizer.find_params(linear.weight.detach(), weight=True)
        with torch.no_grad():
            linear.weight.copy_(
                quantize(
                    linear.weight,
                    quantizer.scale,
                    quantizer.zero,
                    quantizer.maxq,
                )
            )
        wrapper = QuantLinear(
            linear,
            qcfg=PTQConfig(weight=affine(DType.uint(target_bits))),
            fp_name="linear",
        )

        if source_bits != target_bits:
            with check.assertRaisesRegex(ValueError, "linear"):
                example.inject_gptq_qparams(wrapper, {"linear": quantizer})
            check.assertFalse(wrapper.obs_weight.has_qparams)
            check.assertFalse(wrapper.obs_weight._qparams_locked)
        else:
            result = example.inject_gptq_qparams(wrapper, {"linear": quantizer})
            check.assertIsNone(result)
            check.assertTrue(wrapper.obs_weight._qparams_locked)
            check.assertEqual(wrapper.obs_weight.dtype, DType.uint(target_bits))
            torch.testing.assert_close(
                wrapper.obs_weight.fake_quant(linear.weight), linear.weight
            )


def _check_qwen_config() -> None:
    """Keep patch-embedding precision aligned across GPTQ and PTQ CLI policy."""
    example = importlib.import_module(f"{_EXAMPLE_PACKAGE}.quantize_qwen3_vl_with_gptq")
    check = unittest.TestCase()
    key = "model.visual.patch_embed.proj"

    for patch_bits in (8, 6):
        argv = [
            "quantize_qwen3_vl_with_gptq",
            "--model",
            "local-test-model",
            "--linear_weight_bits",
            "4",
            "--vision_patch_embed_weight_bits",
            str(patch_bits),
        ]
        with patch.object(sys, "argv", argv):
            args = example.parse_args()
        config = example.build_qwen3_vl_gptq_config(args)
        check.assertEqual(config.weight_bits, 4)
        check.assertEqual(config.weight_bits_overrides, {key: patch_bits})
        check.assertFalse(config.quantize_lm_head)

        args.gptq_lm_head = True
        args.lm_head_weight_bits = 8
        config = example.build_qwen3_vl_gptq_config(args)
        check.assertEqual(config.weight_bits_overrides, {key: patch_bits, "lm_head": 8})

        args.gptq_vision = False
        config = example.build_qwen3_vl_gptq_config(args)
        check.assertEqual(config.weight_bits, 4)
        check.assertEqual(config.weight_bits_overrides, {"lm_head": 8})
        check.assertFalse(config.quantize_vision_patch_embed)


class TestLegacyGPTQQParamHandoff(unittest.TestCase):
    """Run production example functions with isolated optional imports."""

    def _run_probe(self, probe: str) -> None:
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), probe],
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True,
            text=True,
            check=False,
            timeout=45,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_basic_example_rejects_bit_mismatch(self) -> None:
        """The basic GPTQ example must use the checked shared handoff."""
        self._run_probe("quantize_with_gptq")

    def test_full_model_example_rejects_bit_mismatch(self) -> None:
        """The full-model example must not retain an unchecked private copy."""
        self._run_probe("quantize_full_qmodel_with_gptq")

    def test_qwen_example_rejects_bit_mismatch(self) -> None:
        """The Qwen3-VL example must reject mismatched GPTQ/PTQ grids."""
        self._run_probe("quantize_qwen3_vl_with_gptq")

    def test_qwen_patch_embedding_uses_requested_gptq_bits(self) -> None:
        """Vision precision overrides must not alter linear or LM-head policy."""
        self._run_probe("qwen_config")


if __name__ == "__main__":
    # Absolute script execution otherwise exposes only the test directory.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    _install_example_dependencies()
    if sys.argv[1] == "qwen_config":
        _check_qwen_config()
    else:
        _check_helper(sys.argv[1])
