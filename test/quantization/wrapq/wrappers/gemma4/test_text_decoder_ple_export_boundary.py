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

"""Regression tests for the split Gemma4 text PLE quantization boundary."""

import unittest

import torch
from tico.passes.decompose_fake_quantize_tensor_qparams import (
    DecomposeFakeQuantizeTensorQParams,
)
from tico.quantization.passes.fold_quant_ops import FoldQuantOps
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.wrappers.gemma4.export_adapters import (
    Gemma4TextDecoderLayerDecodeExportAdapter,
    Gemma4TextDecoderLayerPrefillExportAdapter,
)
from tico.serialize.quant_param import QPARAM_KEY


class _FrozenPLEObserver(torch.nn.Module):
    """Expose the frozen affine fake-quant operation used at the PLE boundary."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("_cached_scale", torch.tensor(0.125))
        self.register_buffer("_cached_zp", torch.tensor(0, dtype=torch.int))

    def fake_quant(self, tensor: torch.Tensor) -> torch.Tensor:
        """Fake-quantize with a fixed signed 16-bit per-tensor domain."""
        return torch.fake_quantize_per_tensor_affine(
            tensor,
            scale=self._cached_scale,
            zero_point=self._cached_zp,
            quant_min=-32768,
            quant_max=32767,
        )


class _FakeDecoderLayer(torch.nn.Module):
    """Consume the decoder adapter ABI while keeping the test graph minimal."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        per_layer_input: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        past_key_value=None,
        shared_key_value=None,
        use_cache: bool = False,
        cache_output_mode: str = "delta",
    ) -> torch.Tensor:
        """Add PLE to hidden states and ignore unrelated decoder inputs."""
        del (
            attention_mask,
            position_embeddings,
            past_key_value,
            shared_key_value,
            use_cache,
            cache_output_mode,
        )
        if per_layer_input is None:
            raise RuntimeError("per_layer_input is required by this test layer.")
        return hidden_states + per_layer_input


class TestGemma4TextPLEExportBoundary(unittest.TestCase):
    """Ensure split decoder inputs retain the packed PLE producer qparam."""

    @staticmethod
    def _fold_quantization_metadata(
        module: torch.nn.Module,
        args: tuple,
        kwargs: dict,
    ):
        """Export a module and fold fake-quant operations into node metadata."""
        exported_program = torch.export.export(module.eval(), args, kwargs=kwargs)
        DecomposeFakeQuantizeTensorQParams().call(exported_program)
        FoldQuantOps().call(exported_program)
        return exported_program

    def _assert_ple_placeholder_is_quantized(
        self,
        module: torch.nn.Module,
        args: tuple,
        per_layer_input: torch.Tensor,
    ) -> None:
        """Assert that folding attaches qparam metadata to the PLE placeholder."""
        exported_program = self._fold_quantization_metadata(
            module,
            args,
            {"per_layer_input": per_layer_input},
        )
        ple_placeholders = [
            node
            for node in exported_program.graph.nodes
            if node.op == "placeholder" and node.target == "per_layer_input"
        ]
        self.assertEqual(len(ple_placeholders), 1)
        self.assertIn(QPARAM_KEY, ple_placeholders[0].meta)

    def test_prefill_and_decode_attach_qparam_to_ple_placeholder(self) -> None:
        """Both split text graphs should quantize their external PLE input."""
        observer = _FrozenPLEObserver()
        hidden = torch.randn(1, 2, 4)
        attention_mask = torch.zeros(1, 1, 2, 2)
        position_embeddings = (
            torch.randn(1, 2, 4),
            torch.randn(1, 2, 4),
        )
        per_layer_input = torch.randn_like(hidden)

        prefill = Gemma4TextDecoderLayerPrefillExportAdapter(
            _FakeDecoderLayer(),
            return_kv=False,
            mode=Mode.QUANT,
            per_layer_input_observer=observer,
        )
        self._assert_ple_placeholder_is_quantized(
            prefill,
            (hidden, attention_mask, position_embeddings),
            per_layer_input,
        )

        decode = Gemma4TextDecoderLayerDecodeExportAdapter(
            _FakeDecoderLayer(),
            return_kv=False,
            mode=Mode.QUANT,
            per_layer_input_observer=observer,
        )
        self._assert_ple_placeholder_is_quantized(
            decode,
            (
                hidden[:, :1, :],
                attention_mask[:, :, :1, :],
                (
                    position_embeddings[0][:, :1, :],
                    position_embeddings[1][:, :1, :],
                ),
            ),
            per_layer_input[:, :1, :],
        )

    def test_no_quant_export_does_not_require_frozen_qparams(self) -> None:
        """Floating-point split export should bypass the boundary observer."""

        class _FailIfCalledObserver(torch.nn.Module):
            def fake_quant(self, tensor: torch.Tensor) -> torch.Tensor:
                del tensor
                raise AssertionError("NO_QUANT export must not call fake_quant().")

        hidden = torch.randn(1, 2, 4)
        per_layer_input = torch.randn_like(hidden)
        module = Gemma4TextDecoderLayerPrefillExportAdapter(
            _FakeDecoderLayer(),
            return_kv=False,
            mode=Mode.NO_QUANT,
            per_layer_input_observer=_FailIfCalledObserver(),
        )

        output = module(
            hidden,
            torch.zeros(1, 1, 2, 2),
            (torch.zeros(1, 2, 4), torch.zeros(1, 2, 4)),
            per_layer_input=per_layer_input,
        )
        torch.testing.assert_close(output, hidden + per_layer_input)

    def test_calibration_mode_is_rejected_at_export_boundary(self) -> None:
        """A split artifact must be produced only before or after calibration."""
        with self.assertRaisesRegex(RuntimeError, "NO_QUANT or QUANT"):
            Gemma4TextDecoderLayerPrefillExportAdapter(
                _FakeDecoderLayer(),
                return_kv=False,
                mode=Mode.CALIB,
                per_layer_input_observer=_FrozenPLEObserver(),
            )


if __name__ == "__main__":
    unittest.main()
