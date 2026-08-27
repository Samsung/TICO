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

"""Unit tests for the Gemma4 vision patch embedder PTQ wrapper."""

import unittest
from unittest import mock

import tico
import torch
from circle_schema import circle
from tico.circle.io import model_from_bytes
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.qscheme import QScheme
from tico.quantization.wrapq.wrappers.gemma4.quant_vision_patch_embedder import (
    QuantGemma4VisionPatchEmbedder,
)


_SKIP_MSG = "required transformers Gemma4 modules are not installed"


def _has_gemma4() -> bool:
    """Return whether the installed transformers package provides Gemma4 support."""
    try:
        from transformers.models.gemma4.modeling_gemma4 import (  # noqa: F401
            Gemma4VisionPatchEmbedder,
        )
    except Exception:
        return False
    return True


def _make_patch_embedder(
    hidden_size=32,
    patch_size=4,
    position_embedding_size=8,
):
    """Create a tiny Gemma4VisionPatchEmbedder for testing."""
    from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionPatchEmbedder

    config = Gemma4VisionConfig(
        hidden_size=hidden_size,
        patch_size=patch_size,
        position_embedding_size=position_embedding_size,
    )
    return Gemma4VisionPatchEmbedder(config).eval()


def _builtin_code(model, operator) -> int:
    """Return the builtin code referenced by one Circle operator."""
    return model.operatorCodes[operator.opcodeIndex].builtinCode


@unittest.skipUnless(_has_gemma4(), _SKIP_MSG)
class TestQuantGemma4VisionPatchEmbedder(unittest.TestCase):
    """Validate Gemma4 vision patch embedder wrapper behavior."""

    def setUp(self):
        """Create deterministic inputs."""
        torch.manual_seed(2026)
        self.hidden_size = 32
        self.patch_size = 4
        self.position_embedding_size = 8
        self.batch_size = 1
        self.num_patches = 16

    def _sample_inputs(self):
        """Create synthetic inputs."""
        patch_dim = 3 * self.patch_size**2
        pixel_values = torch.randn(self.batch_size, self.num_patches, patch_dim)
        pixel_position_ids = torch.randint(
            0, self.position_embedding_size, (self.batch_size, self.num_patches, 2)
        )
        padding_positions = torch.zeros(
            self.batch_size, self.num_patches, dtype=torch.bool
        )
        return pixel_values, pixel_position_ids, padding_positions

    # ------------------------------------------------------------------
    # NO_QUANT mode
    # ------------------------------------------------------------------

    def test_no_quant_forward_matches_fp(self):
        """In NO_QUANT mode the wrapper should match the floating-point module."""
        fp_module = _make_patch_embedder(
            hidden_size=self.hidden_size,
            patch_size=self.patch_size,
            position_embedding_size=self.position_embedding_size,
        )
        q_module = QuantGemma4VisionPatchEmbedder(fp_module).eval()

        self.assertIs(q_module._mode, Mode.NO_QUANT)

        pixel_values, pixel_position_ids, padding_positions = self._sample_inputs()
        with torch.no_grad():
            q_out = q_module(pixel_values, pixel_position_ids, padding_positions)
            fp_out = fp_module(pixel_values, pixel_position_ids, padding_positions)

        # Shapes must match
        self.assertEqual(q_out.shape, fp_out.shape)

        # The folded projection must remain numerically equivalent.
        self.assertTrue(torch.allclose(q_out, fp_out, atol=1e-5, rtol=1e-5))

    def test_pixel_normalization_is_folded_into_input_projection(self):
        """Fold pixel normalization into a cloned projection before PTQ."""
        fp_module = _make_patch_embedder(
            hidden_size=self.hidden_size,
            patch_size=self.patch_size,
            position_embedding_size=self.position_embedding_size,
        )
        original_weight = fp_module.input_proj.weight.detach().clone()
        self.assertIsNone(fp_module.input_proj.bias)

        q_module = QuantGemma4VisionPatchEmbedder(fp_module).eval()
        folded_linear = q_module.input_proj.wrapped.module
        pixel_values, _, _ = self._sample_inputs()

        self.assertIsNot(folded_linear, fp_module.input_proj)
        torch.testing.assert_close(folded_linear.weight, original_weight * 2.0)
        self.assertIsNotNone(folded_linear.bias)
        torch.testing.assert_close(
            folded_linear.bias,
            -original_weight.sum(dim=1),
        )
        with torch.no_grad():
            expected = fp_module.input_proj((pixel_values - 0.5) * 2.0)
            actual = folded_linear(pixel_values)
        torch.testing.assert_close(actual, expected)

        torch.testing.assert_close(fp_module.input_proj.weight, original_weight)
        self.assertIsNone(fp_module.input_proj.bias)

    def test_no_quant_output_shape(self):
        """Check that the output has the expected static shape."""
        fp_module = _make_patch_embedder(
            hidden_size=self.hidden_size,
            patch_size=self.patch_size,
            position_embedding_size=self.position_embedding_size,
        )
        q_module = QuantGemma4VisionPatchEmbedder(fp_module).eval()

        pixel_values, pixel_position_ids, padding_positions = self._sample_inputs()
        with torch.no_grad():
            output = q_module(pixel_values, pixel_position_ids, padding_positions)

        expected_shape = (self.batch_size, self.num_patches, self.hidden_size)
        self.assertEqual(output.shape, expected_shape)

    # ------------------------------------------------------------------
    # Mode transitions
    # ------------------------------------------------------------------

    def test_mode_transitions(self):
        """Check the calibration lifecycle: NO_QUANT → CALIB → QUANT."""
        fp_module = _make_patch_embedder(
            hidden_size=self.hidden_size,
            patch_size=self.patch_size,
            position_embedding_size=self.position_embedding_size,
        )
        q_module = QuantGemma4VisionPatchEmbedder(fp_module).eval()

        self.assertIs(q_module._mode, Mode.NO_QUANT)

        q_module.enable_calibration()
        self.assertIs(q_module._mode, Mode.CALIB)

        pixel_values, pixel_position_ids, padding_positions = self._sample_inputs()
        with torch.no_grad():
            _ = q_module(pixel_values, pixel_position_ids, padding_positions)

        q_module.freeze_qparams()
        self.assertIs(q_module._mode, Mode.QUANT)

    def test_observers_are_collected(self):
        """Check that _all_observers returns the five remaining observers."""
        fp_module = _make_patch_embedder(
            hidden_size=self.hidden_size,
            patch_size=self.patch_size,
            position_embedding_size=self.position_embedding_size,
        )
        q_module = QuantGemma4VisionPatchEmbedder(fp_module).eval()

        all_obs = list(q_module._all_observers())
        self.assertEqual(len(all_obs), 5)
        self.assertIs(all_obs[0], q_module.obs_emb_table)
        self.assertIs(all_obs[1], q_module.obs_act_in)
        self.assertIs(all_obs[2], q_module.obs_hidden_states)
        self.assertIs(all_obs[3], q_module.obs_position_embeddings)
        self.assertIs(all_obs[4], q_module.obs_output)

    # ------------------------------------------------------------------
    # Calibration and fake quantization
    # ------------------------------------------------------------------

    def test_emb_table_is_observed_in_calib_mode(self):
        """In CALIB mode the position_embedding_table should be observed."""
        fp_module = _make_patch_embedder(
            hidden_size=self.hidden_size,
            patch_size=self.patch_size,
            position_embedding_size=self.position_embedding_size,
        )
        q_module = QuantGemma4VisionPatchEmbedder(fp_module).eval()

        q_module.enable_calibration()

        # Check that the embedding-table observer collected min/max statistics.
        self.assertIsNotNone(q_module.obs_emb_table.min_val)
        self.assertIsNotNone(q_module.obs_emb_table.max_val)

    def test_quant_mode_output_is_finite(self):
        """In QUANT mode the output should be finite and have the correct shape."""
        fp_module = _make_patch_embedder(
            hidden_size=self.hidden_size,
            patch_size=self.patch_size,
            position_embedding_size=self.position_embedding_size,
        )
        q_module = QuantGemma4VisionPatchEmbedder(fp_module).eval()
        q_module.enable_calibration()

        pixel_values, pixel_position_ids, padding_positions = self._sample_inputs()
        with torch.no_grad():
            _ = q_module(pixel_values, pixel_position_ids, padding_positions)
        q_module.freeze_qparams()

        with torch.no_grad():
            output = q_module(pixel_values, pixel_position_ids, padding_positions)

        expected_shape = (self.batch_size, self.num_patches, self.hidden_size)
        self.assertEqual(output.shape, expected_shape)
        self.assertTrue(torch.isfinite(output).all())

    # ------------------------------------------------------------------
    # dtype override
    # ------------------------------------------------------------------

    def test_emb_table_uses_per_tensor_symm_by_default(self):
        """Check that emb_table observer uses PER_TENSOR_SYMM by default."""
        fp_module = _make_patch_embedder(
            hidden_size=self.hidden_size,
            patch_size=self.patch_size,
            position_embedding_size=self.position_embedding_size,
        )
        q_module = QuantGemma4VisionPatchEmbedder(fp_module).eval()

        self.assertEqual(q_module.obs_emb_table.qscheme, QScheme.PER_TENSOR_SYMM)

    # ------------------------------------------------------------------
    # Static buffers
    # ------------------------------------------------------------------

    def test_position_embedding_table_is_registered_as_buffer(self):
        """position_embedding_table should be registered as a buffer on the wrapper."""
        fp_module = _make_patch_embedder(
            hidden_size=self.hidden_size,
            patch_size=self.patch_size,
            position_embedding_size=self.position_embedding_size,
        )
        q_module = QuantGemma4VisionPatchEmbedder(fp_module).eval()

        self.assertTrue(hasattr(q_module, "position_embedding_table"))
        self.assertIsInstance(q_module.position_embedding_table, torch.Tensor)
        self.assertEqual(
            q_module.position_embedding_table.shape,
            (2, self.position_embedding_size, self.hidden_size),
        )

    def test_position_embedding_table_matches_original(self):
        """position_embedding_table buffer should match the original module."""
        fp_module = _make_patch_embedder(
            hidden_size=self.hidden_size,
            patch_size=self.patch_size,
            position_embedding_size=self.position_embedding_size,
        )
        q_module = QuantGemma4VisionPatchEmbedder(fp_module).eval()

        self.assertTrue(
            torch.allclose(
                q_module.position_embedding_table, fp_module.position_embedding_table
            )
        )

    def test_pixel_normalization_constants_are_not_registered_as_buffers(self):
        """Avoid retaining normalization constants after parameter folding."""
        fp_module = _make_patch_embedder(
            hidden_size=self.hidden_size,
            patch_size=self.patch_size,
            position_embedding_size=self.position_embedding_size,
        )
        q_module = QuantGemma4VisionPatchEmbedder(fp_module).eval()

        buffers = dict(q_module.named_buffers())
        self.assertNotIn("pixel_center", buffers)
        self.assertNotIn("pixel_rescale", buffers)

    # ------------------------------------------------------------------
    # Config attributes
    # ------------------------------------------------------------------

    def test_config_attributes_are_stored(self):
        """Check that config attributes are stored on the wrapper."""
        fp_module = _make_patch_embedder(
            hidden_size=64,
            patch_size=8,
            position_embedding_size=16,
        )
        q_module = QuantGemma4VisionPatchEmbedder(fp_module).eval()

        self.assertEqual(q_module.hidden_size, 64)
        self.assertEqual(q_module.patch_size, 8)
        self.assertEqual(q_module.position_embedding_size, 16)

    # ------------------------------------------------------------------
    # as_export_module
    # ------------------------------------------------------------------

    def test_as_export_module_supports_no_quant_mode(self):
        """Floating-point export should bake coordinates into a static adapter."""
        from tico.quantization.wrapq.wrappers.gemma4.export_adapters import (
            Gemma4VisionPatchEmbedderPrefillExportAdapter,
        )

        fp_module = _make_patch_embedder(
            hidden_size=self.hidden_size,
            patch_size=self.patch_size,
            position_embedding_size=self.position_embedding_size,
        )
        q_module = QuantGemma4VisionPatchEmbedder(fp_module).eval()
        pixel_values, pixel_position_ids, padding_positions = self._sample_inputs()

        export_module = q_module.as_export_module(
            mode="prefill",
            pixel_position_ids=pixel_position_ids,
            padding_positions=padding_positions,
        )

        self.assertIsInstance(
            export_module,
            Gemma4VisionPatchEmbedderPrefillExportAdapter,
        )
        with torch.no_grad():
            expected = q_module(
                pixel_values,
                pixel_position_ids,
                padding_positions,
            )
            actual = export_module(pixel_values)
        torch.testing.assert_close(actual, expected)

        exported = torch.export.export(
            export_module,
            (pixel_values,),
            strict=False,
        )
        self.assertEqual(len(exported.graph_signature.user_inputs), 1)
        call_targets = {
            str(node.target)
            for node in exported.graph.nodes
            if node.op == "call_function"
        }
        self.assertNotIn("aten.embedding.default", call_targets)
        with torch.no_grad():
            exported_output = exported.module()(pixel_values)
        torch.testing.assert_close(exported_output, actual)

    def test_as_export_module_rejects_calibration_mode(self):
        """Export should reject CALIB mode because its qparams are incomplete."""
        fp_module = _make_patch_embedder(
            hidden_size=self.hidden_size,
            patch_size=self.patch_size,
            position_embedding_size=self.position_embedding_size,
        )
        q_module = QuantGemma4VisionPatchEmbedder(fp_module).eval()
        q_module.enable_calibration()
        _, pixel_position_ids, padding_positions = self._sample_inputs()

        with self.assertRaisesRegex(RuntimeError, "NO_QUANT or QUANT"):
            q_module.as_export_module(
                mode="prefill",
                pixel_position_ids=pixel_position_ids,
                padding_positions=padding_positions,
            )

    def test_export_adapter_bakes_profile_and_has_one_user_input(self):
        """The exported patch embedder should not read position IDs at runtime."""
        fp_module = _make_patch_embedder(
            hidden_size=self.hidden_size,
            patch_size=self.patch_size,
            position_embedding_size=self.position_embedding_size,
        )
        q_module = QuantGemma4VisionPatchEmbedder(fp_module).eval()
        q_module.enable_calibration()

        pixel_values, pixel_position_ids, padding_positions = self._sample_inputs()
        pixel_position_ids[:, -2:] = -1
        padding_positions[:, -2:] = True
        with torch.no_grad():
            _ = q_module(pixel_values, pixel_position_ids, padding_positions)
        q_module.freeze_qparams()

        with torch.no_grad():
            expected = q_module(
                pixel_values,
                pixel_position_ids,
                padding_positions,
            )
        export_module = q_module.as_export_module(
            mode="prefill",
            pixel_position_ids=pixel_position_ids,
            padding_positions=padding_positions,
        )

        export_buffers = dict(export_module.named_buffers())
        self.assertIn("position_embeddings_template", export_buffers)
        self.assertNotIn("padding_positions_template", export_buffers)
        self.assertEqual(
            torch.count_nonzero(
                export_buffers["position_embeddings_template"][:, -2:]
            ).item(),
            0,
        )

        with mock.patch.object(
            q_module,
            "_lookup_position_embeddings",
            side_effect=AssertionError("runtime position lookup was used"),
        ):
            with torch.no_grad():
                actual = export_module(pixel_values)
        torch.testing.assert_close(actual, expected)

        exported = torch.export.export(
            export_module,
            (pixel_values,),
            strict=False,
        )
        self.assertEqual(len(exported.graph_signature.user_inputs), 1)
        call_targets = {
            str(node.target)
            for node in exported.graph.nodes
            if node.op == "call_function"
        }
        self.assertNotIn("aten.where.self", call_targets)
        self.assertNotIn("aten.full_like.default", call_targets)
        self.assertNotIn("aten.zeros_like.default", call_targets)

        bool_nodes = [
            node.name
            for node in exported.graph.nodes
            if isinstance(node.meta.get("val"), torch.Tensor)
            and node.meta["val"].dtype == torch.bool
        ]
        self.assertEqual(bool_nodes, [])

        with torch.no_grad():
            exported_output = exported.module()(pixel_values)
        torch.testing.assert_close(exported_output, actual)

    def test_circle_export_folds_pixel_normalization_into_fully_connected(self):
        """Export one biased FC without pixel-normalization Sub or Mul ops."""
        fp_module = _make_patch_embedder(
            hidden_size=self.hidden_size,
            patch_size=self.patch_size,
            position_embedding_size=self.position_embedding_size,
        )
        q_module = QuantGemma4VisionPatchEmbedder(fp_module).eval()
        q_module.enable_calibration()

        pixel_values, pixel_position_ids, padding_positions = self._sample_inputs()
        with torch.no_grad():
            _ = q_module(pixel_values, pixel_position_ids, padding_positions)
        q_module.freeze_qparams()

        export_module = q_module.as_export_module(
            mode="prefill",
            pixel_position_ids=pixel_position_ids,
            padding_positions=padding_positions,
        ).eval()
        circle_model = tico.convert(export_module, (pixel_values,))
        model = model_from_bytes(circle_model.circle_binary)
        graph = model.subgraphs[0]

        sub_code = circle.BuiltinOperator.BuiltinOperator.SUB
        mul_code = circle.BuiltinOperator.BuiltinOperator.MUL
        add_code = circle.BuiltinOperator.BuiltinOperator.ADD
        fc_code = circle.BuiltinOperator.BuiltinOperator.FULLY_CONNECTED
        operators_by_code = {
            code: [
                operator
                for operator in graph.operators
                if _builtin_code(model, operator) == code
            ]
            for code in (sub_code, mul_code, add_code, fc_code)
        }

        self.assertEqual(operators_by_code[sub_code], [])
        self.assertEqual(operators_by_code[mul_code], [])
        self.assertEqual(len(operators_by_code[add_code]), 1)
        self.assertEqual(len(operators_by_code[fc_code]), 1)

        expected_type = circle.TensorType.TensorType.UINT8
        add = operators_by_code[add_code][0]
        self.assertEqual(len(add.inputs), 2)
        self.assertEqual(len(add.outputs), 1)
        for tensor_index in (*add.inputs, *add.outputs):
            self.assertEqual(graph.tensors[tensor_index].type, expected_type)

        fc = operators_by_code[fc_code][0]
        self.assertEqual(len(fc.inputs), 3)
        self.assertEqual(len(fc.outputs), 1)
        bias = graph.tensors[fc.inputs[2]]
        self.assertNotEqual(bias.buffer, 0)
        self.assertIsNotNone(bias.quantization)


if __name__ == "__main__":
    unittest.main()
