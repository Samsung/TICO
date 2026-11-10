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

import contextlib
import io
import unittest
from unittest.mock import patch

import tico.quantization.recipes.qparams as qparams
import tico.quantization.recipes.stages.ptq as ptq_mod

import torch
from tico.quantization.algorithm.gptq.quant import quantize, Quantizer
from tico.quantization.config import affine, PTQConfig
from tico.quantization.public_interface import QUANTIZER_ATTRIBUTE_NAME
from tico.quantization.recipes.context import RecipeContext
from tico.quantization.recipes.stages.ptq import PTQStage
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.quantizer import PTQQuantizer
from tico.quantization.wrapq.wrappers.nn.quant_linear import QuantLinear


class DummyAdapter:
    """Adapter fake used by the GPTQ-to-PTQ handoff regression tests."""

    family = "llama"

    def __init__(self, weight_bits=4, calibration_input=None):
        self.calibrated_model = None
        self.weight_bits = weight_bits
        self.calibration_input = calibration_input

    def build_ptq_config(self, ctx, stage_cfg):
        """Build the independent target PTQ weight policy."""
        return PTQConfig(weight=affine(DType.uint(self.weight_bits)))

    def calibrate_prepared_model(self, ctx, model, stage_cfg):
        """Record the prepared model passed to calibration."""
        self.calibrated_model = model
        if self.calibration_input is not None:
            model(self.calibration_input)


class TestPTQQParamHandoff(unittest.TestCase):
    @staticmethod
    def _prepare_linear_model(model, config):
        """Supply a real prepared tree without loading unrelated model wrappers."""
        wrapper = QuantLinear(model[0], qcfg=config, fp_name="0")
        wrapper.enable_calibration()
        prepared = torch.nn.Sequential(wrapper)
        if hasattr(model, "quantizers"):
            prepared.quantizers = model.quantizers
        setattr(prepared, QUANTIZER_ATTRIBUTE_NAME, PTQQuantizer(config))
        return prepared

    @staticmethod
    def _quantized_model(bits=4):
        """Build GPTQ-grid weights for stage calibration and conversion tests."""
        linear = torch.nn.Linear(2, 1, bias=False)
        model = torch.nn.Sequential(linear)
        with torch.no_grad():
            linear.weight.copy_(torch.tensor([[-1.0, 2.0]]))
        quantizer = Quantizer()
        quantizer.configure(bits, perchannel=True, sym=False)
        quantizer.find_params(linear.weight, weight=True)
        with torch.no_grad():
            linear.weight.copy_(
                quantize(linear.weight, quantizer.scale, quantizer.zero, quantizer.maxq)
            )
        setattr(model, "quantizers", {"0": quantizer})
        return model, quantizer

    @staticmethod
    def _build_handoff_models(
        *,
        quantizer_name: str = "linear",
        observer_name: str = "linear",
    ):
        """Build a nested GPTQ owner and a prepared PTQ observer tree."""
        linear = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            linear.weight.copy_(torch.tensor([[-1.0, 2.0]]))
        quantizer = Quantizer()
        quantizer.configure(4, perchannel=True, sym=False)
        quantizer.find_params(linear.weight, weight=True)
        scale, zero = quantizer.scale, quantizer.zero
        quantizers = {quantizer_name: quantizer}

        source_model = torch.nn.Module()
        setattr(source_model, "quantizers", quantizers)

        top_level_wrapper = torch.nn.Module()
        top_level_wrapper.module = source_model
        top_level_wrapper.linear = QuantLinear(
            linear,
            qcfg=PTQConfig(weight=affine(DType.uint(4))),
            fp_name=observer_name,
        )
        top_level_wrapper.linear.enable_calibration()
        observer = top_level_wrapper.linear.obs_weight

        prepared_model = torch.nn.Module()
        prepared_model.wrapped = top_level_wrapper
        return source_model, prepared_model, observer, quantizers, scale, zero

    def test_stage_reuses_nested_owner_qparams_in_prepared_tree(self):
        """PTQStage should reuse GPTQ qparams in the prepared observer tree."""
        (
            source_model,
            prepared_model,
            observer,
            quantizers,
            scale,
            zero,
        ) = self._build_handoff_models()

        owner, found_quantizers = qparams.find_gptq_quantizers(prepared_model)
        self.assertIs(owner, source_model)
        self.assertIs(found_quantizers, quantizers)
        self.assertFalse(hasattr(prepared_model.wrapped, "quantizers"))

        adapter = DummyAdapter()
        ctx = RecipeContext(cfg={}, adapter=adapter, model=source_model)
        stdout = io.StringIO()

        with (
            patch.object(ptq_mod, "prepare", lambda model, config: prepared_model),
            patch.object(ptq_mod, "convert", lambda model: model),
        ):
            with contextlib.redirect_stdout(stdout):
                result = PTQStage().run(ctx, {"name": "ptq"})

        self.assertIs(result.model, prepared_model)
        self.assertTrue(observer._qparams_locked)
        torch.testing.assert_close(
            observer._cached_scale, scale.flatten(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            observer._cached_zp, zero.flatten().to(torch.int), rtol=0, atol=0
        )
        self.assertFalse(hasattr(source_model, "quantizers"))
        self.assertIs(adapter.calibrated_model, prepared_model)
        self.assertIn(
            "[Info] Reused GPTQ qparams for 1 PTQ weight observer(s).",
            stdout.getvalue(),
        )

    def test_stage_can_disable_gptq_qparam_reuse(self):
        """PTQStage should recompute PTQ qparams when reuse is disabled."""
        source_model, prepared_model, observer, _, _, _ = self._build_handoff_models()
        adapter = DummyAdapter()
        ctx = RecipeContext(cfg={}, adapter=adapter, model=source_model)
        stdout = io.StringIO()

        with (
            patch.object(ptq_mod, "prepare", lambda model, config: prepared_model),
            patch.object(ptq_mod, "convert", lambda model: model),
            patch.object(ptq_mod, "inject_gptq_qparams") as inject_mock,
        ):
            with contextlib.redirect_stdout(stdout):
                result = PTQStage().run(
                    ctx,
                    {"name": "ptq", "reuse_gptq_qparams": False},
                )

        inject_mock.assert_not_called()
        self.assertIs(result.model, prepared_model)
        self.assertFalse(observer.has_qparams)
        self.assertFalse(hasattr(source_model, "quantizers"))
        self.assertIs(adapter.calibrated_model, prepared_model)
        self.assertIn(
            "reuse_gptq_qparams=false",
            stdout.getvalue(),
        )

    def test_stage_rejects_zero_matched_qparam_reuse(self):
        """PTQStage should fail when GPTQ metadata matches no PTQ observer."""
        source_model, prepared_model, _, _, _, _ = self._build_handoff_models(
            quantizer_name="other",
        )
        adapter = DummyAdapter()
        ctx = RecipeContext(cfg={}, adapter=adapter, model=source_model)

        with (
            patch.object(ptq_mod, "prepare", lambda model, config: prepared_model),
            patch.object(ptq_mod, "convert", lambda model: model),
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "no PTQ weight observer reused their qparams",
                ):
                    PTQStage().run(ctx, {"name": "ptq"})

        self.assertFalse(hasattr(source_model, "quantizers"))
        self.assertIsNone(adapter.calibrated_model)

    def test_stage_requires_metadata_after_enabled_gptq_stage(self):
        """PTQStage should fail if enabled GPTQ produced no reusable metadata."""
        source_model = torch.nn.Module()
        prepared_model = torch.nn.Module()
        adapter = DummyAdapter()
        ctx = RecipeContext(
            cfg={
                "pipeline": [
                    {"name": "gptq", "enabled": True},
                    {"name": "ptq", "enabled": True},
                ]
            },
            adapter=adapter,
            model=source_model,
        )

        with (
            patch.object(ptq_mod, "prepare", lambda model, config: prepared_model),
            patch.object(ptq_mod, "convert", lambda model: model),
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "no GPTQ quantizers were found",
                ):
                    PTQStage().run(ctx, {"name": "ptq"})

        self.assertIsNone(adapter.calibrated_model)

    def test_stage_rejects_non_boolean_reuse_option(self):
        """PTQStage should require a boolean qparam reuse option."""
        source_model = torch.nn.Module()
        prepared_model = torch.nn.Module()
        adapter = DummyAdapter()
        ctx = RecipeContext(cfg={}, adapter=adapter, model=source_model)

        with (
            patch.object(ptq_mod, "prepare") as prepare_mock,
            patch.object(ptq_mod, "convert", lambda model: model),
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(
                    TypeError,
                    "reuse_gptq_qparams must be a boolean",
                ):
                    PTQStage().run(
                        ctx,
                        {"name": "ptq", "reuse_gptq_qparams": "false"},
                    )

        prepare_mock.assert_not_called()
        self.assertIsNone(adapter.calibrated_model)

    def test_real_stage_rejects_bit_mismatch_before_calibration(self):
        """Both width mismatches fail before adapter calibration or convert."""
        for source_bits, target_bits in ((4, 8), (8, 4)):
            with self.subTest(source_bits=source_bits, target_bits=target_bits):
                model, _ = self._quantized_model(source_bits)
                adapter = DummyAdapter(target_bits, torch.tensor([[0.5, 1.0]]))
                ctx = RecipeContext(cfg={}, adapter=adapter, model=model)
                with (
                    patch.object(
                        ptq_mod, "prepare", side_effect=self._prepare_linear_model
                    ),
                    patch.object(ptq_mod, "convert") as convert_mock,
                ):
                    with contextlib.redirect_stdout(io.StringIO()):
                        with self.assertRaisesRegex(ValueError, "GPTQ"):
                            PTQStage().run(ctx, {"name": "ptq"})

                self.assertIsNone(adapter.calibrated_model)
                convert_mock.assert_not_called()
                self.assertIs(ctx.model, model)
                self.assertTrue(hasattr(model, "quantizers"))

    def test_real_stage_preserves_matching_grid_through_convert(self):
        """Real injection/calibration/convert keeps the prepared GPTQ weight grid."""
        model, quantizer = self._quantized_model()
        weight = model[0].weight.detach().clone()
        adapter = DummyAdapter(4, torch.tensor([[0.5, 1.0]]))
        ctx = RecipeContext(cfg={}, adapter=adapter, model=model)

        with (
            patch.object(ptq_mod, "prepare", side_effect=self._prepare_linear_model),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            result = PTQStage().run(ctx, {"name": "ptq"})

        observer = result.model[0].get_observer("weight")
        self.assertTrue(observer._qparams_locked)
        self.assertFalse(observer.enabled)
        self.assertFalse(hasattr(result.model, "quantizers"))
        torch.testing.assert_close(
            observer._cached_scale, quantizer.scale.flatten(), rtol=0, atol=0
        )
        torch.testing.assert_close(observer.fake_quant(weight), weight, rtol=0, atol=0)

    def test_real_stage_opt_out_recomputes_target_weight_grid(self):
        """Explicit opt-out permits a new bit width and computes new qparams."""
        model, quantizer = self._quantized_model(4)
        adapter = DummyAdapter(8, torch.tensor([[0.5, 1.0]]))
        ctx = RecipeContext(cfg={}, adapter=adapter, model=model)

        with (
            patch.object(ptq_mod, "prepare", side_effect=self._prepare_linear_model),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            result = PTQStage().run(ctx, {"name": "ptq", "reuse_gptq_qparams": False})

        observer = result.model[0].get_observer("weight")
        self.assertEqual(observer.dtype, DType.uint(8))
        self.assertFalse(observer._qparams_locked)
        torch.testing.assert_close(
            observer._cached_scale, torch.tensor([3.0 / 255.0]), rtol=0, atol=0
        )
        self.assertFalse(torch.equal(observer._cached_scale, quantizer.scale.flatten()))
        self.assertIs(adapter.calibrated_model, result.model)

    def test_real_pure_ptq_needs_no_gptq_metadata(self):
        """Pure PTQ remains valid when no enabled GPTQ stage precedes it."""
        model, _ = self._quantized_model()
        del model.quantizers
        adapter = DummyAdapter(8, torch.tensor([[0.5, 1.0]]))
        ctx = RecipeContext(
            cfg={"pipeline": [{"name": "gptq", "enabled": False}]},
            adapter=adapter,
            model=model,
        )

        with (
            patch.object(ptq_mod, "prepare", side_effect=self._prepare_linear_model),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            result = PTQStage().run(ctx, {"name": "ptq"})

        observer = result.model[0].get_observer("weight")
        self.assertTrue(observer.has_qparams)
        self.assertFalse(observer._qparams_locked)
        self.assertIs(adapter.calibrated_model, result.model)


if __name__ == "__main__":
    unittest.main()
