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

"""Export preflight follows declared computation, independently of grid storage."""

import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import patch

import torch

from tico.quantization.recipes.export.circle import (
    convert_and_save_circle,
    export_full_circle,
)
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.observers.affine_base import AffineObserverBase
from tico.quantization.wrapq.qscheme import QScheme
from tico.quantization.wrapq.utils.static_export import (
    ExportCapabilityGuard,
    iter_export_execution_sites,
    validate_export_execution,
)
from tico.quantization.wrapq.wrappers.gemma4.export_adapters import (
    Gemma4TokenEmbeddingExportAdapter,
)
from tico.quantization.wrapq.wrappers.gemma4.quant_text_scaled_word_embedding import (
    QuantGemma4TextScaledWordEmbedding,
)
from tico.quantization.wrapq.wrappers.llama.export_adapters import (
    LlamaTokenEmbeddingExportAdapter,
)
from tico.quantization.wrapq.wrappers.nn.quant_linear import QuantLinear
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.qwen_vl.export_adapters import (
    Qwen3VLTextEmbeddingExportAdapter,
)
from torch import nn

from test.quantization.quant_spec_helpers import make_affine_ptq_config


class _TokenOwner(QuantModuleBase):
    def __init__(self):
        super().__init__(make_affine_ptq_config())
        self.embed_tokens = PTQWrapper(nn.Embedding(4, 2), qcfg=self.qcfg)
        self.dormant = PTQWrapper(nn.Linear(2, 2), qcfg=self.qcfg)

    def _all_observers(self):
        return ()

    def _conditional_quant_subtrees(self):
        return {"token_embeddings": ("embed_tokens",), "dormant": ("dormant",)}

    def forward(self, tokens):
        return self.embed_tokens(tokens)


class _SelectedStage(nn.Module):
    guard: ExportCapabilityGuard

    def __init__(self, selected, dormant):
        super().__init__()
        self.selected = selected
        self.dormant = dormant

    def forward(self, x):
        return self.selected(x)

    def _export_execution_sites(self, example_inputs, kwargs):
        return iter_export_execution_sites(
            self.selected, example_inputs, kwargs=kwargs, module_path="selected"
        )


class _ScaledEmbedding(nn.Embedding):
    def __init__(self):
        super().__init__(4, 2)
        self.register_buffer("embed_scale", torch.tensor(2.0))


class TestExportExecutionPreflight(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1097)
        self.x = torch.tensor([[0.25, -0.5]])

    def _linear(self, *, bf16_weight=False, bf16_input=False):
        overrides = {}
        if bf16_input:
            overrides["act_in"] = {
                "qscheme": QScheme.PER_CHANNEL_ASYMM,
                "channel_axis": 1,
            }
        wrapper = PTQWrapper(
            nn.Linear(2, 2), qcfg=make_affine_ptq_config(overrides=overrides)
        )
        wrapper.enable_calibration()
        wrapper(self.x)
        linear = cast(QuantLinear, wrapper.wrapped)
        if bf16_weight:
            cast(AffineObserverBase, linear.obs_weight).load_qparams(
                torch.tensor([0.125, 0.25], dtype=torch.bfloat16),
                torch.tensor([2, 3], dtype=torch.int32),
            )
        if bf16_input:
            cast(AffineObserverBase, linear.obs_act_in).load_qparams(
                torch.tensor([0.125, 0.25], dtype=torch.bfloat16),
                torch.tensor([2, 3], dtype=torch.int32),
            )
        wrapper.freeze_qparams()
        return wrapper

    def _write(self, module, inputs=None, *, kwargs=None):
        with tempfile.TemporaryDirectory() as tmpdir:
            convert_and_save_circle(
                module,
                (self.x,) if inputs is None else inputs,
                Path(tmpdir) / "stage.circle",
                kwargs=kwargs,
            )

    def test_selected_cpu_weight_rejects_before_forward_and_conversion(self):
        module = self._linear(bf16_weight=True)
        before = module.wrapped.obs_weight.require_frozen()
        with (
            patch("tico.quantization.recipes.export.circle.tico.convert") as convert,
            patch.object(module, "forward", wraps=module.forward) as forward,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                r"wrapped.obs_weight.*fake_quantize_per_channel_affine.*device cpu.*torch.bfloat16",
            ):
                self._write(module)
            forward.assert_not_called()
            convert.assert_not_called()
        after = module.wrapped.obs_weight.require_frozen()
        self.assertEqual(after.scale.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(before.scale, after.scale))
        self.assertEqual(before.policy, after.policy)
        self.assertEqual(
            (before.provenance, before.locked), (after.provenance, after.locked)
        )

    def test_known_input_activation_rejects_positional_and_keyword_routes(self):
        module = self._linear(bf16_input=True)
        for inputs, kwargs in (((self.x,), None), ((), {"x": self.x})):
            with (
                self.subTest(kwargs=kwargs is not None),
                patch(
                    "tico.quantization.recipes.export.circle.tico.convert"
                ) as convert,
            ):
                with self.assertRaisesRegex(
                    RuntimeError, r"wrapped.obs_act_in.*device cpu.*torch.bfloat16"
                ):
                    self._write(module, inputs, kwargs=kwargs)
                convert.assert_not_called()

    def test_fp32_scales_and_disabled_bf16_reach_conversion(self):
        supported = self._linear()
        supported_bf16 = self._linear()
        supported_bf16.wrapped.module.to(dtype=torch.bfloat16)
        bf16_input = self.x.bfloat16()
        actual = supported_bf16(bf16_input)
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertEqual(
            supported_bf16.wrapped.obs_weight.require_frozen().scale.dtype,
            torch.float32,
        )
        bypassed = self._linear(bf16_weight=True)
        bypassed.wrapped.obs_weight.disable_fake_quant()
        for module, x in (
            (supported, self.x),
            (supported_bf16, bf16_input),
            (bypassed, self.x),
        ):
            with (
                self.subTest(bypassed=module is bypassed, input_dtype=x.dtype),
                patch(
                    "tico.quantization.recipes.export.circle.tico.convert"
                ) as convert,
            ):
                self._write(module, (x,))
                convert.assert_called_once()
                convert.return_value.save.assert_called_once()
        self.assertFalse(
            bypassed.wrapped.obs_weight.require_frozen().fake_quant_enabled
        )

    def test_weight_execution_device_is_not_inferred_from_stage_input(self):
        module = self._linear(bf16_weight=True)
        with patch("tico.quantization.recipes.export.circle.tico.convert") as convert:
            with self.assertRaisesRegex(RuntimeError, r"obs_weight.*device cpu"):
                self._write(module, (torch.empty(1, 2, device="meta"),))
            convert.assert_not_called()

    def test_metadata_check_uses_actual_weight_device_not_cpu_grid_storage(self):
        module = self._linear(bf16_weight=True)
        module.wrapped.module.to("meta")
        self.assertEqual(
            module.wrapped.obs_weight.require_frozen().scale.device.type, "cpu"
        )
        sites = list(iter_export_execution_sites(module, (self.x,)))
        self.assertEqual(
            {name: device.type for name, _, device in sites},
            {
                "wrapped.obs_act_in": "cpu",
                "wrapped.obs_weight": "meta",
                "wrapped.obs_act_out": "meta",
            },
        )
        validate_export_execution(module, (self.x,))
        # Metadata-only success for Meta is not a claim that a native kernel ran.
        self.assertEqual(
            module.wrapped.obs_weight.require_frozen().scale.dtype, torch.bfloat16
        )

    def test_registered_nonparticipating_branch_is_not_preflighted(self):
        selected = self._linear()
        dormant = self._linear(bf16_weight=True)
        stage = _SelectedStage(selected, dormant)
        with (
            patch.object(
                dormant.wrapped.obs_weight,
                "validate_execution",
                wraps=dormant.wrapped.obs_weight.validate_execution,
            ) as dormant_check,
            patch("tico.quantization.recipes.export.circle.tico.convert") as convert,
        ):
            self._write(stage)
            convert.assert_called_once()
            dormant_check.assert_not_called()

    def _owner(self):
        owner = _TokenOwner()
        owner.enable_calibration()
        owner(torch.tensor([[0, 1]]))
        owner.dormant(self.x)
        owner.freeze_qparams()
        return owner

    @staticmethod
    def _adapter(family, owner):
        text = nn.Module()
        text.add_module("wrapped", owner)
        model = nn.Module()
        if family == "llama":
            model.add_module("model", text)
            return LlamaTokenEmbeddingExportAdapter(model)
        if family == "gemma4":
            return Gemma4TokenEmbeddingExportAdapter(owner)
        multimodal = nn.Module()
        multimodal.add_module("language_model", text)
        wrapped = nn.Module()
        wrapped.add_module("wrapped", multimodal)
        model.add_module("model", wrapped)
        return Qwen3VLTextEmbeddingExportAdapter(model)

    def test_all_token_adapter_families_reject_before_lookup(self):
        for family in ("llama", "gemma4", "qwen3_vl"):
            with self.subTest(family=family):
                owner = self._owner()
                observer = owner.embed_tokens.wrapped.weight_obs
                observer.load_qparams(
                    torch.full((4,), 0.125, dtype=torch.bfloat16),
                    torch.zeros(4, dtype=torch.int32),
                )
                adapter = self._adapter(family, owner)
                with (
                    patch.object(
                        owner.embed_tokens, "forward", wraps=owner.embed_tokens.forward
                    ) as lookup,
                    patch(
                        "tico.quantization.recipes.export.circle.tico.convert"
                    ) as convert,
                ):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        r"embed_tokens.wrapped.weight_obs.*device cpu.*torch.bfloat16",
                    ):
                        self._write(adapter, (torch.tensor([[0, 1]]),))
                    lookup.assert_not_called()
                    convert.assert_not_called()

    def test_token_adapter_does_not_traverse_retained_capability_owner(self):
        for family in ("llama", "gemma4", "qwen3_vl"):
            with self.subTest(family=family):
                owner = self._owner()
                dormant = owner.dormant.wrapped.obs_weight
                dormant.load_qparams(
                    torch.full((2,), 0.125, dtype=torch.bfloat16),
                    torch.zeros(2, dtype=torch.int32),
                )
                adapter = self._adapter(family, owner)
                with (
                    patch.object(
                        dormant, "validate_execution", wraps=dormant.validate_execution
                    ) as dormant_check,
                    patch(
                        "tico.quantization.recipes.export.circle.tico.convert"
                    ) as convert,
                ):
                    self._write(adapter, (torch.tensor([[0, 1]]),))
                    convert.assert_called_once()
                    dormant_check.assert_not_called()

    def test_unavailable_capability_rejects_before_site_collection(self):
        owner = _TokenOwner()
        adapter = self._adapter("llama", owner)
        owner.enable_calibration()
        owner.dormant(self.x)
        owner.freeze_qparams()
        with (
            patch.object(
                adapter,
                "_export_execution_sites",
                wraps=adapter._export_execution_sites,
            ) as collect,
            patch("tico.quantization.recipes.export.circle.tico.convert") as convert,
        ):
            with self.assertRaisesRegex(RuntimeError, "token_embeddings.*unavailable"):
                self._write(adapter, (torch.tensor([[0, 1]]),))
            collect.assert_not_called()
            convert.assert_not_called()

    def test_stale_mode_rejects_before_site_collection(self):
        owner = self._linear()
        stage = _SelectedStage(owner, nn.Identity())
        stage.guard = ExportCapabilityGuard(owner, "required", mode=Mode.QUANT)
        owner._mode = Mode.NO_QUANT
        with (
            patch.object(
                stage, "_export_execution_sites", wraps=stage._export_execution_sites
            ) as collect,
            patch("tico.quantization.recipes.export.circle.tico.convert") as convert,
        ):
            with self.assertRaisesRegex(RuntimeError, "mode does not match"):
                self._write(stage)
            collect.assert_not_called()
            convert.assert_not_called()

    def test_gemma_scaled_embedding_declares_lookup_sites(self):
        module = QuantGemma4TextScaledWordEmbedding(
            _ScaledEmbedding(), qcfg=make_affine_ptq_config()
        )
        module.enable_calibration()
        module(torch.tensor([[0, 1]]))
        module.freeze_qparams()
        cast(AffineObserverBase, module.obs_weight).load_qparams(
            torch.full((4,), 0.125, dtype=torch.bfloat16),
            torch.zeros(4, dtype=torch.int32),
        )
        with patch("tico.quantization.recipes.export.circle.tico.convert") as convert:
            with self.assertRaisesRegex(
                RuntimeError, r"obs_weight.*device cpu.*torch.bfloat16"
            ):
                self._write(module, (torch.tensor([[0, 1]]),))
            convert.assert_not_called()

    def test_no_quant_mode_does_not_preflight_inactive_observers(self):
        module = self._linear(bf16_weight=True)
        module.wrapped._mode = Mode.NO_QUANT
        with patch("tico.quantization.recipes.export.circle.tico.convert") as convert:
            self._write(module)
            convert.assert_called_once()

    def test_full_export_preflight_rejects_before_conversion(self):
        module = self._linear(bf16_weight=True)
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("tico.quantization.recipes.export.circle.tico.convert") as convert,
        ):
            with self.assertRaisesRegex(
                RuntimeError, r"wrapped.obs_weight.*device cpu"
            ):
                export_full_circle(
                    model=module, example_input=self.x, output_dir=tmpdir
                )
            convert.assert_not_called()
