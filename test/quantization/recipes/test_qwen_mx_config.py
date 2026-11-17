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

try:
    from quantization.recipes.optional_dependency_stubs import (
        install_optional_dependency_stubs,
    )
except ModuleNotFoundError:
    from test.quantization.recipes.optional_dependency_stubs import (
        install_optional_dependency_stubs,
    )

install_optional_dependency_stubs()

import unittest
from pathlib import Path
from types import SimpleNamespace

from tico.quantization.recipes.adapters.qwen3_vl import Qwen3VLAdapter
from tico.quantization.recipes.config import load_recipe_config
from tico.quantization.recipes.context import RecipeContext
from tico.quantization.recipes.override_policies import apply_ptq_override_policies
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.observers.mx import MXObserver


class TestQwenMxConfig(unittest.TestCase):
    def test_existing_observers_define_affine_layout_and_mx_compute(self):
        cfg = load_recipe_config(
            Path(__file__).resolve().parents[3]
            / "tico/quantization/examples/configs/qwen3_vl_eval_suite_mx_override_polices.yaml"
        )
        model = SimpleNamespace(
            config=SimpleNamespace(
                text_config=SimpleNamespace(num_hidden_layers=1),
                vision_config=SimpleNamespace(num_hidden_layers=1),
            )
        )
        adapter = Qwen3VLAdapter()
        ctx = RecipeContext(cfg=cfg, adapter=adapter, model=model)
        stage = next(s for s in cfg["pipeline"] if s["name"] == "ptq")
        qcfg = adapter.build_ptq_config(ctx, stage)
        apply_ptq_override_policies(
            qcfg, stage, family="qwen3_vl", model=model, adapter=adapter
        )
        text = qcfg.child("model").child("language_model").child("layers").child("0")
        vision = qcfg.child("model").child("visual").child("blocks").child("0")
        for parent, modules in (
            (text, ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")),
            (vision, ("attn.qkv",)),
        ):
            for module in modules:
                with self.subTest(module=module):
                    local = parent
                    for scope in module.split("."):
                        local = local.child(scope)
                    self.assertEqual(
                        local.get_kwargs("act_out")["dtype"], DType.int(16)
                    )
                    self.assertIs(local.get_kwargs("act_in")["observer"], MXObserver)
        for parent, attention, output in (
            (text, "self_attn", "o_proj"),
            (vision, "attn", "proj"),
        ):
            local = parent.child(attention)
            self.assertIs(local.get_kwargs("softmax")["observer"], MXObserver)
            self.assertIs(
                local.child(output).get_kwargs("act_in")["observer"], MXObserver
            )
            self.assertEqual(local.get_role_kwargs("attn_out")["dtype"], DType.int(16))
        self.assertIs(
            text.child("self_attn").get_kwargs("mask_add")["observer"], MXObserver
        )
        self.assertIs(vision.child("attn").get_kwargs("logits")["observer"], MXObserver)
        self.assertTrue(cfg["export"]["move_affine_quantize_before_layout"])
        self.assertTrue(cfg["export"]["legalize_mx_affine_domains"])
