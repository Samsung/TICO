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

import unittest

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.recipes.override_policies import (
    apply_ptq_override_policies_to_config,
    ComponentQuantTargetInfo,
    QuantTargetResolverContext,
)
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.observers.mx import MXObserver


class TestPtqOverridePolicies(unittest.TestCase):
    def _text_context(self, num_layers: int = 2) -> QuantTargetResolverContext:
        return QuantTargetResolverContext(
            components={
                "text": ComponentQuantTargetInfo(
                    name="text",
                    layer_path_prefixes=(("model", "layers"),),
                    num_layers=num_layers,
                    op_aliases={
                        "linear": (
                            "self_attn.q_proj",
                            "self_attn.k_proj",
                            "self_attn.v_proj",
                            "self_attn.o_proj",
                            "mlp.gate_proj",
                            "mlp.up_proj",
                            "mlp.down_proj",
                        )
                    },
                )
            }
        )

    def _text_vision_context(self) -> QuantTargetResolverContext:
        return QuantTargetResolverContext(
            components={
                "text": ComponentQuantTargetInfo(
                    name="text",
                    layer_path_prefixes=(("model", "language_model", "layers"),),
                    num_layers=2,
                    op_aliases={"linear": ("mlp.down_proj",)},
                ),
                "vision": ComponentQuantTargetInfo(
                    name="vision",
                    layer_path_prefixes=(("model", "visual", "blocks"),),
                    num_layers=1,
                    op_aliases={"linear": ("attn.qkv",)},
                ),
            }
        )

    def test_named_mx_spec_applies_to_all_linear_activations(self):
        qcfg = PTQConfig()
        stage_cfg = {
            "specs": {
                "mx_fp8_act": {
                    "kind": "mx",
                    "elem_format": "fp8_e4m3",
                    "axis": -1,
                }
            },
            "override_policies": [
                {
                    "name": "all_text_linear_activations_mx",
                    "target": {
                        "component": "text",
                        "layers": "all",
                        "op_type": "linear",
                        "observer_role": "activation",
                    },
                    "spec": "mx_fp8_act",
                }
            ],
        }

        apply_ptq_override_policies_to_config(qcfg, stage_cfg, self._text_context())

        act_out = qcfg.overrides["model"]["layers"]["1"]["mlp"]["down_proj"][  # type: ignore[index]
            "act_out"
        ]
        act_in = qcfg.overrides["model"]["layers"]["0"]["self_attn"]["q_proj"][  # type: ignore[index]
            "act_in"
        ]
        self.assertIs(act_out["observer"], MXObserver)
        self.assertIs(act_in["observer"], MXObserver)

    def test_specific_policy_wins_over_broader_policy(self):
        qcfg = PTQConfig()
        stage_cfg = {
            "specs": {
                "mx_fp8_act": {
                    "kind": "mx",
                    "elem_format": "fp8_e4m3",
                    "axis": -1,
                },
                "int16_act": {"kind": "affine", "dtype": "int16"},
            },
            "override_policies": [
                {
                    "name": "layer_1_down_proj_output_int16",
                    "target": {
                        "component": "text",
                        "layers": [1],
                        "module": "mlp.down_proj",
                        "observers": ["act_out"],
                    },
                    "spec": "int16_act",
                },
                {
                    "name": "all_text_linear_activations_mx",
                    "target": {
                        "component": "text",
                        "layers": "all",
                        "op_type": "linear",
                        "observer_role": "activation",
                    },
                    "spec": "mx_fp8_act",
                },
            ],
        }

        apply_ptq_override_policies_to_config(qcfg, stage_cfg, self._text_context())

        act_out = qcfg.overrides["model"]["layers"]["1"]["mlp"]["down_proj"][  # type: ignore[index]
            "act_out"
        ]
        self.assertEqual(act_out["dtype"], DType.int(16))

    def test_raw_overrides_win_over_selector_policies(self):
        qcfg = PTQConfig()
        stage_cfg = {
            "specs": {
                "mx_fp8_act": {
                    "kind": "mx",
                    "elem_format": "fp8_e4m3",
                    "axis": -1,
                }
            },
            "override_policies": [
                {
                    "name": "all_text_linear_activations_mx",
                    "target": {
                        "component": "text",
                        "layers": "all",
                        "op_type": "linear",
                        "observer_role": "activation",
                    },
                    "spec": "mx_fp8_act",
                }
            ],
            "raw_overrides": {
                "model.layers.1.mlp.down_proj.act_out": {
                    "kind": "affine",
                    "dtype": "int16",
                }
            },
        }

        apply_ptq_override_policies_to_config(qcfg, stage_cfg, self._text_context())

        act_out = qcfg.overrides["model"]["layers"]["1"]["mlp"]["down_proj"][  # type: ignore[index]
            "act_out"
        ]
        self.assertEqual(act_out["dtype"], DType.int(16))

    def test_component_all_targets_text_and_vision_components(self):
        qcfg = PTQConfig()
        stage_cfg = {
            "override_policies": [
                {
                    "name": "all_linear_inputs_int16",
                    "target": {
                        "component": "all",
                        "layers": "all",
                        "op_type": "linear",
                        "observer_role": "input_activation",
                    },
                    "spec": {"kind": "affine", "dtype": "int16"},
                }
            ]
        }

        apply_ptq_override_policies_to_config(
            qcfg,
            stage_cfg,
            self._text_vision_context(),
        )

        text_act = qcfg.overrides["model"]["language_model"]["layers"]["1"]["mlp"][  # type: ignore[index]
            "down_proj"
        ][
            "act_in"
        ]
        vision_act = qcfg.overrides["model"]["visual"]["blocks"]["0"]["attn"][  # type: ignore[index]
            "qkv"
        ][
            "act_in"
        ]
        self.assertEqual(text_act["dtype"], DType.int(16))
        self.assertEqual(vision_act["dtype"], DType.int(16))

    def test_missing_component_requires_allow_empty(self):
        qcfg = PTQConfig()
        stage_cfg = {
            "override_policies": [
                {
                    "name": "vision_policy",
                    "target": {
                        "component": "vision",
                        "layers": "all",
                        "op_type": "linear",
                        "observer_role": "activation",
                    },
                    "spec": "int16",
                }
            ]
        }

        with self.assertRaises(ValueError):
            apply_ptq_override_policies_to_config(qcfg, stage_cfg, self._text_context())

        stage_cfg["override_policies"][0]["allow_empty"] = True  # type: ignore[assignment]
        apply_ptq_override_policies_to_config(qcfg, stage_cfg, self._text_context())
        self.assertEqual(qcfg.overrides, {})

    def test_rejects_out_of_range_layers(self):
        qcfg = PTQConfig()
        stage_cfg = {
            "override_policies": [
                {
                    "name": "bad_layer",
                    "target": {
                        "component": "text",
                        "layers": [3],
                        "module": "mlp.down_proj",
                        "observers": ["act_out"],
                    },
                    "spec": "int16",
                }
            ]
        }

        with self.assertRaises(ValueError):
            apply_ptq_override_policies_to_config(qcfg, stage_cfg, self._text_context())


if __name__ == "__main__":
    unittest.main()
