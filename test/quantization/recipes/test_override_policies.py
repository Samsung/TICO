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
from types import SimpleNamespace
from typing import Any

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.config.specs import affine
from tico.quantization.recipes.override_policies import (
    apply_ptq_override_policies,
    apply_ptq_override_policies_to_config,
    ComponentQuantTargetInfo,
    QuantTargetResolverContext,
)
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.observers.mx import MXObserver


_LAYER0_DOWN_OUT = ("model", "layers", "0", "mlp", "down_proj", "act_out")
_LAYER1_DOWN_OUT = ("model", "layers", "1", "mlp", "down_proj", "act_out")
_LAYER0_Q_WEIGHT = ("model", "layers", "0", "self_attn", "q_proj", "weight")


def _down_proj_out_policy(name: str, layers, spec) -> dict:
    """Return a selector policy targeting ``mlp.down_proj.act_out``."""
    return {
        "name": name,
        "target": {
            "component": "text",
            "layers": layers,
            "module": "mlp.down_proj",
            "observers": ["act_out"],
        },
        "spec": spec,
    }


def _leaf(qcfg: PTQConfig, path: tuple[str, ...]) -> Any:
    """Return the normalized override leaf stored at ``path``."""
    current: Any = qcfg.overrides
    for key in path:
        current = current[key]
    return current


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

    def test_applied_paths_collects_exact_recipe_targets(self):
        """applied_paths should hold exactly the paths recipe overrides targeted."""
        cases: list[
            tuple[
                str,
                dict[str, Any],
                dict[tuple[str, ...], Any] | None,
                set[tuple[str, ...]],
                dict[tuple[str, ...], DType],
            ]
        ] = [
            (
                "selector_only",
                {"override_policies": [_down_proj_out_policy("p", [0], "int16")]},
                None,
                {_LAYER0_DOWN_OUT},
                {_LAYER0_DOWN_OUT: DType.int(16)},
            ),
            (
                "raw_only",
                {"raw_overrides": {"model.layers.1.mlp.down_proj.act_out": "int16"}},
                None,
                {_LAYER1_DOWN_OUT},
                {_LAYER1_DOWN_OUT: DType.int(16)},
            ),
            (
                "selector_and_raw_same_path",
                {
                    "override_policies": [_down_proj_out_policy("p", [0], "int8")],
                    "raw_overrides": {
                        "model.layers.0.mlp.down_proj.act_out": "int16",
                    },
                },
                None,
                {_LAYER0_DOWN_OUT},
                {_LAYER0_DOWN_OUT: DType.int(16)},
            ),
            (
                "broad_and_specific_same_path",
                {
                    "override_policies": [
                        _down_proj_out_policy("specific", [1], "int16"),
                        _down_proj_out_policy("broad", "all", "int8"),
                    ]
                },
                None,
                {_LAYER0_DOWN_OUT, _LAYER1_DOWN_OUT},
                {_LAYER0_DOWN_OUT: DType.int(8), _LAYER1_DOWN_OUT: DType.int(16)},
            ),
            (
                "explicit_target_equal_to_existing_value",
                {"override_policies": [_down_proj_out_policy("p", [0], "int16")]},
                {_LAYER0_DOWN_OUT: affine(DType.int(16))},
                {_LAYER0_DOWN_OUT},
                {_LAYER0_DOWN_OUT: DType.int(16)},
            ),
            (
                "unrelated_adapter_default_not_collected",
                {"override_policies": [_down_proj_out_policy("p", [0], "int16")]},
                {_LAYER0_Q_WEIGHT: affine(DType.uint(4))},
                {_LAYER0_DOWN_OUT},
                {_LAYER0_DOWN_OUT: DType.int(16), _LAYER0_Q_WEIGHT: DType.uint(4)},
            ),
            (
                "disabled_policy",
                {
                    "override_policies": [
                        {**_down_proj_out_policy("p", [0], "int16"), "enabled": False}
                    ]
                },
                None,
                set(),
                {},
            ),
            (
                "allow_empty_policy",
                {
                    "override_policies": [
                        {
                            "name": "vision",
                            "allow_empty": True,
                            "target": {
                                "component": "vision",
                                "layers": "all",
                                "op_type": "linear",
                                "observer_role": "activation",
                            },
                            "spec": "int16",
                        }
                    ]
                },
                None,
                set(),
                {},
            ),
            (
                "empty_overrides",
                {"override_policies": [], "raw_overrides": {}},
                None,
                set(),
                {},
            ),
        ]

        for label, stage_cfg, preset, expected_paths, expected_dtypes in cases:
            with self.subTest(label):
                qcfg = PTQConfig()
                for path, value in (preset or {}).items():
                    qcfg.set_override(path, value)
                applied: set[tuple[str, ...]] = set()

                result = apply_ptq_override_policies_to_config(
                    qcfg,
                    stage_cfg,
                    self._text_context(),
                    applied_paths=applied,
                )

                self.assertIs(result, qcfg)
                self.assertEqual(applied, expected_paths)
                for path, dtype in expected_dtypes.items():
                    self.assertEqual(_leaf(qcfg, path)["dtype"], dtype)
                if not expected_dtypes:
                    self.assertEqual(qcfg.overrides, {})

    def test_apply_apis_return_same_config_object(self):
        """Both apply APIs should return the input PTQConfig with or without collection."""
        llama_model = SimpleNamespace(
            model=SimpleNamespace(layers=[object(), object()])
        )
        stage_cfg = {"override_policies": [_down_proj_out_policy("p", [1], "int16")]}

        with self.subTest("to_config_without_collection"):
            qcfg = PTQConfig()
            result = apply_ptq_override_policies_to_config(
                qcfg, stage_cfg, self._text_context()
            )
            self.assertIs(result, qcfg)
            self.assertEqual(_leaf(qcfg, _LAYER1_DOWN_OUT)["dtype"], DType.int(16))

        with self.subTest("to_config_with_collection"):
            qcfg = PTQConfig()
            applied: set[tuple[str, ...]] = set()
            result = apply_ptq_override_policies_to_config(
                qcfg, stage_cfg, self._text_context(), applied_paths=applied
            )
            self.assertIs(result, qcfg)
            self.assertEqual(applied, {_LAYER1_DOWN_OUT})

        with self.subTest("family_api_without_collection"):
            qcfg = PTQConfig()
            result = apply_ptq_override_policies(
                qcfg, stage_cfg, family="llama", model=llama_model
            )
            self.assertIs(result, qcfg)
            self.assertEqual(_leaf(qcfg, _LAYER1_DOWN_OUT)["dtype"], DType.int(16))

        with self.subTest("family_api_with_collection"):
            qcfg = PTQConfig()
            applied = set()
            result = apply_ptq_override_policies(
                qcfg,
                stage_cfg,
                family="llama",
                model=llama_model,
                applied_paths=applied,
            )
            self.assertIs(result, qcfg)
            self.assertEqual(applied, {_LAYER1_DOWN_OUT})

        with self.subTest("family_api_specs_only_is_still_ignored"):
            # No override payload: the family API returns early without
            # validating specs or the model family, and collects nothing.
            qcfg = PTQConfig()
            applied = set()
            result = apply_ptq_override_policies(
                qcfg,
                {"specs": {"bad": "not-a-spec"}},
                family="unsupported_family",
                model=object(),
                applied_paths=applied,
            )
            self.assertIs(result, qcfg)
            self.assertEqual(qcfg.overrides, {})
            self.assertEqual(applied, set())

        with self.subTest("family_api_raw_only_still_validates_family"):
            applied = set()
            with self.assertRaises(ValueError) as cm:
                apply_ptq_override_policies(
                    PTQConfig(),
                    {
                        "raw_overrides": {
                            "model.layers.0.mlp.down_proj.act_out": "int16"
                        }
                    },
                    family="unsupported_family",
                    model=object(),
                    applied_paths=applied,
                )
            self.assertIn("not supported for model family", str(cm.exception))
            self.assertEqual(applied, set())

    def test_applied_paths_preserves_existing_entries(self):
        """Collection should add paths without clearing what the caller stored."""
        sentinel = ("caller", "sentinel")
        applied: set[tuple[str, ...]] = {sentinel}

        apply_ptq_override_policies_to_config(
            PTQConfig(),
            {"override_policies": [_down_proj_out_policy("p", [0], "int16")]},
            self._text_context(),
            applied_paths=applied,
        )

        self.assertEqual(applied, {sentinel, _LAYER0_DOWN_OUT})

    def test_raw_override_failure_keeps_applied_selector_mutation(self):
        """A raw override error should surface after selector overrides were applied."""
        qcfg = PTQConfig()
        applied: set[tuple[str, ...]] = set()
        stage_cfg = {
            "override_policies": [_down_proj_out_policy("p", [0], "int16")],
            "raw_overrides": ["not-a-mapping"],
        }

        with self.assertRaises(TypeError) as cm:
            apply_ptq_override_policies_to_config(
                qcfg, stage_cfg, self._text_context(), applied_paths=applied
            )

        self.assertIn("ptq.raw_overrides must be a mapping", str(cm.exception))
        self.assertEqual(_leaf(qcfg, _LAYER0_DOWN_OUT)["dtype"], DType.int(16))
        self.assertEqual(applied, {_LAYER0_DOWN_OUT})

    def test_invalid_selector_failure_applies_nothing(self):
        """A selector error should surface before any override is applied."""
        qcfg = PTQConfig()
        applied: set[tuple[str, ...]] = set()
        stage_cfg = {
            "override_policies": [
                _down_proj_out_policy("ok", [0], "int16"),
                _down_proj_out_policy("bad", [5], "int16"),
            ],
            "raw_overrides": {"model.layers.1.mlp.down_proj.act_out": "int16"},
        }

        with self.assertRaises(ValueError) as cm:
            apply_ptq_override_policies_to_config(
                qcfg, stage_cfg, self._text_context(), applied_paths=applied
            )

        self.assertIn("out of range", str(cm.exception))
        self.assertEqual(qcfg.overrides, {})
        self.assertEqual(applied, set())


if __name__ == "__main__":
    unittest.main()
