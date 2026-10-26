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
    from optional_dependency_stubs import install_optional_dependency_stubs

install_optional_dependency_stubs()

import unittest
from types import SimpleNamespace
from typing import Any

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.config.specs import affine
from tico.quantization.recipes.override_policies import (
    apply_ptq_override_policies,
    apply_ptq_override_policies_to_config,
    build_quant_target_resolver_context,
    ComponentQuantTargetInfo,
    QuantTargetResolverContext,
)
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.observers.mx import MXObserver

_LAYER0_DOWN_OUT = ("model", "layers", "0", "mlp", "down_proj", "act_out")
_LAYER1_DOWN_OUT = ("model", "layers", "1", "mlp", "down_proj", "act_out")
_LAYER0_Q_WEIGHT = ("model", "layers", "0", "self_attn", "q_proj", "weight")

# Hand-written structure expectations for the built-in families. These pin the
# selector metadata the adapters declare; they are deliberately not derived
# from the resolver under test.
_HF_DECODER_LINEARS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
_GEMMA4_TEXT_LINEARS = _HF_DECODER_LINEARS + (
    "per_layer_input_gate",
    "per_layer_projection",
)
_QWEN3_VL_VISION_LINEARS = (
    "attn.qkv",
    "attn.proj",
    "mlp.linear_fc1",
    "mlp.linear_fc2",
)
_OBSERVER_ROLE_ALIASES = {
    "activation": ("act_in", "act_out"),
    "parameter": ("weight",),
    "input_activation": ("act_in",),
    "output_activation": ("act_out",),
}
_GEMMA4_TEXT_PREFIXES = (
    ("layers",),
    ("model", "layers"),
    ("language_model", "layers"),
    ("model", "language_model", "layers"),
)
_GEMMA4_VISION_PREFIXES = (
    ("vision_tower", "encoder", "layers"),
    ("model", "vision_tower", "encoder", "layers"),
)


def _llama_model(num_layers: int) -> SimpleNamespace:
    return SimpleNamespace(model=SimpleNamespace(layers=[object()] * num_layers))


def _qwen3_vl_model(
    *,
    text_config: Any = None,
    vision_config: Any = None,
    config_num_hidden_layers: int | None = None,
    structure_text_layers: int | None = None,
    structure_vision_blocks: int | None = None,
) -> SimpleNamespace:
    config_fields: dict[str, Any] = {}
    if text_config is not None:
        config_fields["text_config"] = text_config
    if vision_config is not None:
        config_fields["vision_config"] = vision_config
    if config_num_hidden_layers is not None:
        config_fields["num_hidden_layers"] = config_num_hidden_layers
    inner_fields: dict[str, Any] = {}
    if structure_text_layers is not None:
        inner_fields["language_model"] = SimpleNamespace(
            layers=[object()] * structure_text_layers
        )
    if structure_vision_blocks is not None:
        inner_fields["visual"] = SimpleNamespace(
            blocks=[object()] * structure_vision_blocks
        )
    return SimpleNamespace(
        config=SimpleNamespace(**config_fields), model=SimpleNamespace(**inner_fields)
    )


def _gemma4_model(
    *,
    get_text_config: Any = None,
    text_config: Any = None,
    vision_config: Any = None,
) -> SimpleNamespace:
    config_fields: dict[str, Any] = {}
    if get_text_config is not None:
        config_fields["get_text_config"] = lambda: get_text_config
    if text_config is not None:
        config_fields["text_config"] = text_config
    if vision_config is not None:
        config_fields["vision_config"] = vision_config
    return SimpleNamespace(config=SimpleNamespace(**config_fields))


class TestBuiltinFamilyQuantTargetContexts(unittest.TestCase):
    """Characterize the structure metadata resolved for each built-in family."""

    def _assert_component(
        self,
        component: ComponentQuantTargetInfo,
        *,
        name: str,
        prefixes: tuple[tuple[str, ...], ...],
        num_layers: int,
        linear: tuple[str, ...],
    ) -> None:
        self.assertEqual(component.name, name)
        self.assertEqual(component.layer_path_prefixes, prefixes)
        self.assertEqual(component.num_layers, num_layers)
        self.assertEqual(dict(component.op_aliases), {"linear": linear})
        self.assertEqual(dict(component.observer_aliases), _OBSERVER_ROLE_ALIASES)

    def test_llama_context_uses_actual_model_layers_length(self):
        context = build_quant_target_resolver_context(
            family="llama", model=_llama_model(3)
        )
        self.assertIsInstance(context, QuantTargetResolverContext)
        self.assertEqual(tuple(context.components), ("text",))
        self._assert_component(
            context.components["text"],
            name="text",
            prefixes=(("model", "layers"),),
            num_layers=3,
            linear=_HF_DECODER_LINEARS,
        )

    def test_llama_context_requires_model_layers(self):
        for label, model in (
            ("missing_layers", SimpleNamespace(model=SimpleNamespace())),
            ("missing_model", SimpleNamespace(config=SimpleNamespace())),
        ):
            with self.subTest(label):
                with self.assertRaises(ValueError) as cm:
                    build_quant_target_resolver_context(family="llama", model=model)
                self.assertEqual(
                    str(cm.exception),
                    "Cannot determine llama layers; missing attribute model.layers.",
                )

    def test_qwen3_vl_context_prefixes_and_aliases(self):
        model = _qwen3_vl_model(
            text_config=SimpleNamespace(num_hidden_layers=2),
            vision_config=SimpleNamespace(num_hidden_layers=1),
        )
        context = build_quant_target_resolver_context(family="qwen3_vl", model=model)
        self.assertEqual(tuple(context.components), ("text", "vision"))
        self._assert_component(
            context.components["text"],
            name="text",
            prefixes=(("model", "language_model", "layers"),),
            num_layers=2,
            linear=_HF_DECODER_LINEARS,
        )
        self._assert_component(
            context.components["vision"],
            name="vision",
            prefixes=(("model", "visual", "blocks"),),
            num_layers=1,
            linear=_QWEN3_VL_VISION_LINEARS,
        )

    def test_qwen3_vl_layer_count_lookup_order(self):
        vision_ok = SimpleNamespace(num_hidden_layers=1)
        text_ok = SimpleNamespace(num_hidden_layers=2)
        text_cases: list[tuple[str, SimpleNamespace, int]] = [
            (
                "text_config_wins_over_config_and_structure",
                _qwen3_vl_model(
                    text_config=SimpleNamespace(num_hidden_layers=2),
                    vision_config=vision_ok,
                    config_num_hidden_layers=5,
                    structure_text_layers=7,
                ),
                2,
            ),
            (
                "config_num_hidden_layers_when_text_config_lacks_it",
                _qwen3_vl_model(
                    text_config=SimpleNamespace(),
                    vision_config=vision_ok,
                    config_num_hidden_layers=5,
                    structure_text_layers=7,
                ),
                5,
            ),
            (
                "structure_when_no_config_count",
                _qwen3_vl_model(vision_config=vision_ok, structure_text_layers=7),
                7,
            ),
        ]
        for label, model, expected in text_cases:
            with self.subTest(label):
                context = build_quant_target_resolver_context(
                    family="qwen3_vl", model=model
                )
                self.assertEqual(context.components["text"].num_layers, expected)

        vision_cases: list[tuple[str, SimpleNamespace, int]] = [
            (
                "num_hidden_layers_first",
                _qwen3_vl_model(
                    text_config=text_ok,
                    vision_config=SimpleNamespace(
                        num_hidden_layers=3, num_layers=4, depth=5
                    ),
                    structure_vision_blocks=6,
                ),
                3,
            ),
            (
                "num_layers_second",
                _qwen3_vl_model(
                    text_config=text_ok,
                    vision_config=SimpleNamespace(num_layers=4, depth=5),
                    structure_vision_blocks=6,
                ),
                4,
            ),
            (
                "depth_third",
                _qwen3_vl_model(
                    text_config=text_ok,
                    vision_config=SimpleNamespace(depth=5),
                    structure_vision_blocks=6,
                ),
                5,
            ),
            (
                "structure_last",
                _qwen3_vl_model(text_config=text_ok, structure_vision_blocks=6),
                6,
            ),
        ]
        for label, model, expected in vision_cases:
            with self.subTest(label):
                context = build_quant_target_resolver_context(
                    family="qwen3_vl", model=model
                )
                self.assertEqual(context.components["vision"].num_layers, expected)

        with self.subTest("missing_text_structure"):
            with self.assertRaises(ValueError) as cm:
                build_quant_target_resolver_context(
                    family="qwen3_vl",
                    model=_qwen3_vl_model(vision_config=vision_ok),
                )
            self.assertEqual(
                str(cm.exception),
                "Cannot determine Qwen3-VL text layers; missing attribute "
                "model.language_model.layers.",
            )
        with self.subTest("missing_vision_structure"):
            with self.assertRaises(ValueError) as cm:
                build_quant_target_resolver_context(
                    family="qwen3_vl",
                    model=_qwen3_vl_model(text_config=text_ok),
                )
            self.assertEqual(
                str(cm.exception),
                "Cannot determine Qwen3-VL vision blocks; missing attribute "
                "model.visual.blocks.",
            )

    def test_gemma4_context_prefixes_and_aliases(self):
        # get_text_config() wins over a differing text_config attribute.
        model = _gemma4_model(
            get_text_config=SimpleNamespace(num_hidden_layers=4),
            text_config=SimpleNamespace(num_hidden_layers=9),
            vision_config=SimpleNamespace(num_hidden_layers=2),
        )
        context = build_quant_target_resolver_context(family="gemma4", model=model)
        self.assertEqual(tuple(context.components), ("text", "vision"))
        self._assert_component(
            context.components["text"],
            name="text",
            prefixes=_GEMMA4_TEXT_PREFIXES,
            num_layers=4,
            linear=_GEMMA4_TEXT_LINEARS,
        )
        self._assert_component(
            context.components["vision"],
            name="vision",
            prefixes=_GEMMA4_VISION_PREFIXES,
            num_layers=2,
            linear=_HF_DECODER_LINEARS,
        )

    def test_gemma4_text_config_fallback_and_errors(self):
        vision_ok = SimpleNamespace(num_hidden_layers=2)
        with self.subTest("text_config_attribute_without_get_text_config"):
            context = build_quant_target_resolver_context(
                family="gemma4",
                model=_gemma4_model(
                    text_config=SimpleNamespace(num_hidden_layers=6),
                    vision_config=vision_ok,
                ),
            )
            self.assertEqual(context.components["text"].num_layers, 6)

        error_cases: list[tuple[str, Any, str]] = [
            (
                "no_text_config",
                _gemma4_model(vision_config=vision_ok),
                "Cannot determine Gemma4 text layer count.",
            ),
            (
                "text_config_without_layer_count",
                _gemma4_model(text_config=SimpleNamespace(), vision_config=vision_ok),
                "Cannot determine Gemma4 text layer count.",
            ),
            (
                "no_config",
                SimpleNamespace(),
                "Cannot determine Gemma4 text layer count.",
            ),
            (
                "no_vision_config",
                _gemma4_model(text_config=SimpleNamespace(num_hidden_layers=6)),
                "Cannot determine Gemma4 vision layer count.",
            ),
        ]
        for label, model, message in error_cases:
            with self.subTest(label):
                with self.assertRaises(ValueError) as cm:
                    build_quant_target_resolver_context(family="gemma4", model=model)
                self.assertEqual(str(cm.exception), message)

    def test_family_string_handling(self):
        model = _llama_model(1)
        with self.subTest("strip_and_lower_accepted"):
            context = build_quant_target_resolver_context(
                family="  LLaMA ", model=model
            )
            self.assertEqual(context.components["text"].num_layers, 1)

        for family in ("qwen3-vl", "gemma4_assistant", "gemma4-assistant", "", "x"):
            with self.subTest(family=family):
                with self.assertRaises(ValueError) as cm:
                    build_quant_target_resolver_context(family=family, model=model)
                self.assertEqual(
                    str(cm.exception),
                    "PTQ override policies are not supported for model family "
                    f"{family!r}.",
                )

    def test_representative_selectors_resolve_to_exact_paths(self):
        def policy(name: str, target: dict) -> dict:
            return {"name": name, "target": target, "spec": "int16"}

        cases: list[tuple[str, str, Any, dict, set[tuple[str, ...]]]] = [
            (
                "llama_layer_1_linear_weights",
                "llama",
                _llama_model(2),
                policy(
                    "p",
                    {
                        "component": "text",
                        "layers": [1],
                        "op_type": "linear",
                        "observer_role": "parameter",
                    },
                ),
                {
                    ("model", "layers", "1") + tuple(module.split(".")) + ("weight",)
                    for module in _HF_DECODER_LINEARS
                },
            ),
            (
                "qwen3_vl_all_components_first_layer_outputs",
                "qwen3_vl",
                _qwen3_vl_model(
                    text_config=SimpleNamespace(num_hidden_layers=2),
                    vision_config=SimpleNamespace(depth=2),
                ),
                policy(
                    "p",
                    {
                        "component": "all",
                        "layers": "0:1",
                        "op_type": "linear",
                        "observer_role": "output_activation",
                    },
                ),
                {
                    ("model", "language_model", "layers", "0")
                    + tuple(module.split("."))
                    + ("act_out",)
                    for module in _HF_DECODER_LINEARS
                }
                | {
                    ("model", "visual", "blocks", "0")
                    + tuple(module.split("."))
                    + ("act_out",)
                    for module in _QWEN3_VL_VISION_LINEARS
                },
            ),
            (
                "gemma4_text_projection_across_all_prefixes",
                "gemma4",
                _gemma4_model(
                    get_text_config=SimpleNamespace(num_hidden_layers=2),
                    vision_config=SimpleNamespace(num_hidden_layers=2),
                ),
                policy(
                    "p",
                    {
                        "component": "text",
                        "layers": 0,
                        "module": "per_layer_projection",
                        "observers": ["act_in"],
                    },
                ),
                {
                    ("layers", "0", "per_layer_projection", "act_in"),
                    ("model", "layers", "0", "per_layer_projection", "act_in"),
                    ("language_model", "layers", "0", "per_layer_projection", "act_in"),
                    (
                        "model",
                        "language_model",
                        "layers",
                        "0",
                        "per_layer_projection",
                        "act_in",
                    ),
                },
            ),
            (
                "gemma4_vision_down_proj_weight_across_prefixes",
                "gemma4",
                _gemma4_model(
                    get_text_config=SimpleNamespace(num_hidden_layers=2),
                    vision_config=SimpleNamespace(num_hidden_layers=2),
                ),
                policy(
                    "p",
                    {
                        "component": "vision",
                        "layers": [1],
                        "module": "mlp.down_proj",
                        "observers": ["weight"],
                    },
                ),
                {
                    (
                        "vision_tower",
                        "encoder",
                        "layers",
                        "1",
                        "mlp",
                        "down_proj",
                        "weight",
                    ),
                    (
                        "model",
                        "vision_tower",
                        "encoder",
                        "layers",
                        "1",
                        "mlp",
                        "down_proj",
                        "weight",
                    ),
                },
            ),
        ]
        for label, family, model, policy_cfg, expected_paths in cases:
            with self.subTest(label):
                qcfg = PTQConfig()
                applied: set[tuple[str, ...]] = set()
                result = apply_ptq_override_policies(
                    qcfg,
                    {"override_policies": [policy_cfg]},
                    family=family,
                    model=model,
                    applied_paths=applied,
                )
                self.assertIs(result, qcfg)
                self.assertEqual(applied, expected_paths)
                for path in expected_paths:
                    self.assertEqual(_leaf(qcfg, path)["dtype"], DType.int(16))


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
        ]["act_in"]
        vision_act = qcfg.overrides["model"]["visual"]["blocks"]["0"]["attn"][  # type: ignore[index]
            "qkv"
        ]["act_in"]
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
