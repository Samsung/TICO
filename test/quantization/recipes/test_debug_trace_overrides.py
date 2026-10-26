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

"""Regression tests: ``trace_ptq_parity`` must prepare with recipe PTQ overrides."""

try:
    from quantization.recipes.optional_dependency_stubs import (
        install_optional_dependency_stubs,
    )
except ModuleNotFoundError:
    from optional_dependency_stubs import install_optional_dependency_stubs

install_optional_dependency_stubs()

import contextlib
import copy
import io
import unittest
from typing import Any
from unittest.mock import patch

import tico.quantization.recipes.debug.trace as trace_mod
import tico.quantization.recipes.override_policies as override_policies_mod
import tico.quantization.recipes.stages.ptq as ptq_mod

import torch
import torch.nn as nn
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.config.specs import affine
from tico.quantization.recipes.context import RecipeContext
from tico.quantization.recipes.debug.trace import trace_ptq_parity
from tico.quantization.recipes.stages.ptq import PTQStage
from tico.quantization.recipes.utils import quant_spec_from_config
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.observers.minmax import MinMaxObserver
from tico.quantization.wrapq.observers.mx import MXObserver

_LAYER0_DOWN_OUT = ("model", "layers", "0", "mlp", "down_proj", "act_out")
_LAYER1_DOWN_OUT = ("model", "layers", "1", "mlp", "down_proj", "act_out")
_LAYER0_Q_WEIGHT = ("model", "layers", "0", "self_attn", "q_proj", "weight")
_MX_INT8_SPEC = {"kind": "mx", "elem_format": "int8", "axis": -1}


class _TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(2, 2)

    def forward(self, x):
        return self.q_proj(x)


class _TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_proj = nn.Linear(2, 2)

    def forward(self, x):
        return self.down_proj(x)


class _TinyDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _TinyAttention()
        self.mlp = _TinyMLP()

    def forward(self, x):
        return self.mlp(self.self_attn(x))


class _TinyLlamaInner(nn.Module):
    def __init__(self, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(_TinyDecoderLayer() for _ in range(num_layers))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class _TinyLlama(nn.Module):
    """Minimal ``model.layers[i].{self_attn,mlp}`` tree for the llama resolver."""

    def __init__(self, num_layers: int = 2):
        super().__init__()
        self.model = _TinyLlamaInner(num_layers)

    def forward(self, x):
        return self.model(x)


class _LlamaPTQAdapter:
    """Adapter fake that builds a real PTQConfig with one adapter-only override."""

    def __init__(self, family: str = "llama", events: list[str] | None = None):
        self.family = family
        self.events = events if events is not None else []
        self.built_configs: list[PTQConfig] = []
        self.calibrated: list[Any] = []

    def build_ptq_config(self, ctx, stage_cfg):
        self.events.append("build_ptq_config")
        activation = quant_spec_from_config(stage_cfg.get("activation"))
        config = PTQConfig(
            activation=activation if activation is not None else affine(DType.uint(8)),
            strict_wrap=bool(stage_cfg.get("strict_wrap", True)),
        )
        # Adapter default that no recipe override in these tests targets.
        config.set_override(_LAYER0_Q_WEIGHT, affine(DType.uint(4)))
        self.built_configs.append(config)
        return config

    def calibrate_prepared_model(self, ctx, prepared_model, stage_cfg):
        self.events.append("calibrate")
        self.calibrated.append(prepared_model)
        with torch.no_grad():
            prepared_model(ctx.calibration_inputs[0])


def _down_proj_out_policy(name: str, layers, spec, **extra) -> dict:
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
        **extra,
    }


def _leaf(qcfg: PTQConfig, path: tuple[str, ...]) -> Any:
    """Return the normalized override leaf at ``path`` or ``None`` when absent."""
    current: Any = qcfg.overrides
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _make_ctx(stage_cfg: dict, adapter: _LlamaPTQAdapter) -> RecipeContext:
    torch.manual_seed(0)
    ctx = RecipeContext(
        cfg={"pipeline": [stage_cfg]},
        adapter=adapter,
        model=_TinyLlama(),
        calibration_inputs=[torch.tensor([[0.5, -1.5]])],
    )
    ctx.device = torch.device("cpu")
    return ctx


def _run_trace(
    ctx: RecipeContext,
    *,
    enable_quantization: bool = False,
    events: list[str] | None = None,
    prepare=None,
) -> tuple[dict[str, Any], str]:
    """Run trace with prepare/convert replaced by recorders at the heavy boundary."""
    calls: dict[str, Any] = {}

    def recording_prepare(model, config):
        if events is not None:
            events.append("prepare")
        calls["prepare"] = (model, config)
        return model

    def recording_convert(model):
        if events is not None:
            events.append("convert")
        calls["convert"] = calls.get("convert", 0) + 1
        return model

    stdout = io.StringIO()
    with (
        patch.object(trace_mod, "prepare", prepare or recording_prepare),
        patch.object(trace_mod, "convert", recording_convert),
    ):
        with contextlib.redirect_stdout(stdout):
            trace_ptq_parity(ctx, enable_quantization=enable_quantization)
    return calls, stdout.getvalue()


def _run_ptq_stage(ctx: RecipeContext, stage_cfg: dict) -> PTQConfig:
    """Run PTQStage and return the PTQConfig handed to prepare."""
    calls: dict[str, Any] = {}

    def recording_prepare(model, config):
        calls["prepare"] = (model, config)
        return model

    with (
        patch.object(ptq_mod, "prepare", recording_prepare),
        patch.object(ptq_mod, "convert", lambda model: model),
        patch.object(ptq_mod, "find_gptq_quantizers", lambda model: (None, None)),
    ):
        with contextlib.redirect_stdout(io.StringIO()):
            PTQStage().run(ctx, stage_cfg)
    return calls["prepare"][1]


def _state_snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


class TestTracePtqParityOverrides(unittest.TestCase):
    def _assert_adapter_default_preserved(self, qcfg: PTQConfig) -> None:
        leaf = _leaf(qcfg, _LAYER0_Q_WEIGHT)
        self.assertIsNotNone(leaf)
        self.assertEqual(leaf["dtype"], DType.uint(4))

    def _assert_unchanged(
        self,
        ctx: RecipeContext,
        *,
        cfg_snapshot: dict,
        model: nn.Module,
        weights: dict[str, torch.Tensor],
    ) -> None:
        self.assertEqual(ctx.cfg, cfg_snapshot)
        self.assertIs(ctx.model, model)
        current = model.state_dict()
        self.assertEqual(set(current), set(weights))
        for name, tensor in weights.items():
            self.assertTrue(torch.equal(current[name], tensor), name)

    # ------------------------------------------------------------------ #
    # A. core regression: raw override reaches prepare
    # ------------------------------------------------------------------ #
    def test_prepare_receives_raw_override_from_recipe(self):
        """The PTQConfig handed to prepare must carry the recipe raw override."""
        stage_cfg = {
            "name": "ptq",
            "raw_overrides": {"model.layers.0.mlp.down_proj.act_out": "int16"},
        }
        adapter = _LlamaPTQAdapter()
        ctx = _make_ctx(stage_cfg, adapter)

        with patch.object(
            override_policies_mod,
            "build_quant_target_resolver_context",
            wraps=override_policies_mod.build_quant_target_resolver_context,
        ) as resolver_spy:
            calls, _ = _run_trace(ctx)

        prepared_model, prepared_cfg = calls["prepare"]
        self.assertIsInstance(prepared_cfg, PTQConfig)
        leaf = _leaf(prepared_cfg, _LAYER0_DOWN_OUT)
        self.assertIsNotNone(leaf, "recipe raw override was not applied before prepare")
        self.assertEqual(leaf["dtype"], DType.int(16))
        self.assertIs(leaf["observer"], MinMaxObserver)
        self._assert_adapter_default_preserved(prepared_cfg)
        # The same object the adapter built is updated and handed to prepare.
        self.assertIs(prepared_cfg, adapter.built_configs[0])

        # Model structure is resolved from the traced copy that prepare receives,
        # never from ctx.model itself.
        self.assertEqual(resolver_spy.call_count, 1)
        self.assertIs(resolver_spy.call_args.kwargs["model"], prepared_model)
        self.assertIsNot(prepared_model, ctx.model)
        self.assertEqual(resolver_spy.call_args.kwargs["family"], "llama")
        # The selected adapter is handed over so its own declaration is used.
        self.assertIs(resolver_spy.call_args.kwargs["adapter"], adapter)

    # ------------------------------------------------------------------ #
    # B. policy combinations
    # ------------------------------------------------------------------ #
    def test_policy_combinations_reach_prepare(self):
        """Selector/raw combinations must land in prepare with hand-set final values."""
        cases: list[tuple[str, dict[str, Any], dict[tuple[str, ...], Any]]] = [
            (
                "raw_only",
                {"raw_overrides": {"model.layers.1.mlp.down_proj.act_out": "int16"}},
                {
                    _LAYER1_DOWN_OUT: {
                        "dtype": DType.int(16),
                        "observer": MinMaxObserver,
                    },
                    _LAYER0_DOWN_OUT: None,
                },
            ),
            (
                "named_spec_selector",
                {
                    "specs": {"mx_int8": _MX_INT8_SPEC},
                    "override_policies": [_down_proj_out_policy("p", [0], "mx_int8")],
                },
                {
                    _LAYER0_DOWN_OUT: {"observer": MXObserver, "elem_format": "int8"},
                    _LAYER1_DOWN_OUT: None,
                },
            ),
            (
                "raw_wins_over_selector_and_replaces_mx_fields",
                {
                    "specs": {"mx_int8": _MX_INT8_SPEC},
                    "override_policies": [_down_proj_out_policy("p", "all", "mx_int8")],
                    "raw_overrides": {
                        "model.layers.0.mlp.down_proj.act_out": "int16",
                    },
                },
                {
                    _LAYER0_DOWN_OUT: {
                        "dtype": DType.int(16),
                        "observer": MinMaxObserver,
                        "elem_format": None,
                    },
                    _LAYER1_DOWN_OUT: {"observer": MXObserver, "elem_format": "int8"},
                },
            ),
            (
                "disabled_policy",
                {
                    "override_policies": [
                        _down_proj_out_policy("p", [0], "int16", enabled=False)
                    ]
                },
                {_LAYER0_DOWN_OUT: None, _LAYER1_DOWN_OUT: None},
            ),
            (
                "allow_empty_missing_component",
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
                {_LAYER0_DOWN_OUT: None, _LAYER1_DOWN_OUT: None},
            ),
            (
                "no_overrides",
                {},
                {_LAYER0_DOWN_OUT: None, _LAYER1_DOWN_OUT: None},
            ),
        ]

        for label, payload, expected in cases:
            with self.subTest(label):
                adapter = _LlamaPTQAdapter()
                ctx = _make_ctx({"name": "ptq", **payload}, adapter)
                calls, _ = _run_trace(ctx)
                prepared_cfg = calls["prepare"][1]

                self.assertIs(prepared_cfg, adapter.built_configs[0])
                self._assert_adapter_default_preserved(prepared_cfg)
                for path, fields in expected.items():
                    leaf = _leaf(prepared_cfg, path)
                    if fields is None:
                        self.assertIsNone(leaf, ".".join(path))
                        continue
                    self.assertIsNotNone(leaf, ".".join(path))
                    for key, value in fields.items():
                        if value is None:
                            self.assertNotIn(key, leaf, ".".join(path))
                        elif isinstance(value, type):
                            self.assertIs(leaf[key], value, ".".join(path))
                        else:
                            self.assertEqual(leaf[key], value, ".".join(path))

    # ------------------------------------------------------------------ #
    # C. parity with the regular PTQ stage at the prepare boundary
    # ------------------------------------------------------------------ #
    def test_trace_and_ptq_stage_hand_matching_config_to_prepare(self):
        """PTQStage.run and trace must hand prepare the same effective PTQ policy."""
        stage_cfg = {
            "name": "ptq",
            "activation": "int16",
            "strict_wrap": False,
            "specs": {"mx_int8": _MX_INT8_SPEC},
            "override_policies": [_down_proj_out_policy("all_down", "all", "mx_int8")],
            "raw_overrides": {"model.layers.0.mlp.down_proj.act_out": "int8"},
        }

        stage_ctx = _make_ctx(copy.deepcopy(stage_cfg), _LlamaPTQAdapter())
        stage_cfg_prepared = _run_ptq_stage(stage_ctx, stage_ctx.cfg["pipeline"][0])

        trace_ctx = _make_ctx(copy.deepcopy(stage_cfg), _LlamaPTQAdapter())
        trace_calls, _ = _run_trace(trace_ctx)
        trace_prepared = trace_calls["prepare"][1]

        self.assertIsNot(stage_cfg_prepared, trace_prepared)
        for label, qcfg in (("stage", stage_cfg_prepared), ("trace", trace_prepared)):
            with self.subTest(label):
                self.assertEqual(qcfg.activation, affine(DType.int(16)))
                self.assertEqual(qcfg.weight, affine(DType.uint(8)))
                self.assertFalse(qcfg.strict_wrap)
                layer0 = _leaf(qcfg, _LAYER0_DOWN_OUT)
                layer1 = _leaf(qcfg, _LAYER1_DOWN_OUT)
                self.assertEqual(layer0["dtype"], DType.int(8))
                self.assertIs(layer0["observer"], MinMaxObserver)
                self.assertNotIn("elem_format", layer0)
                self.assertIs(layer1["observer"], MXObserver)
                self.assertEqual(layer1["elem_format"], "int8")
                self._assert_adapter_default_preserved(qcfg)

        self.assertEqual(stage_cfg_prepared.overrides, trace_prepared.overrides)
        self.assertEqual(stage_cfg_prepared.activation, trace_prepared.activation)
        self.assertEqual(stage_cfg_prepared.weight, trace_prepared.weight)
        self.assertEqual(stage_cfg_prepared.strict_wrap, trace_prepared.strict_wrap)

    # ------------------------------------------------------------------ #
    # D. ordering with and without --enable-quantization
    # ------------------------------------------------------------------ #
    def test_override_applied_once_before_prepare_in_both_modes(self):
        """Overrides apply exactly once before prepare; convert only when enabled."""
        for enable_quantization in (False, True):
            with self.subTest(enable_quantization=enable_quantization):
                events: list[str] = []
                adapter = _LlamaPTQAdapter(events=events)
                ctx = _make_ctx(
                    {
                        "name": "ptq",
                        "raw_overrides": {
                            "model.layers.0.mlp.down_proj.act_out": "int16"
                        },
                    },
                    adapter,
                )

                real_apply = override_policies_mod.apply_ptq_override_policies

                def recording_apply(*args, **kwargs):
                    events.append("apply_overrides")
                    return real_apply(*args, **kwargs)

                real_collect = trace_mod.collect_forward_outputs

                def recording_collect(*args, **kwargs):
                    events.append("collect")
                    return real_collect(*args, **kwargs)

                with (
                    patch.object(
                        trace_mod, "apply_ptq_override_policies", recording_apply
                    ),
                    patch.object(
                        trace_mod, "collect_forward_outputs", recording_collect
                    ),
                ):
                    calls, output = _run_trace(
                        ctx,
                        enable_quantization=enable_quantization,
                        events=events,
                    )

                expected = [
                    "build_ptq_config",
                    "apply_overrides",
                    "prepare",
                    "calibrate",
                ]
                if enable_quantization:
                    expected.append("convert")
                expected.extend(["collect", "collect"])
                self.assertEqual(events, expected)
                self.assertEqual(calls.get("convert", 0), int(enable_quantization))
                self.assertEqual(
                    _leaf(calls["prepare"][1], _LAYER0_DOWN_OUT)["dtype"],
                    DType.int(16),
                )
                self.assertLess(output.index("FP trace"), output.index("PTQ trace"))
                self.assertLess(
                    output.index("PTQ trace"), output.index("Side-by-side diff")
                )

    # ------------------------------------------------------------------ #
    # E. error and no-op contracts
    # ------------------------------------------------------------------ #
    def test_override_errors_propagate_before_prepare(self):
        """Common helper errors must surface before prepare, calibration, or trace."""

        def failing_prepare(model, config):
            raise AssertionError("prepare must not run after an override failure")

        def failing_convert(model):
            raise AssertionError("convert must not run after an override failure")

        cases: list[tuple[str, str, dict[str, Any], type[Exception], str]] = [
            (
                "bad_named_spec",
                "llama",
                {
                    "specs": {"bad": None},
                    "override_policies": [_down_proj_out_policy("p", [0], "bad")],
                },
                ValueError,
                "ptq.specs.bad must not resolve to None",
            ),
            (
                "bad_selector",
                "llama",
                {"override_policies": [_down_proj_out_policy("p", [7], "int16")]},
                ValueError,
                "out of range",
            ),
            (
                "bad_raw",
                "llama",
                {"raw_overrides": ["model.layers.0.mlp.down_proj.act_out"]},
                TypeError,
                "ptq.raw_overrides must be a mapping",
            ),
            (
                "unsupported_family_with_raw",
                "synthetic_family",
                {"raw_overrides": {"model.layers.0.mlp.down_proj.act_out": "int16"}},
                ValueError,
                "not supported for model family",
            ),
        ]

        for label, family, payload, exc_type, fragment in cases:
            with self.subTest(label):
                adapter = _LlamaPTQAdapter(family=family)
                ctx = _make_ctx({"name": "ptq", **payload}, adapter)
                model = ctx.model
                cfg_snapshot = copy.deepcopy(ctx.cfg)
                weights = _state_snapshot(model)

                stdout = io.StringIO()
                with (
                    self.assertRaises(exc_type) as cm,
                    contextlib.redirect_stdout(stdout),
                    patch.object(trace_mod, "prepare", failing_prepare),
                    patch.object(trace_mod, "convert", failing_convert),
                ):
                    trace_ptq_parity(ctx)

                self.assertIn(fragment, str(cm.exception))
                self.assertEqual(len(adapter.built_configs), 1)
                self.assertEqual(adapter.calibrated, [])
                self.assertNotIn("FP trace", stdout.getvalue())
                self.assertNotIn("PTQ trace", stdout.getvalue())
                self._assert_unchanged(
                    ctx, cfg_snapshot=cfg_snapshot, model=model, weights=weights
                )

    def test_no_payload_keeps_adapter_config_for_unsupported_family(self):
        """Without selector/raw payload the adapter config reaches prepare untouched."""
        cases: list[tuple[str, dict[str, Any]]] = [
            ("empty", {}),
            ("specs_only", {"specs": {"bad": "not-a-spec"}}),
        ]
        for label, payload in cases:
            with self.subTest(label):
                adapter = _LlamaPTQAdapter(family="synthetic_family")
                ctx = _make_ctx({"name": "ptq", **payload}, adapter)
                calls, output = _run_trace(ctx, enable_quantization=True)

                prepared_cfg = calls["prepare"][1]
                self.assertIs(prepared_cfg, adapter.built_configs[0])
                self._assert_adapter_default_preserved(prepared_cfg)
                self.assertIsNone(_leaf(prepared_cfg, _LAYER0_DOWN_OUT))
                self.assertEqual(len(adapter.calibrated), 1)
                self.assertEqual(calls["convert"], 1)
                self.assertIn("Side-by-side diff", output)

    def test_repeated_runs_do_not_accumulate_or_mutate_state(self):
        """Repeated and failed runs must not leak config or model state across calls."""
        adapter = _LlamaPTQAdapter()
        ctx = _make_ctx(
            {
                "name": "ptq",
                "raw_overrides": {"model.layers.0.mlp.down_proj.act_out": "int16"},
            },
            adapter,
        )
        model = ctx.model
        cfg_snapshot = copy.deepcopy(ctx.cfg)
        weights = _state_snapshot(model)

        first, _ = _run_trace(ctx, enable_quantization=True)
        second, _ = _run_trace(ctx, enable_quantization=False)

        first_cfg = first["prepare"][1]
        second_cfg = second["prepare"][1]
        self.assertIsNot(first_cfg, second_cfg)
        self.assertEqual(first_cfg.overrides, second_cfg.overrides)
        self.assertEqual(_leaf(second_cfg, _LAYER0_DOWN_OUT)["dtype"], DType.int(16))
        self.assertIsNone(_leaf(second_cfg, _LAYER1_DOWN_OUT))
        self.assertEqual(len(adapter.built_configs), 2)
        self._assert_unchanged(
            ctx, cfg_snapshot=cfg_snapshot, model=model, weights=weights
        )

        # A failing run in between must not change what a later run prepares.
        ctx.cfg["pipeline"][0]["override_policies"] = [
            _down_proj_out_policy("bad", [9], "int16")
        ]
        with self.assertRaises(ValueError), contextlib.redirect_stdout(io.StringIO()):
            trace_ptq_parity(ctx)
        del ctx.cfg["pipeline"][0]["override_policies"]

        third, _ = _run_trace(ctx)
        self.assertEqual(third["prepare"][1].overrides, first_cfg.overrides)
        self.assertEqual(len(adapter.built_configs), 4)
        self._assert_unchanged(
            ctx, cfg_snapshot=cfg_snapshot, model=model, weights=weights
        )


if __name__ == "__main__":
    unittest.main()
