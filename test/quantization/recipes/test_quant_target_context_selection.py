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

"""Adapter-owned PTQ selector structure and its selection through the registry.

These tests pin which adapter declares the ``QuantTargetResolverContext`` used
by selector override policies: the explicitly selected adapter first, then the
canonical family adapter, and finally the legacy unsupported-family error. The
external-adapter cases only prove that a declaration flows from the hook through
override compilation into the config handed to ``prepare``; they are not a claim
that the synthetic family is quantizable or exportable. Heavy ``prepare`` and
``convert`` calls are replaced by recorders at that boundary.
"""

try:
    from quantization.recipes.optional_dependency_stubs import (
        install_optional_dependency_stubs,
    )
except ModuleNotFoundError:
    from optional_dependency_stubs import install_optional_dependency_stubs

install_optional_dependency_stubs()

import contextlib
import io
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import tico.quantization.recipes.adapters as adapters_mod
import tico.quantization.recipes.debug.trace as trace_mod
import tico.quantization.recipes.stages.ptq as ptq_mod

import torch
import torch.nn as nn

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.recipes.adapters import (
    available_adapters,
    register_adapter,
    resolve_adapter,
    resolve_quant_target_context,
)
from tico.quantization.recipes.adapters.base import ModelAdapter
from tico.quantization.recipes.adapters.gemma4 import Gemma4Adapter
from tico.quantization.recipes.adapters.gemma4_assistant import Gemma4AssistantAdapter
from tico.quantization.recipes.adapters.llama import LlamaAdapter
from tico.quantization.recipes.adapters.qwen3_vl import Qwen3VLAdapter
from tico.quantization.recipes.context import RecipeContext
from tico.quantization.recipes.debug.trace import trace_ptq_parity
from tico.quantization.recipes.override_policies import (
    apply_ptq_override_policies,
    build_quant_target_resolver_context,
    ComponentQuantTargetInfo,
    QuantTargetResolverContext,
)
from tico.quantization.recipes.stages.ptq import PTQStage
from tico.quantization.wrapq.dtypes import DType


# --------------------------------------------------------------------------- #
# Tiny models
# --------------------------------------------------------------------------- #
class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 2)

    def forward(self, x):
        return self.fc(x)


class _Encoder(nn.Module):
    def __init__(self, num_blocks: int):
        super().__init__()
        self.blocks = nn.ModuleList(_Block() for _ in range(num_blocks))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class _TinyEncoderModel(nn.Module):
    """``encoder.blocks[i].fc`` tree that no built-in family describes."""

    def __init__(self, num_blocks: int = 2):
        super().__init__()
        self.encoder = _Encoder(num_blocks)

    def forward(self, x):
        return self.encoder(x)


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(2, 2)

    def forward(self, x):
        return self.q_proj(x)


class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_proj = nn.Linear(2, 2)

    def forward(self, x):
        return self.down_proj(x)


class _DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attention()
        self.mlp = _MLP()

    def forward(self, x):
        return self.mlp(self.self_attn(x))


class _LlamaInner(nn.Module):
    def __init__(self, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(_DecoderLayer() for _ in range(num_layers))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class _TinyLlama(nn.Module):
    """Minimal ``model.layers[i]`` tree accepted by the llama declaration."""

    def __init__(self, num_layers: int = 2):
        super().__init__()
        self.model = _LlamaInner(num_layers)

    def forward(self, x):
        return self.model(x)


def _llama_stub(num_layers: int = 2) -> SimpleNamespace:
    return SimpleNamespace(model=SimpleNamespace(layers=[object()] * num_layers))


_ENCODER_CONTEXT_BLOCKS = 2
_ENC_BLOCK0_FC_IN = ("encoder", "blocks", "0", "fc", "act_in")
_ENC_BLOCK1_FC_OUT = ("encoder", "blocks", "1", "fc", "act_out")
_LLAMA_L0_DOWN_OUT = ("model", "layers", "0", "mlp", "down_proj", "act_out")
_LLAMA_L0_Q_OUT = ("model", "layers", "0", "self_attn", "q_proj", "act_out")


def _encoder_context(num_blocks: int) -> QuantTargetResolverContext:
    return QuantTargetResolverContext(
        components={
            "vision": ComponentQuantTargetInfo(
                name="vision",
                layer_path_prefixes=(("encoder", "blocks"),),
                num_layers=num_blocks,
                op_aliases={"linear": ("fc",)},
            )
        }
    )


# --------------------------------------------------------------------------- #
# Adapters
# --------------------------------------------------------------------------- #
class _DirectAdapter(ModelAdapter):
    """Out-of-tree adapter subclassing ``ModelAdapter`` without the new hook."""

    family = "synthetic"

    def __init__(self, family: str | None = None):
        if family is not None:
            self.family = family
        self.built_configs: list[PTQConfig] = []
        self.calibrated: list[Any] = []

    def load_model(self, ctx):
        return ctx

    def build_calibration_inputs(self, ctx):
        return []

    def forward_calibration(self, ctx, model, calibration_inputs, *, desc):
        return None

    def calibrate_prepared_model(self, ctx, prepared_model, stage_cfg):
        self.calibrated.append(prepared_model)
        with torch.no_grad():
            prepared_model(ctx.calibration_inputs[0])

    def build_ptq_config(self, ctx, stage_cfg):
        config = PTQConfig()
        self.built_configs.append(config)
        return config

    def evaluate(self, ctx):
        return None

    def export(self, ctx):
        return None


class _HookAdapter(_DirectAdapter):
    """Direct adapter whose hook returns a fixed result and records its calls."""

    def __init__(
        self, family: str, result: Any = None, *, error: Exception | None = None
    ):
        super().__init__(family)
        self.result = result
        self.error = error
        self.hook_models: list[Any] = []

    def build_quant_target_context(self, model):
        self.hook_models.append(model)
        if self.error is not None:
            raise self.error
        return self.result


class _EncoderAdapter(_HookAdapter):
    """External family adapter declaring the ``encoder.blocks`` structure."""

    def __init__(self):
        super().__init__("synthetic_enc")

    def build_quant_target_context(self, model):
        self.hook_models.append(model)
        return _encoder_context(len(model.encoder.blocks))


class _DuckTypedAdapter:
    """Duck-typed stage double without the ``ModelAdapter`` base class."""

    def __init__(self, family: str):
        self.family = family


# --------------------------------------------------------------------------- #
# Runners with prepare/convert replaced at the heavy boundary
# --------------------------------------------------------------------------- #
def _make_ctx(adapter, model: nn.Module, stage_cfg: dict) -> RecipeContext:
    torch.manual_seed(0)
    ctx = RecipeContext(
        cfg={"pipeline": [stage_cfg]},
        adapter=adapter,
        model=model,
        calibration_inputs=[torch.tensor([[0.5, -1.5]])],
    )
    ctx.device = torch.device("cpu")
    return ctx


def _run_ptq_stage(ctx: RecipeContext, *, prepare=None) -> tuple[dict[str, Any], str]:
    calls: dict[str, Any] = {}

    def recording_prepare(model, config):
        calls["prepare"] = (model, config)
        return model

    stdout = io.StringIO()
    with (
        patch.object(ptq_mod, "prepare", prepare or recording_prepare),
        patch.object(ptq_mod, "convert", lambda model: model),
        patch.object(ptq_mod, "find_gptq_quantizers", lambda model: (None, None)),
    ):
        with contextlib.redirect_stdout(stdout):
            PTQStage().run(ctx, ctx.cfg["pipeline"][0])
    return calls, stdout.getvalue()


def _run_trace(ctx: RecipeContext, *, prepare=None) -> dict[str, Any]:
    calls: dict[str, Any] = {}

    def recording_prepare(model, config):
        calls["prepare"] = (model, config)
        return model

    with (
        patch.object(trace_mod, "prepare", prepare or recording_prepare),
        patch.object(trace_mod, "convert", lambda model: model),
    ):
        with contextlib.redirect_stdout(io.StringIO()):
            trace_ptq_parity(ctx)
    return calls


def _leaf(qcfg: PTQConfig, path: tuple[str, ...]) -> Any:
    current: Any = qcfg.overrides
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _policy(component: str, observer_role: str, **extra) -> dict:
    return {
        "name": "p",
        "target": {
            "component": component,
            "layers": "all",
            "op_type": "linear",
            "observer_role": observer_role,
        },
        "spec": "int16",
        **extra,
    }


def _forbidden_prepare(model, config):
    raise AssertionError("prepare must not run after a resolution failure")


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
class _RegistryIsolation(unittest.TestCase):
    def setUp(self):
        self._registry_patch = patch.dict(adapters_mod._ADAPTERS, clear=False)
        self._registry_patch.start()
        # Each test starts from the built-in registry only.
        self.assertNotIn("synthetic_enc", available_adapters())
        self.assertNotIn("llama_ext", available_adapters())

    def tearDown(self):
        self._registry_patch.stop()


class TestAdapterQuantTargetHook(_RegistryIsolation):
    """Adapter declarations and the registry-backed resolver."""

    def test_builtin_adapters_declare_their_structure(self):
        llama = LlamaAdapter().build_quant_target_context(_llama_stub(3))
        self.assertEqual(tuple(llama.components), ("text",))
        self.assertEqual(llama.components["text"].num_layers, 3)
        self.assertEqual(
            llama.components["text"].layer_path_prefixes, (("model", "layers"),)
        )

        qwen = Qwen3VLAdapter().build_quant_target_context(
            SimpleNamespace(
                config=SimpleNamespace(
                    text_config=SimpleNamespace(num_hidden_layers=2),
                    vision_config=SimpleNamespace(depth=1),
                )
            )
        )
        self.assertEqual(tuple(qwen.components), ("text", "vision"))
        self.assertEqual(
            qwen.components["vision"].layer_path_prefixes,
            (("model", "visual", "blocks"),),
        )

        gemma4 = Gemma4Adapter().build_quant_target_context(
            SimpleNamespace(
                config=SimpleNamespace(
                    get_text_config=lambda: SimpleNamespace(num_hidden_layers=2),
                    vision_config=SimpleNamespace(num_hidden_layers=1),
                )
            )
        )
        self.assertEqual(tuple(gemma4.components), ("text", "vision"))
        self.assertIn(
            "per_layer_projection", gemma4.components["text"].op_aliases["linear"]
        )

        # Families without selector support and the base class declare nothing.
        self.assertIsNone(Gemma4AssistantAdapter().build_quant_target_context(object()))
        self.assertIsNone(_DirectAdapter().build_quant_target_context(object()))

    def test_builtin_subclass_inherits_declaration(self):
        class _LlamaVariant(LlamaAdapter):
            pass

        variant = _LlamaVariant()
        register_adapter("llama_ext", variant)
        selected = resolve_adapter(
            {"model": {"family": "llama", "adapter": "llama_ext"}}
        )
        self.assertIs(selected, variant)
        context = resolve_quant_target_context(
            _llama_stub(2), family="llama", adapter=selected
        )
        assert context is not None
        self.assertEqual(context.components["text"].num_layers, 2)

    def test_resolver_priority_and_fallbacks(self):
        model = _llama_stub(2)
        cases: list[tuple[str, Any, str, bool]] = [
            ("family_only", None, "llama", True),
            ("direct_subclass_without_hook", _DirectAdapter("llama"), "llama", True),
            ("duck_typed_double", _DuckTypedAdapter("llama"), "llama", True),
            ("stripped_lowered_family", None, "  LLaMA ", True),
            ("unregistered_family", _DirectAdapter("synthetic"), "synthetic", False),
            ("family_alias_rejected", None, "qwen3-vl", False),
            ("family_without_support", None, "gemma4_assistant", False),
            (
                "duck_typed_unsupported",
                _DuckTypedAdapter("gemma4_assistant"),
                "gemma4_assistant",
                False,
            ),
        ]
        for label, adapter, family, expect_context in cases:
            with self.subTest(label):
                context = resolve_quant_target_context(
                    model, family=family, adapter=adapter
                )
                if expect_context:
                    self.assertIsInstance(context, QuantTargetResolverContext)
                    assert context is not None
                    self.assertEqual(context.components["text"].num_layers, 2)
                else:
                    self.assertIsNone(context)

    def test_explicit_declaration_wins_and_family_default_is_not_consulted(self):
        custom = _encoder_context(1)
        adapter = _HookAdapter("llama", result=custom)

        def forbidden(self, model):
            raise AssertionError("family default must not be consulted")

        with patch.object(LlamaAdapter, "build_quant_target_context", forbidden):
            context = resolve_quant_target_context(
                object(), family="llama", adapter=adapter
            )
        self.assertIs(context, custom)
        self.assertEqual(len(adapter.hook_models), 1)

    def test_none_defers_to_family_default_and_hook_runs_once(self):
        adapter = _HookAdapter("llama", result=None)
        context = resolve_quant_target_context(
            _llama_stub(2), family="llama", adapter=adapter
        )
        assert context is not None
        self.assertEqual(context.components["text"].num_layers, 2)
        self.assertEqual(len(adapter.hook_models), 1)

        # When the selected adapter is itself the registered family adapter the
        # hook is not asked a second time.
        own = _HookAdapter("counted_family", result=None)
        register_adapter("counted_family", own)
        self.assertIsNone(
            resolve_quant_target_context(object(), family="counted_family", adapter=own)
        )
        self.assertEqual(len(own.hook_models), 1)

    def test_empty_context_is_a_declaration_not_none(self):
        empty = QuantTargetResolverContext(components={})
        adapter = _HookAdapter("llama", result=empty)
        context = resolve_quant_target_context(
            _llama_stub(2), family="llama", adapter=adapter
        )
        self.assertIs(context, empty)

        stage_cfg = {"override_policies": [_policy("text", "activation")]}
        with self.assertRaisesRegex(ValueError, "not available for this model"):
            apply_ptq_override_policies(
                PTQConfig(),
                stage_cfg,
                family="llama",
                model=_llama_stub(2),
                adapter=adapter,
            )
        qcfg = apply_ptq_override_policies(
            PTQConfig(),
            {"override_policies": [_policy("text", "activation", allow_empty=True)]},
            family="llama",
            model=_llama_stub(2),
            adapter=adapter,
        )
        self.assertEqual(qcfg.overrides, {})

    def test_hook_errors_and_invalid_returns_propagate(self):
        with self.assertRaisesRegex(RuntimeError, "hook exploded"):
            resolve_quant_target_context(
                _llama_stub(2),
                family="llama",
                adapter=_HookAdapter("llama", error=RuntimeError("hook exploded")),
            )
        for bad in ({"text": 1}, 42, "text", _encoder_context(1).components["vision"]):
            with self.subTest(bad=type(bad).__name__):
                with self.assertRaisesRegex(
                    TypeError, "build_quant_target_context\\(\\) must return"
                ):
                    resolve_quant_target_context(
                        _llama_stub(2),
                        family="llama",
                        adapter=_HookAdapter("llama", result=bad),
                    )

    def test_family_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "serves family 'llama'"):
            resolve_quant_target_context(
                _llama_stub(2), family="gemma4", adapter=LlamaAdapter()
            )
        with self.assertRaisesRegex(ValueError, "serves family 'llama'"):
            build_quant_target_resolver_context(
                family="gemma4", model=_llama_stub(2), adapter=LlamaAdapter()
            )

    def test_unsupported_family_keeps_legacy_error_with_explicit_adapter(self):
        with self.assertRaises(ValueError) as cm:
            build_quant_target_resolver_context(
                family="synthetic", model=object(), adapter=_DirectAdapter("synthetic")
            )
        self.assertEqual(
            str(cm.exception),
            "PTQ override policies are not supported for model family 'synthetic'.",
        )


class TestSelectedAdapterReachesPrepare(_RegistryIsolation):
    """Real PTQStage and trace use the selected adapter's declaration."""

    def _register_encoder_adapter(self) -> _EncoderAdapter:
        adapter = _EncoderAdapter()
        register_adapter("synthetic_enc", adapter)
        selected = resolve_adapter({"model": {"family": "synthetic_enc"}})
        self.assertIs(selected, adapter)
        return adapter

    def test_ptq_stage_prepares_with_external_family_declaration(self):
        for print_overrides in (False, True):
            with self.subTest(print_overrides=print_overrides):
                adapter = self._register_encoder_adapter()
                ctx = _make_ctx(
                    adapter,
                    _TinyEncoderModel(_ENCODER_CONTEXT_BLOCKS),
                    {
                        "name": "ptq",
                        "print_overrides": print_overrides,
                        "override_policies": [_policy("vision", "activation")],
                    },
                )
                calls, output = _run_ptq_stage(ctx)
                prepared_model, prepared_cfg = calls["prepare"]

                self.assertIs(prepared_model, ctx.model)
                self.assertIs(prepared_cfg, adapter.built_configs[-1])
                self.assertEqual(
                    _leaf(prepared_cfg, _ENC_BLOCK0_FC_IN)["dtype"], DType.int(16)
                )
                self.assertEqual(
                    _leaf(prepared_cfg, _ENC_BLOCK1_FC_OUT)["dtype"], DType.int(16)
                )
                # Structure is resolved once from the stage model, also when the
                # applied paths are printed afterwards.
                self.assertEqual(adapter.hook_models, [ctx.model])
                self.assertEqual(len(adapter.calibrated), 1)
                if print_overrides:
                    self.assertIn("encoder.blocks.1.fc.act_out: ", output)
                    self.assertIn("encoder.blocks.0.fc.act_in: ", output)
                else:
                    self.assertNotIn("Effective PTQ overrides", output)
                adapters_mod._ADAPTERS.pop("synthetic_enc")

    def test_trace_hands_copy_to_hook_and_prepare(self):
        adapter = self._register_encoder_adapter()
        ctx = _make_ctx(
            adapter,
            _TinyEncoderModel(_ENCODER_CONTEXT_BLOCKS),
            {"name": "ptq", "override_policies": [_policy("vision", "activation")]},
        )
        calls = _run_trace(ctx)
        prepared_model, prepared_cfg = calls["prepare"]

        self.assertEqual(len(adapter.hook_models), 1)
        self.assertIs(adapter.hook_models[0], prepared_model)
        self.assertIsNot(prepared_model, ctx.model)
        self.assertEqual(
            _leaf(prepared_cfg, _ENC_BLOCK1_FC_OUT)["dtype"], DType.int(16)
        )

    def test_stage_and_trace_hand_matching_config_to_prepare(self):
        stage_cfg = {
            "name": "ptq",
            "override_policies": [_policy("vision", "output_activation")],
            "raw_overrides": {"encoder.blocks.0.fc.act_in": "int8"},
        }
        adapter = self._register_encoder_adapter()
        stage_calls, _ = _run_ptq_stage(
            _make_ctx(adapter, _TinyEncoderModel(), dict(stage_cfg))
        )
        trace_calls = _run_trace(
            _make_ctx(adapter, _TinyEncoderModel(), dict(stage_cfg))
        )
        stage_cfg_prepared = stage_calls["prepare"][1]
        trace_prepared = trace_calls["prepare"][1]
        self.assertIsNot(stage_cfg_prepared, trace_prepared)
        self.assertEqual(stage_cfg_prepared.overrides, trace_prepared.overrides)
        self.assertEqual(
            _leaf(trace_prepared, _ENC_BLOCK0_FC_IN)["dtype"], DType.int(8)
        )
        self.assertEqual(
            _leaf(trace_prepared, _ENC_BLOCK1_FC_OUT)["dtype"], DType.int(16)
        )
        self.assertIsNone(
            _leaf(trace_prepared, ("encoder", "blocks", "1", "fc", "act_in"))
        )

    def test_external_declaration_is_not_overwritten_by_family_default(self):
        """A ModelAdapter variant for a built-in family keeps its own aliases."""

        class _LlamaExternal(_DirectAdapter):
            def build_quant_target_context(self, model):
                return QuantTargetResolverContext(
                    components={
                        "text": ComponentQuantTargetInfo(
                            name="text",
                            layer_path_prefixes=(("model", "layers"),),
                            num_layers=len(model.model.layers),
                            op_aliases={"linear": ("mlp.down_proj",)},
                        )
                    }
                )

        register_adapter("llama_ext", _LlamaExternal("llama"))
        adapter = resolve_adapter(
            {"model": {"family": "llama", "adapter": "llama_ext"}}
        )
        self.assertIsInstance(adapter, _LlamaExternal)

        for runner in (_run_ptq_stage, _run_trace):
            with self.subTest(runner=runner.__name__):
                ctx = _make_ctx(
                    adapter,
                    _TinyLlama(),
                    {
                        "name": "ptq",
                        "override_policies": [_policy("text", "output_activation")],
                    },
                )
                result = runner(ctx)
                calls = result[0] if isinstance(result, tuple) else result
                prepared_cfg = calls["prepare"][1]
                self.assertEqual(
                    _leaf(prepared_cfg, _LLAMA_L0_DOWN_OUT)["dtype"], DType.int(16)
                )
                # The built-in llama alias would also have targeted q_proj.
                self.assertIsNone(_leaf(prepared_cfg, _LLAMA_L0_Q_OUT))

    def test_direct_adapter_for_builtin_family_keeps_family_structure(self):
        """A hook-less ModelAdapter subclass serving llama resolves like llama."""
        register_adapter("llama_ext", _DirectAdapter("llama"))
        adapter = resolve_adapter(
            {"model": {"family": "llama", "adapter": "llama_ext"}}
        )
        ctx = _make_ctx(
            adapter,
            _TinyLlama(),
            {
                "name": "ptq",
                "override_policies": [_policy("text", "output_activation")],
            },
        )
        calls, _ = _run_ptq_stage(ctx)
        prepared_cfg = calls["prepare"][1]
        self.assertEqual(
            _leaf(prepared_cfg, _LLAMA_L0_DOWN_OUT)["dtype"], DType.int(16)
        )
        self.assertEqual(_leaf(prepared_cfg, _LLAMA_L0_Q_OUT)["dtype"], DType.int(16))

    def test_no_payload_never_calls_hook(self):
        cases: list[tuple[str, dict[str, Any]]] = [
            ("empty", {}),
            ("specs_only", {"specs": {"bad": "nope"}}),
        ]
        for label, payload in cases:
            for runner in (_run_ptq_stage, _run_trace):
                with self.subTest(label, runner=runner.__name__):
                    adapter = _EncoderAdapter()
                    ctx = _make_ctx(
                        adapter, _TinyEncoderModel(), {"name": "ptq", **payload}
                    )
                    result = runner(ctx)
                    calls = result[0] if isinstance(result, tuple) else result
                    self.assertEqual(adapter.hook_models, [])
                    self.assertIs(calls["prepare"][1], adapter.built_configs[0])
                    self.assertEqual(calls["prepare"][1].overrides, {})

    def test_resolution_failures_stop_before_prepare(self):
        cases: list[tuple[str, Any, type[Exception], str]] = [
            (
                "hook_error",
                _HookAdapter("synthetic_enc", error=RuntimeError("hook exploded")),
                RuntimeError,
                "hook exploded",
            ),
            (
                "invalid_return",
                _HookAdapter("synthetic_enc", result={"vision": 1}),
                TypeError,
                "must return a QuantTargetResolverContext",
            ),
            (
                "no_declaration",
                _DirectAdapter("synthetic_enc"),
                ValueError,
                "not supported for model family 'synthetic_enc'",
            ),
        ]
        for label, adapter, exc_type, fragment in cases:
            for runner in (_run_ptq_stage, _run_trace):
                with self.subTest(label, runner=runner.__name__):
                    ctx = _make_ctx(
                        adapter,
                        _TinyEncoderModel(),
                        {
                            "name": "ptq",
                            "raw_overrides": {"encoder.blocks.0.fc.act_in": "int16"},
                        },
                    )
                    with self.assertRaises(exc_type) as cm:
                        runner(ctx, prepare=_forbidden_prepare)
                    self.assertIn(fragment, str(cm.exception))
                    self.assertEqual(adapter.calibrated, [])


if __name__ == "__main__":
    unittest.main()
