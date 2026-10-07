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

"""GPTQ config class selection through ``GPTQStage`` and the adapter layer.

These tests pin which GPTQ config class the stage instantiates for each
built-in family, for out-of-tree adapters, and for the ``universal`` variant.
Selecting a config class is not a claim that the full GPTQ run is supported
for that family; heavy ``prepare``/``convert`` calls are replaced by fakes and
only the config construction and stage orchestration are checked.
"""

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
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import tico.quantization.recipes.adapters as adapters_mod
import tico.quantization.recipes.stages.gptq as gptq_mod

import torch

from tico.quantization.config.gemma4_gptq import Gemma4GPTQConfig
from tico.quantization.config.gptq import GPTQConfig, UniversalGPTQConfig
from tico.quantization.config.qwen3_vl_gptq import Qwen3VLGPTQConfig
from tico.quantization.recipes.adapters import (
    register_adapter,
    resolve_adapter,
    resolve_gptq_config_class,
)
from tico.quantization.recipes.adapters.base import ModelAdapter
from tico.quantization.recipes.adapters.gemma4 import Gemma4Adapter
from tico.quantization.recipes.adapters.gemma4_assistant import Gemma4AssistantAdapter
from tico.quantization.recipes.adapters.llama import LlamaAdapter
from tico.quantization.recipes.adapters.qwen3_vl import Qwen3VLAdapter
from tico.quantization.recipes.context import RecipeContext
from tico.quantization.recipes.stages.gptq import GPTQStage


class _DirectAdapter(ModelAdapter):
    """Out-of-tree adapter that subclasses ``ModelAdapter`` directly.

    It implements only the abstract contract, so it stands for external
    adapters written before any GPTQ-specific hook existed.
    """

    family = "synthetic"

    def __init__(self, family: str | None = None):
        if family is not None:
            self.family = family
        self.forwarded: tuple[Any, ...] | None = None

    def load_model(self, ctx):
        return ctx

    def build_calibration_inputs(self, ctx):
        return []

    def forward_calibration(self, ctx, model, calibration_inputs, *, desc):
        self.forwarded = (ctx, model, calibration_inputs, desc)

    def calibrate_prepared_model(self, ctx, prepared_model, stage_cfg):
        return None

    def build_ptq_config(self, ctx, stage_cfg):
        return None

    def evaluate(self, ctx):
        return None

    def export(self, ctx):
        return None


class _DuckTypedAdapter:
    """Duck-typed stage test double without the ``ModelAdapter`` base class."""

    def __init__(self, family: str):
        self.family = family

    def forward_calibration(self, ctx, model, calibration_inputs, *, desc):
        return None


def _run_gptq_stage(adapter, stage_cfg, *, cfg=None):
    """Run ``GPTQStage`` with fake prepare/convert and a recording calibration.

    Returns a namespace with the built config, the ordered list of recorded
    calls, the source and prepared models, the context, and the stage result.
    """
    source_model = torch.nn.Linear(2, 2)
    prepared_model = torch.nn.Linear(2, 2)
    ctx = RecipeContext(
        cfg=cfg if cfg is not None else {},
        adapter=adapter,
        model=source_model,
        calibration_inputs=[torch.randn(1, 2)],
    )
    calls: list[tuple[Any, ...]] = []

    def fake_prepare(model_arg, config, inplace=False):
        calls.append(("prepare", model_arg, config, inplace))
        return prepared_model

    def fake_convert(model_arg, inplace=False):
        calls.append(("convert", model_arg, inplace))
        return "converted"

    def fake_forward_calibration(self, ctx_arg, model_arg, calibration_inputs, *, desc):
        calls.append(("calibrate", ctx_arg, model_arg, calibration_inputs, desc))

    with patch.object(gptq_mod, "prepare", fake_prepare), patch.object(
        gptq_mod, "convert", fake_convert
    ), patch.object(
        type(adapter), "forward_calibration", fake_forward_calibration
    ), contextlib.redirect_stdout(
        io.StringIO()
    ):
        result = GPTQStage().run(ctx, stage_cfg)

    prepare_calls = [call for call in calls if call[0] == "prepare"]
    assert len(prepare_calls) == 1, calls
    return SimpleNamespace(
        ctx=ctx,
        result=result,
        calls=calls,
        config=prepare_calls[0][2],
        source_model=source_model,
        prepared_model=prepared_model,
    )


class TestGPTQConfigSelectionCharacterization(unittest.TestCase):
    """Pin the config class selected for each existing family and variant."""

    def setUp(self):
        self._registry_patch = patch.dict(adapters_mod._ADAPTERS, clear=False)
        self._registry_patch.start()

    def tearDown(self):
        self._registry_patch.stop()

    def test_llama_selects_generic_gptq_config(self):
        """LLaMA uses the generic GPTQConfig and drops model-specific keys."""
        run = _run_gptq_stage(
            LlamaAdapter(),
            {"name": "gptq", "weight_bits": 4, "gptq_v2": True, "unknown_key": 1},
        )
        self.assertIs(type(run.config), GPTQConfig)
        self.assertEqual(run.config.weight_bits, 4)
        self.assertEqual(run.config.name, "gptq")
        # Qwen3-VL-only and unknown keys are filtered out for the generic class.
        self.assertFalse(hasattr(run.config, "gptq_v2"))
        self.assertFalse(hasattr(run.config, "unknown_key"))

    def test_qwen3_vl_selects_qwen3_vl_gptq_config_with_its_fields(self):
        """Qwen3-VL uses Qwen3VLGPTQConfig and keeps its dedicated fields."""
        cfg = {
            "calibration": {"dataset": "vqav2", "n_samples": 4, "seq_len": 64},
            "runtime": {"seed": 7},
        }
        run = _run_gptq_stage(
            Qwen3VLAdapter(),
            {
                "name": "gptq",
                "weight_bits": 4,
                "gptq_v2": True,
                "fp_inputs_cache_path": "cache-dir",
                "unknown_key": 1,
            },
            cfg=cfg,
        )
        self.assertIs(type(run.config), Qwen3VLGPTQConfig)
        self.assertEqual(run.config.weight_bits, 4)
        self.assertTrue(run.config.gptq_v2)
        self.assertEqual(run.config.fp_inputs_cache_path, "cache-dir")
        self.assertEqual(
            run.config.calibration_dataset_spec,
            GPTQStage._calibration_dataset_spec(cfg["calibration"], cfg["runtime"]),
        )
        self.assertFalse(hasattr(run.config, "unknown_key"))

    def test_qwen3_vl_explicit_calibration_dataset_spec_wins(self):
        """An explicit calibration_dataset_spec is passed through unchanged."""
        run = _run_gptq_stage(
            Qwen3VLAdapter(),
            {"name": "gptq", "calibration_dataset_spec": "explicit-spec"},
            cfg={"calibration": {"dataset": "vqav2"}},
        )
        self.assertEqual(run.config.calibration_dataset_spec, "explicit-spec")

    def test_gemma4_selects_gemma4_gptq_config_with_its_fields(self):
        """Gemma4 uses Gemma4GPTQConfig and keeps its dedicated fields."""
        run = _run_gptq_stage(
            Gemma4Adapter(),
            {"name": "gptq", "weight_bits": 8, "quantize_vision_pooler": False},
        )
        self.assertIs(type(run.config), Gemma4GPTQConfig)
        self.assertEqual(run.config.weight_bits, 8)
        self.assertFalse(run.config.quantize_vision_pooler)
        self.assertEqual(run.config.name, "gemma4_gptq")

    def test_gemma4_assistant_selects_generic_gptq_config(self):
        """The Gemma4 assistant family currently maps to the generic GPTQConfig."""
        run = _run_gptq_stage(Gemma4AssistantAdapter(), {"name": "gptq"})
        self.assertIs(type(run.config), GPTQConfig)

    def test_unregistered_family_selects_generic_gptq_config(self):
        """A family without a dedicated config falls back to GPTQConfig."""
        run = _run_gptq_stage(_DirectAdapter("synthetic"), {"name": "gptq"})
        self.assertIs(type(run.config), GPTQConfig)

    def test_universal_variant_selects_universal_config_for_every_family(self):
        """variant=universal selects UniversalGPTQConfig regardless of family."""
        adapters = [
            LlamaAdapter(),
            Qwen3VLAdapter(),
            Gemma4Adapter(),
            Gemma4AssistantAdapter(),
            _DirectAdapter("synthetic"),
            _DirectAdapter("qwen3_vl"),
        ]
        for adapter in adapters:
            with self.subTest(adapter=type(adapter).__name__, family=adapter.family):
                run = _run_gptq_stage(
                    adapter,
                    {"name": "gptq", "variant": "universal", "weight_bits": 4},
                )
                self.assertIs(type(run.config), UniversalGPTQConfig)
                self.assertEqual(run.config.weight_bits, 4)
                self.assertEqual(run.config.name, "universal_gptq")

    def test_family_alias_resolved_through_registry_keeps_selection(self):
        """model.family aliases resolve to the built-in adapter and its config."""
        expectations = {
            "qwen3-vl": Qwen3VLGPTQConfig,
            "qwen3_vl": Qwen3VLGPTQConfig,
            "gemma4": Gemma4GPTQConfig,
            "gemma4-assistant": GPTQConfig,
            "llama": GPTQConfig,
        }
        for family, expected in expectations.items():
            with self.subTest(family=family):
                adapter = resolve_adapter({"model": {"family": family}})
                run = _run_gptq_stage(adapter, {"name": "gptq"})
                self.assertIs(type(run.config), expected)

    def test_external_subclass_of_builtin_adapter_keeps_family_selection(self):
        """A registered subclass of a built-in adapter inherits its selection."""

        class _QwenVariant(Qwen3VLAdapter):
            pass

        class _Gemma4Variant(Gemma4Adapter):
            pass

        register_adapter("qwen_variant", _QwenVariant())
        register_adapter("gemma4_variant", _Gemma4Variant())

        qwen = resolve_adapter({"model": {"family": "qwen3_vl", "adapter": "qwen_variant"}})
        self.assertIsInstance(qwen, _QwenVariant)
        self.assertIs(type(_run_gptq_stage(qwen, {"name": "gptq"}).config), Qwen3VLGPTQConfig)

        gemma4 = resolve_adapter(
            {"model": {"family": "gemma4", "adapter": "gemma4_variant"}}
        )
        self.assertIsInstance(gemma4, _Gemma4Variant)
        self.assertIs(type(_run_gptq_stage(gemma4, {"name": "gptq"}).config), Gemma4GPTQConfig)

    def test_external_direct_adapter_with_builtin_family_keeps_family_selection(self):
        """A direct ModelAdapter subclass serving a built-in family keeps that family's config."""
        expectations = {
            "qwen3_vl": Qwen3VLGPTQConfig,
            "gemma4": Gemma4GPTQConfig,
            "llama": GPTQConfig,
        }
        for family, expected in expectations.items():
            with self.subTest(family=family):
                key = f"{family}_direct"
                register_adapter(key, _DirectAdapter(family))
                adapter = resolve_adapter({"model": {"family": family, "adapter": key}})
                self.assertIs(type(adapter), _DirectAdapter)
                run = _run_gptq_stage(adapter, {"name": "gptq", "weight_bits": 4})
                self.assertIs(type(run.config), expected)
                self.assertEqual(run.config.weight_bits, 4)

    def test_duck_typed_double_without_base_class_keeps_family_selection(self):
        """Duck-typed stage doubles select by family exactly like before."""
        self.assertIs(
            type(_run_gptq_stage(_DuckTypedAdapter("llama"), {"name": "gptq"}).config),
            GPTQConfig,
        )
        self.assertIs(
            type(
                _run_gptq_stage(_DuckTypedAdapter("qwen3_vl"), {"name": "gptq"}).config
            ),
            Qwen3VLGPTQConfig,
        )

    def test_stage_orchestration_and_inputs_are_unchanged(self):
        """prepare -> adapter calibration -> convert runs in order without mutating config."""
        stage_cfg = {
            "name": "gptq",
            "enabled": True,
            "weight_bits": 4,
            "gptq_v2": True,
            "unknown_key": {"nested": [1, 2]},
        }
        snapshot = copy.deepcopy(stage_cfg)
        cfg = {"calibration": {"dataset": "vqav2"}, "runtime": {"seed": 3}}
        cfg_snapshot = copy.deepcopy(cfg)

        run = _run_gptq_stage(Qwen3VLAdapter(), stage_cfg, cfg=cfg)

        self.assertEqual([call[0] for call in run.calls], ["prepare", "calibrate", "convert"])
        prepare, calibrate, convert = run.calls
        self.assertIs(prepare[1], run.source_model)
        self.assertIs(prepare[2], run.config)
        self.assertTrue(prepare[3])
        self.assertIs(calibrate[1], run.ctx)
        self.assertIs(calibrate[2], run.prepared_model)
        self.assertIs(calibrate[3], run.ctx.calibration_inputs)
        self.assertEqual(calibrate[4], "GPTQ calibration")
        self.assertIs(convert[1], run.prepared_model)
        self.assertTrue(convert[2])
        self.assertIs(run.result, run.ctx)
        self.assertEqual(run.ctx.model, "converted")
        self.assertEqual(stage_cfg, snapshot)
        self.assertEqual(cfg, cfg_snapshot)


@dataclass
class _ExternalGPTQConfig(Qwen3VLGPTQConfig):
    """Out-of-tree GPTQ config with a field the built-in classes lack."""

    external_knob: int = 1

    @property
    def name(self) -> str:
        return "external_gptq"


class _ExternalQwenAdapter(Qwen3VLAdapter):
    """Out-of-tree Qwen3-VL variant that selects its own GPTQ config."""

    def get_gptq_config_class(self):
        return _ExternalGPTQConfig


class _CountingDirectAdapter(_DirectAdapter):
    """Direct adapter whose hook records how often it is consulted."""

    def __init__(self, family: str, result: Any = None):
        super().__init__(family)
        self.result = result
        self.hook_calls = 0

    def get_gptq_config_class(self):
        self.hook_calls += 1
        return self.result


class TestAdapterGPTQConfigHook(unittest.TestCase):
    """Adapter-owned GPTQ config selection and its resolver."""

    def setUp(self):
        self._registry_patch = patch.dict(adapters_mod._ADAPTERS, clear=False)
        self._registry_patch.start()

    def tearDown(self):
        self._registry_patch.stop()

    def test_builtin_adapters_declare_expected_selection(self):
        """Built-in adapters select their dedicated config or nothing."""
        self.assertIs(Qwen3VLAdapter().get_gptq_config_class(), Qwen3VLGPTQConfig)
        self.assertIs(Gemma4Adapter().get_gptq_config_class(), Gemma4GPTQConfig)
        self.assertIsNone(LlamaAdapter().get_gptq_config_class())
        self.assertIsNone(Gemma4AssistantAdapter().get_gptq_config_class())
        self.assertIsNone(_DirectAdapter("synthetic").get_gptq_config_class())

    def test_resolver_priority_and_fallbacks(self):
        """Own selection, then the registered family adapter, then None."""
        self.assertIs(resolve_gptq_config_class(Qwen3VLAdapter()), Qwen3VLGPTQConfig)
        self.assertIs(resolve_gptq_config_class(Gemma4Adapter()), Gemma4GPTQConfig)
        self.assertIsNone(resolve_gptq_config_class(LlamaAdapter()))
        self.assertIsNone(resolve_gptq_config_class(Gemma4AssistantAdapter()))
        # Direct adapters and duck-typed doubles inherit the family default.
        self.assertIs(
            resolve_gptq_config_class(_DirectAdapter("gemma4")), Gemma4GPTQConfig
        )
        self.assertIs(
            resolve_gptq_config_class(_DuckTypedAdapter("qwen3_vl")),
            Qwen3VLGPTQConfig,
        )
        self.assertIsNone(resolve_gptq_config_class(_DirectAdapter("synthetic")))
        self.assertIsNone(resolve_gptq_config_class(_DuckTypedAdapter("llama")))

    def test_explicit_none_defers_to_family_default(self):
        """An adapter hook returning None keeps the registered family selection."""
        adapter = _CountingDirectAdapter("qwen3_vl", result=None)
        self.assertIs(resolve_gptq_config_class(adapter), Qwen3VLGPTQConfig)
        self.assertEqual(adapter.hook_calls, 1)

    def test_resolver_consults_each_hook_at_most_once(self):
        """The family adapter is not re-asked when it is the selected adapter."""
        adapter = _CountingDirectAdapter("counted_family", result=None)
        register_adapter("counted_family", adapter)
        self.assertIs(resolve_adapter({"model": {"family": "counted_family"}}), adapter)
        self.assertIsNone(resolve_gptq_config_class(adapter))
        self.assertEqual(adapter.hook_calls, 1)

        family_adapter = _CountingDirectAdapter("counted_other", result=None)
        register_adapter("counted_other", family_adapter)
        variant = _CountingDirectAdapter("counted_other", result=None)
        register_adapter("counted_other_variant", variant)
        self.assertIsNone(resolve_gptq_config_class(variant))
        self.assertEqual(variant.hook_calls, 1)
        self.assertEqual(family_adapter.hook_calls, 1)

    def test_external_override_reaches_stage_and_keeps_external_fields(self):
        """An out-of-tree selection is instantiated with its own fields intact."""
        register_adapter("qwen_external", _ExternalQwenAdapter())
        adapter = resolve_adapter(
            {"model": {"family": "qwen3_vl", "adapter": "qwen_external"}}
        )
        self.assertIsInstance(adapter, _ExternalQwenAdapter)

        cfg = {"calibration": {"dataset": "vqav2", "n_samples": 2}, "runtime": {"seed": 5}}
        run = _run_gptq_stage(
            adapter,
            {
                "name": "gptq",
                "weight_bits": 4,
                "gptq_v2": True,
                "external_knob": 7,
                "unknown_key": "ignored",
            },
            cfg=cfg,
        )
        self.assertIs(type(run.config), _ExternalGPTQConfig)
        self.assertEqual(run.config.external_knob, 7)
        self.assertEqual(run.config.weight_bits, 4)
        self.assertTrue(run.config.gptq_v2)
        self.assertEqual(
            run.config.calibration_dataset_spec,
            GPTQStage._calibration_dataset_spec(cfg["calibration"], cfg["runtime"]),
        )
        self.assertFalse(hasattr(run.config, "unknown_key"))
        self.assertEqual([call[0] for call in run.calls], ["prepare", "calibrate", "convert"])
        self.assertIs(run.calls[0][2], run.config)

    def test_external_override_on_direct_adapter_wins_over_family_default(self):
        """An explicit selection beats the registered family default."""
        adapter = _CountingDirectAdapter("gemma4", result=_ExternalGPTQConfig)
        run = _run_gptq_stage(adapter, {"name": "gptq", "external_knob": 3})
        self.assertIs(type(run.config), _ExternalGPTQConfig)
        self.assertEqual(run.config.external_knob, 3)
        self.assertEqual(adapter.hook_calls, 1)

    def test_universal_variant_does_not_consult_adapter_hook(self):
        """variant=universal uses UniversalGPTQConfig without calling the hook."""

        class _RaisingHookAdapter(_DirectAdapter):
            def get_gptq_config_class(self):
                raise AssertionError("universal must not consult the GPTQ hook")

        run = _run_gptq_stage(
            _RaisingHookAdapter("qwen3_vl"),
            {"name": "gptq", "variant": "universal", "weight_bits": 4},
        )
        self.assertIs(type(run.config), UniversalGPTQConfig)
        self.assertEqual(run.config.weight_bits, 4)

    def test_hook_error_is_not_swallowed(self):
        """A failing hook aborts the stage before prepare instead of falling back."""

        class _FailingHookAdapter(_DirectAdapter):
            def get_gptq_config_class(self):
                raise RuntimeError("hook exploded")

        ctx = RecipeContext(
            cfg={},
            adapter=_FailingHookAdapter("llama"),
            model=torch.nn.Linear(2, 2),
            calibration_inputs=[],
        )

        def forbidden_prepare(*args, **kwargs):
            raise AssertionError("prepare must not run after a hook failure")

        with patch.object(gptq_mod, "prepare", forbidden_prepare):
            with self.assertRaisesRegex(RuntimeError, "hook exploded"):
                GPTQStage().run(ctx, {"name": "gptq"})

    def test_invalid_hook_return_is_rejected(self):
        """Non-config return values raise TypeError instead of selecting GPTQConfig."""
        for bad_value in (object, GPTQConfig(), "GPTQConfig", 42):
            with self.subTest(bad_value=bad_value):
                adapter = _CountingDirectAdapter("llama", result=bad_value)
                ctx = RecipeContext(
                    cfg={},
                    adapter=adapter,
                    model=torch.nn.Linear(2, 2),
                    calibration_inputs=[],
                )

                def forbidden_prepare(*args, **kwargs):
                    raise AssertionError("prepare must not run on an invalid hook")

                with patch.object(gptq_mod, "prepare", forbidden_prepare):
                    with self.assertRaisesRegex(
                        TypeError, "get_gptq_config_class\\(\\) must return"
                    ):
                        GPTQStage().run(ctx, {"name": "gptq"})

    def test_stage_module_holds_no_model_specific_config_imports(self):
        """The generic stage no longer imports or maps model-specific configs."""
        for name in ("Qwen3VLGPTQConfig", "Gemma4GPTQConfig", "_FAMILY_CONFIG_MAP"):
            self.assertFalse(hasattr(gptq_mod, name), name)
        self.assertIs(gptq_mod.GPTQConfig, GPTQConfig)
        self.assertIs(gptq_mod.UniversalGPTQConfig, UniversalGPTQConfig)


if __name__ == "__main__":
    unittest.main()
