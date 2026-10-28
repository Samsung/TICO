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

"""Behavioral contracts for the common model freeze transaction."""

import copy
import gc
import unittest
import weakref
from unittest.mock import patch

import torch

from tico.quantization import convert, prepare
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.lifecycle import enable_calibration, freeze_qparams
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.observers.minmax import MinMaxObserver
from tico.quantization.wrapq.observers.percentile import PercentileObserver
from tico.quantization.wrapq.quantizer import PTQQuantizer
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from torch import nn


class Pair(QuantModuleBase):
    def __init__(self):
        super().__init__()
        self.first = MinMaxObserver(name="first")
        self.last = PercentileObserver(name="last")
        self.preparations = 0

    def _all_observers(self):
        return self.first, self.last

    def _prepare_calibration(self):
        self.preparations += 1

    def forward(self, x):
        return self._fq(self._fq(x, self.first), self.last)


class Conditional(Pair):
    def local_observer_groups(self):
        return {"required": (self.first,), "cache": (self.last,)}

    def forward(self, x, use_cache=False):
        if use_cache:
            self.require_capability("cache")
            x = self._fq(x, self.last)
        return self._fq(x, self.first)


class AliasHolder(QuantModuleBase):
    def __init__(self, *, required_alias=False, leaf=None):
        super().__init__()
        self.optional_ref = leaf if leaf is not None else Pair()
        self.alternative_ref = self.optional_ref
        if required_alias:
            self.required_ref = self.optional_ref
        self.first_trigger = MinMaxObserver(name="first_trigger")
        self.second_trigger = MinMaxObserver(name="second_trigger")

    def _all_observers(self):
        return self.first_trigger, self.second_trigger

    def local_observer_groups(self):
        return {
            "required": (),
            "first": (self.first_trigger,),
            "second": (self.second_trigger,),
        }

    def _conditional_quant_subtrees(self):
        return {"first": ("optional_ref",), "second": ("alternative_ref",)}


class TestFrozenLifecycle(unittest.TestCase):
    def test_once_only_arbitrary_containers_and_aliases(self):
        leaf = Pair()
        root = nn.Module()
        root.plain = nn.Module()
        root.plain.leaf = leaf
        root.tied = leaf
        leaf.alias = leaf.first
        with patch.object(leaf.first, "reset", wraps=leaf.first.reset) as reset:
            enable_calibration(root)
            self.assertEqual(reset.call_count, 1)
        self.assertEqual(leaf.preparations, 1)
        leaf(torch.tensor([-1.0, 2.0]))
        with patch.object(
            leaf.first, "derive_candidate", wraps=leaf.first.derive_candidate
        ) as derive:
            PTQQuantizer(PTQConfig()).convert(root)
            self.assertEqual(derive.call_count, 1)
        before = leaf.first.require_frozen()
        freeze_qparams(root)
        after = leaf.first.require_frozen()
        assert before.scale is not None and after.scale is not None
        self.assertTrue(torch.equal(before.scale, after.scale))

    def test_planning_does_not_retain_the_model_until_cyclic_gc(self):
        was_enabled = gc.isenabled()
        gc.disable()
        try:
            model = Pair()
            model.enable_calibration()
            model(torch.tensor([-1.0, 2.0]))
            model.freeze_qparams()
            reference = weakref.ref(model)
            del model
            self.assertIsNone(reference())
        finally:
            if was_enabled:
                gc.enable()

    def test_declared_unregistered_observer_rejects(self):
        model = Pair()
        observer = model._modules.pop("first")
        object.__setattr__(model, "first", observer)
        with self.assertRaisesRegex(RuntimeError, "registered local observer"):
            model.enable_calibration()

    def test_required_alias_is_not_hidden_by_an_optional_reference(self):
        class Holder(QuantModuleBase):
            def __init__(self, branch):
                super().__init__()
                self.branch = branch

            def _all_observers(self):
                return ()

            def _conditional_quant_subtrees(self):
                return {"branch": ("branch",)}

        leaf = Pair()
        root = nn.ModuleList([Holder(leaf), leaf])
        enable_calibration(root)
        with self.assertRaisesRegex(RuntimeError, "calibration"):
            freeze_qparams(root)
        self.assertIs(leaf._mode, Mode.CALIB)

    def test_tied_optional_module_is_ready_through_either_capability(self):
        class Holder(QuantModuleBase):
            def __init__(self, branch):
                super().__init__()
                self.branch = branch
                self.trigger = MinMaxObserver(name="trigger")

            def _all_observers(self):
                return (self.trigger,)

            def local_observer_groups(self):
                return {"required": (), "branch": (self.trigger,)}

            def _conditional_quant_subtrees(self):
                return {"branch": ("branch",)}

        leaf = Pair()
        first, second = Holder(leaf), Holder(leaf)
        root = nn.ModuleList([first, second])
        enable_calibration(root)
        x = torch.tensor([-1.0, 2.0])
        leaf(x)
        first.trigger.collect(x)
        freeze_qparams(root)
        self.assertTrue(first.has_capability("branch"))
        self.assertFalse(second.has_capability("branch"))
        self.assertIsNotNone(leaf.first.frozen_state)
        self.assertFalse(leaf._conditional_unavailable)
        root.load_state_dict(copy.deepcopy(root.state_dict()))

    def test_same_parent_required_alias_cannot_hide_missing_calibration(self):
        model = AliasHolder(required_alias=True)
        model.enable_calibration()
        with self.assertRaisesRegex(RuntimeError, "calibration"):
            model.freeze_qparams()
        self.assertIs(model.optional_ref, model.required_ref)
        self.assertIs(model.required_ref._mode, Mode.CALIB)
        self.assertIsNone(model.required_ref.first.frozen_state)

    def test_same_parent_optional_aliases_admit_either_complete_capability(self):
        for selected in ("first", "second", None):
            with self.subTest(selected=selected):
                model = AliasHolder()
                model.enable_calibration()
                x = torch.tensor([-1.0, 2.0])
                model.optional_ref(x)
                if selected is not None:
                    getattr(model, selected + "_trigger").collect(x)
                with patch.object(
                    model.optional_ref.first,
                    "install_candidate",
                    wraps=model.optional_ref.first.install_candidate,
                ) as install:
                    model.freeze_qparams()
                self.assertEqual(install.call_count, int(selected is not None))
                self.assertEqual(model.has_capability("first"), selected == "first")
                self.assertEqual(model.has_capability("second"), selected == "second")
                self.assertEqual(
                    model.optional_ref._conditional_unavailable, selected is None
                )
                restored = AliasHolder()
                restored.load_state_dict(copy.deepcopy(model.state_dict()))
                self.assertIs(restored.optional_ref, restored.alternative_ref)
                self.assertEqual(
                    restored._ready_capabilities, model._ready_capabilities
                )
                self.assertEqual(
                    restored.optional_ref._ready_capabilities,
                    model.optional_ref._ready_capabilities,
                )
                if selected is not None:
                    self.assertTrue(
                        torch.equal(model.optional_ref(x), restored.optional_ref(x))
                    )
                else:
                    self.assertFalse(restored.optional_ref.has_capability("required"))

    def test_required_alias_topology_roundtrip_and_readiness_validation(self):
        model = AliasHolder(required_alias=True)
        model.enable_calibration()
        x = torch.tensor([-1.0, 2.0])
        model.required_ref(x)
        model.first_trigger.collect(x)
        model.freeze_qparams()
        state = copy.deepcopy(model.state_dict())
        restored = AliasHolder(required_alias=True)
        restored.load_state_dict(state)
        self.assertIs(restored.required_ref, restored.optional_ref)
        self.assertIs(restored.required_ref, restored.alternative_ref)
        self.assertTrue(restored.has_capability("first"))
        self.assertFalse(restored.has_capability("second"))
        self.assertFalse(restored.required_ref._conditional_unavailable)
        self.assertTrue(torch.equal(model.required_ref(x), restored.required_ref(x)))
        for alias in ("required_ref", "optional_ref", "alternative_ref"):
            state[alias + "._extra_state"]["conditional_unavailable"] = True
        with self.assertRaisesRegex(RuntimeError, "[Cc]onditional|[Uu]navailable"):
            restored.load_state_dict(state, strict=False)

    def test_conditional_paths_preserve_aliases_of_intermediate_containers(self):
        containers = {
            "plain": lambda leaf: (nn.Module(), "leaf"),
            "list": lambda leaf: (nn.ModuleList([leaf]), "0"),
            "dict": lambda leaf: (nn.ModuleDict({"leaf": leaf}), "leaf"),
            "sequential": lambda leaf: (nn.Sequential(leaf), "0"),
        }
        for name, build in containers.items():
            with self.subTest(container=name):

                class Holder(QuantModuleBase):
                    def __init__(self):
                        super().__init__()
                        leaf = Pair()
                        container, child_name = build(leaf)
                        if name == "plain":
                            container.leaf = leaf
                        self.required_container = container
                        self.optional_container = container
                        self.path = "optional_container." + child_name

                    def _all_observers(self):
                        return ()

                    def _conditional_quant_subtrees(self):
                        return {"optional": (self.path,)}

                model = Holder()
                model.enable_calibration()
                with self.assertRaisesRegex(RuntimeError, "calibration"):
                    model.freeze_qparams()

    def test_conditional_paths_through_plain_containers_can_be_unavailable(self):
        class Holder(QuantModuleBase):
            def __init__(self):
                super().__init__()
                self.container = nn.Module()
                self.container.entries = nn.ModuleDict({"leaf": Pair()})

            def _all_observers(self):
                return ()

            def _conditional_quant_subtrees(self):
                return {"optional": ("container.entries.leaf",)}

        model = Holder()
        model.enable_calibration()
        model.freeze_qparams()
        self.assertFalse(model.has_capability("optional"))
        leaf = model.get_submodule("container.entries.leaf")
        assert isinstance(leaf, Pair)
        self.assertTrue(leaf._conditional_unavailable)
        restored = Holder()
        restored.load_state_dict(copy.deepcopy(model.state_dict()))
        self.assertFalse(restored.has_capability("optional"))

    def test_registration_cycles_do_not_duplicate_or_reclassify_required_work(self):
        model = AliasHolder(required_alias=True)
        model.plain = nn.Module()
        model.plain.loop = model.plain
        model.loop = model
        model.enable_calibration()
        x = torch.tensor([-1.0, 2.0])
        model.required_ref(x)
        with patch.object(
            model.required_ref.first,
            "install_candidate",
            wraps=model.required_ref.first.install_candidate,
        ) as install:
            model.freeze_qparams()
        self.assertEqual(install.call_count, 1)
        self.assertFalse(model.required_ref._conditional_unavailable)

    def test_nested_optional_local_group_does_not_disable_required_child_route(self):
        child = Conditional()
        model = AliasHolder(leaf=child)
        model.enable_calibration()
        x = torch.tensor([-1.0, 2.0])
        child(x)
        model.first_trigger.collect(x)
        model.freeze_qparams()
        self.assertTrue(model.has_capability("first"))
        self.assertFalse(child._conditional_unavailable)
        self.assertTrue(child.has_capability("required"))
        self.assertFalse(child.has_capability("cache"))
        self.assertIsNotNone(child.first.frozen_state)
        self.assertIsNone(child.last.frozen_state)
        child(x)
        with self.assertRaisesRegex(RuntimeError, "capability 'cache'"):
            child(x, use_cache=True)
        restored = AliasHolder(leaf=Conditional())
        restored.load_state_dict(copy.deepcopy(model.state_dict()))
        self.assertTrue(restored.has_capability("first"))
        self.assertFalse(restored.optional_ref.has_capability("cache"))

    def test_nested_conditional_boundary_does_not_disable_outer_required_route(self):
        class Nested(Pair):
            def __init__(self):
                super().__init__()
                self.optional_child = Pair()

            def _conditional_quant_subtrees(self):
                return {"nested": ("optional_child",)}

        model = AliasHolder(leaf=Nested())
        model.enable_calibration()
        x = torch.tensor([-1.0, 2.0])
        model.optional_ref(x)
        model.first_trigger.collect(x)
        model.freeze_qparams()
        self.assertTrue(model.has_capability("first"))
        self.assertFalse(model.optional_ref.has_capability("nested"))
        nested_child = model.optional_ref.get_submodule("optional_child")
        assert isinstance(nested_child, Pair)
        self.assertTrue(nested_child._conditional_unavailable)
        restored = AliasHolder(leaf=Nested())
        restored.load_state_dict(copy.deepcopy(model.state_dict()))
        self.assertTrue(restored.has_capability("first"))
        self.assertFalse(restored.optional_ref.has_capability("nested"))
        state = copy.deepcopy(model.state_dict())
        for alias in ("optional_ref", "alternative_ref"):
            state[alias + "._extra_state"]["capabilities"].append("nested")
        with self.assertRaisesRegex(RuntimeError, "Corrupt frozen capability"):
            restored.load_state_dict(state, strict=False)

    def test_unavailable_stateless_child_has_no_vacuously_ready_groups(self):
        class Stateless(QuantModuleBase):
            def _all_observers(self):
                return ()

        model = AliasHolder(leaf=Stateless())
        model.enable_calibration()
        model.freeze_qparams()
        self.assertTrue(model.optional_ref._conditional_unavailable)
        self.assertEqual(model.optional_ref._ready_capabilities, set())
        restored = AliasHolder(leaf=Stateless())
        restored.load_state_dict(copy.deepcopy(model.state_dict()))
        self.assertEqual(restored.optional_ref._ready_capabilities, set())

    def test_optional_groups_can_share_an_observer_without_sharing_readiness(self):
        class Shared(Pair):
            def __init__(self):
                super().__init__()
                self.shared = MinMaxObserver(name="shared")

            def _all_observers(self):
                return super()._all_observers() + (self.shared,)

            def local_observer_groups(self):
                return {
                    "required": (),
                    "first": (self.first, self.shared),
                    "last": (self.last, self.shared),
                }

        model = Shared()
        model.enable_calibration()
        x = torch.tensor([-1.0, 2.0])
        model.first.collect(x)
        model.shared.collect(x)
        with patch.object(
            model.shared, "install_candidate", wraps=model.shared.install_candidate
        ) as install:
            model.freeze_qparams()
        self.assertEqual(install.call_count, 1)
        self.assertTrue(model.has_capability("first"))
        self.assertFalse(model.has_capability("last"))
        restored = Shared()
        restored.load_state_dict(copy.deepcopy(model.state_dict()))
        self.assertTrue(restored.has_capability("first"))
        self.assertFalse(restored.has_capability("last"))

    def test_invalid_conditional_declarations_reject_with_migration_guidance(self):
        class Invalid(Pair):
            declaration: object

            def _conditional_quant_subtrees(self):
                return {"invalid": self.declaration}

        empty_paths: tuple[str, ...] = ()
        for declaration, message in (
            ((Pair(),), "registration-path strings"),
            ((), "registered module paths"),
            ([], "nonempty tuple"),
            ((item for item in empty_paths), "nonempty tuple"),
            ((item for item in ("first",)), "nonempty tuple"),
            (("missing",), "not a registered module path"),
            (("first..missing",), "registration-path strings"),
        ):
            with self.subTest(message=message):
                model = Invalid()
                model.declaration = declaration
                with self.assertRaisesRegex(RuntimeError, message):
                    model.enable_calibration()

    def test_one_registration_edge_cannot_have_multiple_capability_owners(self):
        class SharedEdge(Pair):
            def __init__(self):
                super().__init__()
                self.child = Pair()

            def _conditional_quant_subtrees(self):
                return {"first_route": ("child",), "second_route": ("child",)}

        class Nested(Pair):
            def __init__(self):
                super().__init__()
                self.child = Pair()

            def _conditional_quant_subtrees(self):
                return {"child_route": ("child",)}

        class Ancestor(Pair):
            def __init__(self):
                super().__init__()
                self.nested = Nested()

            def _conditional_quant_subtrees(self):
                return {"ancestor_route": ("nested.child",)}

        for model in (SharedEdge(), Ancestor()):
            with self.subTest(model=type(model).__name__):
                with self.assertRaisesRegex(RuntimeError, "one capability owner"):
                    model.enable_calibration()

    def test_conditional_registration_cycles_reject_instead_of_claiming_readiness(self):
        class Cycle(Pair):
            def _conditional_quant_subtrees(self):
                return {"cycle": ("parent",)}

        model = Cycle()
        model.parent = model
        with self.assertRaisesRegex(RuntimeError, "registration cycles"):
            model.enable_calibration()
        root = nn.Module()
        root.child = Cycle()
        root.child.parent = root
        with self.assertRaisesRegex(RuntimeError, "registration cycles"):
            enable_calibration(root)

    def test_empty_optional_group_without_subtree_is_rejected(self):
        class Empty(QuantModuleBase):
            def _all_observers(self):
                return ()

            def local_observer_groups(self):
                return {"required": (), "missing": ()}

        with self.assertRaisesRegex(RuntimeError, "Empty optional capability"):
            Empty().enable_calibration()

    def test_conflicting_observer_owners_reject(self):
        first, second = Pair(), Pair()
        second.first = first.first
        root = nn.ModuleList([first, second])
        with self.assertRaisesRegex(RuntimeError, "conflicting wrapper owners"):
            enable_calibration(root)

    def test_partial_optional_subtree_has_no_partial_frozen_grids(self):
        class OptionalTree(Pair):
            def __init__(self):
                super().__init__()
                self.branch = Pair()

            def _conditional_quant_subtrees(self):
                return {"branch": ("branch",)}

        model = OptionalTree()
        model.enable_calibration()
        x = torch.tensor([-1.0, 2.0])
        model(x)
        model.branch.first.collect(x)
        model.freeze_qparams()
        self.assertFalse(model.has_capability("branch"))
        self.assertIsNone(model.branch.first.frozen_state)
        self.assertIsNone(model.branch.last.frozen_state)
        restored = OptionalTree()
        restored.load_state_dict(copy.deepcopy(model.state_dict()))
        self.assertFalse(restored.has_capability("branch"))

    def test_real_percentile_failure_preserves_all_state_and_retry_handle(self):
        for inplace in (True, False):
            with self.subTest(inplace=inplace):
                model = Pair()
                enable_calibration(model)
                setattr(model, "tico_quantizer", PTQQuantizer(PTQConfig()))
                model.first.collect(torch.tensor([-1.0, 2.0]))
                before = copy.deepcopy(model.state_dict())
                with self.assertRaisesRegex(RuntimeError, "calibration"):
                    convert(model, inplace=inplace)
                self.assertTrue(hasattr(model, "tico_quantizer"))
                self.assertIs(model._mode, Mode.CALIB)
                self.assertTrue(model.first.enabled)
                self.assertFalse(model.first.has_qparams)
                self.assertEqual(before.keys(), model.state_dict().keys())
                for key, value in before.items():
                    if isinstance(value, torch.Tensor):
                        self.assertTrue(
                            torch.equal(value, model.state_dict()[key]), key
                        )
                model(torch.tensor([-1.0, 2.0]))
                frozen = convert(model, inplace=inplace)
                self.assertIs(frozen._mode, Mode.QUANT)
                self.assertFalse(hasattr(frozen, "tico_quantizer"))
                self.assertEqual(hasattr(model, "tico_quantizer"), not inplace)
                self.assertIs(model._mode, Mode.QUANT if inplace else Mode.CALIB)

    def test_install_failure_rolls_back_small_state(self):
        model = Pair()
        model.enable_calibration()
        model(torch.tensor([-1.0, 2.0]))
        original = model.last.install_candidate

        def fail(state):
            original(state)
            raise RuntimeError("install failure")

        with patch.object(model.last, "install_candidate", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "install failure"):
                model.freeze_qparams()
        self.assertIs(model._mode, Mode.CALIB)
        self.assertTrue(model.first.enabled)
        self.assertTrue(model.last.enabled)
        self.assertIsNone(model.first.frozen_state)
        self.assertIsNone(model.last.frozen_state)
        model.freeze_qparams()
        self.assertIs(model._mode, Mode.QUANT)

    def test_install_failure_restores_existing_clipping_provenance(self):
        model = Pair()
        model.enable_calibration()
        model(torch.tensor([-1.0, 2.0]))
        model.freeze_qparams()
        before = model.last.require_frozen()
        assert before.clipping_bounds is not None

        def fail(state):
            del state
            bounds = model.last._frozen_clipping_bounds
            assert bounds is not None
            bounds[0].fill_(333.0)
            raise RuntimeError("in-place install failure")

        with patch.object(model.last, "install_candidate", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "in-place install failure"):
                model.freeze_qparams()
        after = model.last.require_frozen()
        assert after.clipping_bounds is not None
        for actual, expected in zip(after.clipping_bounds, before.clipping_bounds):
            self.assertTrue(torch.equal(actual, expected))
        self.assertIs(model._mode, Mode.QUANT)
        self.assertFalse(model.last.enabled)

    def test_parent_semantic_failure_does_not_commit_children(self):
        class Semantic(Pair):
            reject = True

            def _validate_freeze(self, plan, available):
                if self.reject:
                    raise RuntimeError("invalid semantic binding")

        model = Semantic()
        model.child = Pair()
        model.enable_calibration()
        x = torch.tensor([-1.0, 2.0])
        model(x)
        model.child(x)
        with self.assertRaisesRegex(RuntimeError, "invalid semantic binding"):
            model.freeze_qparams()
        for module in (model, model.child):
            self.assertIs(module._mode, Mode.CALIB)
            self.assertIsNone(module.first.frozen_state)
            self.assertIsNone(module.last.frozen_state)
        model.reject = False
        model.freeze_qparams()
        self.assertIs(model.child._mode, Mode.QUANT)

    def test_invalid_optional_statistics_are_not_silently_unavailable(self):
        model = Conditional()
        model.enable_calibration()
        model.first.collect(torch.tensor([-1.0, 2.0]))
        # Percentile explicitly filters nonfinite samples; use a min/max site to
        # distinguish invalid observed data from the untouched-range sentinel.
        setattr(model, "last", MinMaxObserver(name="last"))
        model.last.collect(torch.tensor([float("nan")]))
        with self.assertRaisesRegex(RuntimeError, "invalid calibration"):
            model.freeze_qparams()
        self.assertIs(model._mode, Mode.CALIB)

    def test_plain_root_postload_rejects_missing_frozen_site_nonstrict(self):
        root = nn.Module()
        root.child = Pair()
        enable_calibration(root)
        root.child(torch.tensor([-1.0, 2.0]))
        freeze_qparams(root)
        state = copy.deepcopy(root.state_dict())
        del state["child.first._extra_state"]
        del state["child.first._cached_scale"]
        del state["child.first._cached_zp"]
        with self.assertRaisesRegex(RuntimeError, "(Missing frozen site|not frozen)"):
            root.load_state_dict(state, strict=False)

    def test_fresh_plain_receiver_checks_promised_local_readiness(self):
        source = Pair()
        source.enable_calibration()
        source(torch.tensor([-1.0, 2.0]))
        source.freeze_qparams()
        root = nn.Module()
        root.child = source
        state = copy.deepcopy(root.state_dict())
        for suffix in ("_extra_state", "_cached_scale", "_cached_zp"):
            del state["child.first." + suffix]
        receiver = nn.Module()
        receiver.child = Pair()
        with self.assertRaisesRegex(RuntimeError, "not frozen"):
            receiver.load_state_dict(state, strict=False)

    def test_fresh_plain_receiver_cannot_hide_required_sites_in_capabilities(self):
        source = Pair()
        source.enable_calibration()
        source(torch.tensor([-1.0, 2.0]))
        source.freeze_qparams()
        root = nn.Module()
        root.child = source
        state = copy.deepcopy(root.state_dict())
        state["child._extra_state"]["capabilities"] = []
        for suffix in ("_extra_state", "_cached_scale", "_cached_zp"):
            del state["child.first." + suffix]
        receiver = nn.Module()
        receiver.child = Pair()
        with self.assertRaisesRegex(RuntimeError, "not frozen"):
            receiver.load_state_dict(state, strict=False)

    def test_whole_module_boundary_rejects_corrupt_mode_instead_of_float_execution(
        self,
    ):
        from tico.quantization.wrapq.lifecycle import validate_frozen_model

        model = Pair()
        model.enable_calibration()
        model(torch.tensor([-1.0, 2.0]))
        model.freeze_qparams()
        object.__setattr__(model, "_mode", "QUANT")
        with self.assertRaisesRegex(RuntimeError, "Corrupt WrapQ frozen readiness"):
            validate_frozen_model(model)

    def test_nonstrict_load_cannot_demote_a_child_of_a_frozen_model(self):
        model = Pair()
        model.child = Pair()
        model.enable_calibration()
        x = torch.tensor([-1.0, 2.0])
        model(x)
        model.child(x)
        model.freeze_qparams()
        for mode in ("CALIB", "NO_QUANT"):
            with self.subTest(mode=mode):
                state = copy.deepcopy(model.state_dict())
                state["child._extra_state"]["mode"] = mode
                with self.assertRaisesRegex(
                    RuntimeError, "Inconsistent frozen wrapper modes"
                ):
                    model.load_state_dict(state, strict=False)

    def test_nonstrict_cannot_demote_wrapper_missing_mode_schema(self):
        model = prepare(nn.Linear(2, 2, bias=False).eval(), PTQConfig())
        model(torch.tensor([[-1.0, 2.0]]))
        frozen = convert(model)
        state = copy.deepcopy(frozen.state_dict())
        del state["wrapped._extra_state"]
        receiver = prepare(nn.Linear(2, 2, bias=False).eval(), PTQConfig())
        with self.assertRaisesRegex(RuntimeError, "Incomplete WrapQ wrapper state"):
            receiver.load_state_dict(state, strict=False)

    def test_optional_capability_is_explicit_and_persistent(self):
        model = Conditional()
        model.enable_calibration()
        x = torch.tensor([-1.0, 2.0])
        model(x)
        model.freeze_qparams()
        self.assertIsNone(model.last.frozen_state)
        model(x)
        with self.assertRaisesRegex(RuntimeError, "capability 'cache'"):
            model(x, use_cache=True)
        restored = Conditional()
        restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=False)
        self.assertTrue(torch.equal(model(x), restored(x)))
        with self.assertRaisesRegex(RuntimeError, "capability 'cache'"):
            restored(x, use_cache=True)

    def test_uncalibrated_linear_rejects_before_quant(self):
        model = prepare(nn.Linear(2, 2, bias=False).eval(), PTQConfig())
        with self.assertRaisesRegex(RuntimeError, "calibration"):
            convert(model)
        self.assertTrue(hasattr(model, "tico_quantizer"))
        self.assertIs(model.wrapped._mode, Mode.CALIB)

    def test_legacy_mutating_lifecycle_override_rejected(self):
        class Old(Pair):
            def freeze_qparams(self):
                super().freeze_qparams()

        with self.assertRaisesRegex(RuntimeError, "migrate lifecycle overrides"):
            freeze_qparams(Old())


if __name__ == "__main__":
    unittest.main()
