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

"""Gemma4 RoPE split/reorder/concat inherit the frozen Q/K norm output grid."""

import copy
import hashlib
import io
import unittest
from unittest import mock

import numpy as np

import tico
import torch
from tico.circle import CircleDocument
from tico.circle.runtime import CircleReferenceRuntime, ExecutionMode
from tico.circle.runtime.quantization import fake_quantize_array
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.config.specs import affine, mx, no_quant
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.observers.base import ObserverBase
from tico.quantization.wrapq.observers.ema import EMAObserver
from tico.quantization.wrapq.observers.minmax import MinMaxObserver
from tico.quantization.wrapq.qscheme import QScheme
from tico.quantization.wrapq.wrappers.gemma4.quant_text_attention import (
    QuantGemma4TextAttention,
)
from tico.quantization.wrapq.wrappers.gemma4.rope import prepare_gemma4_rope_sin

from test.quantization.quant_spec_helpers import make_affine_ptq_config
from test.quantization.wrapq.wrappers.gemma4.test_quant_text_attention import (
    _has_gemma4,
    _make_text_config,
)

UINT8 = DType.uint(8)
INT16 = DType.int(16)
Q_SITES = ("q_x1", "q_x2", "q_cat")
K_SITES = ("k_x1", "k_x2", "k_cat")
ROPE_SITES = Q_SITES + K_SITES


def _independent_sites(dtype=UINT8, sites=ROPE_SITES):
    """Explicit site overrides keep the independent grids of the prior policy."""
    return {name: affine(dtype) for name in sites}


def _fingerprint(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(value.dtype).encode() if hasattr(value, "dtype") else b"")
        digest.update(repr(value).encode() if not torch.is_tensor(value) else b"")
        if torch.is_tensor(value):
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    for name, buffer in sorted(module.named_buffers()):
        digest.update(name.encode())
        digest.update(buffer.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@unittest.skipUnless(
    _has_gemma4(), "required transformers Gemma4 modules are not installed"
)
class TestGemma4RopeDomainInheritance(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026)

    # ------------------------------------------------------------------ fixtures
    @staticmethod
    def _fp(config=None, layer_idx=0):
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention

        torch.manual_seed(77 + layer_idx)
        config = _make_text_config() if config is None else config
        return Gemma4TextAttention(config, layer_idx=layer_idx).eval()

    @staticmethod
    def _qcfg(
        *, rope="pre_negated_sin", dtype=UINT8, observer=MinMaxObserver, overrides=None
    ):
        return make_affine_ptq_config(
            dtype=dtype,
            observer=observer,
            overrides={} if overrides is None else overrides,
            model_args={
                "attention": {"layout": "unrolled", "rope": rope},
                "text": {"max_seq": 32},
            },
        )

    def _attention(self, fp=None, qcfg=None, **qcfg_kwargs):
        fp = self._fp() if fp is None else fp
        qcfg = self._qcfg(**qcfg_kwargs) if qcfg is None else qcfg
        return QuantGemma4TextAttention(copy.deepcopy(fp), qcfg=qcfg).eval()

    @staticmethod
    def _inputs(attention, length, offset=0):
        hidden = torch.linspace(-1.3, 1.6, length * 16).reshape(1, length, 16)
        position = torch.arange(offset, offset + length).float().view(1, length, 1)
        inv = torch.pow(10000.0, -torch.arange(2).float() / 2).view(1, 1, 2)
        phase = (position * inv).repeat(1, 1, 2)
        sin = prepare_gemma4_rope_sin(phase.sin(), attention.attn_options.rope)
        return hidden, (phase.cos(), sin)

    def _forward(self, attention, length, *, past=None, offset=0, shared=None):
        hidden, rope = self._inputs(attention, length, offset)
        k_len = length + (0 if past is None else past[0].size(-2))
        with torch.no_grad():
            return attention(
                hidden,
                rope,
                attention_mask=torch.zeros(1, 1, length, k_len),
                past_key_value=past,
                shared_kv_states=shared,
                use_cache=True,
            )

    def _calibrate(self, attention, *, decode=True):
        attention.enable_calibration()
        out = self._forward(attention, 3)
        if decode and not attention.is_kv_shared_layer:
            self._forward(attention, 1, past=out[2], offset=3)
        attention.freeze_qparams()

    @staticmethod
    def _active_names(attention):
        return [observer.name for observer in attention._all_observers()]

    # ------------------------------------------------------------- eligibility
    def test_default_npu_profile_inherits_both_branches(self):
        attention = QuantGemma4TextAttention(self._fp(), qcfg=PTQConfig()).eval()
        self.assertEqual(attention._rope_domains, {"q": True, "k": True})
        active = self._active_names(attention)
        for name in ROPE_SITES:
            self.assertNotIn(name, active)
            # Dormant sites stay registered so persistence remains exact.
            self.assertIn(f"obs_{name}", dict(attention.named_modules()))
        self.assertIn("q_cos", active)
        self.assertIn("k_rot", active)

    def test_hf_convention_keeps_independent_grids(self):
        attention = self._attention(rope="hf")
        self.assertEqual(attention._rope_domains, {"q": False, "k": False})
        active = self._active_names(attention)
        for name in ROPE_SITES:
            self.assertIn(name, active)
        self._calibrate(attention)
        for name in ROPE_SITES:
            self.assertIsNotNone(getattr(attention, f"obs_{name}").frozen_state)

    def test_branches_are_decided_independently(self):
        cases = {
            "k_cat override": (_independent_sites(sites=("k_cat",)), True, False),
            "q_x1 dtype override": ({"q_x1": affine(INT16)}, False, True),
            "per-channel q_norm output": (
                {
                    "q_norm": {
                        "act_out": affine(
                            UINT8, qscheme=QScheme.PER_CHANNEL_ASYMM, channel_axis=3
                        )
                    }
                },
                False,
                True,
            ),
            "mx k_norm output": ({"k_norm": {"act_out": mx()}}, True, False),
            "float k_norm output": ({"k_norm": {"act_out": no_quant()}}, True, False),
            "mx k sites": ({name: mx() for name in K_SITES}, True, False),
        }
        for label, (overrides, q_expected, k_expected) in cases.items():
            with self.subTest(label=label):
                attention = self._attention(overrides=overrides)
                self.assertEqual(
                    attention._rope_domains, {"q": q_expected, "k": k_expected}
                )
                active = self._active_names(attention)
                for name in Q_SITES:
                    self.assertEqual(name in active, not q_expected)
                for name in K_SITES:
                    self.assertEqual(name in active, not k_expected)
                self._calibrate(attention)
                for name in Q_SITES:
                    frozen = getattr(attention, f"obs_{name}").frozen_state
                    self.assertEqual(frozen is not None, not q_expected)
                for name in K_SITES:
                    frozen = getattr(attention, f"obs_{name}").frozen_state
                    self.assertEqual(frozen is not None, not k_expected)
                self.assertTrue(torch.isfinite(self._forward(attention, 2)[0]).all())

    def test_shared_kv_consumer_inherits_only_the_query_branch(self):
        config = _make_text_config(
            num_hidden_layers=2,
            layer_types=["full_attention"] * 2,
            num_kv_shared_layers=1,
        )
        producer = self._attention(fp=self._fp(config, 0))
        consumer = self._attention(fp=self._fp(config, 1))
        self.assertFalse(hasattr(consumer, "k_norm"))
        self.assertEqual(consumer._rope_domains, {"q": True, "k": False})
        self.assertEqual(producer._rope_domains, {"q": True, "k": True})
        active = self._active_names(consumer)
        for name in ROPE_SITES:
            self.assertNotIn(name, active)
        consumer.bind_cache_source(producer)
        dependencies = consumer._freeze_dependencies()
        self.assertEqual(
            dependencies,
            (
                producer.obs_new_k,
                producer.obs_new_v,
                consumer.q_norm.wrapped.obs_act_out,
            ),
        )
        self.assertEqual(
            producer._freeze_dependencies(),
            (producer.q_norm.wrapped.obs_act_out, producer.k_norm.wrapped.obs_act_out),
        )
        producer.enable_calibration()
        consumer.enable_calibration()
        shared: dict = {}
        self._forward(producer, 3, shared=shared)
        self._forward(consumer, 3, shared=shared)
        producer.freeze_qparams()
        consumer.freeze_qparams()
        shared = {}
        self._forward(producer, 3, shared=shared)
        output = self._forward(consumer, 3, shared=shared)[0]
        self.assertTrue(torch.isfinite(output).all())
        consumer.as_export_module("decode", cache_source=producer)

    # ------------------------------------------------------------- collection
    def test_no_quant_and_calibration_match_the_independent_path(self):
        fp = self._fp()
        inherited = self._attention(fp=fp)
        independent = self._attention(fp=fp, overrides=_independent_sites())
        self.assertEqual(independent._rope_domains, {"q": False, "k": False})
        reference = self._forward(inherited, 4)
        for actual, expected in zip(self._forward(independent, 4), reference):
            for a, b in zip(
                torch.utils._pytree.tree_leaves(actual),
                torch.utils._pytree.tree_leaves(expected),
            ):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        for module in (inherited, independent):
            module.enable_calibration()
        calib_inherited = self._forward(inherited, 4)
        calib_independent = self._forward(independent, 4)
        # CALIB passes float values through: outputs equal NO_QUANT exactly.
        torch.testing.assert_close(calib_inherited[0], reference[0], rtol=0, atol=0)
        torch.testing.assert_close(
            calib_independent[0], calib_inherited[0], rtol=0, atol=0
        )
        for name in ("q_norm", "k_norm"):
            left = getattr(inherited, name).wrapped.obs_act_out
            right = getattr(independent, name).wrapped.obs_act_out
            torch.testing.assert_close(left.min_val, right.min_val, rtol=0, atol=0)
            torch.testing.assert_close(left.max_val, right.max_val, rtol=0, atol=0)

    def test_canonical_norm_observer_collects_once_per_forward(self):
        observers: tuple[type[ObserverBase], ...] = (MinMaxObserver, EMAObserver)
        for observer in observers:
            with self.subTest(observer=observer.__name__):
                fp = self._fp()
                inherited = self._attention(fp=fp, observer=observer)
                independent = self._attention(
                    fp=fp,
                    observer=observer,
                    overrides={
                        name: affine(UINT8, observer=observer) for name in ROPE_SITES
                    },
                )
                counts = {}
                stats = {}
                for label, module in (("R1", inherited), ("R0", independent)):
                    module.enable_calibration()
                    norm = module.q_norm.wrapped.obs_act_out
                    k_norm = module.k_norm.wrapped.obs_act_out
                    site = module.obs_q_x1
                    with (
                        mock.patch.object(
                            norm, "collect", wraps=norm.collect
                        ) as q_calls,
                        mock.patch.object(
                            k_norm, "collect", wraps=k_norm.collect
                        ) as k_calls,
                        mock.patch.object(
                            site, "collect", wraps=site.collect
                        ) as site_calls,
                    ):
                        out = self._forward(module, 3)
                        self._forward(module, 1, past=out[2], offset=3)
                    counts[label] = (q_calls.call_count, k_calls.call_count)
                    self.assertEqual(site_calls.call_count, 0 if label == "R1" else 2)
                    stats[label] = (
                        norm.min_val.clone(),
                        norm.max_val.clone(),
                        k_norm.min_val.clone(),
                        k_norm.max_val.clone(),
                    )
                self.assertEqual(counts["R1"], (2, 2))
                self.assertEqual(counts["R0"], (2, 2))
                for left, right in zip(stats["R1"], stats["R0"]):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)

    # ------------------------------------------------------------------ QUANT
    def test_inherited_sites_redeclare_the_frozen_norm_grid_only(self):
        attention = self._attention()
        self._calibrate(attention)
        for name in ROPE_SITES:
            site = getattr(attention, f"obs_{name}")
            self.assertIsNone(site.frozen_state)
            self.assertFalse(site.has_qparams)
        norm = attention.q_norm.wrapped.obs_act_out
        before = norm.frozen_state
        with (
            mock.patch.object(norm, "fake_quant", wraps=norm.fake_quant) as grid_calls,
            mock.patch.object(norm, "collect", wraps=norm.collect) as collect_calls,
            mock.patch.object(
                attention.obs_q_x1, "fake_quant", wraps=attention.obs_q_x1.fake_quant
            ) as site_calls,
        ):
            output = self._forward(attention, 2)[0]
        # One call from the norm itself and three annotation-only calls.
        self.assertEqual(grid_calls.call_count, 4)
        self.assertEqual(collect_calls.call_count, 0)
        self.assertEqual(site_calls.call_count, 0)
        after = norm.frozen_state
        torch.testing.assert_close(after.scale, before.scale, rtol=0, atol=0)
        torch.testing.assert_close(after.zero_point, before.zero_point, rtol=0, atol=0)
        # Re-declaring an already quantized value is a no-op on the grid.
        value = norm.fake_quant(torch.linspace(-2, 2, 64).reshape(1, 2, 4, 8))
        torch.testing.assert_close(norm.fake_quant(value), value, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(output).all())

    def test_lifecycle_keeps_inventory_and_dependencies_stable(self):
        attention = self._attention()
        snapshots = [self._active_names(attention)]
        dependencies = [attention._freeze_dependencies()]
        attention.enable_calibration()
        snapshots.append(self._active_names(attention))
        dependencies.append(attention._freeze_dependencies())
        out = self._forward(attention, 3)
        self._forward(attention, 1, past=out[2], offset=3)
        attention.freeze_qparams()
        snapshots.append(self._active_names(attention))
        dependencies.append(attention._freeze_dependencies())
        self.assertEqual(snapshots[0], snapshots[1])
        self.assertEqual(snapshots[1], snapshots[2])
        for dependency in dependencies:
            self.assertEqual(
                dependency,
                (
                    attention.q_norm.wrapped.obs_act_out,
                    attention.k_norm.wrapped.obs_act_out,
                ),
            )
        groups = attention.local_observer_groups()
        for name in ROPE_SITES:
            self.assertNotIn(name, [o.name for o in groups["required"]])
        self.assertEqual(attention._ready_capabilities, {"required", "cache"})

    def test_bypassed_norm_output_is_rejected_instead_of_floating(self):
        attention = self._attention()
        attention.enable_calibration()
        out = self._forward(attention, 3)
        self._forward(attention, 1, past=out[2], offset=3)
        attention.k_norm.wrapped.obs_act_out.disable_fake_quant()
        with self.assertRaisesRegex(ValueError, "k_norm output must keep an affine"):
            attention.freeze_qparams()
        self.assertIs(attention._mode, Mode.CALIB)
        self.assertIsNone(attention.q_norm.wrapped.obs_act_out.frozen_state)
        attention.k_norm.wrapped.obs_act_out.enable_fake_quant()
        attention.freeze_qparams()
        self.assertIs(attention._mode, Mode.QUANT)

    # ------------------------------------------------------------ checkpoints
    def test_state_dict_round_trip_reproduces_outputs_grids_and_capabilities(self):
        fp = self._fp()
        original = self._attention(fp=fp)
        self._calibrate(original)
        restored = self._attention(fp=fp)
        restored.load_state_dict(copy.deepcopy(original.state_dict()))
        self.assertIs(restored._mode, Mode.QUANT)
        self.assertEqual(restored._rope_domains, original._rope_domains)
        self.assertEqual(
            restored.get_extra_state()["gemma_rope"],
            {"version": 1, "inherited": {"q": True, "k": True}},
        )
        self.assertEqual(self._active_names(restored), self._active_names(original))
        self.assertEqual(restored._ready_capabilities, original._ready_capabilities)
        for name, observer in original.named_observers():
            loaded = restored.get_observer(name)
            before, after = observer.frozen_state, loaded.frozen_state
            self.assertEqual(before is None, after is None, name)
            if before is not None:
                torch.testing.assert_close(after.scale, before.scale, rtol=0, atol=0)
                torch.testing.assert_close(
                    after.zero_point, before.zero_point, rtol=0, atol=0
                )
        for name in ROPE_SITES:
            self.assertIsNone(getattr(restored, f"obs_{name}").frozen_state)
        expected = self._forward(original, 3)
        actual = self._forward(restored, 3)
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        for a, b in zip(actual[2], expected[2]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(
            self._forward(restored, 1, past=actual[2], offset=3)[0],
            self._forward(original, 1, past=expected[2], offset=3)[0],
            rtol=0,
            atol=0,
        )

    def test_module_pickle_and_deepcopy_preserve_inherited_domains(self):
        original = self._attention()
        self._calibrate(original)
        checkpoint = io.BytesIO()
        torch.save(original, checkpoint)
        checkpoint.seek(0)
        loaded = torch.load(checkpoint, weights_only=False)
        expected = self._forward(original, 3)
        for restored in (loaded, copy.deepcopy(original)):
            self.assertEqual(restored._rope_domains, {"q": True, "k": True})
            self.assertIsNot(
                restored.q_norm.wrapped.obs_act_out, original.q_norm.wrapped.obs_act_out
            )
            actual = self._forward(restored, 3)
            torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
            self.assertEqual(self._active_names(restored), self._active_names(original))
            restored.as_export_module("prefill")

    def test_checkpoints_of_the_other_rope_policy_are_rejected(self):
        fp = self._fp()
        inherited = self._attention(fp=fp)
        self._calibrate(inherited)
        state = copy.deepcopy(inherited.state_dict())
        legacy = copy.deepcopy(state)
        extra = dict(legacy["_extra_state"])
        del extra["gemma_rope"]
        legacy["_extra_state"] = extra
        for strict in (True, False):
            with self.subTest(
                case="legacy record into inheriting wrapper", strict=strict
            ):
                with self.assertRaisesRegex(RuntimeError, "RoPE domain policy"):
                    self._attention(fp=fp).load_state_dict(legacy, strict=strict)
            with self.subTest(
                case="inherited record into independent wrapper", strict=strict
            ):
                independent = self._attention(fp=fp, overrides=_independent_sites())
                with self.assertRaisesRegex(RuntimeError, "RoPE domain policy"):
                    independent.load_state_dict(copy.deepcopy(state), strict=strict)
        # Independent configurations keep accepting records without the key:
        # their semantics are unchanged by this policy.
        for label, qcfg_kwargs in (
            ("explicit site overrides", {"overrides": _independent_sites()}),
            ("hf convention", {"rope": "hf"}),
        ):
            with self.subTest(case=label):
                source = self._attention(fp=fp, **qcfg_kwargs)
                self._calibrate(source)
                saved = copy.deepcopy(source.state_dict())
                self.assertEqual(
                    saved["_extra_state"]["gemma_rope"]["inherited"],
                    {"q": False, "k": False},
                )
                old = dict(saved["_extra_state"])
                del old["gemma_rope"]
                saved["_extra_state"] = old
                restored = self._attention(fp=fp, **qcfg_kwargs)
                restored.load_state_dict(saved)
                torch.testing.assert_close(
                    self._forward(restored, 3)[0],
                    self._forward(source, 3)[0],
                    rtol=0,
                    atol=0,
                )

    def test_pickled_wrapper_without_policy_record_is_rejected_when_inheriting(self):
        inherited = self._attention()
        self._calibrate(inherited)
        del inherited.__dict__["_rope_domains"]
        with self.assertRaisesRegex(RuntimeError, "RoPE domain policy"):
            self._forward(inherited, 2)
        checkpoint = io.BytesIO()
        torch.save(inherited, checkpoint)
        checkpoint.seek(0)
        with self.assertRaisesRegex(RuntimeError, "RoPE domain policy"):
            torch.load(checkpoint, weights_only=False)
        independent = self._attention(rope="hf")
        self._calibrate(independent)
        expected = self._forward(independent, 2)[0]
        del independent.__dict__["_rope_domains"]
        checkpoint = io.BytesIO()
        torch.save(independent, checkpoint)
        checkpoint.seek(0)
        restored = torch.load(checkpoint, weights_only=False)
        self.assertEqual(restored._rope_domains, {"q": False, "k": False})
        torch.testing.assert_close(
            self._forward(restored, 2)[0], expected, rtol=0, atol=0
        )

    # --------------------------------------------------------------- stability
    def test_export_and_evaluation_leave_frozen_state_unchanged(self):
        attention = self._attention()
        self._calibrate(attention)
        before = _fingerprint(attention)
        out = self._forward(attention, 3)
        self._forward(attention, 1, past=out[2], offset=3)
        hidden, rope = self._inputs(attention, 3)
        prefill = attention.as_export_module("prefill").eval()
        with torch.no_grad():
            tico.convert(prefill, (hidden, torch.zeros(1, 1, 3, 3), rope), strict=False)
        decode = attention.as_export_module("decode").eval()
        hidden_1, rope_1 = self._inputs(attention, 1, 3)
        with torch.no_grad():
            tico.convert(
                decode,
                (hidden_1, torch.zeros(1, 1, 1, 4), rope_1),
                kwargs={"past_key_value": out[2]},
                strict=False,
            )
        with self.assertRaises(Exception):
            with torch.no_grad():
                # A mask shorter than the key length fails inside the wrapper.
                tico.convert(
                    prefill, (hidden, torch.zeros(1, 1, 3, 1), rope), strict=False
                )
        self.assertEqual(_fingerprint(attention), before)
        self.assertIs(attention._mode, Mode.QUANT)
        self.assertEqual(attention._rope_domains, {"q": True, "k": True})
        self.assertFalse(hasattr(attention, "_static_export_inputs"))

    def test_k1_cache_contract_is_preserved(self):
        attention = self._attention(
            dtype=INT16, overrides={"new_k": affine(UINT8), "new_v": affine(UINT8)}
        )
        self.assertEqual(attention._rope_domains, {"q": True, "k": True})
        self._calibrate(attention)
        storage_k, storage_v = attention.obs_new_k, attention.obs_new_v
        compute_k, compute_v = attention.obs_present_key, attention.obs_present_value
        self.assertEqual((storage_k.dtype, compute_k.dtype), (UINT8, INT16))
        prefill = self._forward(attention, 3)
        past = prefill[2]
        torch.testing.assert_close(
            storage_k.fake_quant(past[0]), past[0], rtol=0, atol=0
        )
        captured = {}
        original = attention._build_present_key_value

        def spy(**kwargs):
            captured.update(kwargs)
            result = original(**kwargs)
            captured["result"] = result
            return result

        with mock.patch.object(attention, "_build_present_key_value", side_effect=spy):
            decode = self._forward(attention, 1, past=past, offset=3)
        raw_k, raw_v = captured["current_key_value"]
        new_k, new_v = captured["new_key_value"]
        compute, storage, _ = captured["result"]
        torch.testing.assert_close(new_k, storage_k.fake_quant(raw_k), rtol=0, atol=0)
        torch.testing.assert_close(new_v, storage_v.fake_quant(raw_v), rtol=0, atol=0)
        for actual, past_part, raw, compute_obs in (
            (compute[0], past[0], raw_k, compute_k),
            (compute[1], past[1], raw_v, compute_v),
        ):
            expected = torch.cat(
                (compute_obs.fake_quant(past_part), compute_obs.fake_quant(raw)), dim=-2
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for stored, past_part, delta in zip(storage, past, (new_k, new_v)):
            torch.testing.assert_close(
                stored, torch.cat((past_part, delta), dim=-2), rtol=0, atol=0
            )
        for d, s in zip(decode[2], (new_k, new_v)):
            torch.testing.assert_close(d, s, rtol=0, atol=0)
        # Current compute uses the raw projection, never the storage-rounded delta.
        self.assertFalse(
            torch.equal(compute_k.fake_quant(raw_k), compute_k.fake_quant(new_k))
        )

    # ---------------------------------------------------------- saved Circle
    @staticmethod
    def _runtime(module, args, kwargs=None, *, optimize):
        with torch.no_grad():
            converted = tico.convert(
                module.eval(),
                args,
                kwargs=kwargs,
                strict=False,
                config=tico.CompileConfigV1(circle_optimize=optimize),
            )
        document = CircleDocument.from_bytes(bytes(converted.circle_binary))
        document.verify()
        return CircleReferenceRuntime(document)

    @staticmethod
    def _grid(tensor):
        quantization = tensor.quantization
        if quantization is None or not quantization.scale:
            return None
        return (
            str(tensor.dtype),
            tuple(quantization.scale),
            tuple(quantization.zero_point),
        )

    def _rope_chains(self, program):
        """Return per-branch (norm output, halves, concat output, quantizes)."""
        producers = {
            output: operator
            for operator in program.operators
            for output in operator.outputs
        }
        chains = []
        for norm in program.operators:
            if norm.name != "RMS_NORM":
                continue
            source = norm.outputs[0]
            users = [op for op in program.operators if source in op.inputs]
            split_ops = [
                op for op in users if op.name in ("SPLIT", "SPLIT_V", "STRIDED_SLICE")
            ]
            if not split_ops:
                continue  # v_norm feeds attention without a rotation.
            halves = [out for op in split_ops for out in op.outputs]
            self.assertEqual(len(halves), 2)
            concats = [
                op
                for op in program.operators
                if op.name == "CONCATENATION" and set(op.inputs) == set(halves)
            ]
            self.assertEqual(len(concats), 1)
            concat_out = concats[0].outputs[0]
            consumers = [op for op in program.operators if concat_out in op.inputs]
            self.assertEqual([op.name for op in consumers], ["MUL"])
            region = set(halves) | {concat_out}
            quantizes = [
                op
                for op in program.operators
                if op.name == "QUANTIZE"
                and (op.inputs[0] in region or op.inputs[0] == source)
                and op.outputs[0] in region
            ]
            chains.append((source, halves, concat_out, quantizes))
        return chains

    def _assert_parity(self, module, args, kwargs, runtime, *, label):
        leaves, spec = torch.utils._pytree.tree_flatten((args, kwargs or {}))
        self.assertEqual(len(leaves), len(runtime.program.inputs))
        encoded = []
        for value, index in zip(leaves, runtime.program.inputs):
            tensor = runtime.program.tensor(index)
            array = value.detach().numpy()
            if tensor.is_quantized_integer:
                array = fake_quantize_array(array, tensor)
            encoded.append(np.array(array, copy=True))
        eager_args, eager_kwargs = torch.utils._pytree.tree_unflatten(
            [torch.from_numpy(value.copy()) for value in encoded], spec
        )
        with torch.no_grad():
            eager = module(*eager_args, **eager_kwargs)
        expected = torch.utils._pytree.tree_leaves(eager)
        result = runtime.run(encoded, mode=ExecutionMode.FAKE_QUANTIZE)
        self.assertEqual(len(result.outputs), len(expected))
        for index, (actual, wanted) in enumerate(zip(result.outputs, expected)):
            # Same frozen grids on both sides; only accumulation order differs.
            np.testing.assert_allclose(
                actual,
                wanted.detach().numpy(),
                rtol=1e-4,
                atol=1e-5,
                err_msg=f"{label} output{index}",
            )
        return result.outputs

    def test_serialized_rope_domains_match_the_norm_output(self):
        profiles = {
            "uint8": dict(dtype=UINT8, overrides=None),
            "u8 storage / i16 compute": dict(
                dtype=INT16,
                overrides={"new_k": affine(UINT8), "new_v": affine(UINT8)},
            ),
        }
        for profile, kwargs in profiles.items():
            attention = self._attention(**kwargs)
            self._calibrate(attention)
            prefill_module = attention.as_export_module("prefill").eval()
            decode_module = attention.as_export_module("decode").eval()
            hidden, rope = self._inputs(attention, 3)
            prefill_args = (hidden, torch.zeros(1, 1, 3, 3), rope)
            mixed = kwargs["overrides"] is not None
            for optimize in (False, True):
                with self.subTest(profile=profile, optimize=optimize):
                    prefill = self._runtime(
                        prefill_module, prefill_args, optimize=optimize
                    )
                    chains = self._rope_chains(prefill.program)
                    self.assertEqual(len(chains), 2)
                    for source, halves, concat_out, quantizes in chains:
                        grid = self._grid(prefill.program.tensor(source))
                        self.assertIsNotNone(grid)
                        for index in (*halves, concat_out):
                            self.assertEqual(
                                self._grid(prefill.program.tensor(index)), grid
                            )
                        self.assertEqual(quantizes, [])
                    quantize_count = sum(
                        op.name == "QUANTIZE" for op in prefill.program.operators
                    )
                    # Only the UINT8 storage write-outs remain in the mixed profile.
                    self.assertEqual(quantize_count, 2 if mixed else 0)
                    outputs = self._assert_parity(
                        prefill_module,
                        prefill_args,
                        None,
                        prefill,
                        label=f"{profile} prefill",
                    )
                    past = tuple(
                        torch.from_numpy(value.copy()) for value in outputs[-2:]
                    )
                    hidden_1, rope_1 = self._inputs(attention, 1, 3)
                    decode_args = (hidden_1, torch.zeros(1, 1, 1, 4), rope_1)
                    decode_kwargs = {"past_key_value": past}
                    decode = self._runtime(
                        decode_module, decode_args, decode_kwargs, optimize=optimize
                    )
                    chains = self._rope_chains(decode.program)
                    self.assertEqual(len(chains), 2)
                    for source, halves, concat_out, quantizes in chains:
                        grid = self._grid(decode.program.tensor(source))
                        for index in (*halves, concat_out):
                            self.assertEqual(
                                self._grid(decode.program.tensor(index)), grid
                            )
                        self.assertEqual(quantizes, [])
                    storage_conversions = []
                    for index in decode.program.inputs[-2:]:
                        users = [
                            op for op in decode.program.operators if index in op.inputs
                        ]
                        self.assertEqual(len(users), 1)
                        storage_conversions.append(users[0].name)
                    quantize_count = sum(
                        op.name == "QUANTIZE" for op in decode.program.operators
                    )
                    if mixed:
                        # Past S inputs convert to C; current delta writes out S.
                        self.assertEqual(storage_conversions, ["QUANTIZE", "QUANTIZE"])
                        self.assertEqual(quantize_count, 4)
                    else:
                        self.assertEqual(quantize_count, 0)
                    for index, observer in zip(
                        decode.program.outputs[-2:],
                        (attention.obs_new_k, attention.obs_new_v),
                    ):
                        scale, zero_point = observer.compute_qparams()
                        self.assertEqual(
                            self._grid(decode.program.tensor(index)),
                            ("uint8", (float(scale),), (int(zero_point),)),
                        )
                    self._assert_parity(
                        decode_module,
                        decode_args,
                        decode_kwargs,
                        decode,
                        label=f"{profile} decode",
                    )


if __name__ == "__main__":
    unittest.main()
