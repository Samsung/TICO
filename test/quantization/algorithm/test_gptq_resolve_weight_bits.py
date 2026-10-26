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

"""
Characterization tests for the ``weight_bits_overrides`` selection policy shared
by the GPTQ, Qwen3-VL GPTQ, and Gemma4 GPTQ quantizers.

Every case is a hand-derived expectation; the shared ``resolve_weight_bits``
function and the three quantizer methods that delegate to it are checked
against the same table so that the policy cannot drift between them.
"""

import copy
import inspect
import types
import unittest
import unittest.mock
from typing import Any, Callable, Dict, List, Mapping, NamedTuple, Protocol, Tuple

import torch

from tico.quantization.algorithm.gemma4_gptq.quantizer import Gemma4GPTQQuantizer
from tico.quantization.algorithm.gptq.quantizer import GPTQQuantizer
from tico.quantization.algorithm.gptq.utils import resolve_weight_bits
from tico.quantization.algorithm.qwen3_vl_gptq.quantizer import Qwen3VLGPTQQuantizer
from tico.quantization.config.gemma4_gptq import Gemma4GPTQConfig
from tico.quantization.config.gptq import GPTQConfig
from tico.quantization.config.qwen3_vl_gptq import Qwen3VLGPTQConfig


class _Case(NamedTuple):
    name: str
    default_bits: int
    overrides: Dict[str, int]
    full_module_name: str
    local_module_name: str
    expected: int


# Hand-derived expectations for the documented policy:
#   full exact > local exact > suffix (last match in mapping order) > default.
CASES: List[_Case] = [
    _Case(
        "no_overrides_returns_default",
        4,
        {},
        "model.layers.0.self_attn.q_proj",
        "self_attn.q_proj",
        4,
    ),
    _Case(
        "full_exact_beats_local_and_suffix",
        4,
        {
            "q_proj": 5,
            "self_attn.q_proj": 6,
            "model.layers.0.self_attn.q_proj": 8,
        },
        "model.layers.0.self_attn.q_proj",
        "self_attn.q_proj",
        8,
    ),
    _Case(
        "local_exact_beats_suffix",
        4,
        {"q_proj": 5, "self_attn.q_proj": 6},
        "model.layers.1.self_attn.q_proj",
        "self_attn.q_proj",
        6,
    ),
    _Case(
        "single_suffix_match",
        4,
        {"q_proj": 5},
        "model.layers.2.self_attn.q_proj",
        "self_attn.q_proj",
        5,
    ),
    _Case(
        # Neither key equals the local name, so the suffix branch decides.
        "multiple_suffix_matches_last_wins",
        4,
        {"layers.0.self_attn.q_proj": 8, "q_proj": 4},
        "model.layers.0.self_attn.q_proj",
        "self_attn.q_proj",
        4,
    ),
    _Case(
        "multiple_suffix_matches_reversed_order_flips_winner",
        4,
        {"q_proj": 4, "layers.0.self_attn.q_proj": 8},
        "model.layers.0.self_attn.q_proj",
        "self_attn.q_proj",
        8,
    ),
    _Case(
        # Longer suffix first, shorter suffix last: the shorter one wins
        # because mapping order, not pattern length, decides.
        "longer_suffix_first_shorter_suffix_last",
        4,
        {"layers.3.mlp.down_proj": 6, "down_proj": 3},
        "model.layers.3.mlp.down_proj",
        "mlp.down_proj",
        3,
    ),
    _Case(
        # "proj" is a plain string suffix of "...q_proj" but not a "."-delimited one.
        "plain_string_tail_without_dot_boundary_does_not_match",
        4,
        {"proj": 2},
        "model.layers.0.self_attn.q_proj",
        "self_attn.q_proj",
        4,
    ),
    _Case(
        "no_match_returns_default",
        4,
        {"k_proj": 2, "self_attn.v_proj": 3, "model.layers.9.self_attn.q_proj": 8},
        "model.layers.0.self_attn.q_proj",
        "self_attn.q_proj",
        4,
    ),
    _Case(
        "full_and_local_name_equal_exact_match",
        4,
        {"linear": 8},
        "linear",
        "linear",
        8,
    ),
    _Case(
        "full_and_local_name_equal_no_dot_suffix_match",
        4,
        {"ear": 8},
        "linear",
        "linear",
        4,
    ),
    _Case(
        "short_lm_head_name_exact",
        4,
        {"lm_head": 8},
        "lm_head",
        "lm_head",
        8,
    ),
    _Case(
        "short_lm_head_name_default",
        4,
        {"q_proj": 8},
        "lm_head",
        "lm_head",
        4,
    ),
    _Case(
        "explicit_override_value_is_returned_unchanged",
        4,
        {"self_attn.q_proj": 16},
        "model.layers.0.self_attn.q_proj",
        "self_attn.q_proj",
        16,
    ),
]


Resolver = Callable[[int, Dict[str, int], str, str], int]


def _shared_resolver(default_bits, overrides, full_module_name, local_module_name):
    return resolve_weight_bits(
        default_bits=default_bits,
        weight_bits_overrides=overrides,
        full_module_name=full_module_name,
        local_module_name=local_module_name,
    )


def _gptq_resolver(default_bits, overrides, full_module_name, local_module_name):
    quantizer = GPTQQuantizer(
        GPTQConfig(weight_bits=default_bits, weight_bits_overrides=overrides)
    )
    assert isinstance(quantizer.config, GPTQConfig)
    return quantizer._resolve_weight_bits(
        quantizer.config,
        full_module_name=full_module_name,
        local_module_name=local_module_name,
    )


def _qwen3_vl_resolver(default_bits, overrides, full_module_name, local_module_name):
    quantizer = Qwen3VLGPTQQuantizer(
        Qwen3VLGPTQConfig(weight_bits=default_bits, weight_bits_overrides=overrides)
    )
    assert isinstance(quantizer.config, Qwen3VLGPTQConfig)
    return quantizer._resolve_weight_bits(
        quantizer.config,
        full_module_name=full_module_name,
        local_module_name=local_module_name,
    )


def _gemma4_resolver(default_bits, overrides, full_module_name, local_module_name):
    quantizer = Gemma4GPTQQuantizer(
        Gemma4GPTQConfig(weight_bits=default_bits, weight_bits_overrides=overrides)
    )
    assert isinstance(quantizer.config, Gemma4GPTQConfig)
    return quantizer._resolve_weight_bits(
        quantizer.config,
        full_module_name=full_module_name,
        local_module_name=local_module_name,
    )


class _ResolvesWeightBits(Protocol):
    def _resolve_weight_bits(
        self, gptq_conf: Any, *, full_module_name: str, local_module_name: str
    ) -> int: ...


class _BuildsGPTQObjects(Protocol):
    def _build_gptq_objects(
        self,
        subset: Dict[str, torch.nn.Module],
        module_name: Dict[torch.nn.Module, str],
    ) -> Mapping[str, Any]: ...


RESOLVERS: Dict[str, Resolver] = {
    "resolve_weight_bits": _shared_resolver,
    "GPTQQuantizer": _gptq_resolver,
    "Qwen3VLGPTQQuantizer": _qwen3_vl_resolver,
    "Gemma4GPTQQuantizer": _gemma4_resolver,
}


class TestResolveWeightBitsPolicy(unittest.TestCase):
    """Table-driven policy checks shared by every resolver entry point."""

    def test_policy_table(self):
        for resolver_name, resolver in RESOLVERS.items():
            for case in CASES:
                with self.subTest(resolver=resolver_name, case=case.name):
                    overrides = dict(case.overrides)
                    result = resolver(
                        case.default_bits,
                        overrides,
                        case.full_module_name,
                        case.local_module_name,
                    )
                    self.assertEqual(result, case.expected)
                    self.assertIs(type(result), int)

    def test_shared_function_accepts_read_only_mapping(self):
        overrides = types.MappingProxyType({"q_proj": 4, "self_attn.k_proj": 8})
        self.assertEqual(
            _shared_resolver(6, overrides, "m.layers.0.self_attn.q_proj", "x"), 4
        )
        self.assertEqual(
            _shared_resolver(6, overrides, "m.layers.0.self_attn.k_proj", "x"), 8
        )
        self.assertEqual(_shared_resolver(6, overrides, "m.layers.0.mlp.fc", "x"), 6)

    def test_inputs_are_not_mutated_and_calls_do_not_accumulate_state(self):
        overrides = {"layers.0.self_attn.q_proj": 8, "q_proj": 4}
        snapshot = copy.deepcopy(overrides)
        for resolver_name, resolver in RESOLVERS.items():
            with self.subTest(resolver=resolver_name):
                first = resolver(
                    6, overrides, "model.layers.0.self_attn.q_proj", "self_attn.q_proj"
                )
                second = resolver(
                    6, overrides, "model.layers.0.mlp.up_proj", "mlp.up_proj"
                )
                third = resolver(
                    6, overrides, "model.layers.0.self_attn.q_proj", "self_attn.q_proj"
                )
                self.assertEqual(first, 4)
                self.assertEqual(second, 6)
                self.assertEqual(third, 4)
                self.assertEqual(overrides, snapshot)
                self.assertEqual(list(overrides.keys()), list(snapshot.keys()))


class TestQuantizerMethodContract(unittest.TestCase):
    """The quantizer methods keep their (config, *, full, local) -> int contract."""

    def test_quantizer_methods_delegate_to_shared_resolver(self):
        """The methods must reach the real shared function, not a private copy."""
        overrides = {"q_proj": 4}
        pairs = [
            (GPTQQuantizer, GPTQConfig),
            (Qwen3VLGPTQQuantizer, Qwen3VLGPTQConfig),
            (Gemma4GPTQQuantizer, Gemma4GPTQConfig),
        ]
        for quantizer_cls, config_cls in pairs:
            with self.subTest(quantizer=quantizer_cls.__name__):
                quantizer = quantizer_cls(
                    config_cls(weight_bits=6, weight_bits_overrides=overrides)
                )
                conf = quantizer.config
                assert isinstance(conf, GPTQConfig)
                seen: List[dict] = []

                def _spy(**kwargs):
                    seen.append(kwargs)
                    return resolve_weight_bits(**kwargs)

                module = __import__(
                    quantizer_cls.__module__, fromlist=["resolve_weight_bits"]
                )
                self.assertIs(module.resolve_weight_bits, resolve_weight_bits)
                with unittest.mock.patch.object(module, "resolve_weight_bits", _spy):
                    result = quantizer._resolve_weight_bits(
                        conf,
                        full_module_name="model.layers.0.self_attn.q_proj",
                        local_module_name="self_attn.q_proj",
                    )
                self.assertEqual(result, 4)
                self.assertEqual(
                    seen,
                    [
                        dict(
                            default_bits=6,
                            weight_bits_overrides=overrides,
                            full_module_name="model.layers.0.self_attn.q_proj",
                            local_module_name="self_attn.q_proj",
                        )
                    ],
                )
                self.assertIs(seen[0]["weight_bits_overrides"], overrides)

    def test_repeated_calls_on_one_quantizer_leave_config_untouched(self):
        overrides = {"q_proj": 4, "layers.0.self_attn.q_proj": 8}
        gptq_conf = GPTQConfig(weight_bits=6, weight_bits_overrides=overrides)
        qwen_conf = Qwen3VLGPTQConfig(weight_bits=6, weight_bits_overrides=overrides)
        gemma_conf = Gemma4GPTQConfig(weight_bits=6, weight_bits_overrides=overrides)
        quantizers: List[Tuple[_ResolvesWeightBits, GPTQConfig]] = [
            (GPTQQuantizer(gptq_conf), gptq_conf),
            (Qwen3VLGPTQQuantizer(qwen_conf), qwen_conf),
            (Gemma4GPTQQuantizer(gemma_conf), gemma_conf),
        ]
        for quantizer, conf in quantizers:
            with self.subTest(quantizer=type(quantizer).__name__):
                before = copy.deepcopy(conf.weight_bits_overrides)
                results = [
                    quantizer._resolve_weight_bits(
                        conf,
                        full_module_name="model.layers.0.self_attn.q_proj",
                        local_module_name="self_attn.q_proj",
                    )
                    for _ in range(3)
                ]
                self.assertEqual(results, [8, 8, 8])
                self.assertEqual(conf.weight_bits, 6)
                self.assertEqual(conf.weight_bits_overrides, before)
                self.assertIs(conf.weight_bits_overrides, overrides)
                self.assertEqual(
                    quantizer._resolve_weight_bits(
                        conf,
                        full_module_name="model.layers.1.mlp.gate_proj",
                        local_module_name="mlp.gate_proj",
                    ),
                    6,
                )

    def test_name_arguments_are_keyword_only(self):
        methods: Tuple[Callable[..., int], ...] = (
            GPTQQuantizer._resolve_weight_bits,
            Qwen3VLGPTQQuantizer._resolve_weight_bits,
            Gemma4GPTQQuantizer._resolve_weight_bits,
            resolve_weight_bits,
        )
        for method in methods:
            with self.subTest(method=method.__qualname__):
                signature = inspect.signature(method)
                for name in ("full_module_name", "local_module_name"):
                    self.assertIs(
                        signature.parameters[name].kind,
                        inspect.Parameter.KEYWORD_ONLY,
                    )
                self.assertIs(signature.return_annotation, int)


class TestSelectedBitsReachGPTQConfigure(unittest.TestCase):
    """Module name -> bits -> ``quantizer.configure(bits=...)`` on real GPTQ objects."""

    def test_build_gptq_objects_uses_suffix_last_match(self):
        overrides = {"layers.0.self_attn.q_proj": 8, "q_proj": 4}
        builders: List[Tuple[str, _BuildsGPTQObjects]] = [
            (
                "Qwen3VLGPTQQuantizer",
                Qwen3VLGPTQQuantizer(
                    Qwen3VLGPTQConfig(weight_bits=6, weight_bits_overrides=overrides)
                ),
            ),
            (
                "Gemma4GPTQQuantizer",
                Gemma4GPTQQuantizer(
                    Gemma4GPTQConfig(weight_bits=6, weight_bits_overrides=overrides)
                ),
            ),
        ]
        for name, quantizer in builders:
            with self.subTest(quantizer=name):
                torch.manual_seed(0)
                q_proj = torch.nn.Linear(4, 4)
                k_proj = torch.nn.Linear(4, 4)
                subset: Dict[str, torch.nn.Module] = {
                    "self_attn.q_proj": q_proj,
                    "self_attn.k_proj": k_proj,
                }
                module_name: Dict[torch.nn.Module, str] = {
                    q_proj: "model.layers.0.self_attn.q_proj",
                    k_proj: "model.layers.0.self_attn.k_proj",
                }
                gptq_objs = quantizer._build_gptq_objects(subset, module_name)
                # suffix branch: both keys match, the last mapping entry (4) wins.
                self.assertEqual(
                    gptq_objs["self_attn.q_proj"].quantizer.maxq.item(), 2**4 - 1
                )
                # no match: config default.
                self.assertEqual(
                    gptq_objs["self_attn.k_proj"].quantizer.maxq.item(), 2**6 - 1
                )


if __name__ == "__main__":
    unittest.main()
