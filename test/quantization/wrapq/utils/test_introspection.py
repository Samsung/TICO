# Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

import os
import unittest

import torch
import torch.nn as nn

from tico.quantization import convert, prepare
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.config.smoothquant import SmoothQuantConfig
from tico.quantization.wrapq.utils.introspection import (
    build_fqn_map,
    compare_layer_outputs,
    ModuleInputOutput,
    ModuleName,
    ModuleOutput,
    save_fp_outputs,
    TensorStatistics,
)
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper

IS_INTERNAL_TEST = os.environ.get("RUN_INTERNAL_TESTS", "0") == "1"
MAX_SEQ = 256


def make_fixed_inputs(tokenizer, prompt: str):
    batch = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQ,
    )
    input_ids = batch["input_ids"]  # [1,MAX_SEQ]
    return input_ids


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 4)
        # nn.Sequential gives us numbered sub-modules (0, 1).
        self.block = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
        )


class TestBuildFqnMap(unittest.TestCase):
    def setUp(self):
        # Build the test model once for all test methods.
        self.model = DummyModel()
        self.fqn_map = build_fqn_map(self.model)

    # ---------- basic correctness checks ---------- #
    def test_root_included(self):
        self.assertIn(self.model, self.fqn_map)
        self.assertEqual(self.fqn_map[self.model], "")

    def test_direct_child_name(self):
        self.assertEqual(self.fqn_map[self.model.linear1], "linear1")

    def test_sequential_children(self):
        conv = self.model.block[0]
        relu = self.model.block[1]
        self.assertEqual(self.fqn_map[conv], "block.0")
        self.assertEqual(self.fqn_map[relu], "block.1")

    # ---------- structural sanity tests ---------- #
    def test_total_entries(self):
        expected_count = 5
        self.assertEqual(len(self.fqn_map), expected_count)

    def test_bidirectional_consistency(self):
        inverse = {m: n for n, m in self.model.named_modules()}
        for mod, name in self.fqn_map.items():
            self.assertEqual(name, inverse[mod])


@unittest.skipIf(
    not IS_INTERNAL_TEST, "Internal test — run only if --include-internal is set"
)
class TestSmoothQuantPTQDiff(unittest.TestCase):
    """
    Unit-test: verify that W8A8 SmoothQuant + PTQ does NOT explode layer-wise.

    The test checks per-wrapper activation deltas between
      • CALIB mode (FP32 pass-through)  vs.
      • QUANT mode (fake-/real-quant output)

    For speed it uses "Maykeye/TinyLLama-v0" and a single, short input.
    """

    model_name: str
    device: torch.device
    input_ids: torch.Tensor
    model: torch.nn.Module
    fp_cache: dict[str, torch.Tensor]

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)

        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cls.model_name = "Maykeye/TinyLLama-v0"
        cls.device = torch.device("cpu")

        # tiny model + tokenizer
        tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        fp_model = (
            AutoModelForCausalLM.from_pretrained(cls.model_name, dtype=torch.float32)
            .to(cls.device)
            .eval()
        )
        # Make sure pad token exists (Llama often uses eos as pad)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        fp_model.config.use_cache = False
        fp_model.config.max_position_embeddings = MAX_SEQ
        fqn_map = build_fqn_map(fp_model)

        # SmoothQuant calibration
        sq_model = prepare(fp_model, SmoothQuantConfig(), inplace=True)
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        with torch.inference_mode():
            for i in range(5):
                ids = tokenizer(ds[i]["text"], return_tensors="pt").input_ids.to(
                    cls.device
                )
                sq_model(ids)
        sq_model = convert(sq_model, inplace=True)

        # PTQ-wrap first layer
        qcfg = PTQConfig()
        new_layers = torch.nn.ModuleList()
        for idx, fp_layer in enumerate(sq_model.model.layers):
            if idx >= 1:
                new_layers.append(fp_layer)
                continue
            new_layers.append(
                PTQWrapper(
                    fp_layer,
                    qcfg=qcfg.child(f"layer{idx}"),
                    fp_name=fqn_map.get(fp_layer),
                )
            )
        sq_model.model.layers = new_layers
        cls.model = sq_model

        # prepare static input & capture FP refs
        cls.input_ids = make_fixed_inputs(tokenizer, "Unit-test input sequence.")

        sq_model.model.layers.apply(
            lambda m: getattr(m, "enable_calibration", lambda: None)()
        )
        h_save, cls.fp_cache = save_fp_outputs(cls.model)
        with torch.no_grad():
            cls.model(cls.input_ids)
        for h in h_save:
            h.remove()

        # switch to QUANT mode
        sq_model.model.layers.apply(
            lambda m: getattr(m, "freeze_qparams", lambda: None)()
        )

    # ------------------------------------------------------------------ #
    # 1. Original diff-only assertion (updated for nested dict)          #
    # ------------------------------------------------------------------ #
    def test_layerwise_diff(self):
        h_cmp, stats = compare_layer_outputs(
            self.model,
            self.fp_cache,
            metrics=["diff"],
            rtol=0.0,
            atol=1.0,
            collect=True,
        )
        with torch.no_grad():
            self.model(self.input_ids)
        for h in h_cmp:
            h.remove()

        for name, metric_dict in stats.items():
            self.assertLessEqual(
                metric_dict["diff"],
                3.0,
                msg=f"{name}: diff={metric_dict['diff']:.3e} > 3.0",
            )

    # ------------------------------------------------------------------ #
    # 2. PEIR metric exists & is finite                                  #
    # ------------------------------------------------------------------ #
    def test_layerwise_peir(self):
        _, stats = compare_layer_outputs(
            self.model, self.fp_cache, metrics=["peir"], collect=True
        )
        for name, metric_dict in stats.items():
            val = metric_dict["peir"]
            self.assertTrue(
                torch.isfinite(torch.tensor(val)),
                msg=f"{name}: non-finite PEIR",
            )

    # ------------------------------------------------------------------ #
    # 3. Subset selection ('diff', 'peir')                               #
    # ------------------------------------------------------------------ #
    def test_metric_subset_selection(self):
        _, stats = compare_layer_outputs(
            self.model,
            self.fp_cache,
            metrics=["diff", "peir"],
            collect=True,
        )
        for metric_dict in stats.values():
            self.assertEqual(set(metric_dict), {"diff", "peir"})

    # ------------------------------------------------------------------ #
    # 4. Custom metric (mean-abs-error)                                  #
    # ------------------------------------------------------------------ #
    def test_custom_metric(self):
        def mae(a: torch.Tensor, b: torch.Tensor) -> float:
            return (a - b).abs().mean().item()

        _, stats = compare_layer_outputs(
            self.model,
            self.fp_cache,
            metrics=["mae"],
            custom_metrics={"mae": mae},
            collect=True,
        )
        for metric_dict in stats.values():
            self.assertIn("mae", metric_dict)


class TestTensorStatistics(unittest.TestCase):
    """Unit tests for TensorStatistics class."""

    def test_tensor_statistics_creation(self):
        """Test that TensorStatistics can be created with correct values."""
        from tico.quantization.wrapq.utils.introspection import (
            DifferenceStatistics,
            get_tensor_statistics,
            TensorStatistics,
        )

        stats = TensorStatistics(mean=1.5, min=-2.0, max=3.0, stddev=0.5)

        self.assertEqual(stats.mean, 1.5)
        self.assertEqual(stats.min, -2.0)
        self.assertEqual(stats.max, 3.0)
        self.assertEqual(stats.stddev, 0.5)

    def test_tensor_statistics_asdict(self):
        """Test that _asdict method returns correct dictionary."""
        from tico.quantization.wrapq.utils.introspection import TensorStatistics

        stats = TensorStatistics(mean=1.5, min=-2.0, max=3.0, stddev=0.5)
        result = stats._asdict()

        expected = {"mean": 1.5, "min": -2.0, "max": 3.0, "stddev": 0.5}

        self.assertEqual(result, expected)

    def test_tensor_statistics_frozen(self):
        """Test that TensorStatistics is frozen and cannot be modified."""
        from tico.quantization.wrapq.utils.introspection import TensorStatistics

        stats = TensorStatistics(mean=1.5, min=-2.0, max=3.0, stddev=0.5)

        with self.assertRaises(Exception):
            stats.mean = 2.0  # type: ignore[misc]

    def test_get_tensor_statistics(self):
        """Test get_tensor_statistics function with various tensors."""
        from tico.quantization.wrapq.utils.introspection import get_tensor_statistics

        # Test with simple tensor
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = get_tensor_statistics(tensor)

        self.assertIsInstance(stats, TensorStatistics)
        self.assertAlmostEqual(stats.mean, 3.0, places=5)
        self.assertAlmostEqual(stats.min, 1.0, places=5)
        self.assertAlmostEqual(stats.max, 5.0, places=5)
        self.assertAlmostEqual(stats.stddev, 1.414213538, places=5)

        # Test with 2D tensor
        tensor_2d = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        stats_2d = get_tensor_statistics(tensor_2d)

        self.assertIsInstance(stats_2d, TensorStatistics)
        self.assertAlmostEqual(stats_2d.mean, 2.5, places=5)
        self.assertAlmostEqual(stats_2d.min, 1.0, places=5)
        self.assertAlmostEqual(stats_2d.max, 4.0, places=5)

        # Test with single element tensor
        tensor_single = torch.tensor([42.0])
        stats_single = get_tensor_statistics(tensor_single)

        self.assertIsInstance(stats_single, TensorStatistics)
        self.assertAlmostEqual(stats_single.mean, 42.0, places=5)
        self.assertAlmostEqual(stats_single.min, 42.0, places=5)
        self.assertAlmostEqual(stats_single.max, 42.0, places=5)

    def test_difference_statistics_creation(self):
        """Test that DifferenceStatistics can be created with correct values."""
        from tico.quantization.wrapq.utils.introspection import DifferenceStatistics

        diff_stats = DifferenceStatistics(
            mean=0.1, min=0.0, max=0.5, stddev=0.05, peir=0.02
        )

        self.assertEqual(diff_stats.mean, 0.1)
        self.assertEqual(diff_stats.min, 0.0)
        self.assertEqual(diff_stats.max, 0.5)
        self.assertEqual(diff_stats.stddev, 0.05)
        self.assertEqual(diff_stats.peir, 0.02)

    def test_difference_statistics_inheritance(self):
        """Test that DifferenceStatistics inherits from TensorStatistics."""
        from tico.quantization.wrapq.utils.introspection import (
            DifferenceStatistics,
            TensorStatistics,
        )

        diff_stats = DifferenceStatistics(
            mean=0.1, min=0.0, max=0.5, stddev=0.05, peir=0.02
        )

        # Should have all TensorStatistics attributes
        self.assertTrue(hasattr(diff_stats, "mean"))
        self.assertTrue(hasattr(diff_stats, "min"))
        self.assertTrue(hasattr(diff_stats, "max"))
        self.assertTrue(hasattr(diff_stats, "stddev"))
        # Plus the additional peir attribute
        self.assertTrue(hasattr(diff_stats, "peir"))
        # Should be an instance of TensorStatistics
        self.assertIsInstance(diff_stats, TensorStatistics)

    def test_difference_statistics_asdict(self):
        """Test that DifferenceStatistics _asdict method includes peir."""
        from tico.quantization.wrapq.utils.introspection import DifferenceStatistics

        diff_stats = DifferenceStatistics(
            mean=0.1, min=0.0, max=0.5, stddev=0.05, peir=0.02
        )
        result = diff_stats._asdict()

        expected = {"mean": 0.1, "min": 0.0, "max": 0.5, "stddev": 0.05, "peir": 0.02}

        self.assertEqual(result, expected)


class TestDetachTensors(unittest.TestCase):
    """Unit tests for detach_tensors function."""

    def test_detach_plain_tensor(self):
        """Test that a plain tensor is detached and cloned."""
        from tico.quantization.wrapq.utils.introspection import detach_tensors

        t = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        result = detach_tensors(t)

        # Should be a tensor with the same values
        self.assertTrue(torch.equal(result, t))
        # Should not require grad (detached)
        self.assertFalse(result.requires_grad)
        # Should be a different object (cloned)
        self.assertIsNot(result, t)
        # Modifying the clone should not affect the original
        result[0] = 99.0
        self.assertAlmostEqual(t[0].item(), 1.0)

    def test_detach_tensor_no_grad(self):
        """Test detach_tensors on a tensor that already doesn't require grad."""
        from tico.quantization.wrapq.utils.introspection import detach_tensors

        t = torch.tensor([4.0, 5.0, 6.0])
        result = detach_tensors(t)

        self.assertTrue(torch.equal(result, t))
        self.assertFalse(result.requires_grad)
        self.assertIsNot(result, t)

    def test_detach_dict(self):
        """Test that tensors inside a dict are recursively detached."""
        from tico.quantization.wrapq.utils.introspection import detach_tensors

        t1 = torch.tensor([1.0, 2.0], requires_grad=True)
        t2 = torch.tensor([3.0, 4.0], requires_grad=True)
        d = {"a": t1, "b": t2, "c": "not_a_tensor"}

        result = detach_tensors(d)

        assert isinstance(result, dict)  # for mypy type narrowing
        self.assertEqual(set(result.keys()), {"a", "b", "c"})
        # Tensors should be detached and cloned
        self.assertTrue(torch.equal(result["a"], t1))
        self.assertFalse(result["a"].requires_grad)
        self.assertTrue(torch.equal(result["b"], t2))
        self.assertFalse(result["b"].requires_grad)
        # Non-tensor values should pass through unchanged
        self.assertEqual(result["c"], "not_a_tensor")

    def test_detach_list(self):
        """Test that tensors inside a list are recursively detached."""
        from tico.quantization.wrapq.utils.introspection import detach_tensors

        t1 = torch.tensor([1.0], requires_grad=True)
        t2 = torch.tensor([2.0], requires_grad=True)
        lst = [t1, t2, 42]

        result = detach_tensors(lst)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)
        self.assertTrue(torch.equal(result[0], t1))
        self.assertFalse(result[0].requires_grad)
        self.assertTrue(torch.equal(result[1], t2))
        self.assertFalse(result[1].requires_grad)
        self.assertEqual(result[2], 42)

    def test_detach_tuple(self):
        """Test that tensors inside a tuple are recursively detached."""
        from tico.quantization.wrapq.utils.introspection import detach_tensors

        t1 = torch.tensor([1.0], requires_grad=True)
        t2 = torch.tensor([2.0], requires_grad=True)
        tup = (t1, t2)

        result = detach_tensors(tup)

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertTrue(torch.equal(result[0], t1))
        self.assertFalse(result[0].requires_grad)
        self.assertTrue(torch.equal(result[1], t2))
        self.assertFalse(result[1].requires_grad)

    def test_detach_nested_structure(self):
        """Test that tensors inside nested dicts/lists are recursively detached."""
        from tico.quantization.wrapq.utils.introspection import detach_tensors

        t = torch.tensor([1.0, 2.0], requires_grad=True)
        nested = {"outer": {"inner": [t, 42]}}

        result = detach_tensors(nested)

        assert isinstance(result, dict)  # for mypy type narrowing
        self.assertIsInstance(result["outer"], dict)
        self.assertIsInstance(result["outer"]["inner"], list)
        self.assertTrue(torch.equal(result["outer"]["inner"][0], t))
        self.assertFalse(result["outer"]["inner"][0].requires_grad)
        self.assertEqual(result["outer"]["inner"][1], 42)

    def test_detach_non_tensor_scalar(self):
        """Test that non-tensor, non-iterable values pass through unchanged."""
        from tico.quantization.wrapq.utils.introspection import detach_tensors

        self.assertEqual(detach_tensors(42), 42)
        self.assertEqual(detach_tensors(3.14), 3.14)
        self.assertEqual(detach_tensors("hello"), "hello")
        self.assertIsNone(detach_tensors(None))

    def test_detach_model_output(self):
        """Test that tensors inside a ModelOutput are recursively detached."""
        from tico.quantization.wrapq.utils.introspection import detach_tensors
        from transformers.modeling_outputs import BaseModelOutput

        t = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        output = BaseModelOutput(last_hidden_state=t)

        result = detach_tensors(output)

        self.assertIsInstance(result, BaseModelOutput)
        self.assertTrue(torch.equal(result.last_hidden_state, t))
        self.assertFalse(result.last_hidden_state.requires_grad)


class TestModelOutputToSerializable(unittest.TestCase):
    """Unit tests for model_output_to_serializable function."""

    def test_tensor_with_type(self):
        """Test serialization of a tensor with type information included."""
        from tico.quantization.wrapq.utils.introspection import (
            model_output_to_serializable,
        )

        t = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = model_output_to_serializable(t, include_type=True)

        assert isinstance(result, dict)  # for mypy type narrowing
        self.assertEqual(result["type"], "Tensor")
        self.assertEqual(result["dtype"], str(t.dtype))
        self.assertEqual(result["shape"], str(t.shape))
        # numel > 1, so statistics should be present
        self.assertIn("statistics", result)
        self.assertIsInstance(result["statistics"], dict)
        assert isinstance(result["statistics"], dict)  # for mypy type narrowing
        self.assertIn("mean", result["statistics"])
        self.assertIn("min", result["statistics"])
        self.assertIn("max", result["statistics"])
        self.assertIn("stddev", result["statistics"])
        # numel > 1 and include_tensor_content=False, so no value key
        self.assertNotIn("value", result)

    def test_tensor_without_type(self):
        """Test serialization of a tensor with type information excluded."""
        from tico.quantization.wrapq.utils.introspection import (
            model_output_to_serializable,
        )

        t = torch.tensor([1.0, 2.0, 3.0])
        result = model_output_to_serializable(t, include_type=False)

        assert isinstance(result, dict)  # for mypy type narrowing
        self.assertNotIn("type", result)
        self.assertIn("dtype", result)
        self.assertIn("shape", result)

    def test_tensor_with_content(self):
        """Test serialization of a tensor with content included."""
        from tico.quantization.wrapq.utils.introspection import (
            model_output_to_serializable,
        )

        t = torch.tensor([1.0, 2.0, 3.0])
        result = model_output_to_serializable(t, include_tensor_content=True)

        assert isinstance(result, dict)
        self.assertIn("value", result)
        self.assertEqual(result["value"], [1.0, 2.0, 3.0])

    def test_single_element_tensor(self):
        """Test serialization of a single-element tensor (value always included)."""
        from tico.quantization.wrapq.utils.introspection import (
            model_output_to_serializable,
        )

        t = torch.tensor([42.0])
        result = model_output_to_serializable(t)

        # Single element: value is included even without include_tensor_content
        assert isinstance(result, dict)
        self.assertIn("value", result)
        self.assertEqual(result["value"], [42.0])
        # Single element: no statistics
        self.assertNotIn("statistics", result)

    def test_scalar_tensor(self):
        """Test serialization of a scalar tensor (0-dim)."""
        from tico.quantization.wrapq.utils.introspection import (
            model_output_to_serializable,
        )

        t = torch.tensor(3.14)
        result = model_output_to_serializable(t)

        # Scalar: numel == 1, value is included
        assert isinstance(result, dict)
        self.assertIn("value", result)
        assert isinstance(result["value"], float)  # for mypy type narrowing
        self.assertAlmostEqual(result["value"], 3.14, places=4)
        self.assertNotIn("statistics", result)

    def test_dict_serialization(self):
        """Test serialization of a dict containing tensors."""
        from tico.quantization.wrapq.utils.introspection import (
            model_output_to_serializable,
        )

        t = torch.tensor([1.0, 2.0])
        d = {"key1": t, "key2": "hello"}
        result = model_output_to_serializable(d)

        assert isinstance(result, dict)  # for mypy type narrowing
        self.assertEqual(result["type"], "dict")
        self.assertIn("key1", result)
        self.assertIn("key2", result)
        # key1 should be a serialized tensor
        self.assertEqual(result["key1"]["type"], "Tensor")
        # key2 should be a serialized string
        self.assertEqual(result["key2"], "hello")

    def test_list_serialization(self):
        """Test serialization of a list containing tensors."""
        from tico.quantization.wrapq.utils.introspection import (
            model_output_to_serializable,
        )

        t = torch.tensor([1.0, 2.0])
        lst = [t, 42]
        result = model_output_to_serializable(lst)

        assert isinstance(result, dict)  # for mypy type narrowing
        self.assertEqual(result["type"], "list")
        self.assertIn("0", result)
        self.assertIn("1", result)
        # First element should be a serialized tensor
        self.assertEqual(result["0"]["type"], "Tensor")
        # Second element (int) should be converted to string
        self.assertEqual(result["1"], 42)

    def test_tuple_serialization(self):
        """Test serialization of a tuple containing tensors."""
        from tico.quantization.wrapq.utils.introspection import (
            model_output_to_serializable,
        )

        t = torch.tensor([1.0])
        tup = (t,)
        result = model_output_to_serializable(tup)

        assert isinstance(result, dict)  # for mypy type narrowing
        self.assertEqual(result["type"], "tuple")
        self.assertIn("0", result)

    def test_model_output_serialization(self):
        """Test serialization of a ModelOutput object."""
        from tico.quantization.wrapq.utils.introspection import (
            model_output_to_serializable,
        )
        from transformers.modeling_outputs import BaseModelOutput

        t = torch.tensor([1.0, 2.0, 3.0])
        output = BaseModelOutput(last_hidden_state=t)
        result = model_output_to_serializable(output)

        assert isinstance(result, dict)  # for mypy type narrowing
        self.assertEqual(result["type"], "BaseModelOutput")
        self.assertIn("last_hidden_state", result)
        self.assertEqual(result["last_hidden_state"]["type"], "Tensor")

    def test_namedtuple_serialization(self):
        """Test serialization of a NamedTuple."""
        from typing import NamedTuple

        from tico.quantization.wrapq.utils.introspection import (
            model_output_to_serializable,
        )

        class Point(NamedTuple):
            x: int
            y: int

        p = Point(x=1, y=2)
        result = model_output_to_serializable(p)

        assert isinstance(result, dict)  # for mypy type narrowing
        self.assertEqual(result["type"], "Point")
        # NamedTuple uses _asdict, so fields are inlined
        self.assertIn("x", result)
        self.assertIn("y", result)

    def test_non_tensor_primitive(self):
        """Test serialization of non-tensor primitive types returns string."""
        from tico.quantization.wrapq.utils.introspection import (
            model_output_to_serializable,
        )

        # Integer
        result_int = model_output_to_serializable(42)
        self.assertEqual(result_int, 42)

        # Float
        result_float = model_output_to_serializable(3.14)
        self.assertEqual(result_float, 3.14)

        # String
        result_str = model_output_to_serializable("hello")
        self.assertEqual(result_str, "hello")

    def test_none_serialization(self):
        """Test serialization of None returns string representation."""
        from tico.quantization.wrapq.utils.introspection import (
            model_output_to_serializable,
        )

        result = model_output_to_serializable(None)
        self.assertEqual(result, "None")

    def test_nested_dict_with_tensors(self):
        """Test serialization of a nested dict with tensors."""
        from tico.quantization.wrapq.utils.introspection import (
            model_output_to_serializable,
        )

        t1 = torch.tensor([1.0, 2.0])
        t2 = torch.tensor([3.0])
        nested = {"outer": {"inner": t1}, "single": t2}
        result = model_output_to_serializable(nested)

        assert isinstance(result, dict)  # for mypy type narrowing
        self.assertEqual(result["type"], "dict")
        self.assertIn("outer", result)
        self.assertIn("single", result)
        self.assertEqual(result["outer"]["type"], "dict")  # type: ignore[index]
        self.assertIn("inner", result["outer"])
        self.assertEqual(result["outer"]["inner"]["type"], "Tensor")  # type: ignore[index]


class TestModuleInputOutput(unittest.TestCase):
    """Unit tests for ModuleInputOutput NamedTuple."""

    def setUp(self):
        from tico.quantization.wrapq.utils.introspection import ModuleInputOutput

        self.module = nn.Linear(4, 4)
        self.inputs = (torch.tensor([1.0, 2.0, 3.0, 4.0]),)
        self.kwargs = {"mask": None}
        self.output = torch.tensor([5.0, 6.0, 7.0, 8.0])
        self.quantization: list = []
        self.mio = ModuleInputOutput(
            module=self.module,
            module_name="linear1",
            inputs=self.inputs,
            kwargs=self.kwargs,
            output=self.output,
            quantization=self.quantization,
        )

    def test_field_access(self):
        """Test that all fields of ModuleInputOutput are accessible."""
        self.assertIs(self.mio.module, self.module)
        self.assertEqual(self.mio.module_name, "linear1")
        self.assertEqual(self.mio.inputs, self.inputs)
        self.assertEqual(self.mio.kwargs, self.kwargs)
        self.assertIs(self.mio.output, self.output)
        self.assertIs(self.mio.quantization, self.quantization)

    def test_is_named_tuple(self):
        """Test that ModuleInputOutput is a NamedTuple."""
        from tico.quantization.wrapq.utils.introspection import ModuleInputOutput

        self.assertTrue(hasattr(ModuleInputOutput, "_fields"))
        self.assertEqual(
            ModuleInputOutput._fields,
            ("module", "module_name", "inputs", "kwargs", "output", "quantization"),
        )
        self.assertIsInstance(self.mio, tuple)

    def test_as_serializable_keys(self):
        """Test that as_serializable returns dict with expected top-level keys."""
        result = self.mio.as_serializable()

        assert isinstance(result, dict)  # for mypy type narrowing
        self.assertIn("module_name", result)
        self.assertIn("module_type", result)
        self.assertIn("inputs", result)
        self.assertIn("kwargs", result)
        self.assertIn("output", result)
        self.assertIn("quantization", result)
        self.assertEqual(result["module_name"], "linear1")
        self.assertEqual(result["module_type"], "Linear")

    def test_as_serializable_with_type(self):
        """Test as_serializable with include_type=True (default)."""
        result = self.mio.as_serializable(include_type=True)

        # Input tuple should have type info
        self.assertIsInstance(result["inputs"], dict)
        self.assertIn("type", result["inputs"])
        self.assertEqual(result["inputs"]["type"], "tuple")

    def test_as_serializable_without_type(self):
        """Test as_serializable with include_type=False."""
        result = self.mio.as_serializable(include_type=False)

        # Top-level should not have type, but nested serialization
        # also respects include_type
        self.assertNotIn("type", result)

    def test_as_serializable_with_tensor_content(self):
        """Test as_serializable with include_tensor_content=True."""
        result = self.mio.as_serializable(include_tensor_content=True)

        # The output tensor should include its value when include_tensor_content is True
        self.assertIn("output", result)
        output_data = result["output"]
        self.assertIn("value", output_data)

    def test_as_serializable_empty_kwargs(self):
        """Test as_serializable with empty kwargs."""
        from tico.quantization.wrapq.utils.introspection import ModuleInputOutput

        mio = ModuleInputOutput(
            module=self.module,
            module_name="linear1",
            inputs=self.inputs,
            kwargs={},
            output=self.output,
        )
        result = mio.as_serializable()

        self.assertIsInstance(result["kwargs"], dict)
        self.assertEqual(len(result["kwargs"]), 0)

    def test_as_serializable_tensor_output(self):
        """Test as_serializable correctly serializes a tensor output."""
        result = self.mio.as_serializable()

        output_data = result["output"]
        self.assertIsInstance(output_data, dict)
        self.assertIn("dtype", output_data)
        self.assertIn("shape", output_data)
        # numel > 1 so statistics should be present
        self.assertIn("statistics", output_data)


class TestModuleHook(unittest.TestCase):
    """Unit tests for module_hook context manager."""

    def test_hook_called_during_context(self):
        """Test that the hook is called for module forward passes within the context."""
        from tico.quantization.wrapq.utils.introspection import module_hook

        called = []

        def hook(module, inputs, kwargs, output):
            called.append(type(module).__name__)

        model = nn.Sequential(nn.Linear(2, 2), nn.ReLU())
        x = torch.randn(1, 2)

        with module_hook(hook):
            with torch.no_grad():
                model(x)

        # Hook should have been called for each module in the model
        self.assertGreater(len(called), 0)
        self.assertIn("Linear", called)
        self.assertIn("ReLU", called)

    def test_hook_not_called_after_context(self):
        """Test that the hook is removed after exiting the context."""
        from tico.quantization.wrapq.utils.introspection import module_hook

        called = []

        def hook(module, inputs, kwargs, output):
            called.append(type(module).__name__)

        model = nn.Sequential(nn.Linear(2, 2), nn.ReLU())
        x = torch.randn(1, 2)

        with module_hook(hook):
            with torch.no_grad():
                model(x)

        count_during = len(called)

        # Run again after the context is exited
        with torch.no_grad():
            model(x)

        # No additional calls should have been made
        self.assertEqual(len(called), count_during)

    def test_hook_receives_kwargs(self):
        """Test that the hook receives keyword arguments (with_kwargs=True)."""
        from tico.quantization.wrapq.utils.introspection import module_hook

        received_kwargs = []

        def hook(module, inputs, kwargs, output):
            received_kwargs.append(kwargs)

        model = nn.Linear(2, 2)
        x = torch.randn(1, 2)

        with module_hook(hook):
            with torch.no_grad():
                model(x)

        # Hook should have been called and kwargs should be a dict
        self.assertEqual(len(received_kwargs), 1)
        self.assertIsInstance(received_kwargs[0], dict)

    def test_hook_receives_inputs_and_output(self):
        """Test that the hook receives inputs and output correctly."""
        from tico.quantization.wrapq.utils.introspection import module_hook

        captured = []

        def hook(module, inputs, kwargs, output):
            captured.append({"inputs": inputs, "output": output})

        model = nn.Linear(2, 2)
        x = torch.randn(1, 2)

        with module_hook(hook):
            with torch.no_grad():
                model(x)

        self.assertEqual(len(captured), 1)
        # Inputs should be a tuple of tensors
        self.assertIsInstance(captured[0]["inputs"], tuple)
        self.assertEqual(len(captured[0]["inputs"]), 1)
        self.assertIsInstance(captured[0]["inputs"][0], torch.Tensor)
        # Output should be a tensor
        self.assertIsInstance(captured[0]["output"], torch.Tensor)

    def test_hook_called_on_nested_modules(self):
        """Test that the hook is called on all nested modules."""
        from tico.quantization.wrapq.utils.introspection import module_hook

        called = []

        def hook(module, inputs, kwargs, output):
            called.append(type(module).__name__)

        class InnerModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2)

            def forward(self, x):
                return self.linear(x)

        class OuterModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = InnerModule()
                self.relu = nn.ReLU()

            def forward(self, x):
                return self.relu(self.inner(x))

        model = OuterModule()
        x = torch.randn(1, 2)

        with module_hook(hook):
            with torch.no_grad():
                model(x)

        # Should capture all modules: OuterModule, InnerModule, Linear, ReLU
        self.assertIn("OuterModule", called)
        self.assertIn("InnerModule", called)
        self.assertIn("Linear", called)
        self.assertIn("ReLU", called)

    def test_multiple_contexts_stacked(self):
        """Test that multiple module_hook contexts can be stacked."""
        from tico.quantization.wrapq.utils.introspection import module_hook

        calls_a = []
        calls_b = []

        def hook_a(module, inputs, kwargs, output):
            calls_a.append(type(module).__name__)

        def hook_b(module, inputs, kwargs, output):
            calls_b.append(type(module).__name__)

        model = nn.Linear(2, 2)
        x = torch.randn(1, 2)

        with module_hook(hook_a):
            with module_hook(hook_b):
                with torch.no_grad():
                    model(x)

        # Both hooks should have been called
        self.assertGreater(len(calls_a), 0)
        self.assertGreater(len(calls_b), 0)

        # After both contexts exit, neither hook should be active
        count_a = len(calls_a)
        count_b = len(calls_b)
        with torch.no_grad():
            model(x)
        self.assertEqual(len(calls_a), count_a)
        self.assertEqual(len(calls_b), count_b)


class TestFullTensorPrinting(unittest.TestCase):
    """Unit tests for full_tensor_printing context manager."""

    def test_large_tensor_printing(self):
        """Test that a large tensor is fully printed within the context."""
        from tico.quantization.wrapq.utils.introspection import full_tensor_printing

        # Create a tensor larger than default threshold
        large_tensor = torch.arange(2000)

        # Outside context, printing would truncate
        default_str = str(large_tensor)
        self.assertIn("...", default_str)  # Default truncates

        # Inside context, printing should show all elements
        with full_tensor_printing():
            full_str = str(large_tensor)
            self.assertNotIn("...", full_str)
            # Should contain the first and last elements
            self.assertIn("0", full_str)
            self.assertIn("1999", full_str)


class TestCreateTracingHook(unittest.TestCase):
    """Unit tests for create_tracing_hook function."""

    def setUp(self):
        from tico.quantization.wrapq.utils.introspection import ModuleInputOutput

        self.module = nn.Linear(2, 2)
        self.mio = ModuleInputOutput(
            module=self.module,
            module_name="linear1",
            inputs=(torch.randn(1, 2),),
            kwargs={},
            output=torch.randn(1, 2),
        )

    def test_stores_output_in_module_outputs(self):
        """Test that the hook stores module output in the provided dict."""
        from tico.quantization.wrapq.utils.introspection import create_tracing_hook

        module_outputs: dict[ModuleName, ModuleOutput] = {}
        hook = create_tracing_hook(
            print_input_output=False,
            module_outputs=module_outputs,
        )

        hook(self.mio)

        self.assertIn("linear1", module_outputs)
        self.assertTrue(torch.equal(module_outputs["linear1"], self.mio.output))

    def test_no_store_when_module_outputs_is_none(self):
        """Test that the hook does not crash when module_outputs is None."""
        from tico.quantization.wrapq.utils.introspection import create_tracing_hook

        hook = create_tracing_hook(
            print_input_output=False,
            module_outputs=None,
        )

        # Should not raise
        hook(self.mio)

    def test_print_input_output(self):
        """Test that the hook prints when print_input_output=True."""
        import sys
        from io import StringIO

        from tico.quantization.wrapq.utils.introspection import create_tracing_hook

        captured = StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            hook = create_tracing_hook(
                print_input_output=True,
                module_outputs=None,
            )
            hook(self.mio)
        finally:
            sys.stdout = old_stdout

        output = captured.getvalue()
        self.assertIn("linear1", output)
        self.assertIn("Linear", output)

    def test_no_print_when_print_input_output_false(self):
        """Test that the hook does not print when print_input_output=False."""
        import sys
        from io import StringIO

        from tico.quantization.wrapq.utils.introspection import create_tracing_hook

        captured = StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            hook = create_tracing_hook(
                print_input_output=False,
                module_outputs=None,
            )
            hook(self.mio)
        finally:
            sys.stdout = old_stdout

        output = captured.getvalue()
        self.assertEqual(output, "")

    def test_interesting_modules_include_tensor_content(self):
        """Test that interesting modules get tensor content in the serialized output."""
        import json
        import sys
        from io import StringIO

        from tico.quantization.wrapq.utils.introspection import (
            create_tracing_hook,
            ModuleInputOutput,
        )

        mio_interesting = ModuleInputOutput(
            module=self.module,
            module_name="target_layer",
            inputs=(torch.tensor([1.0, 2.0]),),
            kwargs={},
            output=torch.tensor([3.0, 4.0]),
        )

        captured = StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            hook = create_tracing_hook(
                print_input_output=True,
                module_outputs=None,
                interesting_modules=["target_layer"],
            )
            hook(mio_interesting)
        finally:
            sys.stdout = old_stdout

        output = captured.getvalue()
        # The serialized output for interesting modules should include tensor values
        self.assertIn("target_layer", output)
        # Since include_tensor_content=True for interesting modules, "value" should appear
        self.assertIn("value", output)

    def test_non_interesting_modules_exclude_tensor_content(self):
        """Test that non-interesting modules do not get tensor content by default."""
        import sys
        from io import StringIO

        from tico.quantization.wrapq.utils.introspection import create_tracing_hook

        mio_boring = ModuleInputOutput(
            module=self.module,
            module_name="boring_layer",
            inputs=(torch.tensor([1.0, 2.0]),),
            kwargs={},
            output=torch.tensor([3.0, 4.0, 5.0, 6.0]),
        )

        captured = StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            hook = create_tracing_hook(
                print_input_output=True,
                module_outputs=None,
                interesting_modules=["target_layer"],
            )
            hook(mio_boring)
        finally:
            sys.stdout = old_stdout

        output = captured.getvalue()
        # "boring_layer" should be printed but without "value" key
        # (since numel > 1 and include_tensor_content=False for non-interesting)
        self.assertIn("boring_layer", output)

    def test_multiple_modules_stored(self):
        """Test that the hook stores outputs for multiple modules."""
        from tico.quantization.wrapq.utils.introspection import (
            create_tracing_hook,
            ModuleInputOutput,
        )

        module_outputs: dict[ModuleName, ModuleOutput] = {}
        hook = create_tracing_hook(
            print_input_output=False,
            module_outputs=module_outputs,
        )

        mio1 = ModuleInputOutput(
            module=nn.Linear(2, 2),
            module_name="layer1",
            inputs=(torch.randn(1, 2),),
            kwargs={},
            output=torch.tensor([1.0]),
        )
        mio2 = ModuleInputOutput(
            module=nn.ReLU(),
            module_name="layer2",
            inputs=(torch.randn(1, 2),),
            kwargs={},
            output=torch.tensor([2.0]),
        )

        hook(mio1)
        hook(mio2)

        self.assertIn("layer1", module_outputs)
        self.assertIn("layer2", module_outputs)
        self.assertTrue(torch.equal(module_outputs["layer1"], torch.tensor([1.0])))
        self.assertTrue(torch.equal(module_outputs["layer2"], torch.tensor([2.0])))


class TestTraceModelInputOutput(unittest.TestCase):
    """Unit tests for trace_model_input_output function."""

    def setUp(self):
        """Set up a simple model for testing."""
        self.model = nn.Sequential(
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
        )
        self.model.eval()
        self.model_inputs = {"input": torch.randn(2, 4)}

    def test_hook_called_for_each_module(self):
        """Test that the hook is called for each module in the model."""
        from tico.quantization.wrapq.utils.introspection import (
            ModuleInputOutput,
            trace_model_input_output,
        )

        captured: list[ModuleInputOutput] = []

        def hook(data: ModuleInputOutput):
            captured.append(data)

        trace_model_input_output(self.model, self.model_inputs, hook=hook)

        # Should capture: Linear(4,4), ReLU, Linear(4,2), and the Sequential itself
        captured_names = [d.module_name for d in captured]
        self.assertIn("", captured_names)  # root Sequential
        self.assertIn("0", captured_names)  # Linear(4,4)
        self.assertIn("1", captured_names)  # ReLU
        self.assertIn("2", captured_names)  # Linear(4,2)

    def test_module_output_is_detached(self):
        """Test that captured outputs are detached (no grad tracking)."""
        from tico.quantization.wrapq.utils.introspection import (
            ModuleInputOutput,
            trace_model_input_output,
        )

        captured: list[ModuleInputOutput] = []

        def hook(data: ModuleInputOutput):
            captured.append(data)

        trace_model_input_output(self.model, self.model_inputs, hook=hook)

        for data in captured:
            if isinstance(data.output, torch.Tensor):
                self.assertFalse(data.output.requires_grad)

    def test_module_input_output_fields(self):
        """Test that ModuleInputOutput has correct fields."""
        from tico.quantization.wrapq.utils.introspection import (
            ModuleInputOutput,
            trace_model_input_output,
        )

        captured: list[ModuleInputOutput] = []

        def hook(data: ModuleInputOutput):
            captured.append(data)

        trace_model_input_output(self.model, self.model_inputs, hook=hook)

        # Find the first Linear module's data
        linear_data = [d for d in captured if isinstance(d.module, nn.Linear)][0]
        self.assertIsInstance(linear_data.module, nn.Linear)
        self.assertIsInstance(linear_data.module_name, str)
        self.assertIsInstance(linear_data.inputs, tuple)
        self.assertIsInstance(linear_data.kwargs, dict)
        self.assertIsInstance(linear_data.output, torch.Tensor)

    def test_hook_receives_correct_module_names(self):
        """Test that module names match the expected FQN pattern."""
        from tico.quantization.wrapq.utils.introspection import (
            ModuleInputOutput,
            trace_model_input_output,
        )

        captured: list[ModuleInputOutput] = []

        def hook(data: ModuleInputOutput):
            captured.append(data)

        trace_model_input_output(self.model, self.model_inputs, hook=hook)

        name_to_module = {d.module_name: d.module for d in captured}
        self.assertIsInstance(name_to_module["0"], nn.Linear)
        self.assertIsInstance(name_to_module["1"], nn.ReLU)
        self.assertIsInstance(name_to_module["2"], nn.Linear)


class TestCompareOutputs(unittest.TestCase):
    """Unit tests for compare_outputs function."""

    def test_none_outputs(self):
        """Test that comparing two None values returns None."""
        from tico.quantization.wrapq.utils.introspection import compare_outputs

        result = compare_outputs(None, None)
        self.assertIsNone(result)

    def test_type_mismatch_raises(self):
        """Test that type mismatch raises DataMismatchError."""
        from tico.quantization.wrapq.utils.introspection import (
            compare_outputs,
            DataMismatchError,
        )

        with self.assertRaises(DataMismatchError):
            compare_outputs(torch.tensor([1.0]), [1.0])

        with self.assertRaises(DataMismatchError):
            compare_outputs({"a": 1}, [1])

    def test_key_mismatch_raises(self):
        """Test that key mismatch raises DataMismatchError."""
        from tico.quantization.wrapq.utils.introspection import (
            compare_outputs,
            DataMismatchError,
        )

        with self.assertRaises(DataMismatchError):
            compare_outputs({"a": 1, "b": 2}, {"a": 1, "c": 2})

        with self.assertRaises(DataMismatchError):
            compare_outputs({"a": 1, "b": 2}, {"a": 1})

    def test_tensor_difference_returns_difference_statistics(self):
        """Test that comparing two tensors returns DifferenceStatistics with PEIR."""
        from tico.quantization.wrapq.utils.introspection import (
            compare_outputs,
            DifferenceStatistics,
        )

        lhs = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        rhs = torch.tensor([1.1, 2.2, 3.0, 3.8, 5.3])

        result = compare_outputs(lhs, rhs)

        assert isinstance(result, DifferenceStatistics)  # for mypy type narrowing
        self.assertIn("mean", result._asdict())
        self.assertIn("min", result._asdict())
        self.assertIn("max", result._asdict())
        self.assertIn("stddev", result._asdict())
        self.assertIn("peir", result._asdict())
        # PEIR should be non-negative
        self.assertGreaterEqual(result.peir, 0.0)

    def test_tensor_difference_zero_interval(self):
        """Test that comparing tensors with zero interval returns TensorStatistics (no PEIR)."""
        from tico.quantization.wrapq.utils.introspection import (
            compare_outputs,
            TensorStatistics,
        )

        # All same values → interval = max - min = 0
        lhs = torch.tensor([3.0, 3.0, 3.0])
        rhs = torch.tensor([3.1, 3.2, 3.0])

        result = compare_outputs(lhs, rhs)

        # When interval is 0, PEIR cannot be computed, so TensorStatistics is returned
        self.assertIsInstance(result, TensorStatistics)

    def test_tensor_full_diff(self):
        """Test that full_tensor_diff=True returns the full difference tensor."""
        from tico.quantization.wrapq.utils.introspection import compare_outputs

        lhs = torch.tensor([1.0, 2.0, 3.0])
        rhs = torch.tensor([1.5, 2.5, 3.5])

        result = compare_outputs(lhs, rhs, full_tensor_diff=True)

        self.assertIsInstance(result, torch.Tensor)
        expected = lhs.to(torch.float) - rhs.to(torch.float)
        self.assertTrue(torch.allclose(result, expected))

    def test_identical_tensors(self):
        """Test that comparing identical tensors returns zero difference."""
        from tico.quantization.wrapq.utils.introspection import (
            compare_outputs,
            DifferenceStatistics,
        )

        t = torch.tensor([1.0, 2.0, 3.0])
        result = compare_outputs(t, t)

        assert isinstance(result, DifferenceStatistics)  # for mypy type narrowing
        self.assertAlmostEqual(result.max, 0.0, places=5)
        self.assertAlmostEqual(result.peir, 0.0, places=5)

    def test_number_difference(self):
        """Test that comparing two numbers returns absolute difference."""
        from tico.quantization.wrapq.utils.introspection import compare_outputs

        result = compare_outputs(5.0, 3.0)
        assert isinstance(result, float)
        self.assertAlmostEqual(result, 2.0, places=5)

        result_int = compare_outputs(10, 7)
        assert isinstance(result_int, int)
        self.assertAlmostEqual(result_int, 3, places=5)

    def test_list_difference(self):
        """Test that comparing two lists returns dict of element-wise differences."""
        from tico.quantization.wrapq.utils.introspection import compare_outputs

        lhs = [1.0, 2.0, 3.0]
        rhs = [1.5, 2.0, 2.5]

        result = compare_outputs(lhs, rhs)

        assert isinstance(result, dict)  # for mypy type narrowing
        self.assertIn("0", result)
        self.assertIn("1", result)
        self.assertIn("2", result)
        self.assertAlmostEqual(result["0"], 0.5, places=5)
        self.assertAlmostEqual(result["1"], 0.0, places=5)
        self.assertAlmostEqual(result["2"], 0.5, places=5)

    def test_list_length_mismatch_raises(self):
        """Test that comparing lists of different lengths raises DataMismatchError."""
        from tico.quantization.wrapq.utils.introspection import (
            compare_outputs,
            DataMismatchError,
        )

        with self.assertRaises(DataMismatchError):
            compare_outputs([1.0, 2.0], [1.0])

    def test_dict_difference(self):
        """Test that comparing two dicts returns dict of key-wise differences."""
        from tico.quantization.wrapq.utils.introspection import compare_outputs

        lhs = {"a": 1.0, "b": 2.0}
        rhs = {"a": 1.5, "b": 2.0}

        result = compare_outputs(lhs, rhs)

        assert isinstance(result, dict)  # for mypy type narrowing
        self.assertIn("a", result)
        self.assertIn("b", result)
        self.assertAlmostEqual(result["a"], 0.5, places=5)
        self.assertAlmostEqual(result["b"], 0.0, places=5)

    def test_tuple_difference(self):
        """Test that comparing two tuples returns dict of element-wise differences."""
        from tico.quantization.wrapq.utils.introspection import compare_outputs

        lhs = (10.0, 20.0)
        rhs = (8.0, 22.0)

        result = compare_outputs(lhs, rhs)

        assert isinstance(result, dict)  # for mypy type narrowing
        self.assertAlmostEqual(result["0"], 2.0, places=5)
        self.assertAlmostEqual(result["1"], 2.0, places=5)

    def test_nested_dict_difference(self):
        """Test that comparing nested dicts returns nested difference dict."""
        from tico.quantization.wrapq.utils.introspection import compare_outputs

        lhs = {"outer": {"a": 1.0, "b": 2.0}}
        rhs = {"outer": {"a": 1.5, "b": 2.0}}

        result = compare_outputs(lhs, rhs)

        assert isinstance(result, dict)  # for mypy type narrowing
        self.assertIn("outer", result)
        self.assertIsInstance(result["outer"], dict)
        self.assertAlmostEqual(result["outer"]["a"], 0.5, places=5)
        self.assertAlmostEqual(result["outer"]["b"], 0.0, places=5)

    def test_tensor_in_list_difference(self):
        """Test that comparing lists of tensors returns proper differences."""
        from tico.quantization.wrapq.utils.introspection import (
            compare_outputs,
            DifferenceStatistics,
        )

        lhs = [torch.tensor([1.0, 2.0, 3.0])]
        rhs = [torch.tensor([1.1, 2.2, 3.0])]

        result = compare_outputs(lhs, rhs)

        assert isinstance(result, dict)  # for mypy type narrowing
        self.assertIn("0", result)
        self.assertIsInstance(result["0"], DifferenceStatistics)


class TestCompareSideBySide(unittest.TestCase):
    """Unit tests for compare_side_by_side function."""

    def test_prints_common_modules(self):
        """Test that compare_side_by_side prints differences for common module names."""
        import sys
        from io import StringIO

        from tico.quantization.wrapq.utils.introspection import compare_side_by_side

        t_a = torch.tensor([1.0, 2.0, 3.0])
        t_b = torch.tensor([1.1, 2.2, 3.3])

        outputs_a = {"layer1": t_a, "layer2": t_a}
        outputs_b = {"layer1": t_b, "layer2": t_b}

        captured = StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            compare_side_by_side(outputs_a, outputs_b)
        finally:
            sys.stdout = old_stdout

        output = captured.getvalue()
        # Should print a header line and entries for both layers
        self.assertIn("layer1", output)
        self.assertIn("layer2", output)

    def test_skips_modules_only_in_a(self):
        """Test that modules only in outputs_a are skipped."""
        import sys
        from io import StringIO

        from tico.quantization.wrapq.utils.introspection import compare_side_by_side

        t = torch.tensor([1.0, 2.0, 3.0])

        outputs_a = {"layer1": t, "only_in_a": t}
        outputs_b = {"layer1": t}

        captured = StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            compare_side_by_side(outputs_a, outputs_b)
        finally:
            sys.stdout = old_stdout

        output = captured.getvalue()
        self.assertIn("layer1", output)
        self.assertNotIn("only_in_a", output)

    def test_identical_outputs(self):
        """Test that comparing identical outputs shows zero difference."""
        import sys
        from io import StringIO

        from tico.quantization.wrapq.utils.introspection import compare_side_by_side

        t = torch.tensor([1.0, 2.0, 3.0])
        outputs = {"layer1": t}

        captured = StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            compare_side_by_side(outputs, outputs)
        finally:
            sys.stdout = old_stdout

        output = captured.getvalue()
        self.assertIn("layer1", output)
        self.assertIn("'mean': 0.0", output)
        self.assertIn("'min': 0.0", output)
        self.assertIn("'max': 0.0", output)
        self.assertIn("'stddev': 0.0", output)
        self.assertIn("'peir': 0.0", output)

    def test_interesting_module_gets_full_tensor_diff(self):
        """Test that interesting modules get full tensor diff in the output."""
        import sys
        from io import StringIO

        from tico.quantization.wrapq.utils.introspection import compare_side_by_side

        t_a = torch.tensor([1.0, 2.0, 3.0])
        t_b = torch.tensor([1.1, 2.2, 3.3])

        outputs_a = {"target": t_a}
        outputs_b = {"target": t_b}

        captured = StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            compare_side_by_side(outputs_a, outputs_b, interesting_modules=["target"])
        finally:
            sys.stdout = old_stdout

        output = captured.getvalue()
        self.assertIn("target", output)

    def test_empty_outputs(self):
        """Test that compare_side_by_side handles empty output dicts."""
        import sys
        from io import StringIO

        from tico.quantization.wrapq.utils.introspection import compare_side_by_side

        captured = StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            compare_side_by_side({}, {})
        finally:
            sys.stdout = old_stdout

        output = captured.getvalue()
        # Should print header but no module entries
        self.assertIn("MODULE NAME", output)

    def test_number_outputs(self):
        """Test that compare_side_by_side handles number (non-tensor) outputs."""
        import sys
        from io import StringIO

        from tico.quantization.wrapq.utils.introspection import compare_side_by_side

        outputs_a = {"loss": 1.5}
        outputs_b = {"loss": 1.8}

        captured = StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            compare_side_by_side(outputs_a, outputs_b)
        finally:
            sys.stdout = old_stdout

        output = captured.getvalue()
        self.assertIn("loss", output)
