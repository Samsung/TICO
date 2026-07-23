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

import torch
import torch.nn as nn
import torch.nn.functional as F

from tico.serialize.circle_serializer import (
    _export_tensors,
    _handle_get_attr_node,
    _initialize_model,
)
from tico.utils.errors import NotYetSupportedError


_TIED_WEIGHT_NAMES = {"embed.weight", "lm_head.weight"}


class TiedEmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(8, 4)
        self.lm_head = nn.Linear(4, 8, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(self, tokens):
        hidden = self.embed(tokens)
        return F.linear(hidden, self.lm_head.weight)


class UntiedEmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(8, 4)
        self.lm_head = nn.Linear(4, 8, bias=False)

        with torch.no_grad():
            self.lm_head.weight.copy_(self.embed.weight)

    def forward(self, tokens):
        hidden = self.embed(tokens)
        return F.linear(hidden, self.lm_head.weight)


class TensorAttributeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = torch.ones(4)

    def forward(self, x):
        return x + self.value


class CircleSerializerInvariantTest(unittest.TestCase):
    @staticmethod
    def _make_get_attr_node(name, value):
        """Create a get_attr node owned by a GraphModule."""
        root = nn.Module()
        if isinstance(value, nn.Module):
            root.add_module(name, value)
        elif isinstance(value, torch.Tensor):
            root.register_buffer(name, value)
        else:
            setattr(root, name, value)

        graph = torch.fx.Graph()
        attr_node = graph.get_attr(name)
        graph.output(attr_node)
        graph_module = torch.fx.GraphModule(root, graph)

        return next(node for node in graph_module.graph.nodes if node.op == "get_attr")

    def test_initialize_model_has_single_reserved_empty_buffer(self):
        model, _ = _initialize_model()

        self.assertEqual(len(model.buffers), 1)
        self.assertIsNone(model.buffers[0].data)

    def test_tensor_attribute_is_lifted_as_constant_placeholder(self):
        exported_program = torch.export.export(
            TensorAttributeModel().eval(),
            (torch.zeros(4),),
        )

        self.assertFalse(
            any(node.op == "get_attr" for node in exported_program.graph.nodes)
        )

        constant_inputs = (
            exported_program.graph_signature.inputs_to_lifted_tensor_constants
        )
        self.assertEqual(len(constant_inputs), 1)

        model, graph = _initialize_model()
        _export_tensors(graph, exported_program)

        placeholder_name = next(iter(constant_inputs))
        tensor = graph.tensors[graph.name_to_tid[placeholder_name]]

        self.assertGreater(tensor.buffer, 0)
        self.assertEqual(len(model.buffers), 2)

    def test_tensor_get_attr_is_rejected_as_invariant_violation(self):
        node = self._make_get_attr_node("value", torch.ones(1))

        with self.assertRaisesRegex(RuntimeError, "must be lifted"):
            _handle_get_attr_node(node)

    def test_nested_graph_module_get_attr_is_rejected(self):
        branch = torch.fx.symbolic_trace(nn.Identity())
        node = self._make_get_attr_node("branch", branch)

        with self.assertRaisesRegex(
            NotYetSupportedError,
            "nested GraphModule",
        ):
            _handle_get_attr_node(node)


class CircleSerializerSharedTensorTest(unittest.TestCase):
    @staticmethod
    def _export_model(model):
        """Export a model with a fixed token input."""
        model.eval()
        tokens = torch.tensor([[1, 2, 3]], dtype=torch.long)
        return torch.export.export(model, (tokens,))

    @staticmethod
    def _export_graph(exported_program):
        """Export tensors from an exported program and return the Circle graph."""
        _, graph = _initialize_model()
        _export_tensors(graph, exported_program)
        return graph

    @staticmethod
    def _get_tied_weight_placeholders(exported_program):
        """Return placeholders whose target parameters share tied embedding storage."""
        pairs = [
            (placeholder_name, target_name)
            for placeholder_name, target_name in (
                exported_program.graph_signature.inputs_to_parameters.items()
            )
            if target_name in _TIED_WEIGHT_NAMES
        ]

        if len(pairs) != 2:
            raise AssertionError(
                "Expected two placeholders for tied embedding weights, but got "
                f"{pairs}."
            )

        alias_keys = set()
        for _, target_name in pairs:
            tensor = exported_program.state_dict[target_name]
            alias_keys.add(
                (
                    str(tensor.device),
                    tensor.data_ptr(),
                    tensor.storage_offset(),
                    tuple(tensor.shape),
                    tuple(tensor.stride()),
                    tensor.dtype,
                    tensor.layout,
                )
            )

        if len(alias_keys) != 1:
            raise AssertionError(
                "Expected tied embedding parameters to share storage, but got "
                f"{pairs} with alias keys {alias_keys}."
            )

        return [placeholder_name for placeholder_name, _ in pairs]

    @staticmethod
    def _get_single_parameter_placeholder(exported_program, parameter_name):
        """Return the only placeholder mapped to the given parameter name."""
        placeholder_names = [
            placeholder_name
            for placeholder_name, target_name in (
                exported_program.graph_signature.inputs_to_parameters.items()
            )
            if target_name == parameter_name
        ]

        if len(placeholder_names) != 1:
            raise AssertionError(
                f"Expected one placeholder for {parameter_name}, but got "
                f"{placeholder_names}."
            )

        return placeholder_names[0]

    @staticmethod
    def _materialized_tensor_names(graph, placeholder_names):
        """Return placeholder names that were materialized as Circle tensors."""
        actual_tensor_names = {tensor.name for tensor in graph.tensors}
        return set(placeholder_names) & actual_tensor_names

    def test_tied_embedding_parameter_placeholders_share_one_circle_tensor(self):
        model = TiedEmbeddingModel()

        self.assertEqual(
            model.embed.weight.data_ptr(),
            model.lm_head.weight.data_ptr(),
        )

        exported_program = self._export_model(model)
        placeholder_names = self._get_tied_weight_placeholders(exported_program)

        graph = self._export_graph(exported_program)

        for placeholder_name in placeholder_names:
            self.assertTrue(graph.has_tensor(placeholder_name))

        tensor_ids = {
            graph.name_to_tid[placeholder_name]
            for placeholder_name in placeholder_names
        }
        self.assertEqual(1, len(tensor_ids))

        self.assertEqual(
            1,
            len(self._materialized_tensor_names(graph, placeholder_names)),
        )

    def test_same_value_parameter_tensors_are_not_shared(self):
        model = UntiedEmbeddingModel()

        self.assertNotEqual(
            model.embed.weight.data_ptr(),
            model.lm_head.weight.data_ptr(),
        )
        self.assertTrue(torch.equal(model.embed.weight, model.lm_head.weight))

        exported_program = self._export_model(model)
        graph = self._export_graph(exported_program)

        embed_placeholder = self._get_single_parameter_placeholder(
            exported_program, "embed.weight"
        )
        lm_head_placeholder = self._get_single_parameter_placeholder(
            exported_program, "lm_head.weight"
        )
        placeholder_names = [embed_placeholder, lm_head_placeholder]

        for placeholder_name in placeholder_names:
            self.assertTrue(graph.has_tensor(placeholder_name))

        self.assertNotEqual(
            graph.name_to_tid[embed_placeholder],
            graph.name_to_tid[lm_head_placeholder],
        )

        self.assertEqual(
            2,
            len(self._materialized_tensor_names(graph, placeholder_names)),
        )
