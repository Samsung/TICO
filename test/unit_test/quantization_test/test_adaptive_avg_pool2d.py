import unittest

from typing import Any, Optional

import torch
from tico.experimental.quantization.algorithm.pt2e.annotation.config import (
    QuantizationConfig,
)

from tico.experimental.quantization.algorithm.pt2e.annotation.op.adaptive_avg_pool2d import (
    _annotate_adaptive_avg_pool2d,
)
from tico.experimental.quantization.algorithm.pt2e.utils import get_input_act_qspec
from torch.ao.quantization.quantizer import (
    QuantizationAnnotation,
    SharedQuantizationSpec,
)


class TestAdaptiveAvgPool2dModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 3, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = torch.ops.aten.adaptive_avg_pool2d.default(x, [1, 1])
        return x


class TestAnnotateAdaptiveAvgPool2d(unittest.TestCase):
    def setUp(self):
        # Create a simple model and export it to get real FX graph
        self.model = TestAdaptiveAvgPool2dModel()
        self.model.eval()

        # Create example input and trace the model
        example_input = torch.randn(1, 3, 32, 32)

        # Export the model to get FX GraphModule
        self.exported_program = torch.export.export(self.model, (example_input,))
        self.gm = self.exported_program.graph_module

        # Find the adaptive_avg_pool2d node
        self.adaptive_avg_pool2d_node: Optional[Any] = None
        self.input_node: Optional[Any] = None

        for node in self.gm.graph.nodes:
            if node.target == torch.ops.aten.adaptive_avg_pool2d.default:
                self.adaptive_avg_pool2d_node = node
                self.input_node = node.args[0]  # First argument is the input tensor
                break

        if self.adaptive_avg_pool2d_node is None:
            raise ValueError("Could not find adaptive_avg_pool2d node in the graph")

        # Create quantization config
        input_qspec = torch.ao.quantization.quantizer.FixedQParamsQuantizationSpec(
            dtype=torch.qint8,
            quant_min=-128,
            quant_max=127,
            qscheme=torch.per_tensor_affine,
            scale=1.0,
            zero_point=0,
        )

        self.quantization_config = QuantizationConfig(
            input_activation=input_qspec,
            output_activation=None,
            weight=None,
            bias=None,
        )

    def test_annotate_adaptive_avg_pool2d(self):
        """Test basic annotation functionality"""
        # Ensure the node is not annotated initially
        self.assertFalse(self._is_node_annotated(self.adaptive_avg_pool2d_node))

        # Apply annotation
        _annotate_adaptive_avg_pool2d(
            self.gm, self.adaptive_avg_pool2d_node, self.quantization_config
        )

        # Verify the node is now annotated
        self.assertTrue(self._is_node_annotated(self.adaptive_avg_pool2d_node))

        # Verify input qspec map is set
        quantization_annotation = self.adaptive_avg_pool2d_node.meta[  # type: ignore[union-attr]
            "quantization_annotation"
        ]
        self.assertIn(self.input_node, quantization_annotation.input_qspec_map)

        # Verify output qspec is set
        self.assertIsNotNone(quantization_annotation.output_qspec)

        # Verify the output qspec is SharedQuantizationSpec pointing to input and node
        output_qspec = quantization_annotation.output_qspec
        self.assertIsInstance(output_qspec, SharedQuantizationSpec)
        self.assertEqual(
            output_qspec.edge_or_node, (self.input_node, self.adaptive_avg_pool2d_node)
        )

    def test_filter_fn_false(self):
        """Test that filter_fn=False prevents annotation"""
        # Ensure the node is not annotated initially
        self.assertFalse(self._is_node_annotated(self.adaptive_avg_pool2d_node))

        # Apply annotation with filter_fn=False
        _annotate_adaptive_avg_pool2d(
            self.gm,
            self.adaptive_avg_pool2d_node,
            self.quantization_config,
            filter_fn=lambda n: False,
        )

        # Verify the node is still not annotated
        self.assertFalse(self._is_node_annotated(self.adaptive_avg_pool2d_node))

    def test_filter_fn_true(self):
        """Test that filter_fn=True allows annotation"""
        # Ensure the node is not annotated initially
        self.assertFalse(self._is_node_annotated(self.adaptive_avg_pool2d_node))

        # Apply annotation with filter_fn=True
        _annotate_adaptive_avg_pool2d(
            self.gm,
            self.adaptive_avg_pool2d_node,
            self.quantization_config,
            filter_fn=lambda n: True,
        )

        # Verify the node is now annotated
        self.assertTrue(self._is_node_annotated(self.adaptive_avg_pool2d_node))

    def test_already_annotated(self):
        """Test that already annotated nodes are not re-annotated"""
        # First, annotate the node
        _annotate_adaptive_avg_pool2d(
            self.gm, self.adaptive_avg_pool2d_node, self.quantization_config
        )
        self.assertTrue(self._is_node_annotated(self.adaptive_avg_pool2d_node))

        # Store the original annotation attributes
        original_annotated = self.adaptive_avg_pool2d_node.meta[  # type: ignore[union-attr]
            "quantization_annotation"
        ]._annotated
        original_input_qspec_map = self.adaptive_avg_pool2d_node.meta[  # type: ignore[union-attr]
            "quantization_annotation"
        ].input_qspec_map
        original_output_qspec = self.adaptive_avg_pool2d_node.meta[  # type: ignore[union-attr]
            "quantization_annotation"
        ].output_qspec

        # Try to annotate again
        _annotate_adaptive_avg_pool2d(
            self.gm, self.adaptive_avg_pool2d_node, self.quantization_config
        )

        # Verify the annotation hasn't changed
        current_annotation = self.adaptive_avg_pool2d_node.meta[  # type: ignore[union-attr]
            "quantization_annotation"
        ]
        self.assertEqual(original_annotated, current_annotation._annotated)
        self.assertEqual(original_input_qspec_map, current_annotation.input_qspec_map)
        self.assertEqual(original_output_qspec, current_annotation.output_qspec)

    def test_input_annotated(self):
        """Test behavior when input is already annotated"""
        # First, annotate the input node
        input_qspec = get_input_act_qspec(self.quantization_config)
        input_annotation = QuantizationAnnotation(
            _annotated=True, output_qspec=input_qspec
        )
        self.input_node.meta["quantization_annotation"] = input_annotation  # type: ignore[union-attr]

        # Ensure the adaptive_avg_pool2d node is not annotated initially
        self.assertFalse(self._is_node_annotated(self.adaptive_avg_pool2d_node))

        # Apply annotation to adaptive_avg_pool2d node
        _annotate_adaptive_avg_pool2d(
            self.gm, self.adaptive_avg_pool2d_node, self.quantization_config
        )

        # Verify the node is now annotated
        self.assertTrue(self._is_node_annotated(self.adaptive_avg_pool2d_node))

        # Verify input qspec map uses SharedQuantizationSpec
        quantization_annotation = self.adaptive_avg_pool2d_node.meta[  # type: ignore[union-attr]
            "quantization_annotation"
        ]
        input_act_qspec = quantization_annotation.input_qspec_map[self.input_node]
        self.assertIsInstance(input_act_qspec, SharedQuantizationSpec)
        self.assertEqual(input_act_qspec.edge_or_node, self.input_node)

    def _is_node_annotated(self, node):
        """Helper method to check if a node is annotated"""
        return (
            "quantization_annotation" in node.meta
            and node.meta["quantization_annotation"]._annotated
        )

    def test_invalid_node_target(self):
        """Test that nodes with wrong target are not annotated"""
        # Find a node that is not adaptive_avg_pool2d
        other_node = None
        for node in self.gm.graph.nodes:
            if node.target != torch.ops.aten.adaptive_avg_pool2d.default:
                other_node = node
                break

        if other_node is None:
            self.skipTest("Could not find a non-adaptive_avg_pool2d node")

        # Ensure the node is not annotated initially
        self.assertFalse(self._is_node_annotated(other_node))

        # Try to annotate it
        _annotate_adaptive_avg_pool2d(self.gm, other_node, self.quantization_config)

        # Verify the node is still not annotated
        self.assertFalse(self._is_node_annotated(other_node))

    def test_none_quantization_config(self):
        """Test behavior with None quantization config"""
        # Ensure the node is not annotated initially
        self.assertFalse(self._is_node_annotated(self.adaptive_avg_pool2d_node))

        # Apply annotation with None config
        _annotate_adaptive_avg_pool2d(self.gm, self.adaptive_avg_pool2d_node, None)

        # Verify the node is annotated even with None config
        self.assertTrue(self._is_node_annotated(self.adaptive_avg_pool2d_node))

        # Verify input qspec map is set with None
        quantization_annotation = self.adaptive_avg_pool2d_node.meta[  # type: ignore[union-attr]
            "quantization_annotation"
        ]
        self.assertIn(self.input_node, quantization_annotation.input_qspec_map)
        self.assertIsNone(quantization_annotation.input_qspec_map[self.input_node])
