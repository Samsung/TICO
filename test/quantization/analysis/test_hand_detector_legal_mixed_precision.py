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

"""Tests for legal hand-detector UINT8/INT16 precision regions."""

from __future__ import annotations

import unittest

import torch

from examples.hand_detector._support.analysis import OUTPUT_NAMES
from examples.hand_detector._support.legal_mixed_precision import (
    build_legal_candidate,
    build_observer_policies,
    build_precision_regions,
    make_precision_map,
    Precision,
    precision_transition_edges,
    PrecisionCostWeights,
    run_legal_mixed_precision_search,
    validate_legal_precision_contract,
)
from examples.hand_detector.hand_detector import (
    ConvNode,
    HandDetector,
    NHWCInputAdapter,
)
from tico.quantization.analysis import make_output_adapter
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.wrappers.nn.quant_conv2d import QuantConv2d
from tico.quantization.wrapq.wrappers.nn.quant_prelu import QuantPReLU
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_stub import QuantStubWrapper


class HandDetectorLegalMixedPrecisionTest(unittest.TestCase):
    def test_regions_partition_graph_and_fix_output_domains(self) -> None:
        model = _tiny_model()
        regions = build_precision_regions(model)

        self.assertEqual(
            [region.name for region in regions],
            [
                "stem",
                "regressors_low_resolution_head",
                "regressors_high_resolution_head",
                "regressors_output",
                "classifiers_low_resolution_head",
                "classifiers_high_resolution_head",
                "classifiers_output",
            ],
        )
        self.assertEqual(
            regions[3].fixed_precision,
            Precision.INT16,
        )
        self.assertEqual(
            regions[-1].fixed_precision,
            Precision.UINT8,
        )
        positions = [
            position for region in regions for position in region.operation_positions
        ]
        self.assertEqual(sorted(positions), list(range(8)))
        self.assertEqual(len(positions), len(set(positions)))

    def test_transition_count_reflects_semantic_dtype_boundaries(self) -> None:
        model = _tiny_model()
        regions = build_precision_regions(model)

        one_uint8 = make_precision_map(regions, ("stem",))
        transitions = precision_transition_edges(
            model.detector,
            regions,
            one_uint8,
        )
        self.assertEqual(len(transitions), 6)

        all_variable_uint8 = make_precision_map(
            regions,
            tuple(region.name for region in regions if not region.is_fixed),
        )
        transitions = precision_transition_edges(
            model.detector,
            regions,
            all_variable_uint8,
        )
        self.assertEqual(len(transitions), 2)
        self.assertTrue(
            all(
                transition["target_region"] == "regressors_output"
                for transition in transitions
            )
        )

    def test_candidate_couples_conv_and_prelu_parameters_to_activation_dtype(
        self,
    ) -> None:
        torch.manual_seed(7)
        model = _tiny_model()
        regions = build_precision_regions(model)
        precision_map = make_precision_map(regions, ("stem",))
        policies = build_observer_policies(
            uint8_percentile=99.9,
            int16_observer="minmax",
            int16_percentile=99.99,
            max_samples=256,
            samples_per_batch=64,
            sampling_seed=17,
        )
        sample = torch.randn(1, 4, 4, 3)
        candidate, metadata = build_legal_candidate(
            model,
            (sample,),
            regions=regions,
            precision_map=precision_map,
            policies=policies,
        )

        self.assertEqual(metadata["contract"]["status"], "pass")
        stem_conv = candidate.detector.layers[0]
        self.assertIsInstance(stem_conv, ConvNode)
        self.assertIsInstance(stem_conv.conv, PTQWrapper)
        stem_conv_quant = stem_conv.conv.wrapped
        self.assertIsInstance(stem_conv_quant, QuantConv2d)
        self.assertEqual(stem_conv_quant.obs_weight.dtype, DType.uint(8))
        self.assertEqual(stem_conv_quant.obs_act_out.dtype, DType.uint(8))

        stem_prelu_wrapper = candidate.detector.layers[1]
        self.assertIsInstance(stem_prelu_wrapper, PTQWrapper)
        stem_prelu = stem_prelu_wrapper.wrapped
        self.assertIsInstance(stem_prelu, QuantPReLU)
        self.assertEqual(stem_prelu.obs_weight.dtype, DType.uint(8))
        self.assertEqual(stem_prelu.obs_act_out.dtype, DType.uint(8))

        int16_head = candidate.detector.layers[2]
        self.assertIsInstance(int16_head, ConvNode)
        self.assertIsInstance(int16_head.conv, PTQWrapper)
        int16_head_quant = int16_head.conv.wrapped
        self.assertIsInstance(int16_head_quant, QuantConv2d)
        self.assertEqual(int16_head_quant.obs_weight.dtype, DType.int(16))
        self.assertEqual(int16_head_quant.obs_act_out.dtype, DType.int(16))

        outputs = candidate(sample)
        self.assertEqual(len(outputs), 2)
        self.assertEqual(metadata["contract"]["dtype_transition_count"], 6)
        self.assertEqual(metadata["contract"]["graph_input_precision"], "uint8")
        self.assertEqual(metadata["contract"]["explicit_boundary_count"], 4)

    def test_add_output_uses_the_region_precision_domain(self) -> None:
        model = _tiny_model_with_add()
        regions = build_precision_regions(model)
        precision_map = make_precision_map(regions, ("stem",))
        policies = build_observer_policies(
            uint8_percentile=99.9,
            int16_observer="minmax",
            int16_percentile=99.99,
            max_samples=128,
            samples_per_batch=32,
            sampling_seed=11,
        )
        sample = torch.randn(1, 4, 4, 3)
        candidate, metadata = build_legal_candidate(
            model,
            (sample,),
            regions=regions,
            precision_map=precision_map,
            policies=policies,
        )

        self.assertEqual(metadata["contract"]["add_output_quantizer_count"], 1)
        add_wrapper = candidate.add_output_quantizers["p001"]
        self.assertIsInstance(add_wrapper, PTQWrapper)
        add_quantizer = add_wrapper.wrapped
        self.assertIsInstance(add_quantizer, QuantStubWrapper)
        self.assertEqual(add_quantizer.obs_act_out.dtype, DType.uint(8))

    def test_contract_rejects_parameter_activation_dtype_mismatch(self) -> None:
        model = _tiny_model()
        regions = build_precision_regions(model)
        precision_map = make_precision_map(regions, ())
        policies = build_observer_policies(
            uint8_percentile=99.9,
            int16_observer="minmax",
            int16_percentile=99.99,
            max_samples=128,
            samples_per_batch=32,
            sampling_seed=5,
        )
        sample = torch.randn(1, 4, 4, 3)
        candidate, _ = build_legal_candidate(
            model,
            (sample,),
            regions=regions,
            precision_map=precision_map,
            policies=policies,
        )
        conv_wrapper = candidate.detector.layers[0].conv
        self.assertIsInstance(conv_wrapper, PTQWrapper)
        conv = conv_wrapper.wrapped
        self.assertIsInstance(conv, QuantConv2d)
        conv.obs_weight.dtype = DType.uint(8)

        with self.assertRaisesRegex(RuntimeError, "expected int16"):
            validate_legal_precision_contract(
                candidate,
                regions,
                precision_map,
            )

    def test_floor_report_uses_only_legal_weight_activation_pairs(self) -> None:
        torch.manual_seed(13)
        model = _tiny_model()
        calibration = (torch.randn(1, 4, 4, 3),)
        evaluation = (torch.randn(1, 4, 4, 3),)
        report = run_legal_mixed_precision_search(
            model,
            calibration,
            evaluation,
            uint8_percentile=99.9,
            int16_observer="minmax",
            int16_percentile=99.99,
            max_samples=128,
            samples_per_batch=32,
            sampling_seed=3,
            regressor_output_precision=Precision.INT16,
            classifier_output_precision=Precision.UINT8,
            target_regressor_mae=1000.0,
            target_classifier_mae=1000.0,
            search="none",
            beam_width=2,
            candidate_count=0,
            max_search_steps=0,
            skip_sensitivity=True,
            search_even_if_entry_infeasible=False,
            cost_weights=PrecisionCostWeights(),
            output_adapter=make_output_adapter(OUTPUT_NAMES),
        )

        floors = report["floors"]
        self.assertIn("legal_all_int16_internal", floors)
        self.assertIn("legal_all_uint8_internal", floors)
        for result in floors.values():
            self.assertEqual(result["contract"]["status"], "pass")
        self.assertEqual(
            report["selected_assignment"]["precision_map"]["stem"],
            "int16",
        )


def _tiny_model() -> NHWCInputAdapter:
    specification = {
        "inputs": [0],
        "outputs": [5, 8],
        "operations": [
            _conv_operation(0, 0, 1, 3, 2),
            {
                "index": 1,
                "name": "PRELU",
                "inputs": [1],
                "outputs": [2],
                "config": {"channels": 2},
            },
            _conv_operation(2, 2, 3, 2, 1),
            _conv_operation(3, 2, 4, 2, 1),
            {
                "index": 4,
                "name": "CONCATENATION",
                "inputs": [3, 4],
                "outputs": [5],
                "config": {"axis": 1},
            },
            _conv_operation(5, 2, 6, 2, 1),
            _conv_operation(6, 2, 7, 2, 1),
            {
                "index": 7,
                "name": "CONCATENATION",
                "inputs": [6, 7],
                "outputs": [8],
                "config": {"axis": 1},
            },
        ],
    }
    detector = HandDetector(specification)
    with torch.no_grad():
        for module in detector.modules():
            if isinstance(module, torch.nn.Conv2d):
                module.weight.uniform_(-0.2, 0.2)
                if module.bias is not None:
                    module.bias.zero_()
    return NHWCInputAdapter(detector).eval()


def _tiny_model_with_add() -> NHWCInputAdapter:
    specification = {
        "inputs": [0],
        "outputs": [6, 9],
        "operations": [
            _conv_operation(0, 0, 1, 3, 3),
            {
                "index": 1,
                "name": "ADD",
                "inputs": [1, 0],
                "outputs": [2],
                "config": {},
            },
            {
                "index": 2,
                "name": "PRELU",
                "inputs": [2],
                "outputs": [3],
                "config": {"channels": 3},
            },
            _conv_operation(3, 3, 4, 3, 1),
            _conv_operation(4, 3, 5, 3, 1),
            {
                "index": 5,
                "name": "CONCATENATION",
                "inputs": [4, 5],
                "outputs": [6],
                "config": {"axis": 1},
            },
            _conv_operation(6, 3, 7, 3, 1),
            _conv_operation(7, 3, 8, 3, 1),
            {
                "index": 8,
                "name": "CONCATENATION",
                "inputs": [7, 8],
                "outputs": [9],
                "config": {"axis": 1},
            },
        ],
    }
    return NHWCInputAdapter(HandDetector(specification)).eval()


def _conv_operation(
    index: int,
    input_tensor: int,
    output_tensor: int,
    in_channels: int,
    out_channels: int,
) -> dict[str, object]:
    return {
        "index": index,
        "name": "CONV_2D",
        "inputs": [input_tensor],
        "outputs": [output_tensor],
        "config": {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "kernel_size": [1, 1],
            "stride": [1, 1],
            "dilation": [1, 1],
            "groups": 1,
            "has_bias": True,
            "padding": "valid",
            "pad": [0, 0, 0, 0],
        },
    }


if __name__ == "__main__":
    unittest.main()
