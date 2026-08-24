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

"""Legal UINT8/INT16 mixed-precision search for the hand detector."""

from __future__ import annotations

import copy
import math

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast, TypeVar

import torch
from tico.quantization import convert as freeze_quantization, prepare, QuantStub
from tico.quantization.analysis import evaluate_models, OutputAdapter
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.config.specs import affine, no_quant, QuantSpec
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.observers.base import ObserverBase
from tico.quantization.wrapq.observers.identity import IdentityObserver
from tico.quantization.wrapq.observers.minmax import MinMaxObserver
from tico.quantization.wrapq.observers.percentile import PercentileObserver
from tico.quantization.wrapq.qscheme import QScheme
from tico.quantization.wrapq.wrappers.nn.quant_conv2d import QuantConv2d
from tico.quantization.wrapq.wrappers.nn.quant_maxpool2d import QuantMaxPool2d
from tico.quantization.wrapq.wrappers.nn.quant_prelu import QuantPReLU
from tico.quantization.wrapq.wrappers.ops.quant_concat import QuantConcat
from tico.quantization.wrapq.wrappers.ops.quant_resize_bilinear import (
    QuantResizeBilinear2d,
)
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.quant_stub import QuantStubWrapper
from torch import nn

from examples.hand_detector._support.analysis import OUTPUT_NAMES
from examples.hand_detector.hand_detector import (
    ConvNode,
    HandDetector,
    NHWCInputAdapter,
)


MetricSummary = Mapping[str, Mapping[str, float | int | None]]
_QuantModuleT = TypeVar("_QuantModuleT", bound=QuantModuleBase)


def _unwrap_ptq_wrapper(
    module: nn.Module,
    expected_type: type[_QuantModuleT],
) -> _QuantModuleT | None:
    """Return the specialized WrapQ module hidden by ``PTQWrapper``.

    ``prepare()`` keeps the public tree node as ``PTQWrapper`` and stores
    the registered implementation in ``PTQWrapper.wrapped``.  Contract
    validation must therefore inspect the specialized child rather than
    requiring it to replace the tree node directly.
    """
    current: nn.Module = module
    while isinstance(current, PTQWrapper):
        current = current.wrapped
    if isinstance(current, expected_type):
        return current
    return None


def _wrapped_type_chain(module: nn.Module) -> str:
    """Return a readable outer-to-inner wrapper type chain."""
    names = [type(module).__name__]
    current = module
    while isinstance(current, PTQWrapper):
        current = current.wrapped
        names.append(type(current).__name__)
    return " -> ".join(names)


class Precision(str, Enum):
    """Represent one legal data-operator precision domain."""

    UINT8 = "uint8"
    INT16 = "int16"

    @property
    def bytes_per_element(self) -> int:
        """Return the storage bytes used by one value."""
        return 1 if self is Precision.UINT8 else 2

    @property
    def dtype(self) -> DType:
        """Return the WrapQ dtype for this precision."""
        return DType.uint(8) if self is Precision.UINT8 else DType.int(16)


@dataclass(frozen=True)
class PrecisionRegion:
    """Describe a semantic region whose data operators share one dtype."""

    name: str
    kind: str
    operation_positions: tuple[int, ...]
    operation_indices: tuple[int, ...]
    operation_names: tuple[str, ...]
    external_inputs: tuple[int, ...] = ()
    produced_tensors: tuple[int, ...] = ()
    fixed_precision: Precision | None = None

    @property
    def is_fixed(self) -> bool:
        """Return whether search may change this region's precision."""
        return self.fixed_precision is not None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible region metadata."""
        return {
            "name": self.name,
            "kind": self.kind,
            "operation_positions": list(self.operation_positions),
            "operation_indices": list(self.operation_indices),
            "operation_names": list(self.operation_names),
            "external_inputs": list(self.external_inputs),
            "produced_tensors": list(self.produced_tensors),
            "fixed_precision": (
                self.fixed_precision.value if self.fixed_precision is not None else None
            ),
        }


@dataclass(frozen=True)
class PrecisionObserverPolicies:
    """Bundle activation and parameter specs for both legal precisions."""

    activation: Mapping[Precision, QuantSpec]
    weight: Mapping[Precision, QuantSpec]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class PrecisionCostWeights:
    """Weight normalized deployment-cost components."""

    parameter: float = 1.0
    activation: float = 1.0
    boundary: float = 0.05

    def validate(self) -> None:
        """Validate finite nonnegative cost weights."""
        for name, value in (
            ("parameter", self.parameter),
            ("activation", self.activation),
            ("boundary", self.boundary),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} cost weight must be finite and nonnegative.")
        if self.parameter + self.activation + self.boundary == 0.0:
            raise ValueError("At least one deployment-cost weight must be positive.")

    def to_dict(self) -> dict[str, float]:
        return {
            "parameter": self.parameter,
            "activation": self.activation,
            "boundary": self.boundary,
        }


@dataclass(frozen=True)
class _RawRegion:
    name: str
    kind: str
    positions: tuple[int, ...]
    fixed_precision: Precision | None = None


class LegalMixedPrecisionAdapter(nn.Module):
    """Execute a detector with explicit semantic-region dtype boundaries."""

    def __init__(
        self,
        detector: HandDetector,
        regions: Sequence[PrecisionRegion],
        precision_map: Mapping[str, Precision],
    ) -> None:
        super().__init__()
        self.detector = detector
        self.detector.input_quantizer = nn.Identity()
        self.regions = tuple(regions)
        self.precision_map = dict(precision_map)
        self._producer_by_tensor = _producer_positions(detector.operations)
        self._region_by_position = {
            position: region.name
            for region in self.regions
            for position in region.operation_positions
        }
        input_regions = tuple(
            region.name
            for region in self.regions
            if detector.input_tensor in region.external_inputs
        )
        if len(input_regions) != 1:
            raise RuntimeError(
                "Expected exactly one semantic region to consume the graph input, "
                f"found {input_regions}."
            )
        self.input_region_name = input_regions[0]
        self.input_quantizer = QuantStub()

        boundary_quantizers: dict[str, QuantStub] = {}
        boundary_keys: dict[tuple[str, int], str] = {}
        for region in self.regions:
            if _region_uses_shared_concat_domain(region):
                continue
            for tensor_id in region.external_inputs:
                if not self._requires_explicit_boundary(region, tensor_id):
                    continue
                key = _boundary_key(region.name, tensor_id)
                boundary_quantizers[key] = QuantStub()
                boundary_keys[(region.name, tensor_id)] = key
        self.boundary_quantizers = nn.ModuleDict(boundary_quantizers)
        self._boundary_keys = boundary_keys

        add_output_quantizers: dict[str, QuantStub] = {}
        add_output_keys: dict[int, str] = {}
        for region in self.regions:
            for position in region.operation_positions:
                if detector.operations[position]["name"] != "ADD":
                    continue
                key = f"p{position:03d}"
                add_output_quantizers[key] = QuantStub()
                add_output_keys[position] = key
        self.add_output_quantizers = nn.ModuleDict(add_output_quantizers)
        self._add_output_keys = add_output_keys

    def _requires_explicit_boundary(
        self,
        region: PrecisionRegion,
        tensor_id: int,
    ) -> bool:
        producer = self._producer_by_tensor.get(tensor_id)
        if producer is None:
            return False
        source_region = self._region_by_position[producer]
        return self.precision_map[source_region] != self.precision_map[region.name]

    def forward(self, input_: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the graph from NHWC input with explicit dtype-transition stubs."""
        quantized_input = self.input_quantizer(input_)
        values: dict[int, torch.Tensor] = {
            self.detector.input_tensor: quantized_input.permute(0, 3, 1, 2)
        }
        for region in self.regions:
            local_values = dict(values)
            for tensor_id in region.external_inputs:
                key = self._boundary_keys.get((region.name, tensor_id))
                if key is None:
                    continue
                local_values[tensor_id] = self.boundary_quantizers[key](
                    values[tensor_id]
                )

            for position in region.operation_positions:
                tensor_id, value = self._execute_operation(local_values, position)
                add_key = self._add_output_keys.get(position)
                if add_key is not None:
                    value = self.add_output_quantizers[add_key](value)
                local_values[tensor_id] = value
                values[tensor_id] = value

        return (
            values[self.detector.output_tensors[0]],
            values[self.detector.output_tensors[1]],
        )

    def _execute_operation(
        self,
        values: Mapping[int, torch.Tensor],
        position: int,
    ) -> tuple[int, torch.Tensor]:
        operation = self.detector.operations[position]
        layer = self.detector.layers[position]
        name = str(operation["name"])
        inputs = tuple(int(value) for value in operation["inputs"])
        output = int(operation["outputs"][0])
        config = operation["config"]
        if name in {
            "CONV_2D",
            "DEPTHWISE_CONV_2D",
            "PRELU",
            "MAX_POOL_2D",
            "PAD",
            "RESIZE_BILINEAR",
        }:
            value = layer(values[inputs[0]])
        elif name == "ADD":
            value = values[inputs[0]] + values[inputs[1]]
        elif name == "RESHAPE":
            source = values[inputs[0]]
            if bool(config["nhwc_memory_order"]):
                source = source.permute(0, 2, 3, 1)
            value = source.reshape(tuple(config["shape"]))
        elif name == "CONCATENATION":
            value = layer(tuple(values[index] for index in inputs))
        else:
            raise RuntimeError(f"Unsupported converted operation: {name}")
        return output, value

    def get_example_inputs(self) -> tuple[torch.Tensor]:
        """Return the static NHWC input expected by Circle export."""
        return (torch.zeros(1, 192, 192, 3, dtype=torch.float32),)


def build_precision_regions(
    model: nn.Module,
    *,
    regressor_output_precision: Precision = Precision.INT16,
    classifier_output_precision: Precision = Precision.UINT8,
) -> tuple[PrecisionRegion, ...]:
    """Partition every detector operation into legal semantic regions."""
    detector = _find_detector(model)
    operations = detector.operations
    producers = _producer_positions(operations)

    output_regions: list[_RawRegion] = []
    head_regions: list[_RawRegion] = []
    reserved: set[int] = set()
    output_precision = {
        "regressors": regressor_output_precision,
        "classifiers": classifier_output_precision,
    }
    for output_tensor, output_name in zip(detector.output_tensors, OUTPUT_NAMES):
        final_position = producers.get(output_tensor)
        if final_position is None:
            raise RuntimeError(f"No operation produces output tensor {output_tensor}.")
        final_operation = operations[final_position]
        if final_operation["name"] != "CONCATENATION":
            raise RuntimeError(
                f"Expected {output_name} output to be produced by CONCATENATION."
            )
        reserved.add(final_position)
        output_regions.append(
            _RawRegion(
                f"{output_name}_output",
                "output",
                (final_position,),
                output_precision[output_name],
            )
        )

        branches: list[tuple[int, tuple[int, ...]]] = []
        for input_tensor in final_operation["inputs"]:
            source_position, positions = _trace_head_branch(
                int(input_tensor),
                operations,
                producers,
            )
            branches.append((source_position, positions))
            reserved.update(positions)
        branches.sort(key=lambda item: item[0])
        for branch_index, (_, positions) in enumerate(branches):
            resolution = _resolution_name(branch_index, len(branches))
            head_regions.append(
                _RawRegion(
                    f"{output_name}_{resolution}_head",
                    "head",
                    positions,
                )
            )

    feature_regions: list[_RawRegion] = []
    current: list[int] = []
    feature_index = 0
    for position, operation in enumerate(operations):
        if position in reserved:
            continue
        current.append(position)
        if operation["name"] != "PRELU":
            continue
        if not feature_regions:
            name = "stem"
            kind = "stem"
        else:
            name = f"feature_block_{feature_index:02d}"
            kind = "feature"
            feature_index += 1
        feature_regions.append(_RawRegion(name, kind, tuple(current)))
        current = []
    if current:
        feature_regions.append(
            _RawRegion(
                f"feature_block_{feature_index:02d}",
                "feature",
                tuple(current),
            )
        )

    raw_regions = sorted(
        (*feature_regions, *head_regions, *output_regions),
        key=lambda region: min(region.positions),
    )
    assigned = [position for region in raw_regions for position in region.positions]
    expected = list(range(len(operations)))
    if sorted(assigned) != expected or len(set(assigned)) != len(assigned):
        missing = sorted(set(expected).difference(assigned))
        duplicate = sorted(
            position for position in set(assigned) if assigned.count(position) > 1
        )
        raise RuntimeError(
            "Precision regions must partition every operation exactly once; "
            f"missing={missing}, duplicate={duplicate}."
        )

    region_by_position = {
        position: region.name for region in raw_regions for position in region.positions
    }
    results: list[PrecisionRegion] = []
    for raw in raw_regions:
        position_set = frozenset(raw.positions)
        external_inputs: set[int] = set()
        produced_tensors: set[int] = set()
        for position in raw.positions:
            operation = operations[position]
            produced_tensors.update(int(value) for value in operation["outputs"])
            for tensor_id in _runtime_input_tensors(operation):
                producer = producers.get(tensor_id)
                if producer is None or producer not in position_set:
                    external_inputs.add(tensor_id)
        result = PrecisionRegion(
            name=raw.name,
            kind=raw.kind,
            operation_positions=raw.positions,
            operation_indices=tuple(
                int(operations[position]["index"]) for position in raw.positions
            ),
            operation_names=tuple(
                str(operations[position]["name"]) for position in raw.positions
            ),
            external_inputs=tuple(sorted(external_inputs)),
            produced_tensors=tuple(sorted(produced_tensors)),
            fixed_precision=raw.fixed_precision,
        )
        results.append(result)

    if set(region_by_position) != set(expected):
        raise AssertionError("Internal region-position map is incomplete.")
    return tuple(results)


def build_observer_policies(
    *,
    uint8_percentile: float,
    int16_observer: str,
    int16_percentile: float,
    max_samples: int,
    samples_per_batch: int,
    sampling_seed: int,
) -> PrecisionObserverPolicies:
    """Create legal activation/parameter specs for both precision domains."""
    _validate_percentile("uint8_percentile", uint8_percentile)
    _validate_percentile("int16_percentile", int16_percentile)
    if max_samples <= 0 or samples_per_batch <= 0:
        raise ValueError("Observer sample limits must be positive.")
    if int16_observer not in {"minmax", "percentile"}:
        raise ValueError("int16_observer must be 'minmax' or 'percentile'.")

    uint8_activation = affine(
        DType.uint(8),
        qscheme=QScheme.PER_TENSOR_ASYMM,
        observer=PercentileObserver,
        percentile=uint8_percentile,
        max_samples=max_samples,
        samples_per_batch=samples_per_batch,
        seed=sampling_seed,
    )
    if int16_observer == "minmax":
        int16_activation = affine(
            DType.int(16),
            qscheme=QScheme.PER_TENSOR_SYMM,
            observer=MinMaxObserver,  # type: ignore[type-abstract]
        )
    else:
        int16_activation = affine(
            DType.int(16),
            qscheme=QScheme.PER_TENSOR_SYMM,
            observer=PercentileObserver,
            percentile=int16_percentile,
            max_samples=max_samples,
            samples_per_batch=samples_per_batch,
            seed=sampling_seed,
        )
    return PrecisionObserverPolicies(
        activation={
            Precision.UINT8: uint8_activation,
            Precision.INT16: int16_activation,
        },
        weight={
            Precision.UINT8: affine(
                DType.uint(8),
                qscheme=QScheme.PER_CHANNEL_ASYMM,
                observer=MinMaxObserver,  # type: ignore[type-abstract]
            ),
            Precision.INT16: affine(
                DType.int(16),
                qscheme=QScheme.PER_CHANNEL_SYMM,
                observer=MinMaxObserver,  # type: ignore[type-abstract]
            ),
        },
        metadata={
            "uint8_activation_observer": (f"PercentileObserver(P{uint8_percentile:g})"),
            "int16_activation_observer": (
                "MinMaxObserver"
                if int16_observer == "minmax"
                else f"PercentileObserver(P{int16_percentile:g})"
            ),
            "max_samples": max_samples,
            "samples_per_batch": samples_per_batch,
            "sampling_seed": sampling_seed,
        },
    )


def make_precision_map(
    regions: Sequence[PrecisionRegion],
    uint8_regions: Sequence[str] | frozenset[str],
) -> dict[str, Precision]:
    """Create one full assignment from the set of demoted UINT8 regions."""
    selected = frozenset(uint8_regions)
    known = {region.name for region in regions}
    unknown = sorted(selected.difference(known))
    if unknown:
        raise ValueError(f"Unknown precision regions: {unknown}.")
    fixed_conflicts = sorted(
        region.name
        for region in regions
        if region.fixed_precision is Precision.INT16 and region.name in selected
    )
    if fixed_conflicts:
        raise ValueError(
            "Cannot demote fixed INT16 output regions to UINT8: " f"{fixed_conflicts}."
        )
    result: dict[str, Precision] = {}
    for region in regions:
        if region.fixed_precision is not None:
            result[region.name] = region.fixed_precision
        elif region.name in selected:
            result[region.name] = Precision.UINT8
        else:
            result[region.name] = Precision.INT16
    return result


def build_legal_candidate(
    float_model: nn.Module,
    calibration_samples: Sequence[torch.Tensor],
    *,
    regions: Sequence[PrecisionRegion],
    precision_map: Mapping[str, Precision],
    policies: PrecisionObserverPolicies,
) -> tuple[nn.Module, dict[str, Any]]:
    """Build, calibrate, freeze, and validate one legal precision candidate."""
    if not calibration_samples:
        raise ValueError("Legal mixed precision requires calibration samples.")
    detector = copy.deepcopy(_find_detector(float_model)).eval()
    candidate: nn.Module = LegalMixedPrecisionAdapter(
        detector,
        regions,
        precision_map,
    ).eval()
    overrides = build_precision_overrides(
        cast(LegalMixedPrecisionAdapter, candidate),
        regions,
        precision_map,
        policies,
    )
    candidate = prepare(
        candidate,
        PTQConfig(
            activation=no_quant(),
            weight=no_quant(),
            overrides=overrides,
            strict_wrap=False,
        ),
        inplace=True,
    ).eval()
    _calibrate(candidate, calibration_samples)
    candidate = freeze_quantization(candidate, inplace=True).eval()
    contract = validate_legal_precision_contract(
        candidate,
        regions,
        precision_map,
    )
    return candidate, {
        "override_count": len(overrides),
        "contract": contract,
    }


def build_precision_overrides(
    adapter: LegalMixedPrecisionAdapter,
    regions: Sequence[PrecisionRegion],
    precision_map: Mapping[str, Precision],
    policies: PrecisionObserverPolicies,
) -> dict[str, QuantSpec]:
    """Create explicit per-observer policies for all data operators and boundaries."""
    overrides: dict[str, QuantSpec] = {
        "input_quantizer.act_out": policies.activation[
            precision_map[adapter.input_region_name]
        ]
    }
    for (region_name, _), key in adapter._boundary_keys.items():
        precision = precision_map[region_name]
        overrides[f"boundary_quantizers.{key}.act_out"] = policies.activation[precision]
    for position, key in adapter._add_output_keys.items():
        region_name = _region_name_for_position(regions, position)
        precision = precision_map[region_name]
        overrides[f"add_output_quantizers.{key}.act_out"] = policies.activation[
            precision
        ]

    for region in regions:
        precision = precision_map[region.name]
        activation_spec = policies.activation[precision]
        weight_spec = policies.weight[precision]
        for position in region.operation_positions:
            operation = adapter.detector.operations[position]
            name = str(operation["name"])
            if name in {"CONV_2D", "DEPTHWISE_CONV_2D"}:
                base = f"detector.layers.{position}.conv"
                overrides[f"{base}.act_in"] = no_quant()
                overrides[f"{base}.act_out"] = activation_spec
                overrides[f"{base}.weight"] = weight_spec
            elif name == "PRELU":
                base = f"detector.layers.{position}"
                overrides[f"{base}.act_in"] = no_quant()
                overrides[f"{base}.act_out"] = activation_spec
                overrides[f"{base}.weight"] = weight_spec
            elif name in {"MAX_POOL_2D", "RESIZE_BILINEAR"}:
                base = f"detector.layers.{position}"
                overrides[f"{base}.act_in"] = no_quant()
                overrides[f"{base}.act_out"] = activation_spec
            elif name == "CONCATENATION":
                overrides[f"detector.layers.{position}.act_out"] = activation_spec
    return overrides


def validate_legal_precision_contract(
    model: nn.Module,
    regions: Sequence[PrecisionRegion],
    precision_map: Mapping[str, Precision],
) -> dict[str, Any]:
    """Verify that every data operator and parameter obeys one precision domain."""
    if not isinstance(model, LegalMixedPrecisionAdapter):
        raise TypeError("Expected LegalMixedPrecisionAdapter after WrapQ preparation.")
    violations: list[str] = []
    checked_operations = 0
    parameter_sites = 0

    for region in regions:
        precision = precision_map[region.name]
        for position in region.operation_positions:
            operation = model.detector.operations[position]
            name = str(operation["name"])
            layer = model.detector.layers[position]
            checked_operations += 1
            if name in {"CONV_2D", "DEPTHWISE_CONV_2D"}:
                if not isinstance(layer, ConvNode):
                    violations.append(
                        f"{name}@{position} expected ConvNode, found "
                        f"{type(layer).__name__}."
                    )
                    continue
                conv = _unwrap_ptq_wrapper(layer.conv, QuantConv2d)
                if conv is None:
                    violations.append(
                        f"{name}@{position} expected QuantConv2d, found "
                        f"{_wrapped_type_chain(layer.conv)}."
                    )
                    continue
                _require_identity(
                    conv.obs_act_in,
                    f"{name}@{position}.act_in",
                    violations,
                )
                _require_activation_domain(
                    conv.obs_act_out,
                    precision,
                    f"{name}@{position}.act_out",
                    violations,
                )
                _require_parameter_domain(
                    conv.obs_weight,
                    precision,
                    int(conv.module.weight.shape[0]),
                    f"{name}@{position}.weight",
                    violations,
                )
                parameter_sites += 1
            elif name == "PRELU":
                prelu = _unwrap_ptq_wrapper(layer, QuantPReLU)
                if prelu is None:
                    violations.append(
                        f"PRELU@{position} expected QuantPReLU, found "
                        f"{_wrapped_type_chain(layer)}."
                    )
                    continue
                _require_identity(
                    prelu.obs_act_in,
                    f"PRELU@{position}.act_in",
                    violations,
                )
                _require_activation_domain(
                    prelu.obs_act_out,
                    precision,
                    f"PRELU@{position}.act_out",
                    violations,
                )
                _require_parameter_domain(
                    prelu.obs_weight,
                    precision,
                    int(prelu.module.weight.numel()),
                    f"PRELU@{position}.weight",
                    violations,
                )
                parameter_sites += 1
            elif name == "MAX_POOL_2D":
                max_pool = _unwrap_ptq_wrapper(layer, QuantMaxPool2d)
                if max_pool is None:
                    violations.append(
                        f"MAX_POOL_2D@{position} expected QuantMaxPool2d, found "
                        f"{_wrapped_type_chain(layer)}."
                    )
                    continue
                _require_identity(
                    max_pool.obs_act_in,
                    f"MAX_POOL_2D@{position}.act_in",
                    violations,
                )
                _require_activation_domain(
                    max_pool.obs_act_out,
                    precision,
                    f"MAX_POOL_2D@{position}.act_out",
                    violations,
                )
            elif name == "RESIZE_BILINEAR":
                resize = _unwrap_ptq_wrapper(layer, QuantResizeBilinear2d)
                if resize is None:
                    violations.append(
                        f"RESIZE_BILINEAR@{position} expected "
                        "QuantResizeBilinear2d, found "
                        f"{_wrapped_type_chain(layer)}."
                    )
                    continue
                _require_identity(
                    resize.obs_act_in,
                    f"RESIZE_BILINEAR@{position}.act_in",
                    violations,
                )
                _require_activation_domain(
                    resize.obs_act_out,
                    precision,
                    f"RESIZE_BILINEAR@{position}.act_out",
                    violations,
                )
            elif name == "CONCATENATION":
                concat = _unwrap_ptq_wrapper(layer, QuantConcat)
                if concat is None:
                    violations.append(
                        f"CONCATENATION@{position} expected QuantConcat, found "
                        f"{_wrapped_type_chain(layer)}."
                    )
                    continue
                _require_activation_domain(
                    concat.obs_act_out,
                    precision,
                    f"CONCATENATION@{position}.act_out",
                    violations,
                )

    input_quantizer = _unwrap_ptq_wrapper(
        model.input_quantizer,
        QuantStubWrapper,
    )
    if input_quantizer is None:
        violations.append(
            "Graph input boundary expected QuantStubWrapper, found "
            f"{_wrapped_type_chain(model.input_quantizer)}."
        )
    else:
        _require_activation_domain(
            input_quantizer.obs_act_out,
            precision_map[model.input_region_name],
            "graph_input",
            violations,
        )

    for (region_name, tensor_id), key in model._boundary_keys.items():
        module = model.boundary_quantizers[key]
        quantizer = _unwrap_ptq_wrapper(module, QuantStubWrapper)
        if quantizer is None:
            violations.append(
                f"Boundary {key} expected QuantStubWrapper, found "
                f"{_wrapped_type_chain(module)}."
            )
            continue
        _require_activation_domain(
            quantizer.obs_act_out,
            precision_map[region_name],
            f"boundary[{region_name},{tensor_id}]",
            violations,
        )
    for position, key in model._add_output_keys.items():
        module = model.add_output_quantizers[key]
        region_name = _region_name_for_position(regions, position)
        quantizer = _unwrap_ptq_wrapper(module, QuantStubWrapper)
        if quantizer is None:
            violations.append(
                f"ADD output {key} expected QuantStubWrapper, found "
                f"{_wrapped_type_chain(module)}."
            )
            continue
        _require_activation_domain(
            quantizer.obs_act_out,
            precision_map[region_name],
            f"ADD@{position}.output",
            violations,
        )

    expected_boundaries = {
        (region.name, tensor_id)
        for region in regions
        if not _region_uses_shared_concat_domain(region)
        for tensor_id in region.external_inputs
        if model._requires_explicit_boundary(region, tensor_id)
    }
    actual_boundaries = set(model._boundary_keys)
    if actual_boundaries != expected_boundaries:
        violations.append(
            "Explicit dtype-boundary set differs from the semantic assignment: "
            f"missing={sorted(expected_boundaries - actual_boundaries)}, "
            f"extra={sorted(actual_boundaries - expected_boundaries)}."
        )

    transitions = precision_transition_edges(
        model.detector,
        regions,
        precision_map,
    )
    if violations:
        raise RuntimeError(
            "Illegal mixed-precision candidate:\n  - " + "\n  - ".join(violations)
        )
    return {
        "status": "pass",
        "checked_operation_count": checked_operations,
        "parameter_site_count": parameter_sites,
        "graph_input_precision": precision_map[model.input_region_name].value,
        "explicit_boundary_count": len(model._boundary_keys),
        "add_output_quantizer_count": len(model._add_output_keys),
        "dtype_transition_count": len(transitions),
        "dtype_transitions": transitions,
    }


def precision_transition_edges(
    detector: HandDetector,
    regions: Sequence[PrecisionRegion],
    precision_map: Mapping[str, Precision],
) -> list[dict[str, Any]]:
    """List every inter-region edge whose source and target dtypes differ."""
    producer_by_tensor = _producer_positions(detector.operations)
    region_by_position = {
        position: region.name
        for region in regions
        for position in region.operation_positions
    }
    transitions: list[dict[str, Any]] = []
    for region in regions:
        for tensor_id in region.external_inputs:
            producer = producer_by_tensor.get(tensor_id)
            if producer is None:
                continue
            source_region = region_by_position[producer]
            source_precision = precision_map[source_region]
            target_precision = precision_map[region.name]
            if source_precision == target_precision:
                continue
            transitions.append(
                {
                    "tensor_id": tensor_id,
                    "source_region": source_region,
                    "target_region": region.name,
                    "source_precision": source_precision.value,
                    "target_precision": target_precision.value,
                }
            )
    return transitions


def collect_region_cost_metadata(
    float_model: nn.Module,
    regions: Sequence[PrecisionRegion],
    sample: torch.Tensor,
) -> dict[str, Any]:
    """Measure parameter and activation elements owned by each region."""
    detector = _find_detector(float_model)
    with torch.inference_mode():
        values = detector.forward_nhwc_values(sample)
    parameter_elements: dict[str, int] = {}
    activation_elements: dict[str, int] = {}
    for region in regions:
        parameters = 0
        activations = 0
        for position in region.operation_positions:
            layer = detector.layers[position]
            operation = detector.operations[position]
            if isinstance(layer, ConvNode):
                parameters += int(layer.conv.weight.numel())
            elif isinstance(layer, nn.PReLU):
                parameters += int(layer.weight.numel())
            activations += sum(
                int(values[int(tensor_id)].numel())
                for tensor_id in operation["outputs"]
            )
        parameter_elements[region.name] = parameters
        activation_elements[region.name] = activations
    return {
        "parameter_elements": parameter_elements,
        "activation_elements": activation_elements,
        "total_parameter_elements": sum(parameter_elements.values()),
        "total_activation_elements": sum(activation_elements.values()),
    }


def summarize_assignment_cost(
    detector: HandDetector,
    regions: Sequence[PrecisionRegion],
    precision_map: Mapping[str, Precision],
    cost_metadata: Mapping[str, Any],
    cost_weights: PrecisionCostWeights,
) -> dict[str, Any]:
    """Summarize bytes, INT16 fractions, boundaries, and normalized cost."""
    cost_weights.validate()
    parameter_elements = cast(Mapping[str, int], cost_metadata["parameter_elements"])
    activation_elements = cast(Mapping[str, int], cost_metadata["activation_elements"])
    total_parameters = int(cost_metadata["total_parameter_elements"])
    total_activations = int(cost_metadata["total_activation_elements"])
    int16_parameters = sum(
        parameter_elements[region.name]
        for region in regions
        if precision_map[region.name] is Precision.INT16
    )
    int16_activations = sum(
        activation_elements[region.name]
        for region in regions
        if precision_map[region.name] is Precision.INT16
    )
    parameter_bytes = sum(
        parameter_elements[region.name] * precision_map[region.name].bytes_per_element
        for region in regions
    )
    activation_bytes = sum(
        activation_elements[region.name] * precision_map[region.name].bytes_per_element
        for region in regions
    )
    transitions = precision_transition_edges(detector, regions, precision_map)
    inter_region_edges = sum(
        1
        for region in regions
        for tensor_id in region.external_inputs
        if _producer_positions(detector.operations).get(tensor_id) is not None
    )
    parameter_fraction = (
        int16_parameters / total_parameters if total_parameters else 0.0
    )
    activation_fraction = (
        int16_activations / total_activations if total_activations else 0.0
    )
    boundary_fraction = (
        len(transitions) / inter_region_edges if inter_region_edges else 0.0
    )
    normalized_cost = (
        cost_weights.parameter * parameter_fraction
        + cost_weights.activation * activation_fraction
        + cost_weights.boundary * boundary_fraction
    )
    return {
        "parameter_bytes": parameter_bytes,
        "activation_bytes": activation_bytes,
        "int16_parameter_elements": int16_parameters,
        "uint8_parameter_elements": total_parameters - int16_parameters,
        "int16_activation_elements": int16_activations,
        "uint8_activation_elements": total_activations - int16_activations,
        "int16_parameter_fraction": parameter_fraction,
        "int16_activation_fraction": activation_fraction,
        "dtype_transition_count": len(transitions),
        "inter_region_edge_count": inter_region_edges,
        "boundary_fraction": boundary_fraction,
        "normalized_cost": normalized_cost,
    }


def run_legal_mixed_precision_search(
    float_model: nn.Module,
    calibration_samples: Sequence[torch.Tensor],
    evaluation_samples: Sequence[torch.Tensor],
    *,
    uint8_percentile: float,
    int16_observer: str,
    int16_percentile: float,
    max_samples: int,
    samples_per_batch: int,
    sampling_seed: int,
    regressor_output_precision: Precision,
    classifier_output_precision: Precision,
    target_regressor_mae: float,
    target_classifier_mae: float,
    search: str,
    beam_width: int,
    candidate_count: int,
    max_search_steps: int,
    skip_sensitivity: bool,
    search_even_if_entry_infeasible: bool,
    cost_weights: PrecisionCostWeights,
    output_adapter: OutputAdapter,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Evaluate legal floors, reverse sensitivity, and constrained search."""
    if not calibration_samples or not evaluation_samples:
        raise ValueError(
            "Legal mixed precision requires calibration and evaluation data."
        )
    if search not in {"none", "reverse-greedy", "reverse-beam"}:
        raise ValueError(f"Unsupported search strategy: {search!r}.")
    if beam_width <= 0:
        raise ValueError("beam_width must be positive.")
    if candidate_count < 0 or max_search_steps < 0:
        raise ValueError("Candidate and step limits must be nonnegative.")
    for name, value in (
        ("target_regressor_mae", target_regressor_mae),
        ("target_classifier_mae", target_classifier_mae),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
    cost_weights.validate()

    regions = build_precision_regions(
        float_model,
        regressor_output_precision=regressor_output_precision,
        classifier_output_precision=classifier_output_precision,
    )
    policies = build_observer_policies(
        uint8_percentile=uint8_percentile,
        int16_observer=int16_observer,
        int16_percentile=int16_percentile,
        max_samples=max_samples,
        samples_per_batch=samples_per_batch,
        sampling_seed=sampling_seed,
    )
    cost_metadata = collect_region_cost_metadata(
        float_model,
        regions,
        calibration_samples[0],
    )
    detector = _find_detector(float_model)
    variable_regions = tuple(region.name for region in regions if not region.is_fixed)
    cache: dict[frozenset[str], dict[str, Any]] = {}
    evaluation_index = 0

    def evaluate(uint8_regions: frozenset[str]) -> dict[str, Any]:
        nonlocal evaluation_index
        cached = cache.get(uint8_regions)
        if cached is not None:
            return cached
        evaluation_index += 1
        precision_map = make_precision_map(regions, uint8_regions)
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "assignment_start",
                    "index": evaluation_index,
                    "uint8_regions": sorted(uint8_regions),
                }
            )
        candidate, build_metadata = build_legal_candidate(
            float_model,
            calibration_samples,
            regions=regions,
            precision_map=precision_map,
            policies=policies,
        )
        outputs = evaluate_models(
            float_model,
            candidate,
            evaluation_samples,
            output_adapter=output_adapter,
        )
        cost = summarize_assignment_cost(
            detector,
            regions,
            precision_map,
            cost_metadata,
            cost_weights,
        )
        result = {
            "uint8_regions": sorted(uint8_regions),
            "int16_regions": sorted(
                name
                for name, precision in precision_map.items()
                if precision is Precision.INT16
            ),
            "precision_map": {
                name: precision.value for name, precision in precision_map.items()
            },
            "outputs": _copy_outputs(outputs),
            "cost": cost,
            "contract": build_metadata["contract"],
            "override_count": build_metadata["override_count"],
            "meets_targets": _meets_targets(
                outputs,
                target_regressor_mae,
                target_classifier_mae,
            ),
        }
        cache[uint8_regions] = result
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "assignment_finish",
                    "index": evaluation_index,
                    "uint8_region_count": len(uint8_regions),
                    "regressor_mae": _reg_mae(result),
                    "classifier_mae": _cls_mae(result),
                    "meets_targets": bool(result["meets_targets"]),
                    "normalized_cost": float(cost["normalized_cost"]),
                    "dtype_transition_count": int(cost["dtype_transition_count"]),
                }
            )
        del candidate
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result

    all_int16 = evaluate(frozenset())
    all_uint8 = evaluate(frozenset(variable_regions))

    sensitivity: list[dict[str, Any]] = []
    if not skip_sensitivity:
        for region_name in variable_regions:
            candidate = evaluate(frozenset((region_name,)))
            sensitivity.append(_sensitivity_entry(all_int16, candidate, region_name))
        sensitivity.sort(
            key=lambda item: (
                not bool(item["meets_targets"]),
                float(item["regressor_mae_cost"]),
                -float(item["normalized_cost_saving"]),
                str(item["region"]),
            )
        )

    candidate_regions = (
        tuple(str(entry["region"]) for entry in sensitivity)
        if sensitivity
        else variable_regions
    )
    if candidate_count:
        candidate_regions = candidate_regions[:candidate_count]

    search_result: dict[str, Any]
    if search == "none":
        search_result = {
            "strategy": "none",
            "status": "not_run",
            "best": all_int16 if all_int16["meets_targets"] else None,
            "steps": [],
        }
    elif not all_int16["meets_targets"] and not search_even_if_entry_infeasible:
        search_result = {
            "strategy": search,
            "status": "entry_infeasible",
            "reason": (
                "The legal all-INT16 entry does not meet the requested accuracy "
                "targets; reverse demotion search was skipped."
            ),
            "best": None,
            "steps": [],
        }
    elif search == "reverse-greedy":
        search_result = _run_reverse_greedy(
            evaluate,
            candidate_regions,
            all_int16,
            target_regressor_mae,
            target_classifier_mae,
            max_search_steps,
        )
    else:
        search_result = _run_reverse_beam(
            evaluate,
            candidate_regions,
            all_int16,
            target_regressor_mae,
            target_classifier_mae,
            beam_width,
            max_search_steps,
        )

    selected = search_result.get("best")
    if selected is None:
        selected = min(
            (all_int16, all_uint8),
            key=lambda result: (
                _reg_mae(result),
                _cls_mae(result),
            ),
        )
    return {
        "analysis": "legal_mixed_precision_search",
        "metadata": {
            "granularity": "semantic",
            "legal_domains": ["uint8", "int16"],
            "operator_contract": (
                "Every data operator uses one input/output dtype; Conv, "
                "DepthwiseConv, and PReLU parameters use the same dtype."
            ),
            "boundary_model": (
                "A QuantStub is inserted for inter-region dtype transitions; "
                "Concat uses its shared input/output observer."
            ),
            "regressor_output_precision": regressor_output_precision.value,
            "classifier_output_precision": classifier_output_precision.value,
            "target_regressor_mae": target_regressor_mae,
            "target_classifier_mae": target_classifier_mae,
            "search": search,
            "beam_width": beam_width,
            "candidate_count": candidate_count,
            "max_search_steps": max_search_steps,
            "cost_weights": cost_weights.to_dict(),
            "evaluated_assignment_count": len(cache),
            **dict(policies.metadata),
        },
        "regions": [region.to_dict() for region in regions],
        "cost_metadata": dict(cost_metadata),
        "floors": {
            "legal_all_int16_internal": all_int16,
            "legal_all_uint8_internal": all_uint8,
        },
        "reverse_demotion_sensitivity": sensitivity,
        "search": search_result,
        "selected_assignment": selected,
        "recommendation": _recommendation(
            all_int16,
            all_uint8,
            search_result,
            target_regressor_mae,
            target_classifier_mae,
        ),
    }


def print_legal_mixed_precision_report(report: Mapping[str, Any]) -> None:
    """Print legal floors, reverse sensitivity, and the selected assignment."""
    floors = cast(Mapping[str, Mapping[str, Any]], report["floors"])
    print("\nLegal UINT8/INT16 precision floors")
    print(
        f"{'profile':30s} {'REG_MAE':>13s} {'CLS_MAE':>13s} "
        f"{'I16_W%':>9s} {'I16_A%':>9s} {'QBOUND':>7s} {'COST':>10s}"
    )
    for name, result in floors.items():
        cost = cast(Mapping[str, Any], result["cost"])
        print(
            f"{name:30s} "
            f"{_reg_mae(result):13.6e} "
            f"{_cls_mae(result):13.6e} "
            f"{100.0 * float(cost['int16_parameter_fraction']):9.3f} "
            f"{100.0 * float(cost['int16_activation_fraction']):9.3f} "
            f"{int(cost['dtype_transition_count']):7d} "
            f"{float(cost['normalized_cost']):10.6f}"
        )

    sensitivity = cast(
        Sequence[Mapping[str, Any]], report["reverse_demotion_sensitivity"]
    )
    if sensitivity:
        print("\nIndependent INT16-to-UINT8 region demotion sensitivity")
        print(
            f"{'rank':>4s} {'region':38s} {'REG_COST':>13s} "
            f"{'CLS_COST':>13s} {'COST_SAVE':>13s} {'TARGET':>7s}"
        )
        for rank, item in enumerate(sensitivity, start=1):
            print(
                f"{rank:4d} {str(item['region'])[:38]:38s} "
                f"{float(item['regressor_mae_cost']):13.6e} "
                f"{float(item['classifier_mae_cost']):13.6e} "
                f"{float(item['normalized_cost_saving']):13.6e} "
                f"{str(bool(item['meets_targets'])):>7s}"
            )

    search = cast(Mapping[str, Any], report["search"])
    print("\nSearch")
    print(
        f"strategy={search['strategy']}, status={search['status']}, "
        f"steps={len(cast(Sequence[Any], search.get('steps', [])))}"
    )
    selected = cast(Mapping[str, Any], report["selected_assignment"])
    selected_cost = cast(Mapping[str, Any], selected["cost"])
    print(
        "Selected: "
        f"REG_MAE={_reg_mae(selected):.6e}, "
        f"CLS_MAE={_cls_mae(selected):.6e}, "
        f"UINT8_REGIONS={len(cast(Sequence[Any], selected['uint8_regions']))}, "
        f"QBOUND={int(selected_cost['dtype_transition_count'])}, "
        f"COST={float(selected_cost['normalized_cost']):.6f}"
    )
    print("UINT8 regions: " + " ".join(cast(Sequence[str], selected["uint8_regions"])))
    print("Recommendation: " + str(report["recommendation"]))


def _run_reverse_greedy(
    evaluate,
    candidate_regions: Sequence[str],
    entry: Mapping[str, Any],
    target_regressor_mae: float,
    target_classifier_mae: float,
    max_search_steps: int,
) -> dict[str, Any]:
    current_regions = frozenset(cast(Sequence[str], entry["uint8_regions"]))
    current = entry
    best = entry if entry["meets_targets"] else None
    steps: list[dict[str, Any]] = []
    limit = max_search_steps or len(candidate_regions)
    for step in range(1, limit + 1):
        candidates: list[Mapping[str, Any]] = []
        for region_name in candidate_regions:
            if region_name in current_regions:
                continue
            result = evaluate(current_regions.union((region_name,)))
            if not _meets_targets(
                cast(MetricSummary, result["outputs"]),
                target_regressor_mae,
                target_classifier_mae,
            ):
                continue
            candidates.append(result)
        if not candidates:
            return {
                "strategy": "reverse-greedy",
                "status": "stopped",
                "stop_reason": "no_target_feasible_demotion",
                "best": best,
                "steps": steps,
            }
        chosen = min(candidates, key=_deployment_rank)
        chosen_regions = frozenset(cast(Sequence[str], chosen["uint8_regions"]))
        added = sorted(chosen_regions.difference(current_regions))
        steps.append(
            {
                "step": step,
                "added_region": added[0] if len(added) == 1 else added,
                "candidate_count": len(candidates),
                "result": chosen,
            }
        )
        current_regions = chosen_regions
        current = chosen
        if best is None or _deployment_rank(chosen) < _deployment_rank(best):
            best = chosen
    return {
        "strategy": "reverse-greedy",
        "status": "max_steps_reached",
        "best": best,
        "steps": steps,
    }


def _run_reverse_beam(
    evaluate,
    candidate_regions: Sequence[str],
    entry: Mapping[str, Any],
    target_regressor_mae: float,
    target_classifier_mae: float,
    beam_width: int,
    max_search_steps: int,
) -> dict[str, Any]:
    beam = [entry]
    best = entry if entry["meets_targets"] else None
    steps: list[dict[str, Any]] = []
    seen = {frozenset(cast(Sequence[str], entry["uint8_regions"]))}
    limit = max_search_steps or len(candidate_regions)
    for depth in range(1, limit + 1):
        expanded: list[Mapping[str, Any]] = []
        for state in beam:
            selected = frozenset(cast(Sequence[str], state["uint8_regions"]))
            for region_name in candidate_regions:
                if region_name in selected:
                    continue
                next_regions = selected.union((region_name,))
                if next_regions in seen:
                    continue
                seen.add(next_regions)
                result = evaluate(next_regions)
                if not _meets_targets(
                    cast(MetricSummary, result["outputs"]),
                    target_regressor_mae,
                    target_classifier_mae,
                ):
                    continue
                expanded.append(result)
        if not expanded:
            return {
                "strategy": "reverse-beam",
                "status": "stopped",
                "stop_reason": "no_target_feasible_demotion",
                "best": best,
                "steps": steps,
            }
        expanded.sort(key=_deployment_rank)
        beam = expanded[:beam_width]
        for state in beam:
            if best is None or _deployment_rank(state) < _deployment_rank(best):
                best = state
        steps.append(
            {
                "depth": depth,
                "expanded_feasible_count": len(expanded),
                "beam": beam,
            }
        )
    return {
        "strategy": "reverse-beam",
        "status": "max_steps_reached",
        "best": best,
        "steps": steps,
    }


def _sensitivity_entry(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    region_name: str,
) -> dict[str, Any]:
    baseline_cost = cast(Mapping[str, Any], baseline["cost"])
    candidate_cost = cast(Mapping[str, Any], candidate["cost"])
    return {
        "region": region_name,
        "regressor_mae": _reg_mae(candidate),
        "classifier_mae": _cls_mae(candidate),
        "regressor_mae_cost": _reg_mae(candidate) - _reg_mae(baseline),
        "classifier_mae_cost": _cls_mae(candidate) - _cls_mae(baseline),
        "normalized_cost_saving": (
            float(baseline_cost["normalized_cost"])
            - float(candidate_cost["normalized_cost"])
        ),
        "parameter_byte_saving": (
            int(baseline_cost["parameter_bytes"])
            - int(candidate_cost["parameter_bytes"])
        ),
        "activation_byte_saving": (
            int(baseline_cost["activation_bytes"])
            - int(candidate_cost["activation_bytes"])
        ),
        "dtype_transition_delta": (
            int(candidate_cost["dtype_transition_count"])
            - int(baseline_cost["dtype_transition_count"])
        ),
        "meets_targets": bool(candidate["meets_targets"]),
        "result": candidate,
    }


def _recommendation(
    all_int16: Mapping[str, Any],
    all_uint8: Mapping[str, Any],
    search: Mapping[str, Any],
    target_regressor_mae: float,
    target_classifier_mae: float,
) -> str:
    if not all_int16["meets_targets"]:
        return (
            "The legal all-INT16 profile misses the requested targets; fix "
            "calibration or numerical parity before demoting any region."
        )
    best = search.get("best")
    if best is not None and cast(Sequence[str], best["uint8_regions"]):
        return (
            "Use the selected legal precision map as the fixed topology, then "
            "recalibrate and optimize only its UINT8 regions before export."
        )
    if all_uint8["meets_targets"]:
        return "The legal all-UINT8 internal profile already meets both targets."
    return (
        f"Keep the all-INT16 legal floor for now; no UINT8 demotion preserved "
        f"REG<={target_regressor_mae:g} and CLS<={target_classifier_mae:g}."
    )


def _deployment_rank(result: Mapping[str, Any]) -> tuple[float, float, float, int]:
    cost = cast(Mapping[str, Any], result["cost"])
    return (
        float(cost["normalized_cost"]),
        _reg_mae(result),
        _cls_mae(result),
        -len(cast(Sequence[str], result["uint8_regions"])),
    )


def _meets_targets(
    outputs: MetricSummary,
    target_regressor_mae: float,
    target_classifier_mae: float,
) -> bool:
    return (
        _metric(outputs, "regressors", "mae") <= target_regressor_mae
        and _metric(outputs, "classifiers", "mae") <= target_classifier_mae
    )


def _reg_mae(result: Mapping[str, Any]) -> float:
    return _metric(cast(MetricSummary, result["outputs"]), "regressors", "mae")


def _cls_mae(result: Mapping[str, Any]) -> float:
    return _metric(cast(MetricSummary, result["outputs"]), "classifiers", "mae")


def _copy_outputs(outputs: MetricSummary) -> dict[str, dict[str, Any]]:
    return {name: dict(metrics) for name, metrics in outputs.items()}


def _metric(outputs: MetricSummary, output_name: str, metric_name: str) -> float:
    value = outputs[output_name][metric_name]
    if not isinstance(value, (int, float)):
        raise TypeError(
            f"Output {output_name!r} metric {metric_name!r} is not numeric."
        )
    return float(value)


def _calibrate(model: nn.Module, samples: Sequence[torch.Tensor]) -> None:
    model.eval()
    with torch.inference_mode():
        for sample in samples:
            model(sample)


def _require_identity(
    observer: ObserverBase,
    context: str,
    violations: list[str],
) -> None:
    if not isinstance(observer, IdentityObserver):
        violations.append(f"{context} must inherit its already-quantized input.")


def _require_dtype(
    observer: ObserverBase,
    expected_dtype: DType,
    context: str,
    violations: list[str],
) -> None:
    actual = getattr(observer, "dtype", None)
    if actual != expected_dtype:
        violations.append(f"{context} expected {expected_dtype}, found {actual}.")


def _require_activation_domain(
    observer: ObserverBase,
    precision: Precision,
    context: str,
    violations: list[str],
) -> None:
    """Require one per-tensor activation observer for a legal precision."""
    _require_dtype(observer, precision.dtype, context, violations)
    expected_qscheme = (
        QScheme.PER_TENSOR_ASYMM
        if precision is Precision.UINT8
        else QScheme.PER_TENSOR_SYMM
    )
    if observer.qscheme is not expected_qscheme:
        violations.append(
            f"{context} expected {expected_qscheme}, found {observer.qscheme}."
        )
    qparams = observer.compute_qparams()
    if qparams is None:
        violations.append(f"{context} does not expose affine qparams.")
        return
    scale, zero_point = qparams
    if scale.numel() != 1 or zero_point.numel() != 1:
        violations.append(
            f"{context} must have one per-tensor qparam, found "
            f"scale={scale.numel()}, zero_point={zero_point.numel()}."
        )
    if precision is Precision.INT16 and torch.any(zero_point != 0):
        violations.append(f"{context} INT16 zero point must be 0.")


def _require_parameter_domain(
    observer: ObserverBase,
    precision: Precision,
    expected_channels: int,
    context: str,
    violations: list[str],
) -> None:
    """Require a per-output-channel parameter observer of the same dtype."""
    _require_dtype(observer, precision.dtype, context, violations)
    expected_qscheme = (
        QScheme.PER_CHANNEL_ASYMM
        if precision is Precision.UINT8
        else QScheme.PER_CHANNEL_SYMM
    )
    if observer.qscheme is not expected_qscheme:
        violations.append(
            f"{context} expected {expected_qscheme}, found {observer.qscheme}."
        )
    if observer.channel_axis != 0:
        violations.append(
            f"{context} expected channel axis 0, found {observer.channel_axis}."
        )
    qparams = observer.compute_qparams()
    if qparams is None:
        violations.append(f"{context} does not expose affine qparams.")
        return
    scale, zero_point = qparams
    if scale.numel() != expected_channels or zero_point.numel() != expected_channels:
        violations.append(
            f"{context} expected {expected_channels} channel qparams, found "
            f"scale={scale.numel()}, zero_point={zero_point.numel()}."
        )
    if precision is Precision.INT16 and torch.any(zero_point != 0):
        violations.append(f"{context} INT16 zero points must all be 0.")


def _validate_percentile(name: str, value: float) -> None:
    if not math.isfinite(value) or not 0.0 < value <= 100.0:
        raise ValueError(f"{name} must be finite and in (0, 100].")


def _find_detector(model: nn.Module) -> HandDetector:
    if isinstance(model, NHWCInputAdapter):
        return model.detector
    if isinstance(model, HandDetector):
        return model
    if isinstance(model, LegalMixedPrecisionAdapter):
        return model.detector
    detector = getattr(model, "detector", None)
    if isinstance(detector, HandDetector):
        return detector
    raise TypeError("Expected HandDetector, NHWCInputAdapter, or mixed adapter.")


def _producer_positions(
    operations: Sequence[Mapping[str, Any]],
) -> dict[int, int]:
    result: dict[int, int] = {}
    for position, operation in enumerate(operations):
        for output in operation["outputs"]:
            tensor_id = int(output)
            if tensor_id in result:
                raise RuntimeError(f"Tensor {tensor_id} has multiple producers.")
            result[tensor_id] = position
    return result


def _trace_head_branch(
    tensor_id: int,
    operations: Sequence[Mapping[str, Any]],
    producers: Mapping[int, int],
) -> tuple[int, tuple[int, ...]]:
    positions: list[int] = []
    current_tensor = tensor_id
    while True:
        position = producers.get(current_tensor)
        if position is None:
            raise RuntimeError(f"No operation produces head tensor {current_tensor}.")
        operation = operations[position]
        positions.append(position)
        if operation["name"] != "RESHAPE":
            break
        current_tensor = int(operation["inputs"][0])
    if operations[position]["name"] not in {"CONV_2D", "DEPTHWISE_CONV_2D"}:
        raise RuntimeError(
            "Expected each output branch to originate from a convolution, but "
            f"found {operations[position]['name']!r}."
        )
    return position, tuple(sorted(positions))


def _resolution_name(index: int, count: int) -> str:
    if count == 1:
        return "single_resolution"
    if count == 2:
        return "low_resolution" if index == 0 else "high_resolution"
    return f"resolution_{index:02d}"


def _runtime_input_tensors(operation: Mapping[str, Any]) -> tuple[int, ...]:
    inputs = tuple(int(value) for value in operation["inputs"])
    name = str(operation["name"])
    if name == "CONCATENATION":
        return inputs
    if name == "ADD":
        return inputs[:2]
    return inputs[:1]


def _region_uses_shared_concat_domain(region: PrecisionRegion) -> bool:
    return region.operation_names == ("CONCATENATION",)


def _region_name_for_position(
    regions: Sequence[PrecisionRegion],
    position: int,
) -> str:
    for region in regions:
        if position in region.operation_positions:
            return region.name
    raise KeyError(f"Operation position {position} is not assigned to a region.")


def _boundary_key(region_name: str, tensor_id: int) -> str:
    safe_name = "".join(
        character if character.isalnum() or character == "_" else "_"
        for character in region_name
    )
    return f"{safe_name}__t{tensor_id}"
