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

"""Reusable numerical analysis for post-training quantization."""

from tico.quantization.analysis.ablation import QuantizationAblation
from tico.quantization.analysis.clipping import (
    build_clipping_candidates,
    ClippingCandidate,
    collect_output_calibration_data,
    evaluate_clipping_candidates,
    EvaluatedClippingCandidate,
    OutputCalibrationData,
)
from tico.quantization.analysis.inputs import ModelInput, ModelInvocation
from tico.quantization.analysis.metrics import evaluate_models, TensorErrorMetrics
from tico.quantization.analysis.output_quantization import (
    AffineQParams,
    AffineQuantizationPolicy,
    calculate_qparams,
    OutputCodeStatistics,
    OutputTensorQuantizer,
)
from tico.quantization.analysis.outputs import (
    make_output_adapter,
    normalize_outputs,
    OutputAdapter,
)
from tico.quantization.analysis.profile import QuantizationProfile
from tico.quantization.analysis.report import (
    QuantizationProfileResult,
    QuantizationReport,
)
from tico.quantization.analysis.selector import QuantizationBoundaries, SiteSelector
from tico.quantization.analysis.sensitivity import (
    QuantizationGroup,
    QuantizationSensitivity,
    SensitivityMode,
    SensitivityResult,
)

__all__ = [
    "AffineQParams",
    "AffineQuantizationPolicy",
    "ClippingCandidate",
    "EvaluatedClippingCandidate",
    "ModelInput",
    "ModelInvocation",
    "OutputAdapter",
    "OutputCalibrationData",
    "OutputCodeStatistics",
    "OutputTensorQuantizer",
    "QuantizationAblation",
    "QuantizationBoundaries",
    "QuantizationGroup",
    "QuantizationProfile",
    "QuantizationProfileResult",
    "QuantizationReport",
    "QuantizationSensitivity",
    "SensitivityMode",
    "SensitivityResult",
    "SiteSelector",
    "TensorErrorMetrics",
    "build_clipping_candidates",
    "calculate_qparams",
    "collect_output_calibration_data",
    "evaluate_clipping_candidates",
    "evaluate_models",
    "make_output_adapter",
    "normalize_outputs",
]
