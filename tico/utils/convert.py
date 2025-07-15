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

import operator
import os
from typing import Any, Dict, Optional, Tuple

import torch
from torch.export import export, ExportedProgram

from tico.config import CompileConfigBase, get_default_config
from tico.experimental.quantization.passes.fold_quant_ops import FoldQuantOps
from tico.experimental.quantization.passes.insert_quantize_on_dtype_mismatch import (
    InsertQuantizeOnDtypeMismatch,
)
from tico.experimental.quantization.passes.propagate_qparam_backward import (
    PropagateQParamBackward,
)
from tico.experimental.quantization.passes.propagate_qparam_forward import (
    PropagateQParamForward,
)
from tico.experimental.quantization.passes.quantize_bias import QuantizeBias
from tico.experimental.quantization.passes.remove_weight_dequant_op import (
    RemoveWeightDequantOp,
)
from tico.passes.fill_meta_val import FillMetaVal
from tico.passes.match_attention_pattern import MatchAttentionPattern
from tico.serialize.circle_serializer import build_circle
from tico.serialize.operators.node_visitor import get_support_targets
from tico.utils import logging
from tico.utils.canonicalize import canonicalize
from tico.utils.errors import NotYetSupportedError
from tico.utils.model import CircleModel
from tico.utils.passes import PassManager
from tico.utils.utils import has_quantization_ops, SuppressWarning


def check_unsupported_target(exported_program: ExportedProgram):
    logger = logging.getLogger(__name__)

    supported_target = list(get_support_targets())
    # Ignore `getitem` since it is no-op for multiple outputs.
    supported_target.append(operator.getitem)
    unsupported = []
    for n in exported_program.graph.nodes:
        if n.op != "call_function":
            continue
        if not n.target in supported_target:
            unsupported.append(n)

    if unsupported:
        for node in unsupported:
            logger.error(
                f"NOT SUPPORTED OPERATOR\n\t(op) {node.target.__name__}\n\t(trace) {node.meta.get('stack_trace')}"
            )
        raise NotYetSupportedError("NOT SUPPORTED OPERATOR IN GRAPH MODULE")


def check_training_ops(exported_program: ExportedProgram):
    TRAINING_OPS = {
        torch.ops.aten.dropout.default,
        torch.ops.aten.native_dropout.default,
    }
    found = set()
    for node in exported_program.graph.nodes:
        if node.op == "call_function" and node.target in TRAINING_OPS:
            found.add(node.target)

    if found:
        raise RuntimeError(
            f"Detected training-mode ops {found}. Call `model.eval()` before export."
        )


def convert_exported_module_to_circle(
    exported_program: ExportedProgram,
    config: Optional[CompileConfigBase] = None,
) -> bytes:
    if not config:
        config = get_default_config()

    assert isinstance(config, CompileConfigBase)

    exported_program = canonicalize(exported_program, config)

    pattern_match = PassManager(
        passes=[
            MatchAttentionPattern(config),
            FillMetaVal(),
        ]
    )
    pattern_match.run(exported_program)

    # TODO Give an option to enable quantiztion to user
    enable_quantization = has_quantization_ops(exported_program.graph)
    if enable_quantization:
        quantize_graph = PassManager(
            passes=[
                FoldQuantOps(),
                RemoveWeightDequantOp(),
                PropagateQParamForward(),
                PropagateQParamBackward(),
                QuantizeBias(),
                InsertQuantizeOnDtypeMismatch(),
            ]
        )
        quantize_graph.run(exported_program)

    check_unsupported_target(exported_program)
    check_training_ops(exported_program)
    circle_program = build_circle(exported_program)

    return circle_program


def convert(
    mod: torch.nn.Module,
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]] = None,
    strict: bool = True,
    config: CompileConfigBase = get_default_config(),
) -> CircleModel:
    if hasattr(mod, "training") and mod.training:
        logger = logging.getLogger(__name__)
        logger.fatal(
            "Your model is in TRAINING MODE. PLEASE CHECK IF YOU FORGOT `model.eval()`."
        )

    with torch.no_grad():
        exported_program = export(mod, args, kwargs, strict=strict)
    breakpoint()
    circle_binary = convert_exported_module_to_circle(exported_program, config=config)

    return CircleModel(circle_binary)


def convert_from_exported_program(
    exported_program: ExportedProgram,
    config: CompileConfigBase = get_default_config(),
) -> CircleModel:
    circle_binary = convert_exported_module_to_circle(exported_program, config=config)

    return CircleModel(circle_binary)


def convert_from_pt2(
    pt2_path: str | os.PathLike, config: CompileConfigBase = get_default_config()
) -> CircleModel:
    exported_program = torch.export.load(pt2_path)
    circle_binary = convert_exported_module_to_circle(exported_program, config=config)

    return CircleModel(circle_binary)
