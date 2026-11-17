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

"""Explicit target fake-quant policy, before quantizer provenance is folded.

The target contract is same-family activation operands/output. Weights remain
independently affine. Arithmetic runs in MXINT8 if any activation boundary is MX;
copy operations instead commute the existing output quantizer, only when its
pointwise/block partition permits it. No calibration or inferred affine grids.
"""

from __future__ import annotations

import copy
import math
import operator
from dataclasses import dataclass

import torch
import torch.ao.quantization.fx._decomposed  # register Q/DQ

from torch.export import ExportedProgram

from tico.quantization.passes.move_affine_quantize_before_layout import _arguments
from tico.quantization.passes.mx_qparam import remap_mx_qparam, remap_mx_split_output
from tico.quantization.passes.preserve_embedding_quantization import (
    EXPLICIT_EMBEDDING,
    FLOAT_EMBEDDING,
)
from tico.serialize.quant_param import QPARAM_KEY, QuantParam
from tico.utils.errors import NotYetSupportedError
from tico.utils.mx.dtypes import assert_supported_mx_export_options, normalize_mx_axis
from tico.utils.passes import PassBase, PassResult
from tico.utils.register_custom_op import RegisterOps
from tico.utils.utils import set_new_meta_val

RegisterOps()

_AQ = torch.ops.quantized_decomposed.quantize_per_tensor.default
_ADQ = torch.ops.quantized_decomposed.dequantize_per_tensor.default
_MQ = torch.ops.circle_custom.quantize_mx.default
_MDQ = torch.ops.circle_custom.dequantize_mx.default
_LAYOUTS = {torch.ops.aten.reshape.default, torch.ops.aten.permute.default}
_BINARY = {
    torch.ops.aten.add.Tensor,
    torch.ops.aten.sub.Tensor,
    torch.ops.aten.mul.Tensor,
    torch.ops.aten.bmm.default,
}
_UNARY = {
    torch.ops.aten.relu.default,
    torch.ops.aten.sigmoid.default,
    torch.ops.aten.rms_norm.default,
    torch.ops.circle_custom.rms_norm.default,
    torch.ops.aten.linear.default,
}
_DONE = "_mx_affine_domains_legalized"


@dataclass(frozen=True)
class _Domain:
    """Exact pre-fold quantizer arguments; never reconstruct ranges from dtype."""

    family: str
    params: tuple

    @property
    def dtype(self):
        return "mxint8" if self.family == "mx" else str(self.params[-1])[6:]

    def qparam(self):
        if self.family == "mx":
            return QuantParam(dtype="mxint8", quantized_dimension=self.params[1])
        return QuantParam(
            dtype=self.dtype, scale=[self.params[0]], zero_point=[self.params[1]]
        )


def _error(node, role, source, destination, reason):
    return NotYetSupportedError(
        f"MX/affine legalization {node.name} ({node.target}), {role}: "
        f"source={source}, required={destination}: {reason}"
    )


def _pair(dq):
    if not isinstance(dq, torch.fx.Node) or dq.target not in {_ADQ, _MDQ}:
        return None
    da = _arguments(dq)
    q = da[0]
    expected = _AQ if dq.target == _ADQ else _MQ
    if not isinstance(q, torch.fx.Node) or q.target != expected:
        return None
    qa = _arguments(q)
    params = tuple(qa[1:])
    if params != tuple(da[1 : len(qa)]):
        return None
    if expected == _AQ:
        scale, zp, lo, hi, dtype = params
        if (
            type(scale) not in (int, float)
            or not math.isfinite(scale)
            or scale <= 0
            or any(type(v) is not int for v in (zp, lo, hi))
            or not lo <= zp <= hi
            or dtype not in (torch.int16, torch.uint8)
            or dq.kwargs.get("out_dtype", torch.float32) != torch.float32
        ):
            return None
        domain = _Domain("affine", params)
    else:
        fmt, axis, method, rounding = params
        if fmt != "int8":
            return None
        try:
            assert_supported_mx_export_options(
                elem_format=fmt, shared_exp_method=method, round=rounding
            )
            axis = normalize_mx_axis(axis, qa[0].meta["val"].ndim)
        except (RuntimeError, ValueError) as exc:
            raise _error(
                dq,
                "quantization boundary",
                params,
                "MXINT8 block32/max/nearest with a valid axis",
                str(exc),
            ) from exc
        domain = _Domain("mx", (fmt, axis, method, rounding))
    return q, domain


def _unknown_domain(source):
    if not isinstance(source, torch.fx.Node):
        return "unknown/float"
    if source.target == _MDQ:
        args = _arguments(source)
        return f"unsupported MX format={args[1]}, axis={args[2]}"
    schema = getattr(source.target, "_schema", None)
    if getattr(schema, "name", None) == "quantized_decomposed::dequantize_per_channel":
        return "unsupported per-channel affine activation"
    return source.target


def _mx(axis):
    return _Domain("mx", ("int8", axis, "max", "nearest"))


def _set_operand(node, index, value):
    """Replace one schema argument, including kwargs; never replace every use."""
    if index < len(node.args):
        args = list(node.args)
        args[index] = value
        node.args = tuple(args)
    else:
        name = node.target._schema.arguments[index].name
        node.kwargs = {**node.kwargs, name: value}


def _append_pair(source, domain, before):
    graph = before.graph
    with graph.inserting_before(before):
        q = graph.call_function(
            _MQ if domain.family == "mx" else _AQ, (source, *domain.params)
        )
        dq = graph.call_function(
            _MDQ if domain.family == "mx" else _ADQ, (q, *domain.params)
        )
        if domain.family == "affine" and domain.params[2] == -32767:
            q.meta["_preserve_affine_clipping"] = True
        for new in (q, dq):
            if "nn_module_stack" in before.meta:
                new.meta["nn_module_stack"] = copy.deepcopy(
                    before.meta["nn_module_stack"]
                )
            set_new_meta_val(new)
    return dq


def _preserve_input_clipping(source):
    """Keep narrow affine saturation in the original boundary chain too."""
    while isinstance(source, torch.fx.Node):
        pair = _pair(source)
        if pair is not None:
            q, domain = pair
            if domain.family == "affine" and domain.params[2] == -32767:
                q.meta["_preserve_affine_clipping"] = True
            source = _arguments(q)[0]
        elif source.target == torch.ops.aten.cat.default:
            for operand in _arguments(source)[0]:
                _preserve_input_clipping(operand)
            return
        elif source.target in _LAYOUTS:
            source = _arguments(source)[0]
        elif source.target == operator.getitem:
            split = source.args[0]
            if (
                not isinstance(split, torch.fx.Node)
                or split.target != torch.ops.aten.split_with_sizes.default
            ):
                return
            source = _arguments(split)[0]
        else:
            return


def _supported_range(node, domain, role):
    if domain.family != "affine":
        return
    lo, hi, dtype = domain.params[2:]
    info = torch.iinfo(dtype)
    if (lo, hi) != (info.min, info.max) and not (
        dtype == torch.int16 and (lo, hi) == (-32767, 32767)
    ):
        raise _error(
            node,
            role,
            domain,
            domain,
            "unsupported affine clipping range; only full INT16/UINT8 and narrow "
            "INT16 [-32767,32767] adapters have a serialized clipping contract",
        )


def _boundaries(node):
    """Discover every supported output Q/DQ before selecting a private one."""
    pairs = []
    for q in node.users:
        for dq in q.users:
            pair = _pair(dq)
            if pair is not None and pair[0] is q:
                pairs.append((q, dq, pair[1]))
    return pairs


def _exclusive(node, q, dq):
    if set(node.users) != {q} or set(q.users) != {dq}:
        raise _error(
            node,
            "output",
            "shared raw/float output",
            "native domain",
            "output must have one private Q/DQ boundary; preserve raw consumers "
            "with an explicit independent operation boundary",
        )


def _copy_domain(domain, node, domains):
    """Reuse block-preservation proofs with temporary, restored input metadata."""
    if domain.family == "affine":
        return domain
    saved = []
    for source in node.all_input_nodes:
        if source in domains:
            saved.append((source, source.meta.get(QPARAM_KEY)))
            source.meta[QPARAM_KEY] = domains[source].qparam()
    try:
        mapped = remap_mx_qparam(domain.qparam(), node)
    except NotYetSupportedError as exc:
        raise _error(node, "copy output", domain, domain, str(exc)) from exc
    finally:
        for source, old in saved:
            if old is None:
                source.meta.pop(QPARAM_KEY, None)
            else:
                source.meta[QPARAM_KEY] = old
    assert mapped is not None
    return _mx(mapped.quantized_dimension)


def _affine_weight(node, index):
    value = _arguments(node)[index]
    if value is None:
        raise _error(
            node,
            "weight",
            "None",
            "affine weight",
            "this native signature requires an explicit affine weight boundary",
        )
    if not isinstance(value, torch.fx.Node):
        raise _error(node, "weight", value, "affine weight", "missing tensor boundary")
    # Per-channel/per-group weight lowering has its own implementation. Only
    # inspect the family here, retaining bitwidth, axis, state and sharing.
    schema = getattr(value.target, "_schema", None)
    name = getattr(schema, "name", "")
    qp = value.meta.get(QPARAM_KEY)
    if name in {
        "quantized_decomposed::dequantize_per_tensor",
        "quantized_decomposed::dequantize_per_channel",
    }:
        return
    if (
        isinstance(qp, QuantParam)
        and qp.dtype
        in {"uint4", "uint8", "int8", "int16", "short", "int32", "int", "int64"}
        and qp.scale
        and qp.zero_point
        and len(qp.scale) == len(qp.zero_point)
        and all(
            type(scale) in (int, float) and math.isfinite(scale) and scale > 0
            for scale in qp.scale
        )
        and all(type(zp) is int for zp in qp.zero_point)
    ):
        return
    raise _error(
        node,
        "weight",
        qp if qp is not None else _unknown_domain(value),
        "affine weight",
        "MX native execution requires an existing affine weight quantization "
        "boundary; configure weight calibration before export",
    )


class LegalizeMxAffineDomains(PassBase):
    """Build an executable target Q/DQ reference after #1130 and before folding.

    Existing affine and float paths are untouched. MX/affine arithmetic retains
    every source boundary, adds operand-specific MX adapters, then quantizes the
    native output before the original output boundary. CAT commutes its original
    quantizer instead: affine is pointwise, MX requires intact block partitions.
    Split/layout keep their source domain and independent result boundaries.
    Only per-tensor INT16/UINT8 (including narrow INT16) and default MXINT8 are admitted.
    """

    def call(self, exported_program: ExportedProgram) -> PassResult:
        graph = exported_program.graph
        domains = {}
        modified = False
        for node in list(graph.nodes):
            pair = _pair(node)
            if pair is not None:
                domains[node] = pair[1]
                continue
            if node.op != "call_function":
                continue
            if node.target in _LAYOUTS:
                source = _arguments(node)[0]
                if source in domains:
                    domains[node] = _copy_domain(domains[source], node, domains)
                continue
            if node.target == torch.ops.aten.split_with_sizes.default:
                source, sizes, _ = _arguments(node)
                if source in domains and domains[source].family == "mx":
                    # Validate ALL Circle outputs, including unused slots.
                    for index in range(len(sizes)):
                        remap_mx_split_output(domains[source].qparam(), node, index)
                continue
            if node.target == operator.getitem:
                split, index = node.args[:2]
                if (
                    isinstance(split, torch.fx.Node)
                    and split.target == torch.ops.aten.split_with_sizes.default
                    and type(index) is int
                ):
                    source = _arguments(split)[0]
                    if source in domains:
                        domains[node] = _copy_domain(domains[source], node, domains)
                continue
            if node.target not in _BINARY | _UNARY | {torch.ops.aten.cat.default}:
                continue
            boundaries = _boundaries(node)
            boundary = boundaries[0] if len(boundaries) == 1 else None
            args = _arguments(node)
            sources = (
                list(args[0])
                if node.target == torch.ops.aten.cat.default
                else (list(args[:2]) if node.target in _BINARY else [args[0]])
            )
            source_domains = [domains.get(source) for source in sources]
            output_domains = [pair[2] for pair in boundaries]
            has_mx = any(
                domain and domain.family == "mx"
                for domain in source_domains + output_domains
            )
            # Policy admission must inspect ALL boundaries. Ambiguous output
            # selection must not erase the fact that an MX output was requested.
            if has_mx and len(boundaries) > 1:
                raise _error(
                    node,
                    "output",
                    output_domains,
                    "a single private native output domain",
                    f"multiple output quantization boundaries ({len(boundaries)}) "
                    "share this output; cannot safely select a native output "
                    "domain while preserving every existing quantizer and consumer",
                )
            output_domain = boundary[2] if boundary else None
            if node.meta.get(_DONE):
                domains[node] = node.meta[_DONE]
                continue
            if (
                node.target == torch.ops.aten.cat.default
                and source_domains
                and source_domains[0] is not None
                and all(d == source_domains[0] for d in source_domains)
            ):
                # Keep a proven native copy plus an already-explicit output
                # conversion. A later arithmetic adapter is not a request to
                # retarget this shared CAT on a second invocation.
                try:
                    domains[node] = _copy_domain(source_domains[0], node, domains)
                except NotYetSupportedError:
                    if boundary is None or source_domains[0] == output_domain:
                        raise
                    # A private destination boundary may instead commute to
                    # the inputs and make the native CAT block-preserving.
                else:
                    continue
            if not has_mx:
                continue
            if any(node.meta.get(key) for key in (EXPLICIT_EMBEDDING, FLOAT_EMBEDDING)):
                raise _error(
                    node,
                    "activation",
                    source_domains,
                    output_domain,
                    "explicit float/embedding island must retain its boundaries",
                )
            if any(d is None for d in source_domains):
                index = source_domains.index(None)
                raise _error(
                    node,
                    f"operand[{index}] input boundary",
                    _unknown_domain(sources[index]),
                    "MXINT8 activation from a supported per-tensor boundary",
                    "missing or unsupported explicit input quantization boundary; only "
                    "per-tensor INT16/UINT8 or MXINT8 activations are supported. "
                    "Prepare and calibrate any required affine input independently",
                )
            if boundary is None:
                raise _error(
                    node,
                    "output",
                    "missing quantization boundary",
                    "native MXINT8",
                    "an explicit requested output boundary is required; "
                    "raw shared/float outputs cannot be changed implicitly",
                )
            known_domains = [d for d in source_domains if d is not None]
            q, dq, output_domain = boundary
            for index, domain in enumerate(known_domains):
                _supported_range(node, domain, f"operand[{index}]")
            if node.target == torch.ops.aten.cat.default:
                if all(d == output_domain for d in known_domains):
                    domains[node] = _copy_domain(output_domain, node, domains)
                    continue
                _exclusive(node, q, dq)
                _supported_range(node, output_domain, "output")
                # Prove the destination grouping before inserting any adapters.
                proposed = dict(domains)
                proposed.update({source: output_domain for source in sources})
                _copy_domain(output_domain, node, proposed)
                for source in sources:
                    _preserve_input_clipping(source)
                converted = []
                cache = {}
                for source, domain in zip(sources, known_domains):
                    if domain == output_domain:
                        converted.append(source)
                    else:
                        if source not in cache:
                            cache[source] = _append_pair(source, output_domain, node)
                            domains[cache[source]] = output_domain
                        converted.append(cache[source])
                _set_operand(node, 0, converted)
                dq.replace_all_uses_with(node)
                graph.erase_node(dq)
                graph.erase_node(q)
                domains[node] = output_domain
                node.meta[_DONE] = output_domain
                modified = True
                continue
            if node.target == torch.ops.aten.linear.default:
                _affine_weight(node, 1)
                bias = args[2]
                bias_schema = getattr(getattr(bias, "target", None), "_schema", None)
                if isinstance(bias, torch.fx.Node) and (
                    bias in domains
                    or getattr(bias_schema, "name", None)
                    in {
                        "quantized_decomposed::dequantize_per_tensor",
                        "quantized_decomposed::dequantize_per_channel",
                        "circle_custom::dequantize_mx",
                    }
                    or QPARAM_KEY in bias.meta
                    or bias.meta["val"].dtype != torch.float32
                ):
                    raise _error(
                        node,
                        "bias",
                        bias.meta.get(QPARAM_KEY, domains.get(bias)),
                        "float32 bias",
                        "already integer bias cannot be reinterpreted after changing activation domain",
                    )
            if node.target in {
                torch.ops.aten.rms_norm.default,
                torch.ops.circle_custom.rms_norm.default,
            }:
                _affine_weight(
                    node, 2 if node.target == torch.ops.aten.rms_norm.default else 1
                )
            # Even an unchanged native family must not silently round a raw
            # graph output or a parallel float consumer during folding.
            _exclusive(node, q, dq)
            # Same-family arithmetic permits independent grids. No scale or zero
            # point is copied between operands or from an input to an output.
            if (
                all(d.family == "mx" for d in known_domains)
                and output_domain.family == "mx"
                and (
                    node.target != torch.ops.aten.bmm.default
                    or (known_domains == [_mx(2), _mx(1)] and output_domain == _mx(2))
                )
            ):
                domains[node] = output_domain
                continue
            _supported_range(node, output_domain, "output")
            for index, (source, domain) in enumerate(zip(sources, known_domains)):
                if source.meta["val"].ndim == 0:
                    raise _error(
                        node,
                        f"operand {index}",
                        domain,
                        "MXINT8",
                        "a scalar tensor has no MX block axis",
                    )
            if output_domain.family == "affine" and output_domain.params[2] == -32767:
                q.meta["_preserve_affine_clipping"] = True
            for source in sources:
                _preserve_input_clipping(source)
            adapters = {}
            for index, (source, domain) in enumerate(zip(sources, known_domains)):
                rank = source.meta["val"].ndim
                required = (
                    _mx(2 if index == 0 else 1)
                    if node.target == torch.ops.aten.bmm.default
                    else (domain if domain.family == "mx" else _mx(rank - 1))
                )
                if domain != required:
                    key = (source, required)
                    if key not in adapters:
                        adapters[key] = _append_pair(source, required, node)
                        domains[adapters[key]] = required
                    _set_operand(node, index, adapters[key])
            native = _mx(node.meta["val"].ndim - 1)
            if (
                output_domain.family == "mx"
                and node.target != torch.ops.aten.bmm.default
            ):
                native = output_domain
            if native != output_domain:
                # Change only the existing private boundary's source, retaining
                # the externally requested output grid and all its users.
                adapted = _append_pair(node, native, q)
                _set_operand(q, 0, adapted)
            domains[node] = native
            node.meta[_DONE] = native
            modified = True
        if modified:
            graph.lint()
            exported_program.graph_module.recompile()
        return PassResult(modified)


class _TargetReferenceInterpreter(torch.fx.Interpreter):
    """Execute logical MX Q/DQ as rounded real values on the CPU."""

    def call_function(self, target, args, kwargs):
        if target == _MQ:
            return torch.ops.circle_custom.mx_fake_quantize.default(*args, **kwargs)
        if target == _MDQ:
            return args[0]
        return super().call_function(target, args, kwargs)


def run_target_reference(exported_program: ExportedProgram, *args, **kwargs):
    """Execute the *pre-fold* graph produced by this pass.

    This uses the graph's exact affine Q/DQ and the existing MX eager kernel;
    there is no second legalization or calibration implementation. Do not call
    on folded graphs: those encode a Circle ABI rather than float execution.
    Keyword inputs are flattened by the exported module's input specification.
    """
    from torch.fx._pytree import tree_flatten_spec

    module = exported_program.module()
    inputs = tree_flatten_spec((args, kwargs), module._in_spec)
    return _TargetReferenceInterpreter(module).run(*inputs)
