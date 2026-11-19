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
r"""Whole-model, trace-based segment-major GPTQ quantizer.

Unlike :class:`LlamaGPTQQuantizer` (which assumes a ``model.model.layers``
structure), this quantizer makes no structural assumption. The target model
is traced once with ``torch.export`` (static shapes) and quantization runs
over the FX graph segment-by-segment::

    for segment in graph split at quantized linears (topological order):
        move the live frontier to the GPU      # one batched copy per sample
        for node in segment:
            evaluate the node for every sample          # "reforward"
        at the closing ``aten.linear``:
            Hessian over ALL samples  ->  fasterquant()  ->  write weights back
            reforward the linear for every sample       # after write-back
        move the live frontier back to pinned host memory

Model-agnostic by construction
------------------------------
Nothing below depends on Llama or any particular architecture.  A model
qualifies as long as it can be traced with ``torch.export`` and its
quantized linears appear as ``aten.linear`` call nodes: HF decoder
families (Llama, Qwen, Mistral, SmolLM, ...), encoders, custom stacks.
The ``llama``/``LlamaGPTQQuantizer`` mentions in the comments throughout
this file are *spec citations, not dependencies*: each documents which
established reference-quantizer behavior (CPU-residency granularity,
fp64 cast discipline, Hessian-chain error feedback) is replicated, so
numerical parity can be verified against :class:`LlamaGPTQQuantizer`.
The single naming-level default inherited from the Llama convention is
``TracedGPTQConfig.layer_group_regex`` (``\.layers\.(\d+)\.``), used
solely for ``gptq_v2`` + ``use_orig_model_inference`` decoder-boundary
detection and overridable per model; when no boundary matches, the
float-chain semantics apply unchanged.

Properties
----------
* Every graph node is evaluated exactly once per sample: the whole
  quantization costs roughly one forward pass per calibration sample. No
  re-entry from the model top, no prefix recomputation.
* Sequential error feedback is preserved: ancestors of a linear appear
  earlier in topological node order, so every Hessian is measured through
  already-quantized weights, and every linear is evaluated (after its own
  quantization) so downstream Hessians see quantized weights too.
* Memory: only the *liveness set* of the graph is retained per sample
  (residual stream, masks, current frontier input). An intermediate value is
  dropped as soon as its last consumer has run. Shared weights (tied
  embeddings / tied lm_head) are detected on the graph and excluded.
* Transfers: data crosses PCIe only at segment boundaries (the quantized
  linears -- the GPTQ-relevant boundaries), as one large batched copy per
  boundary into pinned host memory, matching llama_quantizer's per-layer
  ``move_to_device`` granularity. Inside a segment everything stays on GPU.
* ``double_precision``: llama_quantizer semantics -- when enabled, the whole
  activation chain (quantized and float shadow) is evaluated in float64,
  against per-segment fp64 copies of the consumed parameters; Hessian
  accumulation and GPTQ math run in fp64. The fake-quant write-back still
  lands in the shared fp32 parameters.
* The result is the *float* model whose weights were replaced by their
  quantize->dequantize ("fake quant") float values. No packing, no format
  change: the regular TICO convert/export workflow continues unchanged.

Calibration batches must share one static shape (the example script already
pads). Tracing is done with the model config's ``use_cache`` disabled.
"""

import types
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import re
import torch
import torch.fx
import torch.nn as nn
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind
from tqdm.auto import tqdm

from tico.quantization.algorithm.gptq.gptq import GPTQ
from tico.quantization.config.traced_gptq import TracedGPTQConfig
from tico.utils.utils import move_to_device
from tico.quantization.quantizer import BaseQuantizer
from tico.quantization.quantizer_registry import register_quantizer

__all__ = ["StopForward", "TracedGPTQConfig", "TracedGPTQQuantizer"]


class StopForward(Exception):
    """Raised to stop the model forward right after input capture."""


# TracedGPTQConfig lives in tico.quantization.config.traced_gptq (following
# the GPTQConfig/LlamaGPTQConfig layout); re-exported here for convenience.


def _fetch_attr(module: torch.nn.Module, target: str) -> Any:
    obj: Any = module
    for atom in target.split("."):
        obj = getattr(obj, atom)
    return obj


def _arg_nodes(node: torch.fx.Node, out: Optional[List[torch.fx.Node]] = None) -> List[torch.fx.Node]:
    """All fx.Node values consumed by ``node`` (args + kwargs, flattened;
    recurses through lists/tuples/dicts, e.g. aten.index's ``indices``)."""
    if out is None:
        out = []
    def _walk(x: Any) -> None:
        if isinstance(x, torch.fx.Node):
            out.append(x)
        elif isinstance(x, (list, tuple)):
            for y in x:
                _walk(y)
        elif isinstance(x, dict):
            for y in x.values():
                _walk(y)
    _walk(node.args)
    _walk(node.kwargs)
    return out


def _hint_oom(fn, *args, **kwargs):
    """Run a graph-node evaluation with a unified CUDA-OOM diagnosis."""
    try:
        return fn(*args, **kwargs)
    except torch.OutOfMemoryError as e:
        raise RuntimeError(
            "CUDA OOM during the traced GPTQ walk (activations are "
            "CPU-offloaded; the GPU holds the parameters, one "
            "calibration sample's live set (samples are streamed one "
            "at a time) and per-node transients). Reduce the number "
            "samples or the sequence length, or use a smaller model."
        ) from e


def _executable(n: torch.fx.Node) -> bool:
    """Nodes with tensor dataflow the walk must evaluate (skip placeholders,
    the export guard shim, and grad-mode toggles lifted into the graph)."""
    if n.op in ("placeholder", "get_attr"):
        return False
    if n.op == "call_module" and "_guards_fn" in str(n.target):
        return False  # export guard shim, no tensor dataflow
    if n.op == "call_function" and "set_grad_enabled" in str(n.target):
        return False  # grad-mode toggle lifted into the graph
    return True


def _stage_cpu(obj: Any) -> Any:
    """GPU->CPU staging into pinned host memory.  non_blocking is safe
    without a host sync: the follow-up H2D copy of the same buffer is
    issued on the same stream as this D2H copy."""
    if torch.is_tensor(obj):
        if obj.device.type != "cuda":
            return obj
        p = torch.empty(tuple(obj.shape), dtype=obj.dtype, pin_memory=True)
        p.copy_(obj, non_blocking=True)
        return p
    if isinstance(obj, tuple):
        return tuple(_stage_cpu(x) for x in obj)
    if isinstance(obj, list):
        return [_stage_cpu(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _stage_cpu(v) for k, v in obj.items()}
    return obj


@register_quantizer(TracedGPTQConfig)
class TracedGPTQQuantizer(BaseQuantizer):
    """Structure-free, whole-model GPTQ driven by the traced FX graph."""

    def __init__(self, config: TracedGPTQConfig):
        super().__init__(config)
        assert isinstance(config, TracedGPTQConfig)
        # calibration storage: one entry per captured batch
        self.cache_args: List[List[Any]] = []
        self.cache_kwargs: Dict[str, List[Any]] = {}
        self.num_batches: int = 0
        self._orig_model_forward: Optional[Any] = None
        self.orig_use_cache: Optional[bool] = None

    # ------------------------------------------------------------------ #
    # prepare: capture whole-model inputs for every calibration batch     #
    # ------------------------------------------------------------------ #
    def prepare(
        self,
        model: torch.nn.Module,
        args: Optional[Any] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.nn.Module:
        """Wrap ``model.forward`` so that each calibration call stores its
        (positional, keyword) inputs and stops the forward immediately."""

        def catcher(_model, *m_args, **m_kwargs):
            self.cache_args.append(
                [a.cpu() if torch.is_tensor(a) else a for a in m_args]
            )
            for k, v in m_kwargs.items():
                self.cache_kwargs.setdefault(k, []).append(
                    v.cpu() if torch.is_tensor(v) else v
                )
            self.num_batches += 1
            raise StopForward

        assert self._orig_model_forward is None, "prepare() called twice"
        self._orig_model_forward = model.forward

        # The new forward stores the inputs and returns immediately: for
        # whole-model quantization we only need the model-level inputs, so
        # there is no reason to run the network at all during calibration.
        def suppressor(_model, *m_args, **m_kwargs):
            catcher(_model, *m_args, **m_kwargs)  # stores + raises StopForward
            return None                           # unreachable

        def suppressed(_model, *m_args, **m_kwargs):
            try:
                return suppressor(_model, *m_args, **m_kwargs)
            except StopForward:
                return None  # swallow: the calibration loop must not crash

        model.forward = types.MethodType(suppressed, model)

        # Calibration must be a pure prefill without cache state.
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            self.orig_use_cache = model.config.use_cache
            model.config.use_cache = False
        return model

    # ------------------------------------------------------------------ #
    # convert: trace once, then segment-major GPTQ over the FX graph     #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def convert(self, model: torch.nn.Module) -> torch.nn.Module:
        cfg = self.config
        assert isinstance(cfg, TracedGPTQConfig)
        cfg.validate()
        assert self.num_batches > 0, "no calibration data captured (call prepare() first)"

        # 1) restore original forward (stop catching)
        assert self._orig_model_forward is not None
        model.forward = self._orig_model_forward
        self._orig_model_forward = None

        device = next(model.parameters()).device

        # 2) trace the whole model once with static shapes
        args0 = [a.to(device) if torch.is_tensor(a) else a for a in self.cache_args[0]]
        kwargs0 = {
            k: v[0].to(device) if torch.is_tensor(v[0]) else v[0]
            for k, v in self.cache_kwargs.items()
            if v[0] is not None
        }
        self._assert_static_shapes()
        # fp32 snapshot source for the entry-prefix plan; only populated
        # (and only consumed) under double_precision
        fp32_state: Dict[str, torch.Tensor] = {}
        if cfg.double_precision:
            # Trace the model in float64.  Tracing the fp32 model bakes
            # trace-time no-op dtype casts (e.g. LlamaRMSNorm's
            # ``hidden_states.to(torch.float32)`` on an fp32 tensor) into the
            # graph as shared nodes: at trace dtype they are value-neutral,
            # but under dp the walk evaluates them as real fp64->fp32
            # round-trips that silently degrade the residual stream by ~1
            # fp32 ulp per decoder layer (the final residual add consumes
            # the rounded tensor).  The eager/plain module path keeps the
            # residual in fp64, so the traced Hessian chain diverged from
            # plain's at ~1e-7 relative -- enough to flip sparse GPTQ
            # decisions.  Tracing the double model makes every cast in the
            # graph explicit and value-faithful, so the traced dp chain is
            # bitwise-identical to the module dp chain.  finalize() restores
            # fp32 parameters (matching llama_quantizer's per-layer restore).
            #
            # fp32 snapshot of every param/buffer: the graph prefix BEFORE the
            # first quantized linear (embedding, mask build, rotary, layer-0
            # norm -- and, unfused, SpinQuant's rotate_embedding) is evaluated
            # in fp32 against these snapshots so it stays bitwise-identical to
            # the plain quantizer's captured-fp32 layer-0 input.  Evaluating
            # the prefix in fp64 instead (vs plain's fp32) perturbs the first
            # Hessian by ~1e-6 relative and flips sparse-GPTQ decisions on
            # dozens of downstream linears.
            # state_dict() (not named_parameters()) so that TIED aliases are
            # covered too: with a tied embedding the graph's embedding lookup
            # may consume the get_attr lifted from the *lm_head.weight* alias,
            # which named_parameters never yields -- without it the embedding
            # would run fp64 and force the entry rotation up to fp64 with it.
            fp32_state = {
                name: t.detach().clone()
                for name, t in model.state_dict().items()
                if t.is_floating_point()
            }
            model = model.double()
        # strict=False (dynamo tracing) on purpose: it is the mode that
        # traces HuggingFace forwards reliably (data-dependent control flow,
        # dtype casts); it also keeps quantized linears visible as single
        # aten.linear call nodes for segment splitting.  A strict trace is
        # NOT a supported alternative here: it can inline/decompose linears
        # so no aten.linear node remains to quantize.
        ep: ExportedProgram = torch.export.export(
            model, tuple(args0), kwargs0, strict=False
        )
        gm = ep.module()
        graph = gm.graph

        # 3) env maps: params/buffers/constants are shared across samples
        # Naming: an "env" (environment) is the interpreter/evaluator term
        # (also used by torch.fx.Interpreter-style walkers) for a dict that
        # binds graph nodes to their values while one input executes.  It is
        # NOT the sample itself: the sample is the input, the env is the
        # execution state it creates -- one env per sample, grown as nodes
        # evaluate and liveness-pruned as values die.  Shared constants live
        # in base_env, which belongs to no sample.
        param_node_of: Dict[torch.fx.Node, str] = {}   # node -> param FQN
        base_env: Dict[torch.fx.Node, Any] = {}        # node -> shared tensor
        node_by_name = {n.name: n for n in graph.nodes}
        eager_named = dict(model.named_parameters())
        eager_buffers = dict(model.named_buffers())
        # torch>=2.9 lifts params/buffers into get_attr nodes; older versions
        # pass them as placeholders.  Handle both.  base_env prefers the
        # *eager* tensor object so that in-place write-backs stay visible to
        # both the graph evaluations and the returned float model.
        for n in graph.nodes:
            if n.op == "get_attr":
                if n.target in eager_named:
                    base_env[n] = eager_named[n.target]
                    param_node_of[n] = n.target
                elif n.target in eager_buffers:
                    base_env[n] = eager_buffers[n.target]
                else:
                    try:
                        t = _fetch_attr(gm, n.target)
                    except AttributeError:
                        continue
                    base_env[n] = t
                    if isinstance(t, torch.nn.Parameter):
                        param_node_of[n] = n.target
        for spec in ep.graph_signature.input_specs:
            node = node_by_name.get(spec.arg.name)
            if node is None or node in base_env:
                continue
            if spec.kind == InputKind.PARAMETER:
                param_node_of[node] = spec.target
                base_env[node] = eager_named.get(spec.target)
            elif spec.kind in (InputKind.BUFFER, InputKind.CONSTANT_TENSOR):
                base_env[node] = eager_buffers.get(spec.target)
        user_specs = [
            s for s in ep.graph_signature.input_specs if s.kind == InputKind.USER_INPUT
        ]
        user_placeholders = [node_by_name[s.arg.name] for s in user_specs]

        # 4) quantizable linear nodes: aten.linear whose weight is a parameter,
        #    used by exactly one linear (tied/shared weights are excluded).
        linear_nodes: List[torch.fx.Node] = []
        weight_node_of: Dict[torch.fx.Node, torch.fx.Node] = {}
        for n in graph.nodes:
            if n.op == "call_function" and n.target == torch.ops.aten.linear.default:
                w = n.args[1]
                if isinstance(w, torch.fx.Node) and w in param_node_of:
                    linear_nodes.append(n)
                    weight_node_of[n] = w
        if len(linear_nodes) == 0:
            raise RuntimeError(
                "no aten.linear nodes with parameter weights found in the "
                "traced graph; the model may have been traced in a decomposed form"
            )
        # tied weights surface as distinct get_attr nodes sharing one
        # Parameter -> count consumers per underlying object identity
        obj_count = Counter(id(base_env[w]) for w in weight_node_of.values())
        quantizable = [
            n for n in linear_nodes
            if obj_count[id(base_env[weight_node_of[n]])] == 1
        ]
        if not cfg.quantize_lm_head:
            quantizable = [
                n for n in quantizable
                if not param_node_of[weight_node_of[n]].startswith("lm_head")
            ]
        # SpinQuant runtime rotations: with a tied embedding the embedding-side
        # rotation cannot be folded into the table and stays in the graph as a
        # runtime linear (param names rotate_embedding / rotate_lm_head; see
        # spin_llama.py).  The plain (module) GPTQ quantizer enumerates
        # decoder-layer submodules only and therefore never quantizes them --
        # and the rotations must stay high-precision for the rotation to be
        # exact.  Mirror that here so traced and plain quantize the same set.
        quantizable = [
            n for n in quantizable
            if "rotate_embedding" not in param_node_of[weight_node_of[n]]
            and "rotate_lm_head" not in param_node_of[weight_node_of[n]]
        ]

        # 5) liveness: values whose LAST consumer is node i can be pruned after i
        last_use: Dict[torch.fx.Node, int] = {}
        for i, n in enumerate(graph.nodes):
            for a in _arg_nodes(n):
                last_use[a] = i
        deaths: Dict[int, List[torch.fx.Node]] = defaultdict(list)
        for a, i in last_use.items():
            deaths[i].append(a)

        # 6) bind per-sample envs (the only state: the live value set;
        # "env" = interpreter frame binding nodes to one sample's values,
        # see the naming note at base_env above)
        envs = self._bind_samples(user_placeholders, user_specs, len(args0), device)

        # 7) segment-major walk: segments delimited by quantized linears
        quantizers: Dict[str, Any] = {}

        # use_orig_model_inference (without gptq_v2): keep the FLOAT weight
        # of every quantized linear and evaluate that node with it, so the
        # WHOLE activation chain seen by downstream Hessians stays float
        # (quantization errors never propagate -- plain-GPTQ parity).
        gptq_v2 = cfg.gptq_v2

        # gptq_v2 + use_orig_model_inference: decoder-layer boundary
        # injection (the LlamaGPTQQuantizer semantics -- the Hessian chain
        # resets to the float model's chain at every decoder-layer boundary,
        # while within a layer the quantized chain carries the error).  When
        # no boundary matches (see layer_group_regex below), gptq_v2 keeps
        # the quantized chain and uses the float chain ONLY to feed GPTQv2
        # native (FP) inputs, since a float Hessian chain would make dXXT
        # identically zero.  Without gptq_v2 the whole chain stays float
        # regardless of the regex (no injection).
        orig_inference = cfg.use_orig_model_inference

        float_weights: Dict[str, torch.Tensor] = {}

        # Float shadow envs (one per sample) holding the float model's
        # activation chain: the GPTQv2 native-input source and the boundary
        # injection source (gptq_v2 + use_orig_model_inference).  Without
        # gptq_v2 the envs themselves stay on the float chain via
        # float-weight substitution, so no shadow chain exists.  Shallow
        # copies share the user-input tensors; they diverge at the first
        # quantized linear (float-weight substitution below).
        fenvs: Optional[List[Dict[torch.fx.Node, Any]]] = (
            [dict(env) for env in envs] if gptq_v2 else None
        )

        # llama_quantizer-granularity CPU residency (always on): per-sample
        # activation envs live on pinned host memory and cross PCIe only at
        # segment boundaries (the quantized linears -- the GPTQ-relevant
        # boundaries), one large batched copy per boundary, mirroring how
        # LlamaGPTQ moves its cached layer inputs per decoder layer.  Inside
        # a segment all data stays on the GPU; GPU peak is O(per-sample live
        # set x number of calibration batches) plus the parameters.
        quantized: set = set()

        nodes = list(graph.nodes)

        # The walk stops after the LAST quantized linear: everything between
        # it and the output node (final norm, lm_head) feeds only the output,
        # which is never read here. Evaluating that tail would materialize
        # e.g. full-vocab logits (~1 GB/sample for SmolLM2) that are never
        # pruned (the loop breaks at the output node before pruning runs).
        last_quant_idx = max(i for i, n in enumerate(nodes) if n in quantizable)

        # fp32 entry-prefix parameter plan (empty unless double_precision):
        # prefix-only get_attr params evaluate in fp32, the rest in fp64
        fp32_entry_env = self._plan_fp32_entry_env(
            cfg, graph, nodes, quantizable, last_quant_idx, fp32_state
        )

        # gptq_v2 + use_orig_model_inference boundary detection: the first
        # quantized linear of each new decoder-layer group (per
        # layer_group_regex) is a boundary at which the Hessian chain is
        # reset to the float chain.  Unmatched linears are skipped (no
        # boundary, no group change).  Without gptq_v2 no boundaries exist:
        # the whole chain stays float via float-weight substitution.
        boundary_nodes: set = set()
        if orig_inference and gptq_v2 and cfg.layer_group_regex:
            try:
                pat = re.compile(cfg.layer_group_regex)
            except re.error as e:
                raise ValueError(
                    f"invalid layer_group_regex {cfg.layer_group_regex!r}: {e}"
                ) from e
            prev_gid: Any = None
            for n in nodes:
                if n not in quantizable:
                    continue
                fqn = param_node_of[n.args[1]]
                m = pat.search(fqn)
                if m is None:
                    continue  # unmatched linear: never a boundary
                gid = m.group(1)
                if prev_gid is not None and gid != prev_gid:
                    boundary_nodes.add(n)
                prev_gid = gid
        injection_active = gptq_v2 and bool(boundary_nodes)

        # llama_quantizer double_precision semantics: when enabled, the whole
        # activation chain (quantized AND float shadow) is evaluated in
        # float64 -- llama casts every decoder layer and the float orig layer
        # to double per forward, so the Hessian inputs X are fp64.  In the
        # traced walk this means upcasting the frontier at segment entry and
        # evaluating all segment ops against fp64 parameter copies
        # (dbl_base); the fake-quant write-back still lands in the shared
        # fp32 parameters.
        dp = cfg.double_precision

        def _eval_node(node: torch.fx.Node, a: tuple, k: dict) -> Any:
            if node.op == "call_function":
                if "_assert_tensor_metadata" in str(node.target):
                    # trace-time metadata assertion: carries the traced dtype
                    # as a constant, so under double_precision it fires
                    # spuriously. Pure assertion, no dataflow value -- skip
                    # the check, return None (its own result type) so
                    # liveness pruning and env layout stay identical.
                    return None
                return node.target(*a, **k)
            if node.op == "call_method":
                return getattr(a[0], str(node.target))(*a[1:], **k)
            if node.op == "call_module":
                return gm.get_submodule(str(node.target))(*a, **k)
            raise NotImplementedError(f"unsupported node op: {node.op}")

        # split the executable node range into segments delimited by the
        # quantized linears (the GPTQ-relevant boundaries): each segment is
        # walked llama_quantizer-style, one calibration sample at a time --
        # a sample's frontier crosses PCIe just-in-time on the way in and is
        # staged back to pinned CPU memory right after its pass, so GPU
        # activation residency is one live set regardless of sample count.
        segments: List[List[int]] = []
        cur: List[int] = []
        for i, node in enumerate(nodes):
            if i > last_quant_idx or node.op == "output":
                break
            if not _executable(node):
                continue
            cur.append(i)
            if node in quantizable:
                segments.append(cur)
                cur = []
        if cur:
            segments.append(cur)  # defensive: tail without a closing linear

        # segment-entry dtype policy: under double_precision every
        # floating-point frontier tensor is upcast to fp64 here, once per
        # segment (non-floating tensors -- ids, masks -- pass through,
        # mirroring llama's _cast_to_double).
        def _entry(v: Any) -> Any:
            v = move_to_device(v, device)
            if (
                dp
                and torch.is_tensor(v)
                and v.is_floating_point()
                and v.dtype != torch.float64
            ):
                v = v.double()
            return v

        # progress in QUANTIZED-LINEAR units: one tick per GPTQ'd linear
        # (the meaningful unit of work), with the current linear's FQN in
        # the postfix; sample/node granularity is intentionally not shown
        pbar = tqdm(
            total=len(quantizable),
            desc="Quantizing (GPTQ linears)", unit="linear",
            disable=not cfg.show_progress,
        )
        for seg_i, seg in enumerate(segments):
            end_node = nodes[seg[-1]]
            # gptq_v2 + use_orig_model_inference: reset the Hessian chain to
            # the float model's chain at the decoder-layer boundary.  fenvs
            # holds the float values for the whole live set (both envs prune
            # identically), so one update floats the boundary stream
            # (residual add + next-layer norms) retroactively; error feedback
            # then re-accumulates within the layer.  Without gptq_v2 there
            # are no boundaries: the chain is already fully float.
            if end_node in boundary_nodes:
                for s in range(len(envs)):
                    envs[s].update(fenvs[s])

            # fp64 parameter copies for this segment only (freed at exit):
            # under double_precision the activations are fp64, so every
            # param-consuming op in the segment needs fp64 weights/buffers
            if dp:
                dbl_base: Dict[torch.fx.Node, Any] = {}
                for i in seg:
                    for an in _arg_nodes(nodes[i]):
                        if an in dbl_base or an not in base_env:
                            continue
                        if seg_i == 0 and an in fp32_entry_env:
                            dbl_base[an] = fp32_entry_env[an]
                            continue
                        v = base_env[an]
                        if torch.is_tensor(v) and v.is_floating_point():
                            dbl_base[an] = v.double()
                        else:
                            dbl_base[an] = v
                seg_base_env = dbl_base
            else:
                seg_base_env = base_env

            # every segment closes with exactly one quantized linear (the
            # segment split is built on them)
            lin_node = nodes[seg[-1]]
            has_lin = lin_node in quantizable

            # llama_quantizer-style sample streaming: samples pass through
            # the segment one at a time -- a sample's frontier is moved to
            # the GPU just before its pass and staged back to pinned CPU
            # memory right after, so GPU activation residency is ONE live
            # set (plus transients) regardless of the number of calibration
            # samples.  PCIe traffic is unchanged: the same per-sample
            # frontier copies happen, only just-in-time instead of all
            # upfront at segment entry.
            lin_args_stash: List[Tuple[tuple, dict]] = []

            # phase 1: stream every sample through the pre-linear nodes
            for s in range(len(envs)):
                env = envs[s]
                fenv = fenvs[s] if fenvs is not None else None
                # move just this sample's frontier to the GPU
                for key in env:
                    env[key] = _entry(env[key])
                if fenv is not None:
                    for key in fenv:
                        fenv[key] = _entry(fenv[key])

                # Evaluate the PRE-LINEAR nodes of this segment for this
                # sample: seg[:-1] is everything before the closing linear
                # -- i.e. the producers of the linear's input activations
                # (when has_lin is False this is a linear-less tail segment,
                # so every node runs).  Each node's args/kwargs are resolved
                # via _lookup -- the sample's env first, falling back to
                # seg_base_env for params/buffers/constants shared across
                # samples -- materialized on the GPU just in time, evaluated
                # once, and the result stored in the sample's env for the
                # consumers that follow (the closing linear reads these
                # values as its input for the Hessian).
                for i in (seg[:-1] if has_lin else seg):
                    node = nodes[i]
                    a, k = self._resolve_call_args(
                        node, env, seg_base_env, device
                    )
                    out = _hint_oom(_eval_node, node, a, k)
                    env[node] = move_to_device(out, device)

                    # GPTQv2 float-shadow chain: evaluate the same node
                    # against the float chain (quantized linears are
                    # segment-final, so pre-linear nodes never need the
                    # float-weight substitution)
                    if fenv is not None:
                        fa, fk = self._resolve_call_args(
                            node, fenv, seg_base_env, device
                        )
                        fout = _hint_oom(_eval_node, node, fa, fk)
                        fenv[node] = move_to_device(fout, device)

                    # liveness prune: drop values whose last consumer ran
                    for dead in deaths.get(i, ()):
                        env.pop(dead, None)
                        if fenv is not None:
                            fenv.pop(dead, None)

                if has_lin:
                    # stash this sample's gathered linear inputs for the
                    # quantization barrier below (pinned CPU memory)
                    la, lk = self._resolve_call_args(
                        lin_node, env, seg_base_env, device
                    )
                    lin_args_stash.append((_stage_cpu(la), _stage_cpu(lk)))

                # stage this sample's surviving frontier back to CPU: its
                # whole live set is now idle until the next segment
                for key in env:
                    env[key] = _stage_cpu(env[key])
                if fenv is not None:
                    for key in fenv:
                        fenv[key] = _stage_cpu(fenv[key])

            # quantization barrier: GPTQ needs every sample's input at
            # the linear, so quantization runs once after the sample
            # loop, re-uploading the stashed per-sample inputs (llama
            # does the same with its fp_inps stash)
            if has_lin:
                # GPTQv2: native (FP) inputs from the float shadow chain,
                # in the same per-sample order as the add_batch calls
                # below.  Only under gptq_v2 -- plain
                # use_orig_model_inference must not activate the dXXT/P
                # correction (matches LlamaGPTQ).
                native_inputs = None
                if gptq_v2 and fenvs is not None:
                    native_inputs = [
                        self._lookup(lin_node.args[0], fenv, seg_base_env)
                        for fenv in fenvs
                    ]
                self._quantize_linear(
                    lin_node, lin_args_stash, param_node_of, base_env,
                    quantizers, cfg, float_weights, native_inputs,
                )
                quantized.add(lin_node)
                # one tick per quantized linear; postfix shows which one
                pbar.set_postfix_str(param_node_of[weight_node_of[lin_node]])
                pbar.update(1)

                # nothing downstream of the last quantized linear is ever
                # needed: skip its reforward (and the float-shadow eval)
                # entirely -- its output feeds only the graph output.
                if seg[-1] < last_quant_idx:
                    # reforward: evaluate the linear for every sample
                    # *after* write-back, so downstream Hessians are
                    # measured against quantized weights (sequential
                    # error feedback).  With gptq_v2 AND boundary injection
                    # the chain stays quantized here (float injection
                    # happens at boundaries).  Otherwise (including plain
                    # use_orig_model_inference): evaluate the quantized
                    # linear with its stashed FLOAT weight so the whole
                    # chain stays float.
                    # the stashed input is moved to the GPU one sample at a
                    # time; weight/bias come from the live parameter copies
                    # (NOT the pre-quantization stash) so the reforward sees
                    # the just-written-back quantized weight
                    fw_sub = None
                    if orig_inference and not gptq_v2 and not injection_active:
                        fw = float_weights[param_node_of[lin_node.args[1]]]
                        if dp and torch.is_tensor(fw) and fw.is_floating_point():
                            fw = fw.double()
                        fw_sub = fw
                    rest = tuple(
                        move_to_device(
                            seg_base_env[x]
                            if isinstance(x, torch.fx.Node) else x,
                            device,
                        )
                        for x in lin_node.args[1:]
                    )
                    for s in range(len(envs)):
                        la, lk = lin_args_stash[s]
                        a = (move_to_device(la[0], device),) + rest
                        if fw_sub is not None:
                            a = a[:1] + (fw_sub,) + a[2:]
                        k = {
                            kk: move_to_device(v, device)
                            for kk, v in lk.items()
                        }
                        out = _hint_oom(_eval_node, lin_node, a, k)
                        # this IS the segment-exit frontier value: stage
                        # it straight to pinned CPU memory
                        envs[s][lin_node] = _stage_cpu(out)

                    # GPTQv2: evaluate the linear a second time against
                    # the float chain (float weights of already-quantized
                    # linears), keeping fenvs on the pure float-model
                    # activation path.
                    if fenvs is not None:
                        # float weight for the shadow linear; weight and bias
                        # come from the live parameter copies
                        fw = float_weights[param_node_of[lin_node.args[1]]]
                        if dp and fw.is_floating_point():
                            fw = fw.double()
                        for s, fenv in enumerate(fenvs):
                            # input comes from the FLOAT chain (pinned host
                            # memory here), never from envs
                            fa, fk = self._resolve_call_args(
                                lin_node, fenv, seg_base_env, device
                            )
                            fa = fa[:1] + (fw,) + fa[2:]
                            fout = _hint_oom(_eval_node, lin_node, fa, fk)
                            fenvs[s][lin_node] = _stage_cpu(fout)

                    # each stashed float weight is consumed only by its
                    # own node's shadow/legacy reforward above -- free it
                    # immediately so the stash never grows beyond one
                    # weight (was O(model))
                    float_weights.pop(param_node_of[lin_node.args[1]], None)

                    # liveness prune for the linear node itself
                    for dead in deaths.get(seg[-1], ()):
                        for env in envs:
                            env.pop(dead, None)
                        if fenvs is not None:
                            for fenv in fenvs:
                                fenv.pop(dead, None)

                # the stash fed add_batch and the reforward: release the
                # pinned host buffers now
                lin_args_stash.clear()

            if dp:
                dbl_base.clear()  # free this segment's fp64 param copies
        pbar.close()

        float_weights.clear()  # float stashes no longer needed
        if fenvs is not None:
            fenvs.clear()  # float shadow envs no longer needed
        return self.finalize(model, quantizers)

    # ------------------------------------------------------------------ #
    # helpers                                                             #
    # ------------------------------------------------------------------ #
    def _resolve_call_args(
        self,
        node: torch.fx.Node,
        env: Dict[torch.fx.Node, Any],
        base_env: Dict[torch.fx.Node, Any],
        device: torch.device,
    ) -> Tuple[tuple, dict]:
        """Resolve a node's args/kwargs to one sample's values: the sample
        env first, falling back to ``base_env`` for params/buffers/constants
        shared across samples; every tensor is materialized on ``device``
        just in time.  One helper for the quantized chain, the GPTQv2
        float-shadow chain, and the linear-input stash, so all three stay
        in sync (same lookup, same fallback, same device move)."""
        a = tuple(
            move_to_device(self._lookup(x, env, base_env), device)
            for x in node.args
        )
        k = {
            kk: move_to_device(self._lookup(v, env, base_env), device)
            for kk, v in (node.kwargs or {}).items()
        }
        return a, k

    @staticmethod
    def _lookup(x: Any, env: Dict[Any, Any], base_env: Dict[Any, Any]) -> Any:
        """Substitute fx.Node values (recursively through lists/tuples/dicts,
        e.g. aten.index's ``indices=[node, node]``) with their per-sample
        values."""
        if isinstance(x, torch.fx.Node):
            if x in env:
                return env[x]
            return base_env[x]
        if isinstance(x, list):
            return [TracedGPTQQuantizer._lookup(y, env, base_env) for y in x]
        if isinstance(x, tuple):
            return tuple(TracedGPTQQuantizer._lookup(y, env, base_env) for y in x)
        if isinstance(x, dict):
            return {
                k: TracedGPTQQuantizer._lookup(v, env, base_env)
                for k, v in x.items()
            }
        return x

    def _bind_samples(
        self,
        user_placeholders: List[torch.fx.Node],
        user_specs,
        num_args: int,
        device: torch.device,
    ) -> List[Dict[torch.fx.Node, Any]]:
        """Create one env per calibration batch with user inputs bound.

        "Env" follows the interpreter convention (see the naming note at
        ``base_env`` in ``quantize``): a dict binding graph nodes to the
        values they take while that one sample executes, not the sample
        data itself.

        ``torch.export`` keeps user inputs in *call order* (args first, then
        kwargs) after params/buffers/constants, so the i-th user placeholder
        corresponds to the i-th element of ``args + list(kwargs)``.
        """
        # per placeholder: ('arg', idx) or ('kw', name) into the capture store
        sources: List[Tuple[str, Any]] = [("arg", i) for i in range(num_args)]
        for s in user_specs[num_args:]:
            # torch<=2.8: target is the kwarg name; torch>=2.9: target is None
            # and arg.name carries it
            name = s.target if s.target is not None else s.arg.name
            sources.append(("kw", name))
        if len(user_placeholders) != len(sources):
            raise RuntimeError(
                f"user input mismatch: graph has {len(user_placeholders)} "
                f"user placeholders, calibration batch provides {len(sources)}"
            )
        envs: List[Dict[torch.fx.Node, Any]] = []
        for b in range(self.num_batches):
            env: Dict[torch.fx.Node, Any] = {}
            for ph, (kind, key) in zip(user_placeholders, sources):
                val = self.cache_args[b][key] if kind == "arg" \
                    else self.cache_kwargs[key][b]
                env[ph] = val.to(device) if torch.is_tensor(val) else val
            envs.append(env)
        return envs

    def _assert_static_shapes(self) -> None:
        """All calibration batches must share one static shape for tracing."""
        ref: Dict[str, Tuple[int, ...]] = {}
        for b in range(self.num_batches):
            for i, a in enumerate(self.cache_args[b]):
                if torch.is_tensor(a):
                    key = f"arg{i}"
                    if key in ref and ref[key] != tuple(a.shape):
                        raise ValueError(
                            f"calibration batches must share one static shape; "
                            f"{key} batch 0 is {ref[key]} but batch {b} is "
                            f"{tuple(a.shape)}. Pad calibration sequences."
                        )
                    ref[key] = tuple(a.shape)

    def _plan_fp32_entry_env(
        self,
        cfg: TracedGPTQConfig,
        graph: torch.fx.Graph,
        nodes: List[torch.fx.Node],
        quantizable: List[torch.fx.Node],
        last_quant_idx: int,
        fp32_state: Dict[str, torch.Tensor],
    ) -> Dict[torch.fx.Node, torch.Tensor]:
        """fp32 entry-prefix parameters for double_precision runs.

        A get_attr node consumed EXCLUSIVELY by nodes before the first
        quantized linear is evaluated in fp32 (matching plain's
        captured-fp32 semantics); params also consumed at or after the
        prefix (e.g. the first quantized linear's own weight/bias) stay
        fp64 -- the entry segment must not mix dtypes inside dtype-checked
        ops like aten.linear.  Returns {} when ``double_precision`` is off.
        """
        if not cfg.double_precision:
            return {}
        first_quant_pos = min(i for i, n in enumerate(nodes) if n in quantizable)
        # only consumers that are actually evaluated by the walk matter: nodes
        # beyond last_quant_idx (final norm, lm_head, rotate_lm_head) never
        # run, so a get_attr shared with them (e.g. the TIED embedding table
        # shared by embed_tokens and lm_head) can still go fp32 for the
        # prefix -- marking it fp64 would force the fp32 entry rotation to
        # consume an fp64 embedding output and crash aten.linear.
        fp64_touched: set = set()
        for n in nodes[first_quant_pos : last_quant_idx + 1]:
            for a in _arg_nodes(n):
                fp64_touched.add(a)
        fp32_entry_env: Dict[torch.fx.Node, torch.Tensor] = {}
        fp32_nodes: set = set()
        for n in graph.nodes:
            if (
                n.op == "get_attr"
                and n.target in fp32_state
                and n not in fp64_touched
            ):
                fp32_entry_env[n] = fp32_state[n.target]
                fp32_nodes.add(n)
        # reconcile pass: a dtype-checked prefix op (linear/matmul/...)
        # whose input chain is fp64 -- e.g. a get_attr shared with a later
        # evaluated node -- must not keep an fp32 weight.  Drop it from
        # the fp32 set so the whole op runs fp64 (value-exact upcast)
        # instead of crashing on mixed dtypes.
        is_fp64: Dict[torch.fx.Node, bool] = {}
        for n in nodes[:first_quant_pos]:
            if n.op == "get_attr":
                is_fp64[n] = n not in fp32_nodes
                continue
            if n.op != "call_function":
                is_fp64[n] = False
                continue
            tgt = str(n.target)
            if "_to_copy" in tgt or ".to." in tgt:
                dts = [x for x in n.args if isinstance(x, torch.dtype)]
                is_fp64[n] = any(d == torch.float64 for d in dts) or any(
                    is_fp64.get(a, False) for a in _arg_nodes(n)
                )
                continue
            checked = any(
                s in tgt for s in ("linear", "matmul", "addmm", ".mm.", "bmm")
            )
            arg_fp64 = [a for a in _arg_nodes(n) if is_fp64.get(a, False)]
            if checked and arg_fp64:
                for a in _arg_nodes(n):
                    fp32_nodes.discard(a)
                    fp32_entry_env.pop(a, None)
                is_fp64[n] = True
            else:
                is_fp64[n] = bool(arg_fp64)
        return fp32_entry_env

    def _make_gptq(self, w_param: torch.Tensor, name: str,
                   cfg: TracedGPTQConfig) -> GPTQ:
        """GPTQ works on an ``nn.Module``; build a shim carrying a *clone* of
        the parameter so ``fasterquant`` can replace its weight freely."""
        shim = nn.Linear(
            w_param.shape[1], w_param.shape[0], bias=False,
            device=w_param.device, dtype=w_param.dtype,
        )
        shim.weight.data = w_param.detach().clone()
        g = GPTQ(shim, layer_name=name, double_precision=cfg.double_precision)
        g.saturation_threshold = cfg.saturation_threshold
        g.saturation_min_batches = cfg.saturation_min_batches
        return g

    def _resolve_weight_bits(self, cfg: TracedGPTQConfig, fqn: str) -> int:
        """Full-module name first, then longest suffix match (e.g.
        ``self_attn.q_proj`` or ``q_proj``), else the default."""
        module_fqn = fqn.rsplit(".", 1)[0]
        if module_fqn in cfg.weight_bits_overrides:
            return cfg.weight_bits_overrides[module_fqn]
        suffix = [k for k in cfg.weight_bits_overrides if module_fqn.endswith(k)]
        if suffix:
            return cfg.weight_bits_overrides[max(suffix, key=len)]
        return cfg.weight_bits

    def _quantize_linear(
        self,
        node: torch.fx.Node,
        args_per_sample: List[Tuple[tuple, dict]],
        param_node_of: Dict[torch.fx.Node, str],
        base_env: Dict[torch.fx.Node, Any],
        quantizers: Dict[str, Any],
        cfg: TracedGPTQConfig,
        float_weights: Optional[Dict[str, torch.Tensor]] = None,
        native_inputs: Optional[List[torch.Tensor]] = None,
    ) -> None:
        w_target = param_node_of[node.args[1]]
        # the shared tensor object: graph evaluations and the returned float
        # model both reference it, so one in-place copy_ updates everything
        w_param = base_env[node.args[1]]
        device = w_param.device
        g = self._make_gptq(w_param, w_target, cfg)
        try:
            weight_bits = self._resolve_weight_bits(cfg, w_target)
            sensitivity = None
            if cfg.sensitivity is not None:
                # sensitivity dicts are keyed by MODULE FQN (no ".weight"
                # suffix -- see SensitivityCalibrator.compute_sensitivity_info
                # and the plain quantizer's lookup), while w_target is the
                # lifted PARAM name ("...q_proj.weight").  Try the param name
                # first for robustness, then the module name -- without this
                # the lookup always misses and smse_for_gptq crashes on the
                # mse_for_gptq assert in quant.py.
                mod_fqn = w_target.rsplit(".", 1)[0]
                if w_target in cfg.sensitivity:
                    sensitivity = cfg.sensitivity[w_target]
                elif mod_fqn in cfg.sensitivity:
                    sensitivity = cfg.sensitivity[mod_fqn]
            g.quantizer.configure(
                bits=weight_bits,
                perchannel=cfg.perchannel,
                sym=cfg.symmetric,
                mse=cfg.mse,
                sensitivity=sensitivity,
                mse_tolerance=cfg.mse_tolerance,
                chunk_size=cfg.chunk_size,
            )
            # weighted Hessian accumulation (GPTQConfig.sample_weights):
            # one entry per calibration batch -- either a scalar (scales
            # that batch's XX^T contribution) or a tensor([batch_size])
            # (sqrt-weights the batch's rows).  The GPTQ core indexes
            # g.weights[g.batch_id] inside add_batch and self-increments
            # batch_id per call; the walk calls add_batch once per batch
            # in global batch order, so this wiring exactly matches the
            # plain quantizer's `gptq.weights` setup (quantizer.py).
            if cfg.sample_weights is not None:
                if len(cfg.sample_weights) < len(args_per_sample):
                    raise ValueError(
                        f"sample_weights has {len(cfg.sample_weights)} "
                        f"entries but {len(args_per_sample)} calibration "
                        "batches were captured; provide one weight (scalar "
                        "or per-batch tensor) per batch."
                    )
                g.weights = list(cfg.sample_weights)
            # sample inputs may still live in pinned host memory (the
            # segment walk stashes them on CPU): stream one sample at a
            # time so the GPU never holds all samples' inputs at once
            for s, (a, _) in enumerate(args_per_sample):
                # GPTQv2: one native (FP) input per sample, popped by
                # add_batch and weighted with the same per-sample weight
                # as the Hessian input, building the dXXT correction term
                if native_inputs is not None:
                    g.native_inp = [move_to_device(native_inputs[s], device)]
                g.add_batch(move_to_device(a[0], device), None)  # H += X X^T
            g.fasterquant(
                percdamp=cfg.percdamp,
                groupsize=cfg.groupsize,
                actorder=cfg.actorder,
                static_groups=cfg.static_groups,
                verbose=cfg.verbose,
                adaptive_percdamp=cfg.adaptive_percdamp,
                cond_threshold_good=cfg.cond_threshold_good,
                use_iterate=cfg.use_iterate,
                actorder_precision=cfg.actorder_precision,
            )
            # fake-quant float weights, written in place through the shared
            # tensor: graph evaluations and eager model stay in sync
            q_weight = g.layer.weight.data.to(w_param.dtype)
            if (
                cfg.use_orig_model_inference or cfg.gptq_v2
            ) and float_weights is not None:
                float_weights[w_target] = w_param.detach().clone()
            w_param.data.copy_(q_weight)
            quantizers[w_target.rsplit(".", 1)[0]] = g.quantizer
            if cfg.verbose:
                print(f"  quantized {w_target} ({weight_bits}b, nsamples={g.nsamples})")
        finally:
            g.free()

    def finalize(
        self, model: torch.nn.Module, quantizers: Dict[str, Any]
    ) -> torch.nn.Module:
        if self.orig_use_cache is not None:
            model.config.use_cache = self.orig_use_cache
            self.orig_use_cache = None
        self.cache_args.clear()
        self.cache_kwargs.clear()
        self.num_batches = 0
        model.quantizers = quantizers
        if self.config.double_precision:
            model.float()
        return model
