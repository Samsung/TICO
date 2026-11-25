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

"""Layer-bounded PLE compute and frozen-grid Circle contracts."""

import copy
import unittest
from typing import Any

import numpy as np
import torch

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.config.specs import affine
from tico.quantization.wrapq.dtypes import INT16, UINT8
from tico.quantization.wrapq.wrappers.gemma4.export_adapters import (
    Gemma4PLEProjectionExportAdapter,
)
from tico.quantization.wrapq.wrappers.gemma4.utils import (
    project_gemma4_per_layer_inputs,
    validate_gemma4_ple_projection_slicing,
)
from tico.quantization.wrapq.wrappers.llama.export_adapters import (
    register_fake_quant_meta_kernels_for_dynamic_export,
)

from test.quantization.wrapq.wrappers.gemma4.test_ple_export_adapters import (
    _has_gemma4,
    _make_ple_text_config,
    _make_quant_text_model,
)


@unittest.skipUnless(_has_gemma4(), "Transformers Gemma4 is required")
class TestGemma4PLERangeProjection(unittest.TestCase):
    """Compare independent full producers with smaller execution graphs."""

    def setUp(self):
        rng = torch.random.fork_rng(devices=[])
        rng.__enter__()
        self.addCleanup(rng.__exit__, None, None, None)
        torch.manual_seed(918)
        register_fake_quant_meta_kernels_for_dynamic_export()

    def _fixture(self, dtype):
        cfg = _make_ple_text_config(
            num_hidden_layers=3,
            layer_types=["full_attention"] * 3,
            hidden_size_per_layer_input=4,
        )
        _, text = _make_quant_text_model(
            cfg,
            quantized=dtype is not None,
            qcfg=PTQConfig(activation=affine(dtype)) if dtype else None,
        )
        ids = torch.tensor([[3, 3, 7, 0, 0]])
        # Multimodal H0 deliberately differs from token embeddings and from any
        # intermediate decoder output. Prompt p=2 is not the physical last slot.
        h0 = torch.randn(1, 5, cfg.hidden_size) * 0.4 + 0.2
        with torch.no_grad():
            token = text.get_per_layer_inputs(ids, None)
        return text, h0, token

    def _assert_state(self, expected, module):
        left, left_spec = torch.utils._pytree.tree_flatten(expected)
        right, right_spec = torch.utils._pytree.tree_flatten(module.state_dict())
        self.assertEqual(left_spec, right_spec)
        for value, actual in zip(left, right):
            if isinstance(value, torch.Tensor):
                self.assertTrue(torch.equal(value, actual))
            else:
                self.assertEqual(value, actual)

    def test_full_slice_equivalence_and_source_ownership(self):
        for dtype in (None, UINT8, INT16):
            with self.subTest(dtype=dtype):
                text, h0, token = self._fixture(dtype)
                full = Gemma4PLEProjectionExportAdapter(text)
                before = copy.deepcopy(text.state_dict())
                with torch.no_grad():
                    expected = full(h0, token)
                for k in (1, 2):
                    prefix = Gemma4PLEProjectionExportAdapter(text, layer_end=k)
                    suffix = Gemma4PLEProjectionExportAdapter(text, layer_start=k)
                    with torch.no_grad():
                        actual_prefix = prefix(h0, token[:, :, :k])
                        actual_suffix = suffix(h0[:, 2:3], token[:, 2:3, k:])
                    torch.testing.assert_close(
                        actual_prefix, expected[:, :, :k], atol=1e-6, rtol=1e-6
                    )
                    torch.testing.assert_close(
                        actual_suffix, expected[:, 2:3, k:], atol=1e-6, rtol=1e-6
                    )
                    self.assertFalse(
                        any("embed_tokens" in name for name in prefix.state_dict())
                    )
                    self.assertIs(
                        prefix.per_layer_model_projection,
                        text.per_layer_model_projection,
                    )
                self._assert_state(before, text)
                with torch.no_grad():
                    self.assertTrue(torch.equal(full(h0, token), expected))

    def test_none_token_inputs_and_bias_preserve_original_arithmetic(self):
        linear = torch.nn.Linear(7, 12, bias=True)
        norm = torch.nn.RMSNorm(3)
        hidden = torch.randn(1, 5, 7)
        kwargs: dict[str, Any] = dict(
            projection=linear,
            projection_norm=norm,
            projection_scale=0.3,
            input_scale=0.2,
            num_hidden_layers=4,
            hidden_size_per_layer_input=3,
            fq=lambda value, observer: value,
            projection_observer=None,
            sum_observer=None,
            inputs_observer=None,
        )
        expected = norm((linear(hidden) * 0.3).reshape(1, 5, 4, 3))
        for start, end in ((0, 2), (2, 4)):
            actual = project_gemma4_per_layer_inputs(
                hidden, None, layer_start=start, layer_end=end, **kwargs
            )
            torch.testing.assert_close(
                actual, expected[:, :, start:end], atol=1e-6, rtol=1e-6
            )
            self.assertFalse(torch.allclose(actual, expected[:, :, start:end] * 0.2))

    def test_range_rejects_norm_reduction_across_layers(self):
        projection = torch.nn.Linear(7, 12)
        with self.assertRaisesRegex(ValueError, "final P axis"):
            project_gemma4_per_layer_inputs(
                torch.randn(1, 5, 7),
                None,
                projection=projection,
                projection_norm=torch.nn.RMSNorm((2, 3)),
                projection_scale=1.0,
                input_scale=1.0,
                num_hidden_layers=4,
                hidden_size_per_layer_input=3,
                fq=lambda value, observer: value,
                projection_observer=None,
                sum_observer=None,
                inputs_observer=None,
                layer_end=2,
            )

    def test_frozen_none_token_path_omits_final_input_scale(self):
        for dtype in (UINT8, INT16):
            text, hidden, _ = self._fixture(dtype)
            with torch.no_grad():
                full = text.project_per_layer_inputs(hidden, None)
                projected = project_gemma4_per_layer_inputs(
                    hidden[:, 2:3],
                    None,
                    projection=text.per_layer_model_projection,
                    projection_norm=text.per_layer_projection_norm,
                    projection_scale=text.obs_per_layer_model_projection_scale.fake_quant(
                        text.per_layer_model_projection_scale_tensor
                    ),
                    input_scale=torch.tensor(99.0),
                    num_hidden_layers=3,
                    hidden_size_per_layer_input=4,
                    fq=text._fq,
                    projection_observer=text.obs_per_layer_projection,
                    sum_observer=None,
                    inputs_observer=text.obs_per_layer_inputs,
                    layer_start=1,
                )
            self.assertTrue(torch.equal(projected, full[:, 2:3, 1:]))

    def test_frozen_identity_activation_and_per_tensor_weights(self):
        from tico.quantization.config.specs import no_quant
        from tico.quantization.wrapq.qscheme import QScheme

        cfg = _make_ple_text_config()
        _, text = _make_quant_text_model(
            cfg,
            quantized=True,
            qcfg=PTQConfig(
                activation=no_quant(),
                weight=affine(UINT8, qscheme=QScheme.PER_TENSOR_ASYMM),
            ),
        )
        hidden = torch.randn(1, 5, cfg.hidden_size)
        with torch.no_grad():
            token = text.get_per_layer_inputs(torch.tensor([[1, 1, 3, 0, 0]]), None)
            full = Gemma4PLEProjectionExportAdapter(text)(hidden, token)
            selected = Gemma4PLEProjectionExportAdapter(text, layer_start=1)(
                hidden[:, 2:3], token[:, 2:3, 1:]
            )
        torch.testing.assert_close(selected, full[:, 2:3, 1:], atol=1e-6, rtol=1e-6)

    def test_policy_errors_name_upstream_site(self):
        from tico.quantization.wrapq.observers.minmax import MinMaxObserver
        from tico.quantization.wrapq.observers.mx import MXObserver
        from tico.quantization.wrapq.qscheme import QScheme

        for site in (
            "obs_per_layer_projection",
            "obs_per_layer_sum",
            "obs_per_layer_token_inputs",
        ):
            text, _, _ = self._fixture(UINT8)
            setattr(text, site, MXObserver(name=site))
            with self.assertRaisesRegex(ValueError, site):
                validate_gemma4_ple_projection_slicing(text)
        for role in ("projection", "norm"):
            text, _, _ = self._fixture(UINT8)
            module = getattr(
                text,
                (
                    "per_layer_model_projection"
                    if role == "projection"
                    else "per_layer_projection_norm"
                ),
            ).wrapped
            module.obs_act_in = MinMaxObserver(
                name="act_in", qscheme=QScheme.PER_CHANNEL_ASYMM, channel_axis=-1
            )
            with self.assertRaisesRegex(ValueError, role + r"\.obs_act_in"):
                validate_gemma4_ple_projection_slicing(text)
        text, _, _ = self._fixture(UINT8)
        text.per_layer_model_projection.wrapped.obs_weight = MXObserver(name="weight")
        with self.assertRaisesRegex(ValueError, "weight policy"):
            validate_gemma4_ple_projection_slicing(text)

    def test_actual_circle_bounded_linear_shapes_domains_and_nonmutation(self):
        import tico
        from circle_schema import circle
        from tico.circle import CircleDocument
        from tico.circle.runtime import CircleReferenceRuntime, ExecutionMode
        from tico.circle.runtime.quantization import fake_quantize_array

        for dtype in (None, UINT8, INT16):
            text, h0, token = self._fixture(dtype)
            previous = Gemma4PLEProjectionExportAdapter(text)
            with torch.no_grad():
                previous_result = previous(h0, token)
            before = copy.deepcopy(text.state_dict())
            for start, end, hidden, selected in (
                (0, 1, h0, token[:, :, :1]),
                (1, 3, h0[:, 2:3], token[:, 2:3, 1:]),
            ):
                with self.subTest(dtype=dtype, layer_start=start):
                    stage = Gemma4PLEProjectionExportAdapter(
                        text, layer_start=start, layer_end=end
                    )
                    with torch.no_grad():
                        program = torch.export.export(
                            stage, (hidden, selected), strict=False
                        )
                        converted = tico.convert_from_exported_program(program)
                    linear_nodes = [
                        node
                        for node in program.graph.nodes
                        if node.target == torch.ops.aten.linear.default
                    ]
                    self.assertEqual(len(linear_nodes), 1)
                    self.assertEqual(
                        tuple(linear_nodes[0].meta["val"].shape),
                        (1, hidden.shape[1], (end - start) * 4),
                    )
                    self.assertFalse(
                        any(
                            tuple(node.meta["val"].shape) == (1, 5, 3, 4)
                            for node in program.graph.nodes
                            if isinstance(node.meta.get("val"), torch.Tensor)
                        )
                    )
                    document = CircleDocument.from_bytes(converted.circle_binary)
                    document.verify()
                    runtime = CircleReferenceRuntime(document)
                    encoded = []
                    for value, index in zip((hidden, selected), runtime.program.inputs):
                        tensor = runtime.program.tensor(index)
                        array = value.detach().numpy()
                        encoded.append(
                            fake_quantize_array(array, tensor)
                            if tensor.is_quantized_integer
                            else array
                        )
                    with torch.no_grad():
                        expected = stage(
                            *(torch.from_numpy(np.array(x, copy=True)) for x in encoded)
                        ).numpy()
                    result = runtime.run(
                        encoded,
                        mode=(
                            ExecutionMode.FAKE_QUANTIZE
                            if dtype
                            else ExecutionMode.NATIVE
                        ),
                    )
                    output = runtime.program.tensor(runtime.program.outputs[0])
                    if dtype:
                        assert output.quantization is not None
                        step = max(output.quantization.scale)
                    else:
                        step = 0.0
                    np.testing.assert_allclose(
                        result.outputs[0], expected, rtol=1e-5, atol=step + 1e-6
                    )
                    graph = document.model.subgraphs[0]
                    operator_codes = {
                        document.model.operatorCodes[operator.opcodeIndex].builtinCode
                        for operator in graph.operators
                    }
                    self.assertTrue(
                        operator_codes.isdisjoint(
                            {
                                circle.BuiltinOperator.BuiltinOperator.QUANTIZE,
                                circle.BuiltinOperator.BuiltinOperator.DEQUANTIZE,
                            }
                        )
                    )
                    fc = [
                        operator
                        for operator in graph.operators
                        if document.model.operatorCodes[
                            operator.opcodeIndex
                        ].builtinCode
                        == circle.BuiltinOperator.BuiltinOperator.FULLY_CONNECTED
                    ]
                    self.assertEqual(len(fc), 1)
                    weight = graph.tensors[fc[0].inputs[1]]
                    self.assertEqual(tuple(weight.shape), ((end - start) * 4, 16))
                    if dtype:
                        state = (
                            text.per_layer_model_projection.wrapped.obs_weight.require_frozen()
                        )
                        np.testing.assert_array_equal(
                            weight.quantization.scale,
                            state.scale[start * 4 : end * 4].numpy(),
                        )
                        np.testing.assert_array_equal(
                            weight.quantization.zeroPoint,
                            state.zero_point[start * 4 : end * 4].numpy(),
                        )
                    self.assertFalse(
                        any(
                            tuple(tensor.shape) == (12, 16) and tensor.buffer != 0
                            for tensor in graph.tensors
                        )
                    )
                    for index, observer in (
                        (
                            graph.inputs[0],
                            text.per_layer_model_projection.wrapped.obs_act_in,
                        ),
                        (graph.inputs[1], text.obs_per_layer_token_inputs),
                        (graph.outputs[0], text.obs_per_layer_inputs),
                    ):
                        tensor = graph.tensors[index]
                        if dtype:
                            self.assertEqual(
                                tensor.type,
                                (
                                    circle.TensorType.TensorType.UINT8
                                    if dtype == UINT8
                                    else circle.TensorType.TensorType.INT16
                                ),
                            )
                            state = observer.require_frozen()
                            np.testing.assert_array_equal(
                                tensor.quantization.scale,
                                state.scale.numpy().reshape(-1),
                            )
                            np.testing.assert_array_equal(
                                tensor.quantization.zeroPoint,
                                state.zero_point.numpy().reshape(-1),
                            )
                        else:
                            self.assertEqual(
                                tensor.type, circle.TensorType.TensorType.FLOAT32
                            )
                    self._assert_state(before, text)
                    # Shape failure must leave source and previously made stages intact.
                    with self.assertRaises((RuntimeError, ValueError)):
                        torch.export.export(stage, (hidden, token), strict=False)
                    self._assert_state(before, text)
                    with torch.no_grad():
                        self.assertTrue(
                            torch.equal(previous(h0, token), previous_result)
                        )


if __name__ == "__main__":
    unittest.main()
