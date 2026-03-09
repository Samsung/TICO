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

import types
import unittest
from unittest.mock import MagicMock, patch

import torch

from tico.quantization.wrapq.utils.check_missing_qparam import check_missing_qparam
from tico.serialize.quant_param import QPARAM_KEY


class _FakeTarget:
    def __init__(self, name: str):
        self.__name__ = name


class _FakeNode:
    def __init__(
        self,
        *,
        op: str = "call_function",
        target=None,
        meta=None,
        users=None,
        name: str = "node",
    ):
        self.op = op
        self.target = target if target is not None else _FakeTarget("fake_op")
        self.meta = meta if meta is not None else {}
        self.users = users if users is not None else {}
        self.name = name


def _make_exported_program(nodes):
    graph = types.SimpleNamespace(nodes=nodes)
    graph_module = types.SimpleNamespace(graph=graph)
    exported_program = types.SimpleNamespace(graph_module=graph_module)
    return exported_program


class CheckMissingQParamTest(unittest.TestCase):
    def test_no_missing_qparam(self):
        node_quantized = _FakeNode(
            name="linear_0",
            target=_FakeTarget("linear"),
            meta={
                "val": torch.randn(1, 4),
                QPARAM_KEY: object(),
            },
            users={"user0": object()},
        )
        node_non_tensor = _FakeNode(
            name="shape_0",
            target=_FakeTarget("sym_size"),
            meta={"val": 16},
            users={},
        )
        node_non_call = _FakeNode(
            op="placeholder",
            name="input_0",
            meta={"val": torch.randn(1, 4)},
        )

        exported_program = _make_exported_program(
            [node_quantized, node_non_tensor, node_non_call]
        )

        mock_logger = MagicMock()

        with patch(
            "tico.quantization.wrapq.utils.check_missing_qparam.logging.getLogger",
            return_value=mock_logger,
        ), patch("builtins.print") as mock_print:
            check_missing_qparam(exported_program, strict=False)

        mock_logger.debug.assert_any_call("[QuantCheck] quantized nodes : 1")
        mock_logger.debug.assert_any_call("[QuantCheck] fp nodes        : 0")
        mock_logger.warning.assert_not_called()
        mock_print.assert_not_called()

    def test_missing_qparam_prints_warning_and_logs_details(self):
        node_quantized = _FakeNode(
            name="linear_0",
            target=_FakeTarget("linear"),
            meta={
                "val": torch.randn(1, 4),
                QPARAM_KEY: object(),
                "stack_trace": "model.py:10",
            },
            users={"user0": object()},
        )
        node_missing = _FakeNode(
            name="add_0",
            target=_FakeTarget("add"),
            meta={
                "val": torch.randn(1, 4),
                "stack_trace": "model.py:20",
            },
            users={"user0": object(), "user1": object()},
        )

        exported_program = _make_exported_program([node_quantized, node_missing])

        mock_logger = MagicMock()

        with patch(
            "tico.quantization.wrapq.utils.check_missing_qparam.logging.getLogger",
            return_value=mock_logger,
        ), patch("builtins.print") as mock_print:
            check_missing_qparam(exported_program, strict=False)

        mock_logger.debug.assert_any_call("[QuantCheck] quantized nodes : 1")
        mock_logger.debug.assert_any_call("[QuantCheck] fp nodes        : 1")

        mock_print.assert_called_once_with(
            "[QuantCheck] WARNING: 1 nodes without qparam detected " "(see logs)."
        )

        dummy_name = "add_0"
        dummy_target = "add"
        dummy_users = 2
        dummy_trace = "model.py:20"
        mock_logger.debug.assert_any_call(
            f"[QuantCheck] Missing qparam:\n"
            f"  name   : {dummy_name}\n"
            f"  target : {dummy_target}\n"
            f"  users  : {dummy_users}\n"
            f"  trace  : {dummy_trace}",
        )

    def test_strict_mode_raises_runtime_error(self):
        node_missing = _FakeNode(
            name="matmul_0",
            target=_FakeTarget("matmul"),
            meta={
                "val": torch.randn(2, 2),
                "stack_trace": "model.py:30",
            },
            users={"user0": object()},
        )

        exported_program = _make_exported_program([node_missing])

        mock_logger = MagicMock()

        with patch(
            "tico.quantization.wrapq.utils.check_missing_qparam.logging.getLogger",
            return_value=mock_logger,
        ), patch("builtins.print") as mock_print:
            with self.assertRaisesRegex(
                RuntimeError,
                r"\[QuantCheck\] 1 nodes without qparam detected\.",
            ):
                check_missing_qparam(exported_program, strict=True)

        mock_logger.debug.assert_any_call("[QuantCheck] quantized nodes : 0")
        mock_logger.debug.assert_any_call("[QuantCheck] fp nodes        : 1")
        mock_print.assert_called_once_with(
            "[QuantCheck] WARNING: 1 nodes without qparam detected " "(see logs)."
        )

    def test_ignore_targets_skips_nodes_from_check_and_counts(self):
        ignored_target = _FakeTarget("getitem")
        checked_target = _FakeTarget("linear")

        node_ignored_missing = _FakeNode(
            name="getitem_0",
            target=ignored_target,
            meta={
                "val": torch.randn(1, 4),
                "stack_trace": "model.py:40",
            },
            users={"user0": object()},
        )
        node_checked_quantized = _FakeNode(
            name="linear_0",
            target=checked_target,
            meta={
                "val": torch.randn(1, 4),
                QPARAM_KEY: object(),
                "stack_trace": "model.py:41",
            },
            users={"user0": object()},
        )

        exported_program = _make_exported_program(
            [node_ignored_missing, node_checked_quantized]
        )

        mock_logger = MagicMock()

        with patch(
            "tico.quantization.wrapq.utils.check_missing_qparam.logging.getLogger",
            return_value=mock_logger,
        ), patch("builtins.print") as mock_print:
            check_missing_qparam(
                exported_program,
                strict=False,
                ignore_targets={ignored_target},
            )

        mock_logger.debug.assert_any_call("[QuantCheck] quantized nodes : 1")
        mock_logger.debug.assert_any_call("[QuantCheck] fp nodes        : 0")
        mock_logger.warning.assert_not_called()
        mock_print.assert_not_called()

    def test_tensor_meta_without_val_is_still_checked(self):
        node_missing = _FakeNode(
            name="reshape_0",
            target=_FakeTarget("reshape"),
            meta={
                "tensor_meta": object(),
                "stack_trace": "model.py:50",
            },
            users={"user0": object()},
        )

        exported_program = _make_exported_program([node_missing])

        mock_logger = MagicMock()

        with patch(
            "tico.quantization.wrapq.utils.check_missing_qparam.logging.getLogger",
            return_value=mock_logger,
        ), patch("builtins.print") as mock_print:
            check_missing_qparam(exported_program, strict=False)

        mock_logger.debug.assert_any_call("[QuantCheck] quantized nodes : 0")
        mock_logger.debug.assert_any_call("[QuantCheck] fp nodes        : 1")
        mock_print.assert_called_once()
