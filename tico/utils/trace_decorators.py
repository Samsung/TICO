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

from functools import wraps

import torch
from torch.export import ExportedProgram

from tico.utils.diff_graph import capture, capture_const, log, log_const
from tico.utils.passes import PassBase


def trace_const_diff_on_pass(cls):
    """Decorator for PassBase to trace const diff"""

    assert issubclass(cls, PassBase), type(cls)

    def _call_traced(fn):
        @wraps(fn)
        def wrapped(self, exported_program, graph_module):
            assert isinstance(exported_program, ExportedProgram)
            assert isinstance(graph_module, torch.fx.GraphModule), type(graph_module)
            capture_const(exported_program)
            ret = fn(self, exported_program, graph_module)
            log_const(exported_program, title=str(cls.__name__), recapture=False)
            return ret

        return wrapped

    # replace call function it with traced version
    for key, val in vars(cls).items():
        if key == "call":
            setattr(cls, key, _call_traced(val))
    return cls


def trace_graph_diff_on_pass(cls):
    """Decorator for PassBase to trace graph diff"""

    assert issubclass(cls, PassBase), type(cls)

    def _call_traced(fn):
        @wraps(fn)
        def wrapped(self, exported_program, graph_module):
            assert isinstance(exported_program, ExportedProgram)
            assert isinstance(graph_module, torch.fx.GraphModule), type(graph_module)
            capture(graph_module.graph)
            ret = fn(self, exported_program, graph_module)
            log(graph_module.graph, title=str(cls.__name__), recapture=False)
            return ret

        return wrapped

    # replace call function it with traced version
    for key, val in vars(cls).items():
        if key == "call":
            setattr(cls, key, _call_traced(val))
    return cls


def trace_const_diff_on_func(fn):
    """Decorator for function to trace const diff"""

    @wraps(fn)
    def wrapped(ep: torch.export.ExportedProgram):
        assert isinstance(ep, torch.export.ExportedProgram)
        capture_const(ep)
        ret = fn(ep)
        log_const(ret, title=str(fn.__name__), recapture=False)
        return ret

    return wrapped


def trace_graph_diff_on_func(fn):
    """Decorator for function to trace graph diff"""

    @wraps(fn)
    def wrapped(ep: torch.export.ExportedProgram):
        assert isinstance(ep, torch.export.ExportedProgram)
        capture(ep.graph)
        ret = fn(ep)
        log(ret.graph, title=str(fn.__name__), recapture=False)
        return ret

    return wrapped
