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

from pathlib import Path
from typing import ClassVar

import numpy as np
import torch
from cffi import FFI


class Interpreter:
    """
    Python wrapper for C++ luci-interperter class in ONE using CFFI.

    This class provides a Python interface to the underlying C++ luci-interpreter class in ONE,
     preserving the original C++ API. Each method corresponds to a method in the C++ class,
    with additional error handling implemented to ensure that C++ exceptions are captured and
    translated into Python errors.

    Note that each method includes `check_for_errors` at the end of the body to catch any C++
     exceptions and translate them into Python exceptions. This ensures that errors in the C++
    library do not cause undefined behavior in Python.
    """

    LIB_PATH: ClassVar[Path] = Path("/usr/share/one/lib/libcircle_interpreter_cffi.so")

    @classmethod
    def is_available(cls) -> bool:
        return cls.LIB_PATH.is_file()

    def __init__(self, circle_binary: bytes):
        self.ffi = FFI()
        self.ffi.cdef("""
          typedef struct InterpreterWrapper InterpreterWrapper;

          const char *get_last_error(void);
          void clear_last_error(void);
          InterpreterWrapper *Interpreter_new(const uint8_t *data, const size_t data_size);
          void Interpreter_delete(InterpreterWrapper *intp);
          void Interpreter_interpret(InterpreterWrapper *intp);
          void Interpreter_writeInputTensor(InterpreterWrapper *intp, const int input_idx, const void *data, size_t input_size);
          void Interpreter_readOutputTensor(InterpreterWrapper *intp, const int output_idx, void *output, size_t output_size);
        """)
        # TODO Check if one-compiler version is compatible. Whether it has .so file or not for CFFI.
        if not self.is_available():
            raise RuntimeError("Please install one-compiler for circle inference.")
        self.C = self.ffi.dlopen(str(self.LIB_PATH))

        # Initialize interpreter
        self.intp = self.C.Interpreter_new(circle_binary, len(circle_binary))
        self.check_for_errors()

    def delete(self):
        if not hasattr(self, "C") or not hasattr(self, "intp"):
            return
        self.C.Interpreter_delete(self.intp)
        self.check_for_errors()
        del self.intp

    def interpret(self):
        self.C.Interpreter_interpret(self.intp)
        self.check_for_errors()

    def writeInputTensor(self, input_idx: int, input_data: torch.Tensor):
        input_as_numpy = input_data.numpy()
        # cffi.from_buffer() only accepts C-contiguous array.
        input_as_numpy = np.ascontiguousarray(input_as_numpy)
        c_input = self.ffi.from_buffer(input_as_numpy)
        self.C.Interpreter_writeInputTensor(
            self.intp, input_idx, c_input, input_data.nbytes
        )
        self.check_for_errors()

    def readOutputTensor(self, output_idx: int, output: np.ndarray):
        c_output = self.ffi.from_buffer(output)
        self.C.Interpreter_readOutputTensor(
            self.intp, output_idx, c_output, output.nbytes
        )
        self.check_for_errors()

    def check_for_errors(self):
        error_message = self.ffi.string(self.C.get_last_error()).decode("utf-8")
        if error_message:
            self.C.clear_last_error()
            raise RuntimeError(f"C++ Exception: {error_message}")

    def __del__(self):
        self.delete()
