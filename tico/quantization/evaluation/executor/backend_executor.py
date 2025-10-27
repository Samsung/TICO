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

import abc
from typing import Any

from tico.utils.model import CircleModel


class BackendExecutor(abc.ABC):
    """
    Abstract base class for executing a circle model on a specific backend.
    """

    @abc.abstractmethod
    def compile(self, circle_model: CircleModel) -> None:
        """
        Compile the circle model for this backend, if needed.

        Parameters
        -----------
        circle_model
            The circle model to be compiled.
        """
        pass

    @abc.abstractmethod
    def run_inference(self, input_data: Any) -> Any:
        """
        Run inference using the compiled (or directly usable) model
         on the given input data.

        Parameters
        -----------
        input_data
            The input data to be fed to the model.

        Returns
        --------
        Any
            The model's inference output.
        """
        pass
