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

import argparse
import os
from typing import Optional

import torch
import yaml

from tico.config import CompileConfigBase, get_default_config

from tico.utils.convert import convert_exported_module_to_circle


def convert(
    input: str,
    output: str,
    verbose: bool = False,
    config: Optional[CompileConfigBase] = None,
):
    if config is None:
        config = get_default_config()
    # TODO Check input and output

    if verbose:
        os.environ["TICO_LOG"] = "4"

    exported_program = torch.export.load(input)
    circle_program = convert_exported_module_to_circle(exported_program, config=config)
    circle_binary = circle_program
    with open(output, "wb") as f:
        f.write(circle_binary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="provide a path to .pt2 model.",
    )

    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="provide a path to .circle model.",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="print logs.",
    )

    parser.add_argument(
        "-c",
        "--config",
        required=False,
        help="provide a path to config file.",
    )

    args = parser.parse_args()

    config = None
    if args.config:
        with open(args.config) as f:
            config_dict = yaml.safe_load(f)

            version = config_dict.get("version", None)
            latest_version = "1.0"

            if version is None:
                raise ValueError(
                    f"'version' field must be provided in the config file. (lastest: {latest_version})"
                )

            if version == "1.0":
                from tico.config.v1 import CompileConfigV1

                config = CompileConfigV1.from_dict(config_dict)
            else:
                raise ValueError(
                    f"Unsupported version '{version}'. (lastest: {latest_version})"
                )

    convert(args.input, args.output, args.verbose, config)


if __name__ == "__main__":
    main()
