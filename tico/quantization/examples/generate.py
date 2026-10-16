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

"""Generate greedily from an exported LLaMA Circle bundle."""

import argparse
import json
from pathlib import Path

import numpy as np


def _parse_input_ids(value: str) -> list[int]:
    try:
        token_ids = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--input-ids must be a nonempty comma-separated list of integers"
        ) from exc
    if any(not 0 <= token < (1 << 63) for token in token_ids):
        raise argparse.ArgumentTypeError(
            "--input-ids must contain nonnegative int64 token IDs"
        )
    return token_ids


def _parse_max_new_tokens(value: str) -> int:
    try:
        count = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--max-new-tokens must be a nonnegative integer"
        ) from exc
    if count < 0:
        raise argparse.ArgumentTypeError(
            "--max-new-tokens must be a nonnegative integer"
        )
    return count


def _parse_prompt(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("--prompt must contain text")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate greedily from a local LLaMA Circle bundle."
    )
    parser.add_argument("--bundle", type=Path, required=True, help="Bundle directory.")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--prompt",
        type=_parse_prompt,
        help="Prompt text; requires the bundle's local tokenizer and transformers.",
    )
    inputs.add_argument(
        "--input-ids",
        type=_parse_input_ids,
        metavar="1,2,3",
        help="Comma-separated token IDs; does not load a tokenizer.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=_parse_max_new_tokens,
        default=16,
        help="Maximum greedy decoding steps (default: 16); must fit bundle capacity.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Also save the token IDs, decoded continuation, and execution mode as JSON.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    from tico.quantization.recipes.runtime.llama import LlamaCircleRuntime

    runtime = LlamaCircleRuntime(args.bundle)
    tokenizer = None
    if args.prompt is not None:
        if runtime.tokenizer_dir is None:
            raise ValueError(
                "This bundle has no tokenizer. Use --input-ids or export a bundle "
                "with tokenizer files."
            )
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            runtime.tokenizer_dir,
            local_files_only=True,
            trust_remote_code=False,
        )
        input_ids = tokenizer.encode(args.prompt, add_special_tokens=True)
    else:
        input_ids = args.input_ids

    output_ids = runtime.generate(
        np.asarray([input_ids], dtype=np.int64),
        max_new_tokens=args.max_new_tokens,
    )[0].tolist()
    generated_ids = output_ids[len(input_ids) :]
    result = {
        "input_token_ids": input_ids,
        "token_ids": output_ids,
        "generated_token_ids": generated_ids,
        "text": (
            tokenizer.decode(generated_ids, skip_special_tokens=True)
            if tokenizer is not None
            else None
        ),
        "execution_mode": runtime.execution_mode,
    }
    serialized = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output_json is not None:
        args.output_json.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
