# Vendored from lintrunner-adapters
# (https://github.com/justinchuby/lintrunner-adapters),
# Copyright (c) 2022 Justin Chu, licensed under the MIT License.
# lintrunner-adapters derived this adapter from PyTorch
# (https://github.com/pytorch/pytorch), which is licensed under the
# BSD-style PyTorch license. This file is NOT under the repository's
# Apache-2.0 license; the original notices above must be retained.
#
# Local change from the upstream adapter:
# files are formatted in a ProcessPoolExecutor instead of a ThreadPoolExecutor.
# black/usort are CPU-bound pure-Python work, so threads serialize on the GIL
# and formatting runs ~6x slower than process-based parallelism.

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
import sys
from pathlib import Path

from lintrunner_adapters import LintMessage, LintSeverity
from ufmt.core import make_black_config, ufmt_string  # type: ignore[attr-defined]
from usort import Config as UsortConfig

LINTER_CODE = "UFMT"


def format_error_message(filename: str, err: Exception) -> LintMessage:
    return LintMessage(
        path=filename,
        line=None,
        char=None,
        code=LINTER_CODE,
        severity=LintSeverity.ADVICE,
        name="command-failed",
        original=None,
        replacement=None,
        description=(f"Failed due to {err.__class__.__name__}:\n{err}"),
    )


def check_file(
    filename: str,
) -> list[LintMessage]:
    with open(filename, "rb") as f:
        original = f.read().decode("utf-8")

    try:
        path = Path(filename)

        usort_config = UsortConfig.find(path)
        black_config = make_black_config(path)

        # Use UFMT API to call both usort and black
        replacement = ufmt_string(
            path=path,
            content=original,
            usort_config=usort_config,
            black_config=black_config,
        )

        if original == replacement:
            return []

        return [
            LintMessage(
                path=filename,
                line=None,
                char=None,
                code=LINTER_CODE,
                severity=LintSeverity.WARNING,
                name="format",
                original=original,
                replacement=replacement,
                description="Run `lintrunner -a` to apply this patch.",
            )
        ]
    except Exception as err:
        return [format_error_message(filename, err)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Format files with ufmt (black + usort). Linter code: {LINTER_CODE}",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="verbose logging",
    )
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="<%(processName)s:%(levelname)s> %(message)s",
        level=logging.NOTSET
        if args.verbose
        else logging.DEBUG
        if len(args.filenames) < 1000
        else logging.INFO,
        stream=sys.stderr,
    )

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=os.cpu_count(),
    ) as executor:
        futures = {executor.submit(check_file, x): x for x in args.filenames}
        for future in concurrent.futures.as_completed(futures):
            try:
                for lint_message in future.result():
                    lint_message.display()
            except Exception:
                logging.critical('Failed at "%s".', futures[future])
                raise


if __name__ == "__main__":
    main()
