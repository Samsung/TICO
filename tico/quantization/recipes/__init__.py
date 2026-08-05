"""Reusable quantization recipes used by thin example CLIs."""

from typing import Any, TYPE_CHECKING

__all__ = ["QuantizationRunner"]

if TYPE_CHECKING:
    from .runner import QuantizationRunner


def __getattr__(name: str) -> Any:
    if name == "QuantizationRunner":
        from .runner import QuantizationRunner

        return QuantizationRunner

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
