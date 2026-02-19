"""Backward-compatible shim for historical imports.

Existing code often relied on ``local_corex.linearcorex.Corex``. The new package
structure exposes :class:`LinearCorex` from ``local_corex.base`` instead. Importing
this module keeps the old path working while emitting a deprecation warning.
"""

from __future__ import annotations

import warnings

from .base import LinearCorex

warnings.warn(
    "local_corex.linearcorex is deprecated; import LinearCorex from local_corex "
    "or local_corex.base instead.",
    DeprecationWarning,
    stacklevel=2,
)

Corex = LinearCorex

__all__ = ["Corex", "LinearCorex"]
