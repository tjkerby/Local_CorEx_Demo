"""Backward-compatible shim for historical imports.

The modern API exposes :class:`BioCorex` from ``local_corex.base``. Importing this
module allows older code that still does ``from local_corex import biocorex`` to
keep working, while providing a gentle deprecation warning.
"""

from __future__ import annotations

import warnings

from .base import BioCorex

warnings.warn(
    "local_corex.biocorex is deprecated; import BioCorex from local_corex or "
    "local_corex.base instead.",
    DeprecationWarning,
    stacklevel=2,
)

Corex = BioCorex

__all__ = ["Corex", "BioCorex"]
