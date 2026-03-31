"""
Deprecated compatibility shim for the legacy `src` import namespace.

The supported public import surface lives under `voyager_index`.
"""

from __future__ import annotations

import warnings

_WARNED = False


def _warn_deprecated_namespace() -> None:
    global _WARNED
    if _WARNED:
        return
    warnings.warn(
        "`src` is deprecated and will be removed in a future release. "
        "Import from `voyager_index` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    _WARNED = True


_warn_deprecated_namespace()
