"""
Deprecated shim for `src.kernels`.
"""

from __future__ import annotations

from importlib import import_module

from src import _warn_deprecated_namespace

_warn_deprecated_namespace()
_impl = import_module("voyager_index._internal.kernels")

__path__ = list(_impl.__path__)
__all__ = getattr(_impl, "__all__", [])


def __getattr__(name: str):
    return getattr(_impl, name)
