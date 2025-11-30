"""Minimal deprecation helpers for Kafka backend."""

from __future__ import annotations

import inspect
import warnings


def _resolve_user_stacklevel() -> int:
    """Find caller stacklevel outside kafka backends and site-packages."""
    stack = inspect.stack()
    for idx, frame_info in enumerate(stack[1:], start=2):
        filename = frame_info.filename
        if "cryptofeed/backends/kafka" in filename:
            continue
        if "site-packages" in filename:
            continue
        return idx
    return max(2, len(stack))


def emit_class_deprecation_warning(class_name: str, replacement: str) -> None:
    warnings.warn(
        f"{class_name} is deprecated; use {replacement} instead.",
        DeprecationWarning,
        stacklevel=_resolve_user_stacklevel(),
    )


def emit_import_deprecation_warning(old_path: str, new_path: str) -> None:
    warnings.warn(
        f"Import path '{old_path}' is deprecated; use '{new_path}' instead.",
        DeprecationWarning,
        stacklevel=_resolve_user_stacklevel(),
    )
