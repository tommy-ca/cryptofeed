"""
Simplified deprecation warning system for Kafka backend.

Phase 1 dead code removal: Keep only essential warning functions (23 LOC total).
All timeline infrastructure, communication systems, and decision logs removed.
"""

import warnings


def emit_deprecation_warning(old_api: str, new_api: str, *, stacklevel: int = 2) -> None:
    """Emit generic deprecation warning for API changes."""
    warnings.warn(
        f"{old_api} is deprecated; use {new_api} instead",
        DeprecationWarning,
        stacklevel=stacklevel,
    )


def warn_legacy_usage(class_name: str, *, stacklevel: int = 2) -> None:
    """Emit deprecation warning for legacy Kafka class usage."""
    warnings.warn(
        f"{class_name} is deprecated and will be removed in a future version",
        DeprecationWarning,
        stacklevel=stacklevel,
    )


def emit_class_deprecation_warning(old_name: str, new_name: str, *, stacklevel: int = 2) -> None:
    """Emit deprecation warning for class name changes."""
    warnings.warn(
        f"{old_name} is deprecated; use {new_name} instead",
        DeprecationWarning,
        stacklevel=stacklevel,
    )


def emit_import_deprecation_warning(old_path: str, new_path: str, *, stacklevel: int = 2) -> None:
    """Emit deprecation warning for import path changes."""
    warnings.warn(
        f"Importing {old_path} is deprecated; use {new_path} instead",
        DeprecationWarning,
        stacklevel=stacklevel,
    )
