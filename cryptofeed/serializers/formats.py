"""Utilities for managing backend serialization format selection."""

from __future__ import annotations

import os
from typing import Optional

SUPPORTED_SERIALIZATION_FORMATS = {"json", "protobuf"}

DEFAULT_SERIALIZATION_FORMAT = "json"

CALLBACK_FORMAT_ENV_VAR = "CRYPTOFEED_CALLBACK_FORMAT"


def validate_serialization_format(format_str: str) -> str:
    """Normalize and validate a serialization format value."""

    if format_str is None:
        raise ValueError("Serialization format cannot be None")

    normalized = format_str.strip().lower()
    if normalized in SUPPORTED_SERIALIZATION_FORMATS:
        return normalized

    valid = ", ".join(sorted(SUPPORTED_SERIALIZATION_FORMATS))
    raise ValueError(f"Invalid serialization format '{format_str}'. Valid formats: {valid}")


def get_serialization_format_from_env() -> Optional[str]:
    """Return the serialization format defined via environment override."""

    raw_value = os.getenv(CALLBACK_FORMAT_ENV_VAR)
    if not raw_value:
        return None

    return validate_serialization_format(raw_value)


def resolve_serialization_format(preferred: Optional[str], default: str = DEFAULT_SERIALIZATION_FORMAT) -> str:
    """Resolve the active serialization format using env → explicit → default precedence."""

    env_value = get_serialization_format_from_env()
    if env_value is not None:
        return env_value

    if preferred is not None:
        return validate_serialization_format(preferred)

    return validate_serialization_format(default)


__all__ = [
    "CALLBACK_FORMAT_ENV_VAR",
    "DEFAULT_SERIALIZATION_FORMAT",
    "SUPPORTED_SERIALIZATION_FORMATS",
    "get_serialization_format_from_env",
    "resolve_serialization_format",
    "validate_serialization_format",
]

