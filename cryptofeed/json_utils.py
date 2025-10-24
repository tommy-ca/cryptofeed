"""JSON helper utilities with optional :mod:`orjson` acceleration."""
from __future__ import annotations

import json as _stdlib_json
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping

try:  # pragma: no cover - optional dependency
    import orjson as _orjson
except ModuleNotFoundError:  # pragma: no cover - fallback path
    _orjson = None


JSONType = Any


def _coerce_input(data: str | bytes) -> bytes:
    if isinstance(data, str):
        return data.encode("utf-8")
    return data


def loads(data: str | bytes, *, parse_float: Callable[[str], Any] | None = Decimal, **kwargs: Any) -> JSONType:
    """Deserialize ``data`` into Python objects.

    Falls back to :mod:`json` when Decimal precision or advanced options are
    required. Uses :mod:`orjson` when available and configuration permits.
    """

    use_stdlib = kwargs or (parse_float not in (None, float)) or _orjson is None
    if use_stdlib:
        return _stdlib_json.loads(data, parse_float=parse_float, **kwargs)

    return _orjson.loads(_coerce_input(data))


def dumps(
    obj: Any,
    *,
    ensure_ascii: bool = False,
    separators: tuple[str, str] | None = None,
    default: Callable[[Any], Any] | None = None,
    **kwargs: Any,
) -> str:
    """Serialize ``obj`` into a JSON string."""

    def _default(value: Any) -> Any:
        if isinstance(value, Decimal):
            return str(value)
        # Support common non-JSON types used in payloads
        if isinstance(value, (set, frozenset)):
            return list(value)
        if default is not None:
            return default(value)
        raise TypeError(f"Type is not JSON serializable: {type(value)!r}")

    use_stdlib = ensure_ascii or separators is not None or kwargs or _orjson is None
    if use_stdlib:
        return _stdlib_json.dumps(
            obj,
            ensure_ascii=ensure_ascii,
            separators=separators,
            default=_default,
            **kwargs,
        )

    result = _orjson.dumps(obj, default=_default)
    return result.decode("utf-8")


def dumps_bytes(obj: Any, **kwargs: Any) -> bytes:
    if _orjson is not None and not kwargs:
        try:
            return _orjson.dumps(obj)
        except TypeError:
            return dumps(obj, **kwargs).encode("utf-8")
    return dumps(obj, **kwargs).encode("utf-8")


def load_map(data: Mapping[str, Any] | Iterable[tuple[str, Any]]) -> dict[str, Any]:
    if isinstance(data, Mapping):
        return dict(data)
    return {k: v for k, v in data}


class _JsonNamespace(SimpleNamespace):
    def __init__(self) -> None:
        super().__init__(
            loads=loads,
            dumps=dumps,
            JSONDecodeError=_stdlib_json.JSONDecodeError,
        )


json = _JsonNamespace()


__all__ = ["json", "loads", "dumps", "dumps_bytes", "load_map"]
