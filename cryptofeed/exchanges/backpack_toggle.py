from __future__ import annotations

import os
from threading import RLock
from typing import Callable, List

_DEFAULT_ENV_VALUE = os.environ.get("CRYPTOFEED_BACKPACK_NATIVE", "false").lower() in {"1", "true", "yes", "on"}

_lock = RLock()
_enabled = _DEFAULT_ENV_VALUE
_callbacks: List[Callable[[bool], None]] = []


def is_backpack_native_enabled() -> bool:
    with _lock:
        return _enabled


def register_backpack_toggle_callback(callback: Callable[[bool], None]) -> None:
    with _lock:
        _callbacks.append(callback)
        callback(_enabled)


def set_backpack_native_enabled(enabled: bool) -> None:
    global _enabled
    with _lock:
        if _enabled == enabled:
            return
        _enabled = enabled
        for callback in list(_callbacks):
            callback(_enabled)


def coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def maybe_enable_from_config(config) -> None:
    try:
        exchanges = config["exchanges"]
        backpack = exchanges["backpack"]
        native_enabled = backpack["native_enabled"]
    except KeyError:
        return
    if native_enabled == {}:
        return
    set_backpack_native_enabled(coerce_bool(native_enabled))
