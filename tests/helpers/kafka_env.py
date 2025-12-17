"""Shared helpers for Kafka/Redpanda test configuration."""
from __future__ import annotations

import os
from typing import List


DEFAULT_PORT = "19092"
DEFAULT_HOST_INTERNAL = "redpanda"
DEFAULT_HOST_EXTERNAL = "localhost"


def _parse_env_var(value: str | None) -> List[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def get_bootstrap_servers(*, internal: bool = False) -> List[str]:
    """Return bootstrap servers list using a single env contract.

    Precedence:
    1) KAFKA_BOOTSTRAP_SERVERS (comma-separated)
    2) REDPANDA_HOST_PORT (for compose default) combined with localhost/redpanda host
    3) Fallback to localhost:19092
    """

    env_list = _parse_env_var(os.getenv("KAFKA_BOOTSTRAP_SERVERS"))
    if env_list:
        return env_list

    port = os.getenv("REDPANDA_HOST_PORT", DEFAULT_PORT)
    host = DEFAULT_HOST_INTERNAL if internal else DEFAULT_HOST_EXTERNAL
    return [f"{host}:{port}"]


def get_bootstrap_servers_str(*, internal: bool = False) -> str:
    """Convenience string form."""
    return ",".join(get_bootstrap_servers(internal=internal))
