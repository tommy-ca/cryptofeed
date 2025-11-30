"""Shared fixtures for Kafka/Redpanda integration tests."""

from __future__ import annotations

import os
import socket
import subprocess
import time

import pytest


COMPOSE_FILE = os.getenv("REDPANDA_COMPOSE_FILE", "docker/infra/base.yml")
HOST_BOOTSTRAP = os.getenv("REDPANDA_HOST_BOOTSTRAP", "localhost:19092")


def _docker_compose_available() -> bool:
    try:
        result = subprocess.run(
            ["docker", "compose", "version"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return False
    return result.returncode == 0


def _wait_for_port(host: str, port: int, timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            if sock.connect_ex((host, port)) == 0:
                return
        time.sleep(0.5)
    raise TimeoutError(f"Port {host}:{port} not ready after {timeout}s")


@pytest.fixture(scope="session")
def redpanda():
    """Spin up Redpanda via docker compose for Kafka integration tests."""
    if not _docker_compose_available():
        pytest.skip("docker compose not available")

    up = subprocess.run(
        ["docker", "compose", "-f", COMPOSE_FILE, "up", "-d"],
        capture_output=True,
        text=True,
    )
    if up.returncode != 0:
        pytest.skip(f"failed to start redpanda: {up.stderr.strip()}")

    try:
        host, port_str = HOST_BOOTSTRAP.rsplit(":", 1)
        _wait_for_port(host, int(port_str), timeout=30)
        time.sleep(5)
    except Exception as exc:  # pragma: no cover - env-specific
        subprocess.run(["docker", "compose", "-f", COMPOSE_FILE, "logs"])
        subprocess.run(
            ["docker", "compose", "-f", COMPOSE_FILE, "down"], capture_output=True
        )
        raise exc

    yield HOST_BOOTSTRAP

    subprocess.run(
        ["docker", "compose", "-f", COMPOSE_FILE, "down"], capture_output=True
    )
