"""Shared fixtures for Kafka/Redpanda integration tests."""

from __future__ import annotations

import os
import socket
import subprocess
import time

import pytest


COMPOSE_FILE = os.getenv("REDPANDA_COMPOSE_FILE", "docker/infra/base.yml")
DEFAULT_HOST_PORT = os.getenv("REDPANDA_HOST_PORT", "19092")
HOST_BOOTSTRAP = os.getenv("REDPANDA_HOST_BOOTSTRAP", f"localhost:{DEFAULT_HOST_PORT}")


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


def _port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex((host, port)) == 0


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
    bootstrap = (HOST_BOOTSTRAP.split(",", 1)[0] or HOST_BOOTSTRAP).strip()
    host, port_str = bootstrap.rsplit(":", 1)
    port = int(port_str)

    started_here = False
    if not _port_open(host, port):
        if not _docker_compose_available():
            pytest.skip("docker compose not available")

        up = subprocess.run(
            ["docker", "compose", "-f", COMPOSE_FILE, "up", "-d"],
            capture_output=True,
            text=True,
        )
        if up.returncode != 0:
            pytest.skip(f"failed to start redpanda: {up.stderr.strip()}")
        started_here = True

    try:
        _wait_for_port(host, port, timeout=30)
        time.sleep(5)
    except Exception as exc:  # pragma: no cover - env-specific
        if started_here:
            subprocess.run(["docker", "compose", "-f", COMPOSE_FILE, "logs"])
            subprocess.run(
                ["docker", "compose", "-f", COMPOSE_FILE, "down"], capture_output=True
            )
        raise exc

    yield bootstrap

    if started_here:
        subprocess.run(
            ["docker", "compose", "-f", COMPOSE_FILE, "down"], capture_output=True
        )
