"""
Basic health check for Kafka connectivity (Phase 2 - Task 14.3 simplified).
"""

from __future__ import annotations

import time
from typing import Any, Dict, Iterable, Optional

from confluent_kafka import Producer


def check_kafka_health(bootstrap_servers: Iterable[str], timeout_ms: int = 3000) -> Dict[str, Any]:
    """
    Validate Kafka connectivity by instantiating a Producer and calling list_topics.

    Args:
        bootstrap_servers: List of Kafka broker addresses
        timeout_ms: Timeout for connectivity check in milliseconds

    Returns:
        Dictionary with health check results:
        - ok: bool (True if connected)
        - latency_ms: float (connection latency)
        - error: str | None (error message if failed)
        - bootstrap: list (broker addresses used)
    """
    start = time.time()
    try:
        bootstrap = ",".join(bootstrap_servers)
        config = {"bootstrap.servers": bootstrap}
        producer = Producer(config)
        producer.list_topics(timeout=timeout_ms / 1000 if timeout_ms else None)
        latency_ms = (time.time() - start) * 1000
        return {
            "ok": True,
            "latency_ms": latency_ms,
            "error": None,
            "bootstrap": list(bootstrap_servers),
        }
    except Exception as exc:
        latency_ms = (time.time() - start) * 1000
        return {
            "ok": False,
            "latency_ms": latency_ms,
            "error": str(exc),
            "bootstrap": list(bootstrap_servers),
        }


__all__ = ["check_kafka_health"]
