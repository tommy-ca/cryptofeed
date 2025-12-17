"""
Backward compatibility shim for health check functionality (Task 14.3).

Health check logic has been inlined into KafkaCallback.get_health_status().
This module provides compatibility wrappers for existing code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Callable, Optional


@dataclass
class KafkaHealthStatus:
    """
    Health check status result (backward compatibility wrapper).

    New code should use KafkaCallback.get_health_status() directly.
    """
    implementation: str
    ok: bool
    latency_ms: float
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class KafkaHealthCheck:
    """
    Backward compatibility wrapper for health check functionality.

    New code should use KafkaCallback.get_health_status() instead.
    This class provides static methods that wrap the new API.
    """

    @staticmethod
    def check_connectivity(
        bootstrap_servers: list[str],
        implementation: str,
        producer_factory: Any = None,
        timeout_ms: int = 3000
    ) -> KafkaHealthStatus:
        """
        Check Kafka connectivity (backward compatibility wrapper).

        Args:
            bootstrap_servers: List of broker addresses
            implementation: Implementation identifier (e.g., "modern")
            producer_factory: Optional producer factory (for testing)
            timeout_ms: Timeout in milliseconds

        Returns:
            KafkaHealthStatus with connectivity result
        """
        import time
        from confluent_kafka import Producer

        start = time.time()
        try:
            bootstrap = ",".join(bootstrap_servers)
            config = {"bootstrap.servers": bootstrap}
            factory = producer_factory or Producer
            producer = factory(config)

            timeout_sec = timeout_ms / 1000 if timeout_ms else None
            producer.list_topics(timeout=timeout_sec)

            latency_ms = (time.time() - start) * 1000
            return KafkaHealthStatus(
                implementation=implementation,
                ok=True,
                latency_ms=latency_ms,
                error=None,
                details={"bootstrap": list(bootstrap_servers)},
            )
        except Exception as exc:
            latency_ms = (time.time() - start) * 1000
            return KafkaHealthStatus(
                implementation=implementation,
                ok=False,
                latency_ms=latency_ms,
                error=str(exc),
                details={"bootstrap": list(bootstrap_servers)},
            )

    @staticmethod
    def check_modern(
        kafka_config: Any,
        producer_factory: Any = None,
        timeout_ms: int = 3000
    ) -> KafkaHealthStatus:
        """
        Check Kafka connectivity using KafkaConfig (backward compatibility).

        Args:
            kafka_config: KafkaConfig object
            producer_factory: Optional producer factory (for testing)
            timeout_ms: Timeout in milliseconds

        Returns:
            KafkaHealthStatus with connectivity result
        """
        return KafkaHealthCheck.check_connectivity(
            bootstrap_servers=list(kafka_config.bootstrap_servers),
            implementation="modern",
            producer_factory=producer_factory,
            timeout_ms=timeout_ms,
        )


async def _periodic_health_check_loop(
    interval_sec: float,
    check_fn: Callable[[], KafkaHealthStatus],
    max_runs: Optional[int],
    alert_fn: Optional[Callable[[KafkaHealthStatus], None]]
):
    """Internal coroutine for periodic health checks."""
    import asyncio

    run_count = 0
    while max_runs is None or run_count < max_runs:
        status = check_fn()

        if alert_fn and not status.ok:
            alert_fn(status)

        run_count += 1

        if max_runs is None or run_count < max_runs:
            await asyncio.sleep(interval_sec)


async def start_periodic_health_checks(
    interval_sec: float,
    check_fn: Callable[[], KafkaHealthStatus],
    max_runs: Optional[int] = None,
    alert_fn: Optional[Callable[[KafkaHealthStatus], None]] = None
):
    """
    Run periodic health checks (backward compatibility wrapper).

    Args:
        interval_sec: Interval between checks in seconds
        check_fn: Function that performs health check
        max_runs: Maximum number of runs (None for infinite)
        alert_fn: Optional function to call when health check fails

    Returns:
        Asyncio task that can be awaited
    """
    import asyncio

    # Create and return a task that can be awaited
    return asyncio.create_task(
        _periodic_health_check_loop(interval_sec, check_fn, max_runs, alert_fn)
    )


__all__ = [
    "KafkaHealthStatus",
    "KafkaHealthCheck",
    "start_periodic_health_checks",
]
