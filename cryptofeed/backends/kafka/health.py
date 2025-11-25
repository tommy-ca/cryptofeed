"""
Health checks and monitoring helpers for Kafka backends.

Supports connectivity validation for both modern (Phase 2) and legacy
Kafka configurations, lightweight status reporting, and periodic health
loops suitable for wiring into existing logging/metrics.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, Iterable, Optional

from confluent_kafka import Producer

from .callback import KafkaConfig
from .producer import _normalize_bootstrap_servers

LOG = logging.getLogger("feedhandler")


@dataclass
class KafkaHealthStatus:
    implementation: str
    ok: bool
    latency_ms: float
    error: Optional[str] = None
    details: Dict[str, Any] = None

    def as_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Ensure details is a dict even if None
        data["details"] = data.get("details") or {}
        return data


class KafkaHealthCheck:
    """Connectivity checks for Kafka backends."""

    @staticmethod
    def check_connectivity(
        bootstrap_servers: Iterable[str],
        *,
        implementation: str,
        producer_factory: Callable[[Dict[str, Any]], Producer] | None = None,
        timeout_ms: int = 3000,
        **producer_kwargs: Any,
    ) -> KafkaHealthStatus:
        """
        Validate connectivity by instantiating a Producer and calling list_topics.
        """
        producer_factory = producer_factory or Producer
        start = time.time()
        try:
            config: Dict[str, Any] = dict(producer_kwargs)
            config["bootstrap.servers"] = _normalize_bootstrap_servers(bootstrap_servers)
            producer = producer_factory(config)
            producer.list_topics(timeout=timeout_ms / 1000 if timeout_ms else None)
        except Exception as exc:
            latency_ms = (time.time() - start) * 1000
            LOG.warning(
                "Kafka health check failed (%s): %s", implementation, exc, exc_info=False
            )
            return KafkaHealthStatus(
                implementation=implementation,
                ok=False,
                latency_ms=latency_ms,
                error=str(exc),
                details={"bootstrap": list(bootstrap_servers)},
            )
        else:
            latency_ms = (time.time() - start) * 1000
            LOG.debug(
                "Kafka health check OK (%s) in %.1fms", implementation, latency_ms
            )
            return KafkaHealthStatus(
                implementation=implementation,
                ok=True,
                latency_ms=latency_ms,
                details={"bootstrap": list(bootstrap_servers)},
            )

    @staticmethod
    def check_modern(
        config: KafkaConfig,
        *,
        producer_factory: Callable[[Dict[str, Any]], Producer] | None = None,
        timeout_ms: int = 3000,
    ) -> KafkaHealthStatus:
        """Connectivity check for modern KafkaConfig-based setups."""
        return KafkaHealthCheck.check_connectivity(
            bootstrap_servers=config.bootstrap_servers,
            implementation="modern",
            producer_factory=producer_factory,
            timeout_ms=timeout_ms,
            acks=config.acks,
        )

    @staticmethod
    def check_legacy(
        legacy_config: Dict[str, Any],
        *,
        producer_factory: Callable[[Dict[str, Any]], Producer] | None = None,
        timeout_ms: int = 3000,
    ) -> KafkaHealthStatus:
        """Connectivity check for legacy dict-style configs."""
        servers = legacy_config.get("bootstrap_servers") or []
        return KafkaHealthCheck.check_connectivity(
            bootstrap_servers=servers,
            implementation="legacy",
            producer_factory=producer_factory,
            timeout_ms=timeout_ms,
        )


async def start_periodic_health_checks(
    interval_sec: float,
    check_fn: Callable[[], KafkaHealthStatus],
    on_result: Callable[[KafkaHealthStatus], Any] | None = None,
    *,
    max_runs: int | None = None,
    alert_fn: Callable[[KafkaHealthStatus], Any] | None = None,
    alert_threshold_ms: float = 500.0,
) -> asyncio.Task:
    """
    Start a periodic health loop that executes check_fn every interval_sec.

    Args:
        interval_sec: Interval between checks.
        check_fn: Callable returning KafkaHealthStatus.
        on_result: Optional callback invoked with each status.
        max_runs: Optional limit for number of iterations (useful in tests).
    """

    async def _runner():
        runs = 0
        while max_runs is None or runs < max_runs:
            status = check_fn()
            if on_result:
                on_result(status)
            if alert_fn and (not status.ok or status.latency_ms > alert_threshold_ms):
                alert_fn(status)
            runs += 1
            await asyncio.sleep(interval_sec)

    loop = asyncio.get_running_loop()
    return loop.create_task(_runner(), name="kafka-health-check")


__all__ = ["KafkaHealthStatus", "KafkaHealthCheck", "start_periodic_health_checks"]
