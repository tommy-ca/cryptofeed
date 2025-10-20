"""Proxy selection and health helpers."""
from __future__ import annotations

import asyncio
import random
from abc import ABC, abstractmethod
from contextlib import suppress
from datetime import datetime, UTC
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict

from cryptofeed.proxy_config import ProxyPoolConfig, ProxyUrlConfig


class ProxySelector(ABC):
    """Abstract base class for proxy selection strategies."""

    @abstractmethod
    def select(self, proxies: List[ProxyUrlConfig]) -> ProxyUrlConfig:
        raise NotImplementedError

    def record_connection(self, proxy: ProxyUrlConfig) -> None:
        return None

    def record_disconnection(self, proxy: ProxyUrlConfig) -> None:
        return None


class RoundRobinSelector(ProxySelector):
    def __init__(self) -> None:
        self._current_index = 0

    def select(self, proxies: List[ProxyUrlConfig]) -> ProxyUrlConfig:
        if not proxies:
            raise ValueError("No proxies available for selection")
        selected = proxies[self._current_index % len(proxies)]
        self._current_index += 1
        return selected


class RandomSelector(ProxySelector):
    def select(self, proxies: List[ProxyUrlConfig]) -> ProxyUrlConfig:
        if not proxies:
            raise ValueError("No proxies available for selection")
        return random.choice(proxies)


class LeastConnectionsSelector(ProxySelector):
    def __init__(self) -> None:
        self._connection_counts: Dict[str, int] = {}

    def select(self, proxies: List[ProxyUrlConfig]) -> ProxyUrlConfig:
        if not proxies:
            raise ValueError("No proxies available for selection")
        return min(proxies, key=lambda proxy: self._connection_counts.get(proxy.url, 0))

    def record_connection(self, proxy: ProxyUrlConfig) -> None:
        self._connection_counts[proxy.url] = self._connection_counts.get(proxy.url, 0) + 1

    def record_disconnection(self, proxy: ProxyUrlConfig) -> None:
        current = self._connection_counts.get(proxy.url, 0)
        self._connection_counts[proxy.url] = max(0, current - 1)


class HealthCheckResult(BaseModel):
    model_config = ConfigDict(extra='forbid')

    healthy: bool = Field(..., description="Whether the proxy is healthy")
    latency: Optional[float] = Field(default=None, description="Latency in milliseconds")
    error: Optional[str] = Field(default=None, description="Error message if unhealthy")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC), description="Timestamp of check")


class TCPHealthChecker:
    """TCP-based health checker for proxies."""

    def __init__(self, timeout_seconds: int = 5) -> None:
        self.timeout_seconds = timeout_seconds

    async def check_proxy(self, proxy: ProxyUrlConfig) -> HealthCheckResult:
        start_time = datetime.now(UTC)
        try:
            host, port = proxy.host, proxy.port
            if host is None or port is None:
                return self._error_result("Invalid proxy URL - missing host or port", start_time)
            return await self._attempt_connection(host, port, start_time)
        except Exception as exc:
            return self._error_result(f"Health check error: {exc}", start_time)

    async def _attempt_connection(self, host: str, port: int, start_time: datetime) -> HealthCheckResult:
        writer = None
        error_message: Optional[str] = None
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError:
            error_message = f"Connection timeout after {self.timeout_seconds}s"
        except ConnectionRefusedError:
            error_message = "Connection refused"
        except Exception as exc:
            error_message = f"Connection error: {exc}"

        if writer is not None:
            writer.close()
            with suppress(Exception):
                await writer.wait_closed()

        if error_message:
            return self._error_result(error_message, start_time)

        latency_ms = (datetime.now(UTC) - start_time).total_seconds() * 1000
        return self._success_result(latency_ms, start_time)

    def _success_result(self, latency_ms: float, timestamp: datetime) -> HealthCheckResult:
        return HealthCheckResult(healthy=True, latency=latency_ms, timestamp=timestamp)

    def _error_result(self, message: str, timestamp: datetime) -> HealthCheckResult:
        return HealthCheckResult(healthy=False, error=message, timestamp=timestamp)


class ProxyPool:
    """Proxy pool management with selection strategies and health checking."""

    def __init__(self, pool_config: ProxyPoolConfig, selector: Optional[ProxySelector] = None) -> None:
        self.config = pool_config
        self._unhealthy_proxies: set[str] = set()
        self._selector = selector or self._build_selector(pool_config.strategy)

    def _build_selector(self, strategy: str) -> ProxySelector:
        if strategy == 'round_robin':
            return RoundRobinSelector()
        if strategy == 'random':
            return RandomSelector()
        if strategy == 'least_connections':
            return LeastConnectionsSelector()
        raise ValueError(f"Unsupported selection strategy: {strategy}")

    def get_all_proxies(self) -> List[ProxyUrlConfig]:
        return self.config.proxies.copy()

    def get_healthy_proxies(self) -> List[ProxyUrlConfig]:
        return [proxy for proxy in self.config.proxies if proxy.enabled and proxy.url not in self._unhealthy_proxies]

    def select_proxy(self) -> ProxyUrlConfig:
        healthy = self.get_healthy_proxies()
        if healthy:
            selected = self._selector.select(healthy)
        else:
            enabled = [proxy for proxy in self.config.proxies if proxy.enabled]
            if not enabled:
                raise RuntimeError("No enabled proxies available")
            selected = self._selector.select(enabled)
        self._selector.record_connection(selected)
        return selected

    def mark_unhealthy(self, proxy: ProxyUrlConfig) -> None:
        self._unhealthy_proxies.add(proxy.url)

    def mark_healthy(self, proxy: ProxyUrlConfig) -> None:
        self._unhealthy_proxies.discard(proxy.url)

    def release_proxy(self, proxy: ProxyUrlConfig) -> None:
        if hasattr(self._selector, 'record_disconnection'):
            self._selector.record_disconnection(proxy)
