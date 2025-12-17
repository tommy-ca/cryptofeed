"""
Health check HTTP server for k3s probes.

Exposes /health, /ready, and /metrics endpoints for container orchestration.
Integrates with KafkaHealthCheck and exchange connection state.
"""
import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from aiohttp import web

from cryptofeed.backends.kafka.health import KafkaHealthCheck, KafkaHealthStatus


LOG = logging.getLogger("feedhandler")


class ComponentHealth:
    """Component health status for aggregation."""

    @staticmethod
    def from_kafka(kafka_status: KafkaHealthStatus) -> Dict[str, Any]:
        """Convert KafkaHealthStatus to component health dict."""
        return {
            "status": "healthy" if kafka_status.ok else "unhealthy",
            "latency_ms": kafka_status.latency_ms,
            "bootstrap_servers": kafka_status.details.get("bootstrap", []) if kafka_status.details else [],
            "error": kafka_status.error,
        }

    @staticmethod
    def from_exchanges(exchange_status: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Convert exchange connection status to component health dict."""
        if not exchange_status:
            return {}

        components = {}
        for exchange_name, status in exchange_status.items():
            components[exchange_name] = {
                "status": status.get("status", "unknown"),
                "symbols": status.get("symbols", 0),
            }
        return components


class HealthStatus:
    """Health status aggregation logic."""

    @staticmethod
    def aggregate(kafka_status: KafkaHealthStatus, exchange_status: Dict[str, Dict[str, Any]]) -> str:
        """
        Aggregate component health into overall status.

        Returns:
            "healthy": All components OK
            "degraded": Some exchanges disconnected but Kafka OK
            "unhealthy": Kafka failed or no exchanges connected
        """
        # Critical: Kafka must be connected
        if not kafka_status.ok:
            return "unhealthy"

        # Critical: At least one exchange must be connected
        if not exchange_status:
            return "unhealthy"

        # Check for disconnected exchanges
        all_connected = all(
            ex.get("status") == "connected"
            for ex in exchange_status.values()
        )

        if not all_connected:
            return "degraded"

        return "healthy"


class HealthServer:
    """
    HTTP health check server for container orchestration.

    Exposes endpoints:
    - GET /health: Liveness probe (overall health status)
    - GET /ready: Readiness probe (ready to accept traffic)
    - GET /metrics: Prometheus metrics endpoint
    """

    def __init__(
        self,
        port: Optional[int] = None,
        kafka_health_fn: Optional[Callable[[], KafkaHealthStatus]] = None,
        exchange_health_fn: Optional[Callable[[], Dict[str, Dict[str, Any]]]] = None,
        close_websockets_fn: Optional[Callable[[], Any]] = None,
        flush_kafka_fn: Optional[Callable[[], Any]] = None,
        shutdown_timeout: float = 30.0,
    ):
        """
        Initialize health check server.

        Args:
            port: HTTP server port (default from HEALTH_PORT env or 8080)
            kafka_health_fn: Function returning KafkaHealthStatus
            exchange_health_fn: Function returning exchange connection states
            close_websockets_fn: Async function to close WebSocket connections
            flush_kafka_fn: Async function to flush Kafka producer buffers
            shutdown_timeout: Maximum time to wait for graceful shutdown (seconds)
        """
        self.port = port or int(os.environ.get("HEALTH_PORT", "8080"))
        self.kafka_health_fn = kafka_health_fn or self._default_kafka_health
        self.exchange_health_fn = exchange_health_fn or self._default_exchange_health
        self.close_websockets_fn = close_websockets_fn
        self.flush_kafka_fn = flush_kafka_fn
        self.shutdown_timeout = shutdown_timeout

        self.start_time = time.time()
        self.app: Optional[web.Application] = None
        self.runner: Optional[web.AppRunner] = None

    def _default_kafka_health(self) -> KafkaHealthStatus:
        """Default Kafka health check (returns healthy status)."""
        return KafkaHealthStatus(
            implementation="default",
            ok=True,
            latency_ms=0.0,
            details={},
        )

    def _default_exchange_health(self) -> Dict[str, Dict[str, Any]]:
        """Default exchange health check (returns empty status)."""
        return {}

    def get_uptime_seconds(self) -> float:
        """Calculate uptime in seconds since server start."""
        return time.time() - self.start_time

    async def health_handler(self, request: web.Request) -> web.Response:
        """
        GET /health - Liveness probe endpoint.

        Returns 200 OK if healthy, 503 Service Unavailable if unhealthy.
        """
        try:
            # Gather component health
            kafka_status = self.kafka_health_fn()
            exchange_status = self.exchange_health_fn()

            # Aggregate overall status
            overall_status = HealthStatus.aggregate(kafka_status, exchange_status)

            # Build response
            response_data = {
                "status": overall_status,
                "timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                "uptime_seconds": self.get_uptime_seconds(),
                "components": {
                    "kafka": ComponentHealth.from_kafka(kafka_status),
                    "exchanges": ComponentHealth.from_exchanges(exchange_status),
                },
            }

            status_code = 200 if overall_status == "healthy" else 503
            return web.json_response(response_data, status=status_code)

        except Exception as e:
            LOG.error("Health check failed: %s", e, exc_info=True)
            return web.json_response(
                {"status": "unhealthy", "error": str(e)},
                status=503
            )

    async def ready_handler(self, request: web.Request) -> web.Response:
        """
        GET /ready - Readiness probe endpoint.

        Returns 200 OK if ready, 503 Service Unavailable if not ready.

        Readiness criteria:
        - Kafka producer initialized and connected
        - At least 1 exchange connection active
        - Configuration loaded successfully
        """
        try:
            kafka_status = self.kafka_health_fn()
            exchange_status = self.exchange_health_fn()

            # Readiness checks
            kafka_connected = kafka_status.ok
            exchanges_connected = len(exchange_status) > 0
            config_loaded = True  # Assume config loaded if server started

            all_ready = kafka_connected and exchanges_connected and config_loaded

            response_data = {
                "status": "ready" if all_ready else "not_ready",
                "ready_checks": {
                    "kafka_connected": kafka_connected,
                    "exchanges_connected": exchanges_connected,
                    "config_loaded": config_loaded,
                },
            }

            status_code = 200 if all_ready else 503
            return web.json_response(response_data, status=status_code)

        except Exception as e:
            LOG.error("Readiness check failed: %s", e, exc_info=True)
            return web.json_response(
                {"status": "not_ready", "error": str(e)},
                status=503
            )

    async def metrics_handler(self, request: web.Request) -> web.Response:
        """
        GET /metrics - Prometheus metrics endpoint.

        Returns metrics in Prometheus text exposition format.
        """
        try:
            uptime = self.get_uptime_seconds()

            # Basic metrics in Prometheus format
            metrics = [
                "# HELP cryptofeed_uptime_seconds Uptime in seconds",
                "# TYPE cryptofeed_uptime_seconds gauge",
                f"cryptofeed_uptime_seconds {uptime}",
                "",
            ]

            # Add Kafka metrics if available
            kafka_status = self.kafka_health_fn()
            metrics.extend([
                "# HELP cryptofeed_kafka_healthy Kafka connection health (1=healthy, 0=unhealthy)",
                "# TYPE cryptofeed_kafka_healthy gauge",
                f"cryptofeed_kafka_healthy {1 if kafka_status.ok else 0}",
                "",
                "# HELP cryptofeed_kafka_latency_ms Kafka connection latency in milliseconds",
                "# TYPE cryptofeed_kafka_latency_ms gauge",
                f"cryptofeed_kafka_latency_ms {kafka_status.latency_ms}",
                "",
            ])

            # Add exchange metrics
            exchange_status = self.exchange_health_fn()
            if exchange_status:
                metrics.extend([
                    "# HELP cryptofeed_exchange_connected Exchange connection status (1=connected, 0=disconnected)",
                    "# TYPE cryptofeed_exchange_connected gauge",
                ])
                for exchange_name, status in exchange_status.items():
                    connected = 1 if status.get("status") == "connected" else 0
                    metrics.append(f'cryptofeed_exchange_connected{{exchange="{exchange_name}"}} {connected}')
                metrics.append("")

            metrics_text = "\n".join(metrics)
            return web.Response(text=metrics_text, content_type="text/plain; version=0.0.4")

        except Exception as e:
            LOG.error("Metrics collection failed: %s", e, exc_info=True)
            return web.Response(text="# Error collecting metrics\n", status=500)

    async def shutdown(self) -> None:
        """
        Graceful shutdown handler.

        Closes WebSocket connections, flushes Kafka buffers, exits within timeout.
        """
        LOG.info("Health server shutting down gracefully...")

        try:
            # Close WebSocket connections
            if self.close_websockets_fn:
                if asyncio.iscoroutinefunction(self.close_websockets_fn):
                    await asyncio.wait_for(
                        self.close_websockets_fn(),
                        timeout=self.shutdown_timeout / 2
                    )
                else:
                    self.close_websockets_fn()

            # Flush Kafka producer
            if self.flush_kafka_fn:
                if asyncio.iscoroutinefunction(self.flush_kafka_fn):
                    await asyncio.wait_for(
                        self.flush_kafka_fn(),
                        timeout=self.shutdown_timeout / 2
                    )
                else:
                    self.flush_kafka_fn()

            LOG.info("Health server shutdown complete")

        except asyncio.TimeoutError:
            LOG.warning("Shutdown timeout reached, forcing exit")
        except Exception as e:
            LOG.error("Error during shutdown: %s", e, exc_info=True)

    async def start(self) -> None:
        """Start HTTP server."""
        self.app = create_health_app(self)
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()

        site = web.TCPSite(self.runner, "0.0.0.0", self.port)
        await site.start()
        LOG.info(f"Health server started on port {self.port}")

    async def stop(self) -> None:
        """Stop HTTP server."""
        if self.runner:
            await self.runner.cleanup()
            LOG.info("Health server stopped")


def create_health_app(health_server: HealthServer) -> web.Application:
    """
    Create aiohttp application with health check routes.

    Args:
        health_server: HealthServer instance

    Returns:
        Configured aiohttp Application
    """
    app = web.Application()
    app.router.add_get("/health", health_server.health_handler)
    app.router.add_get("/ready", health_server.ready_handler)
    app.router.add_get("/metrics", health_server.metrics_handler)
    return app


__all__ = [
    "HealthServer",
    "HealthStatus",
    "ComponentHealth",
    "create_health_app",
]
