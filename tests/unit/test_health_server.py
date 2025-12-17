"""
Unit tests for health check HTTP server (Task 1.1).

Tests verify health check endpoints, readiness/liveness criteria,
graceful shutdown, and integration with KafkaHealthCheck.
"""
import asyncio
import pytest
import time
from unittest.mock import Mock, AsyncMock, patch
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase

from cryptofeed.health_server import (
    HealthServer,
    HealthStatus,
    ComponentHealth,
    create_health_app,
)
from cryptofeed.backends.kafka.health import KafkaHealthStatus


class TestHealthServer(AioHTTPTestCase):
    """Test health check HTTP server endpoints and lifecycle."""

    async def get_application(self):
        """Create test application with health endpoints."""
        # Mock components for testing
        self.mock_kafka_health_fn = Mock(return_value=KafkaHealthStatus(
            implementation="modern",
            ok=True,
            latency_ms=12.5,
            details={"bootstrap": ["kafka:9092"]}
        ))
        self.mock_exchange_health_fn = Mock(return_value={
            "binance": {"status": "connected", "symbols": 50},
            "coinbase": {"status": "connected", "symbols": 30}
        })

        # Create health server with mocked health functions
        self.health_server = HealthServer(
            port=8080,
            kafka_health_fn=self.mock_kafka_health_fn,
            exchange_health_fn=self.mock_exchange_health_fn,
        )
        return create_health_app(self.health_server)

    async def test_health_endpoint_returns_200_when_healthy(self):
        """Test /health endpoint returns 200 OK when all components healthy."""
        resp = await self.client.request("GET", "/health")
        assert resp.status == 200
        data = await resp.json()

        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "uptime_seconds" in data
        assert data["components"]["kafka"]["status"] == "healthy"
        assert data["components"]["exchanges"]["binance"]["status"] == "connected"

    
    async def test_health_endpoint_returns_503_when_kafka_unhealthy(self):
        """Test /health endpoint returns 503 when Kafka is unhealthy."""
        # Mock Kafka as unhealthy
        self.mock_kafka_health_fn.return_value = KafkaHealthStatus(
            implementation="modern",
            ok=False,
            latency_ms=5000.0,
            error="Connection timeout",
            details={"bootstrap": ["kafka:9092"]}
        )

        resp = await self.client.request("GET", "/health")
        assert resp.status == 503
        data = await resp.json()

        assert data["status"] == "unhealthy"
        assert data["components"]["kafka"]["status"] == "unhealthy"
        assert data["components"]["kafka"]["error"] == "Connection timeout"

    
    async def test_ready_endpoint_returns_200_when_ready(self):
        """Test /ready endpoint returns 200 when readiness criteria met."""
        resp = await self.client.request("GET", "/ready")
        assert resp.status == 200
        data = await resp.json()

        assert data["status"] == "ready"
        assert data["ready_checks"]["kafka_connected"] is True
        assert data["ready_checks"]["exchanges_connected"] is True
        assert data["ready_checks"]["config_loaded"] is True

    
    async def test_ready_endpoint_returns_503_when_kafka_disconnected(self):
        """Test /ready endpoint returns 503 when Kafka not connected."""
        self.mock_kafka_health_fn.return_value = KafkaHealthStatus(
            implementation="modern",
            ok=False,
            latency_ms=0.0,
            error="Not connected",
        )

        resp = await self.client.request("GET", "/ready")
        assert resp.status == 503
        data = await resp.json()

        assert data["status"] == "not_ready"
        assert data["ready_checks"]["kafka_connected"] is False

    
    async def test_ready_endpoint_returns_503_when_no_exchanges_connected(self):
        """Test /ready endpoint returns 503 when no exchanges connected."""
        self.mock_exchange_health_fn.return_value = {}

        resp = await self.client.request("GET", "/ready")
        assert resp.status == 503
        data = await resp.json()

        assert data["status"] == "not_ready"
        assert data["ready_checks"]["exchanges_connected"] is False

    
    async def test_metrics_endpoint_returns_prometheus_format(self):
        """Test /metrics endpoint returns Prometheus text format."""
        resp = await self.client.request("GET", "/metrics")
        assert resp.status == 200

        content_type = resp.headers.get("Content-Type", "")
        assert "text/plain" in content_type

        text = await resp.text()
        # Should contain at least uptime metric
        assert "cryptofeed_uptime_seconds" in text

    
    async def test_health_status_includes_component_details(self):
        """Test health response includes detailed component health."""
        resp = await self.client.request("GET", "/health")
        data = await resp.json()

        # Kafka component details
        kafka = data["components"]["kafka"]
        assert "status" in kafka
        assert "latency_ms" in kafka
        assert "bootstrap_servers" in kafka

        # Exchange component details
        exchanges = data["components"]["exchanges"]
        assert "binance" in exchanges
        assert exchanges["binance"]["symbols"] == 50

    
    async def test_health_response_includes_timestamp_and_uptime(self):
        """Test health response includes ISO timestamp and uptime in seconds."""
        resp = await self.client.request("GET", "/health")
        data = await resp.json()

        assert "timestamp" in data
        # Timestamp should be ISO 8601 format
        assert "T" in data["timestamp"]
        assert "Z" in data["timestamp"]

        assert "uptime_seconds" in data
        assert isinstance(data["uptime_seconds"], (int, float))
        assert data["uptime_seconds"] >= 0

    
    async def test_health_endpoint_response_within_1_second(self):
        """Test health endpoint responds within 1 second (liveness criteria)."""
        start = time.time()
        resp = await self.client.request("GET", "/health")
        elapsed = time.time() - start

        assert resp.status in (200, 503)
        assert elapsed < 1.0, f"Health check took {elapsed:.2f}s, should be < 1s"


class TestHealthServerConfiguration:
    """Test health server configuration and initialization."""

    def test_health_server_uses_default_port_8080(self):
        """Test health server defaults to port 8080."""
        server = HealthServer()
        assert server.port == 8080

    def test_health_server_uses_custom_port_from_env(self):
        """Test health server respects HEALTH_PORT environment variable."""
        with patch.dict('os.environ', {'HEALTH_PORT': '9090'}):
            server = HealthServer()
            assert server.port == 9090

    def test_health_server_accepts_explicit_port_argument(self):
        """Test health server accepts port as constructor argument."""
        server = HealthServer(port=8888)
        assert server.port == 8888

    def test_health_server_tracks_startup_time(self):
        """Test health server tracks startup time for uptime calculation."""
        server = HealthServer()
        assert hasattr(server, 'start_time')
        assert isinstance(server.start_time, float)

        # Uptime should be non-negative
        uptime = server.get_uptime_seconds()
        assert uptime >= 0

    def test_health_server_accepts_health_check_functions(self):
        """Test health server accepts custom health check functions."""
        kafka_fn = Mock()
        exchange_fn = Mock()

        server = HealthServer(
            kafka_health_fn=kafka_fn,
            exchange_health_fn=exchange_fn,
        )

        assert server.kafka_health_fn == kafka_fn
        assert server.exchange_health_fn == exchange_fn


class TestHealthServerGracefulShutdown:
    """Test graceful shutdown handling on SIGTERM."""

    @pytest.mark.asyncio
    async def test_graceful_shutdown_closes_connections(self):
        """Test graceful shutdown closes WebSocket and Kafka connections."""
        # Mock connection cleanup functions
        mock_close_ws = AsyncMock()
        mock_flush_kafka = AsyncMock()

        server = HealthServer(
            close_websockets_fn=mock_close_ws,
            flush_kafka_fn=mock_flush_kafka,
        )

        # Trigger graceful shutdown
        await server.shutdown()

        # Verify cleanup functions were called
        mock_close_ws.assert_awaited_once()
        mock_flush_kafka.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_graceful_shutdown_completes_within_30_seconds(self):
        """Test graceful shutdown completes within 30 seconds."""
        # Mock slow cleanup (but should complete)
        async def slow_cleanup():
            await asyncio.sleep(0.1)

        server = HealthServer(
            close_websockets_fn=slow_cleanup,
            flush_kafka_fn=slow_cleanup,
        )

        start = time.time()
        await asyncio.wait_for(server.shutdown(), timeout=30.0)
        elapsed = time.time() - start

        assert elapsed < 30.0

    @pytest.mark.asyncio
    async def test_graceful_shutdown_handles_cleanup_timeout(self):
        """Test graceful shutdown handles cleanup functions that timeout."""
        # Mock cleanup that takes too long
        async def timeout_cleanup():
            await asyncio.sleep(100)  # Intentionally long

        server = HealthServer(
            close_websockets_fn=timeout_cleanup,
            shutdown_timeout=5.0,  # 5 second timeout for testing
        )

        # Shutdown should timeout gracefully, not raise
        start = time.time()
        try:
            await asyncio.wait_for(server.shutdown(), timeout=10.0)
        except asyncio.TimeoutError:
            pytest.fail("Shutdown should handle timeout internally")
        elapsed = time.time() - start

        # Should complete near the shutdown_timeout (5s), not the full 100s
        assert elapsed < 10.0


class TestHealthStatusAggregation:
    """Test health status aggregation logic."""

    def test_overall_status_healthy_when_all_components_healthy(self):
        """Test overall status is 'healthy' when all components OK."""
        kafka_status = KafkaHealthStatus(
            implementation="modern",
            ok=True,
            latency_ms=10.0,
        )
        exchange_status = {
            "binance": {"status": "connected"},
            "coinbase": {"status": "connected"},
        }

        status = HealthStatus.aggregate(kafka_status, exchange_status)
        assert status == "healthy"

    def test_overall_status_degraded_when_some_exchanges_disconnected(self):
        """Test overall status is 'degraded' when some exchanges disconnected."""
        kafka_status = KafkaHealthStatus(
            implementation="modern",
            ok=True,
            latency_ms=10.0,
        )
        exchange_status = {
            "binance": {"status": "connected"},
            "coinbase": {"status": "disconnected"},
        }

        status = HealthStatus.aggregate(kafka_status, exchange_status)
        assert status == "degraded"

    def test_overall_status_unhealthy_when_kafka_failed(self):
        """Test overall status is 'unhealthy' when Kafka failed."""
        kafka_status = KafkaHealthStatus(
            implementation="modern",
            ok=False,
            latency_ms=0.0,
            error="Connection refused",
        )
        exchange_status = {
            "binance": {"status": "connected"},
        }

        status = HealthStatus.aggregate(kafka_status, exchange_status)
        assert status == "unhealthy"

    def test_overall_status_unhealthy_when_no_exchanges_connected(self):
        """Test overall status is 'unhealthy' when no exchanges connected."""
        kafka_status = KafkaHealthStatus(
            implementation="modern",
            ok=True,
            latency_ms=10.0,
        )
        exchange_status = {}

        status = HealthStatus.aggregate(kafka_status, exchange_status)
        assert status == "unhealthy"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
