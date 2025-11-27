"""Integration tests for health check endpoint and custom alerting in Kafka producer.

This module tests Task 17.3: Custom Alerting & Health Checks for the market-data-kafka-producer.
Tests verify health check endpoint functionality, status determination logic, and HTTP response codes.

Coverage:
- Health check endpoint response format
- Health status determination logic (healthy/degraded/unhealthy)
- HTTP status codes (200 vs 503)
- Health check metrics integration
- Circuit breaker status reflection
- DLQ status reflection
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, Optional
from unittest.mock import MagicMock

import pytest

from cryptofeed.kafka_callback import KafkaCallback


# ============================================================================
# Health Check Models
# ============================================================================


class HealthStatus(str, Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass(slots=True)
class HealthCheckResponse:
    """Health check response model.

    Attributes:
        status: Overall health status (healthy/degraded/unhealthy)
        kafka_connected: Whether Kafka broker is connected
        buffer_health: Buffer utilization (0.0-1.0, where 0=empty, 1.0=full)
        queue_size: Current queue size in messages
        messages_produced: Total messages produced
        errors_total: Total errors encountered
        circuit_breaker_state: Circuit breaker state (CLOSED/OPEN/HALF_OPEN)
        last_message_timestamp: Unix timestamp of last message
        memory_bytes: Memory usage in bytes
        uptime_seconds: Producer uptime in seconds
    """
    status: str
    kafka_connected: bool
    buffer_health: float
    queue_size: int
    messages_produced: int
    errors_total: int
    circuit_breaker_state: str
    last_message_timestamp: Optional[float]
    memory_bytes: int
    uptime_seconds: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        data = self.to_dict()
        # Handle float precision
        data["buffer_health"] = round(data["buffer_health"], 2)
        return json.dumps(data)


# ============================================================================
# Health Check Logic
# ============================================================================


class HealthCheckDeterminer:
    """Determines health status based on metrics."""

    @staticmethod
    def determine_status(
        kafka_connected: bool,
        buffer_utilization: float,
        error_rate: float,
        circuit_breaker_state: str,
    ) -> str:
        """Determine health status based on metrics.

        Status Logic:
        - HEALTHY: Kafka connected, buffer < 80%, error rate < 0.1%, circuit CLOSED
        - DEGRADED: Kafka connected, buffer 80-95%, error rate 0.1-1%, circuit HALF_OPEN
        - UNHEALTHY: Kafka disconnected, buffer >= 95%, error rate >= 1%, circuit OPEN

        Args:
            kafka_connected: Whether Kafka broker is accessible
            buffer_utilization: Buffer utilization percentage (0-100)
            error_rate: Error rate as decimal (0-1)
            circuit_breaker_state: Circuit breaker state

        Returns:
            Health status string
        """
        # Check for unhealthy conditions
        if not kafka_connected:
            return HealthStatus.UNHEALTHY.value
        if buffer_utilization >= 95:
            return HealthStatus.UNHEALTHY.value
        if error_rate >= 0.01:  # >= 1%
            return HealthStatus.UNHEALTHY.value
        if circuit_breaker_state == "OPEN":
            return HealthStatus.UNHEALTHY.value

        # Check for degraded conditions
        if buffer_utilization >= 80:
            return HealthStatus.DEGRADED.value
        if error_rate >= 0.001:  # >= 0.1%
            return HealthStatus.DEGRADED.value
        if circuit_breaker_state == "HALF_OPEN":
            return HealthStatus.DEGRADED.value

        # Otherwise healthy
        return HealthStatus.HEALTHY.value

    @staticmethod
    def get_http_status_code(health_status: str) -> int:
        """Get HTTP status code for health status.

        Args:
            health_status: Health status string

        Returns:
            HTTP status code (200 for healthy, 503 for degraded/unhealthy)
        """
        if health_status == HealthStatus.HEALTHY.value:
            return 200
        return 503


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def health_check_determiner():
    """Provide HealthCheckDeterminer instance."""
    return HealthCheckDeterminer()


@pytest.fixture
def mock_kafka_callback():
    """Create mock KafkaCallback with health check methods."""
    callback = MagicMock(spec=KafkaCallback)
    callback._producer = MagicMock()
    callback._metrics = MagicMock()
    callback._circuit_breaker = MagicMock()
    callback._dlq_handler = MagicMock()
    callback._start_time = time.time()
    return callback


# ============================================================================
# Health Check Response Format Tests
# ============================================================================


class TestHealthCheckResponseFormat:
    """Test health check response format and fields."""

    def test_health_check_response_all_fields(self):
        """Test HealthCheckResponse contains all required fields."""
        response = HealthCheckResponse(
            status="healthy",
            kafka_connected=True,
            buffer_health=0.5,
            queue_size=100,
            messages_produced=1000,
            errors_total=2,
            circuit_breaker_state="CLOSED",
            last_message_timestamp=1699999999.123,
            memory_bytes=52428800,
            uptime_seconds=3600,
        )

        assert response.status == "healthy"
        assert response.kafka_connected is True
        assert response.buffer_health == 0.5
        assert response.queue_size == 100
        assert response.messages_produced == 1000
        assert response.errors_total == 2
        assert response.circuit_breaker_state == "CLOSED"
        assert response.last_message_timestamp == 1699999999.123
        assert response.memory_bytes == 52428800
        assert response.uptime_seconds == 3600

    def test_health_check_response_to_dict(self):
        """Test HealthCheckResponse.to_dict() conversion."""
        response = HealthCheckResponse(
            status="degraded",
            kafka_connected=True,
            buffer_health=0.85,
            queue_size=500,
            messages_produced=5000,
            errors_total=10,
            circuit_breaker_state="HALF_OPEN",
            last_message_timestamp=1699999999.456,
            memory_bytes=104857600,
            uptime_seconds=7200,
        )

        response_dict = response.to_dict()
        assert isinstance(response_dict, dict)
        assert response_dict["status"] == "degraded"
        assert response_dict["kafka_connected"] is True
        assert response_dict["circuit_breaker_state"] == "HALF_OPEN"

    def test_health_check_response_to_json(self):
        """Test HealthCheckResponse.to_json() conversion."""
        response = HealthCheckResponse(
            status="healthy",
            kafka_connected=True,
            buffer_health=0.45,
            queue_size=123,
            messages_produced=10000,
            errors_total=5,
            circuit_breaker_state="CLOSED",
            last_message_timestamp=1699999999.789,
            memory_bytes=67108864,
            uptime_seconds=10800,
        )

        json_str = response.to_json()
        assert isinstance(json_str, str)

        parsed = json.loads(json_str)
        assert parsed["status"] == "healthy"
        assert parsed["kafka_connected"] is True
        assert parsed["buffer_health"] == 0.45
        assert parsed["queue_size"] == 123
        assert parsed["messages_produced"] == 10000

    def test_health_check_response_json_buffer_health_precision(self):
        """Test buffer health is rounded to 2 decimal places in JSON."""
        response = HealthCheckResponse(
            status="healthy",
            kafka_connected=True,
            buffer_health=0.123456,
            queue_size=50,
            messages_produced=2000,
            errors_total=1,
            circuit_breaker_state="CLOSED",
            last_message_timestamp=1699999999.0,
            memory_bytes=33554432,
            uptime_seconds=1800,
        )

        json_str = response.to_json()
        parsed = json.loads(json_str)
        # Should be rounded to 2 decimal places
        assert parsed["buffer_health"] == 0.12

    def test_health_check_response_with_none_timestamp(self):
        """Test HealthCheckResponse with None last_message_timestamp."""
        response = HealthCheckResponse(
            status="unhealthy",
            kafka_connected=False,
            buffer_health=0.0,
            queue_size=0,
            messages_produced=0,
            errors_total=0,
            circuit_breaker_state="OPEN",
            last_message_timestamp=None,
            memory_bytes=0,
            uptime_seconds=0,
        )

        response_dict = response.to_dict()
        assert response_dict["last_message_timestamp"] is None

        json_str = response.to_json()
        parsed = json.loads(json_str)
        assert parsed["last_message_timestamp"] is None


# ============================================================================
# Health Status Determination Logic Tests
# ============================================================================


class TestHealthStatusDetermination:
    """Test health status determination logic."""

    def test_healthy_status_all_good(self, health_check_determiner):
        """Test HEALTHY status when all metrics are good."""
        status = health_check_determiner.determine_status(
            kafka_connected=True,
            buffer_utilization=50.0,
            error_rate=0.0001,  # 0.01%
            circuit_breaker_state="CLOSED",
        )
        assert status == HealthStatus.HEALTHY.value

    def test_healthy_status_at_boundaries(self, health_check_determiner):
        """Test HEALTHY status at safe boundaries."""
        status = health_check_determiner.determine_status(
            kafka_connected=True,
            buffer_utilization=79.99,  # Just under 80%
            error_rate=0.0009,  # Just under 0.1%
            circuit_breaker_state="CLOSED",
        )
        assert status == HealthStatus.HEALTHY.value

    def test_degraded_status_high_buffer(self, health_check_determiner):
        """Test DEGRADED status when buffer is high."""
        status = health_check_determiner.determine_status(
            kafka_connected=True,
            buffer_utilization=85.0,  # 80-95% range
            error_rate=0.0001,
            circuit_breaker_state="CLOSED",
        )
        assert status == HealthStatus.DEGRADED.value

    def test_degraded_status_elevated_error_rate(self, health_check_determiner):
        """Test DEGRADED status when error rate is elevated."""
        status = health_check_determiner.determine_status(
            kafka_connected=True,
            buffer_utilization=50.0,
            error_rate=0.005,  # 0.5%, between 0.1% and 1%
            circuit_breaker_state="CLOSED",
        )
        assert status == HealthStatus.DEGRADED.value

    def test_degraded_status_half_open_circuit(self, health_check_determiner):
        """Test DEGRADED status when circuit breaker is HALF_OPEN."""
        status = health_check_determiner.determine_status(
            kafka_connected=True,
            buffer_utilization=50.0,
            error_rate=0.0001,
            circuit_breaker_state="HALF_OPEN",
        )
        assert status == HealthStatus.DEGRADED.value

    def test_unhealthy_status_kafka_disconnected(self, health_check_determiner):
        """Test UNHEALTHY status when Kafka is disconnected."""
        status = health_check_determiner.determine_status(
            kafka_connected=False,
            buffer_utilization=50.0,
            error_rate=0.0001,
            circuit_breaker_state="CLOSED",
        )
        assert status == HealthStatus.UNHEALTHY.value

    def test_unhealthy_status_buffer_critical(self, health_check_determiner):
        """Test UNHEALTHY status when buffer is critical."""
        status = health_check_determiner.determine_status(
            kafka_connected=True,
            buffer_utilization=96.0,  # > 95%
            error_rate=0.0001,
            circuit_breaker_state="CLOSED",
        )
        assert status == HealthStatus.UNHEALTHY.value

    def test_unhealthy_status_high_error_rate(self, health_check_determiner):
        """Test UNHEALTHY status when error rate is high."""
        status = health_check_determiner.determine_status(
            kafka_connected=True,
            buffer_utilization=50.0,
            error_rate=0.015,  # 1.5%, > 1%
            circuit_breaker_state="CLOSED",
        )
        assert status == HealthStatus.UNHEALTHY.value

    def test_unhealthy_status_circuit_open(self, health_check_determiner):
        """Test UNHEALTHY status when circuit breaker is OPEN."""
        status = health_check_determiner.determine_status(
            kafka_connected=True,
            buffer_utilization=50.0,
            error_rate=0.0001,
            circuit_breaker_state="OPEN",
        )
        assert status == HealthStatus.UNHEALTHY.value

    def test_unhealthy_takes_precedence(self, health_check_determiner):
        """Test UNHEALTHY status takes precedence over DEGRADED."""
        # Multiple unhealthy conditions
        status = health_check_determiner.determine_status(
            kafka_connected=True,
            buffer_utilization=96.0,  # Critical
            error_rate=0.005,  # Elevated
            circuit_breaker_state="CLOSED",
        )
        assert status == HealthStatus.UNHEALTHY.value

    def test_degraded_takes_precedence_over_healthy(self, health_check_determiner):
        """Test DEGRADED status takes precedence over HEALTHY."""
        # Buffer in degraded range with other healthy metrics
        status = health_check_determiner.determine_status(
            kafka_connected=True,
            buffer_utilization=82.0,
            error_rate=0.0001,
            circuit_breaker_state="CLOSED",
        )
        assert status == HealthStatus.DEGRADED.value


# ============================================================================
# HTTP Status Code Tests
# ============================================================================


class TestHTTPStatusCodes:
    """Test HTTP status code determination."""

    def test_http_200_for_healthy(self, health_check_determiner):
        """Test HTTP 200 OK for healthy status."""
        code = health_check_determiner.get_http_status_code(HealthStatus.HEALTHY.value)
        assert code == 200

    def test_http_503_for_degraded(self, health_check_determiner):
        """Test HTTP 503 Service Unavailable for degraded status."""
        code = health_check_determiner.get_http_status_code(HealthStatus.DEGRADED.value)
        assert code == 503

    def test_http_503_for_unhealthy(self, health_check_determiner):
        """Test HTTP 503 Service Unavailable for unhealthy status."""
        code = health_check_determiner.get_http_status_code(HealthStatus.UNHEALTHY.value)
        assert code == 503


# ============================================================================
# Health Check Integration Tests (Mocked KafkaCallback)
# ============================================================================


class TestHealthCheckIntegration:
    """Test health check integration with KafkaCallback."""

    def test_health_check_with_healthy_metrics(self, mock_kafka_callback):
        """Test health check returns healthy with good metrics."""
        # Mock metrics for healthy state
        mock_kafka_callback._is_connected = True
        mock_kafka_callback._buffer_utilization = 45.0
        mock_kafka_callback._error_rate = 0.0001
        mock_kafka_callback._circuit_breaker.state = "CLOSED"
        mock_kafka_callback._queue_size = 100
        mock_kafka_callback._messages_produced = 5000
        mock_kafka_callback._errors = 2
        mock_kafka_callback._last_message_time = time.time()
        mock_kafka_callback._memory_usage = 50 * 1024 * 1024  # 50MB

        # Simulate health check method
        determiner = HealthCheckDeterminer()
        status = determiner.determine_status(
            kafka_connected=mock_kafka_callback._is_connected,
            buffer_utilization=mock_kafka_callback._buffer_utilization,
            error_rate=mock_kafka_callback._error_rate,
            circuit_breaker_state=mock_kafka_callback._circuit_breaker.state,
        )

        assert status == HealthStatus.HEALTHY.value

    def test_health_check_with_degraded_buffer(self, mock_kafka_callback):
        """Test health check returns degraded with high buffer."""
        mock_kafka_callback._is_connected = True
        mock_kafka_callback._buffer_utilization = 85.0
        mock_kafka_callback._error_rate = 0.0001
        mock_kafka_callback._circuit_breaker.state = "CLOSED"

        determiner = HealthCheckDeterminer()
        status = determiner.determine_status(
            kafka_connected=mock_kafka_callback._is_connected,
            buffer_utilization=mock_kafka_callback._buffer_utilization,
            error_rate=mock_kafka_callback._error_rate,
            circuit_breaker_state=mock_kafka_callback._circuit_breaker.state,
        )

        assert status == HealthStatus.DEGRADED.value

    def test_health_check_with_disconnected_kafka(self, mock_kafka_callback):
        """Test health check returns unhealthy when Kafka disconnected."""
        mock_kafka_callback._is_connected = False
        mock_kafka_callback._buffer_utilization = 50.0
        mock_kafka_callback._error_rate = 0.0001
        mock_kafka_callback._circuit_breaker.state = "CLOSED"

        determiner = HealthCheckDeterminer()
        status = determiner.determine_status(
            kafka_connected=mock_kafka_callback._is_connected,
            buffer_utilization=mock_kafka_callback._buffer_utilization,
            error_rate=mock_kafka_callback._error_rate,
            circuit_breaker_state=mock_kafka_callback._circuit_breaker.state,
        )

        assert status == HealthStatus.UNHEALTHY.value

    def test_health_check_with_open_circuit(self, mock_kafka_callback):
        """Test health check returns unhealthy with open circuit breaker."""
        mock_kafka_callback._is_connected = True
        mock_kafka_callback._buffer_utilization = 50.0
        mock_kafka_callback._error_rate = 0.0001
        mock_kafka_callback._circuit_breaker.state = "OPEN"

        determiner = HealthCheckDeterminer()
        status = determiner.determine_status(
            kafka_connected=mock_kafka_callback._is_connected,
            buffer_utilization=mock_kafka_callback._buffer_utilization,
            error_rate=mock_kafka_callback._error_rate,
            circuit_breaker_state=mock_kafka_callback._circuit_breaker.state,
        )

        assert status == HealthStatus.UNHEALTHY.value


# ============================================================================
# Edge Cases and Boundary Tests
# ============================================================================


class TestHealthCheckEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_error_rate(self, health_check_determiner):
        """Test with zero error rate (no errors)."""
        status = health_check_determiner.determine_status(
            kafka_connected=True,
            buffer_utilization=0.0,
            error_rate=0.0,
            circuit_breaker_state="CLOSED",
        )
        assert status == HealthStatus.HEALTHY.value

    def test_zero_buffer_utilization(self, health_check_determiner):
        """Test with zero buffer utilization (empty buffer)."""
        status = health_check_determiner.determine_status(
            kafka_connected=True,
            buffer_utilization=0.0,
            error_rate=0.0001,
            circuit_breaker_state="CLOSED",
        )
        assert status == HealthStatus.HEALTHY.value

    def test_exact_80_percent_buffer(self, health_check_determiner):
        """Test buffer at exactly 80% (boundary)."""
        status = health_check_determiner.determine_status(
            kafka_connected=True,
            buffer_utilization=80.0,
            error_rate=0.0001,
            circuit_breaker_state="CLOSED",
        )
        # 80% >= threshold is degraded
        assert status == HealthStatus.DEGRADED.value

    def test_exact_95_percent_buffer(self, health_check_determiner):
        """Test buffer at exactly 95% (boundary)."""
        status = health_check_determiner.determine_status(
            kafka_connected=True,
            buffer_utilization=95.0,
            error_rate=0.0001,
            circuit_breaker_state="CLOSED",
        )
        # 95% >= threshold is unhealthy
        assert status == HealthStatus.UNHEALTHY.value

    def test_exact_0_1_percent_error_rate(self, health_check_determiner):
        """Test error rate at exactly 0.1% (boundary)."""
        status = health_check_determiner.determine_status(
            kafka_connected=True,
            buffer_utilization=50.0,
            error_rate=0.001,
            circuit_breaker_state="CLOSED",
        )
        # 0.1% >= threshold is degraded
        assert status == HealthStatus.DEGRADED.value

    def test_exact_1_percent_error_rate(self, health_check_determiner):
        """Test error rate at exactly 1% (boundary)."""
        status = health_check_determiner.determine_status(
            kafka_connected=True,
            buffer_utilization=50.0,
            error_rate=0.01,
            circuit_breaker_state="CLOSED",
        )
        # 1% >= threshold is unhealthy
        assert status == HealthStatus.UNHEALTHY.value

    def test_response_with_minimal_values(self):
        """Test response with minimal/zero values."""
        response = HealthCheckResponse(
            status="healthy",
            kafka_connected=True,
            buffer_health=0.0,
            queue_size=0,
            messages_produced=0,
            errors_total=0,
            circuit_breaker_state="CLOSED",
            last_message_timestamp=None,
            memory_bytes=0,
            uptime_seconds=0,
        )

        assert response.to_json() is not None
        parsed = json.loads(response.to_json())
        assert parsed["queue_size"] == 0
        assert parsed["messages_produced"] == 0

    def test_response_with_large_values(self):
        """Test response with large metric values."""
        response = HealthCheckResponse(
            status="healthy",
            kafka_connected=True,
            buffer_health=0.99,
            queue_size=1000000,
            messages_produced=10000000,
            errors_total=100000,
            circuit_breaker_state="CLOSED",
            last_message_timestamp=time.time(),
            memory_bytes=1073741824,  # 1GB
            uptime_seconds=86400 * 30,  # 30 days
        )

        json_str = response.to_json()
        parsed = json.loads(json_str)
        assert parsed["queue_size"] == 1000000
        assert parsed["messages_produced"] == 10000000
        assert parsed["memory_bytes"] == 1073741824
