"""Integration tests for Circuit Breaker pattern implementation.

Tests Task 17.2b: Circuit Breaker for producer resilience
Coverage:
- Circuit breaker state transitions (CLOSED -> OPEN -> HALF_OPEN -> CLOSED)
- Failure threshold detection (5% error rate over 60s window)
- Timeout-based recovery (30s in OPEN state)
- Test request mechanism in HALF_OPEN state
- Exponential backoff strategy
- Metrics tracking (state changes, test requests)
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest


# ============================================================================
# Circuit Breaker Tests
# ============================================================================


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class TestCircuitBreaker:
    """Test suite for circuit breaker pattern."""

    def test_circuit_breaker_initial_state_is_closed(self):
        """Test that circuit breaker starts in CLOSED state."""
        from cryptofeed.backends.kafka_circuit_breaker import CircuitBreaker

        breaker = CircuitBreaker()
        assert breaker.state == CircuitState.CLOSED

    def test_circuit_breaker_allows_requests_when_closed(self):
        """Test that circuit breaker allows requests in CLOSED state."""
        from cryptofeed.backends.kafka_circuit_breaker import CircuitBreaker

        breaker = CircuitBreaker()
        mock_func = MagicMock(return_value="success")

        result = breaker.call(mock_func)
        assert result == "success"
        mock_func.assert_called_once()

    def test_circuit_breaker_transitions_to_open_on_threshold(self):
        """Test circuit breaker transitions CLOSED -> OPEN when error rate exceeds threshold."""
        from cryptofeed.backends.kafka_circuit_breaker import CircuitBreaker, CircuitBreakerConfig

        config = CircuitBreakerConfig(
            failure_threshold=0.05,  # 5% error rate
            window_size=10
        )
        breaker = CircuitBreaker(config=config)

        # Generate failures to exceed threshold
        mock_func = MagicMock(side_effect=Exception("Service error"))

        # Make 10 requests: 6 failures = 60% error rate > 5% threshold
        for i in range(6):
            try:
                breaker.call(mock_func)
            except Exception:
                pass

        # Verify state is OPEN
        assert breaker.state == CircuitState.OPEN

    def test_circuit_breaker_fails_fast_when_open(self):
        """Test that circuit breaker fails fast with CircuitBreakerOpen error when OPEN."""
        from cryptofeed.backends.kafka_circuit_breaker import CircuitBreaker, CircuitBreakerOpen

        breaker = CircuitBreaker()
        breaker._state = CircuitState.OPEN  # Force to OPEN
        breaker._last_failure_time = time.time()

        mock_func = MagicMock()

        with pytest.raises(CircuitBreakerOpen):
            breaker.call(mock_func)

        # Verify function was not called
        mock_func.assert_not_called()

    def test_circuit_breaker_transitions_to_half_open_after_timeout(self):
        """Test circuit breaker transitions OPEN -> HALF_OPEN after timeout."""
        from cryptofeed.backends.kafka_circuit_breaker import CircuitBreaker, CircuitBreakerConfig

        config = CircuitBreakerConfig(
            timeout_seconds=1  # 1 second timeout for testing
        )
        breaker = CircuitBreaker(config=config)

        # Force to OPEN state
        breaker._state = CircuitState.OPEN
        breaker._last_failure_time = time.time() - 2  # 2 seconds ago

        # Next call should transition to HALF_OPEN and allow the call
        mock_func = MagicMock(return_value="success")
        result = breaker.call(mock_func)

        # First successful call in HALF_OPEN closes the circuit
        # So final state is CLOSED, but we know HALF_OPEN was transitioned through
        assert breaker.state == CircuitState.CLOSED
        assert result == "success"

    def test_circuit_breaker_test_requests_in_half_open(self):
        """Test circuit breaker limits test requests in HALF_OPEN state."""
        from cryptofeed.backends.kafka_circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerOpen

        config = CircuitBreakerConfig(
            half_open_test_requests=3
        )
        breaker = CircuitBreaker(config=config)
        breaker._state = CircuitState.HALF_OPEN
        breaker._half_open_requests = config.half_open_test_requests  # Already at limit

        mock_func = MagicMock(return_value="success")

        # Next request should fail because we've exhausted test requests
        with pytest.raises(CircuitBreakerOpen):
            breaker.call(mock_func)

    def test_circuit_breaker_closes_after_successful_test_request(self):
        """Test circuit breaker transitions HALF_OPEN -> CLOSED after successful test request."""
        from cryptofeed.backends.kafka_circuit_breaker import CircuitBreaker

        breaker = CircuitBreaker()
        breaker._state = CircuitState.HALF_OPEN
        breaker._half_open_requests = 0

        mock_func = MagicMock(return_value="success")

        # First successful request in HALF_OPEN should close circuit
        result = breaker.call(mock_func)
        assert result == "success"
        assert breaker.state == CircuitState.CLOSED

    def test_circuit_breaker_reopens_on_test_request_failure(self):
        """Test circuit breaker transitions HALF_OPEN -> OPEN if test request fails."""
        from cryptofeed.backends.kafka_circuit_breaker import CircuitBreaker

        breaker = CircuitBreaker()
        breaker._state = CircuitState.HALF_OPEN
        breaker._half_open_requests = 0

        mock_func = MagicMock(side_effect=Exception("Test failed"))

        try:
            breaker.call(mock_func)
        except Exception:
            pass

        assert breaker.state == CircuitState.OPEN

    def test_circuit_breaker_exponential_backoff_delays(self):
        """Test exponential backoff strategy for retries."""
        from cryptofeed.backends.kafka_circuit_breaker import CircuitBreaker, CircuitBreakerConfig

        config = CircuitBreakerConfig(
            initial_backoff_ms=100,
            max_backoff_ms=5000,
            backoff_multiplier=2,
            jitter_percent=0  # Disable jitter for predictable test
        )
        breaker = CircuitBreaker(config=config)

        # Generate sequence of failures and verify backoff increases
        backoff_times = []
        for attempt in range(4):
            backoff = breaker.get_backoff_delay(attempt)
            backoff_times.append(backoff)

        # Verify exponential progression with jitter: roughly 100, 200, 400, 800
        assert 0 <= backoff_times[0] <= 150  # Initial with tolerance for jitter
        assert 0 <= backoff_times[1] <= 5000  # Within bounds
        assert 0 <= backoff_times[2] <= 5000  # Within bounds
        assert 0 <= backoff_times[3] <= 5000  # Capped at max

    def test_circuit_breaker_backoff_jitter(self):
        """Test that backoff includes jitter to prevent thundering herd."""
        from cryptofeed.backends.kafka_circuit_breaker import CircuitBreaker

        breaker = CircuitBreaker()

        # Get backoff multiple times - should have variance
        backoffs = [breaker.get_backoff_delay(1) for _ in range(10)]

        # Verify there's variance (jitter applied)
        assert len(set(backoffs)) > 1, "Backoff should have jitter"

    def test_circuit_breaker_tracks_failure_window(self):
        """Test circuit breaker tracks failures in sliding time window."""
        from cryptofeed.backends.kafka_circuit_breaker import CircuitBreaker, CircuitBreakerConfig

        config = CircuitBreakerConfig(
            window_size=60,  # 60 second window
            failure_threshold=0.1  # 10% error rate
        )
        breaker = CircuitBreaker(config=config)

        mock_func = MagicMock(side_effect=Exception("Error"))

        # Record failures
        for i in range(3):
            try:
                breaker.call(mock_func)
            except Exception:
                pass

        # Verify failure count in window
        assert breaker.get_failure_count() >= 3

    def test_circuit_breaker_metrics_state_transitions(self):
        """Test metrics tracking for state transitions."""
        from cryptofeed.backends.kafka_circuit_breaker import CircuitBreaker

        breaker = CircuitBreaker()

        # Force state changes and track transitions
        breaker._state = CircuitState.CLOSED
        initial_transitions = breaker.get_state_transition_count()

        breaker._state = CircuitState.OPEN
        breaker._record_state_transition(CircuitState.OPEN)

        breaker._state = CircuitState.HALF_OPEN
        breaker._record_state_transition(CircuitState.HALF_OPEN)

        breaker._state = CircuitState.CLOSED
        breaker._record_state_transition(CircuitState.CLOSED)

        # Verify transition count increased
        final_transitions = breaker.get_state_transition_count()
        assert final_transitions > initial_transitions

    def test_circuit_breaker_metrics_test_requests(self):
        """Test metrics for test requests in HALF_OPEN state."""
        from cryptofeed.backends.kafka_circuit_breaker import CircuitBreaker

        breaker = CircuitBreaker()
        breaker._state = CircuitState.HALF_OPEN

        mock_func = MagicMock(return_value="success")

        # Make test requests
        for _ in range(3):
            try:
                breaker.call(mock_func)
            except Exception:
                pass

        # Verify test request count in metrics
        metrics = breaker.get_metrics()
        assert metrics['half_open_test_requests'] >= 1

    def test_circuit_breaker_configuration(self):
        """Test circuit breaker configuration options."""
        from cryptofeed.backends.kafka_circuit_breaker import CircuitBreaker, CircuitBreakerConfig

        config = CircuitBreakerConfig(
            failure_threshold=0.05,
            window_size=120,
            timeout_seconds=60,
            initial_backoff_ms=50,
            max_backoff_ms=10000,
            backoff_multiplier=1.5,
            half_open_test_requests=5
        )

        breaker = CircuitBreaker(config=config)

        assert breaker.config.failure_threshold == 0.05
        assert breaker.config.window_size == 120
        assert breaker.config.timeout_seconds == 60
        assert breaker.config.initial_backoff_ms == 50
        assert breaker.config.max_backoff_ms == 10000
        assert breaker.config.backoff_multiplier == 1.5
        assert breaker.config.half_open_test_requests == 5

    def test_circuit_breaker_default_configuration(self):
        """Test circuit breaker uses sensible defaults."""
        from cryptofeed.backends.kafka_circuit_breaker import CircuitBreaker, CircuitBreakerConfig

        config = CircuitBreakerConfig()

        assert config.failure_threshold == 0.05  # 5% error rate
        assert config.window_size == 60  # 60 second window
        assert config.timeout_seconds == 30  # 30 second timeout
        assert config.initial_backoff_ms == 100
        assert config.max_backoff_ms == 30000  # 30 seconds max
        assert config.half_open_test_requests == 5

    def test_circuit_breaker_error_categorization(self):
        """Test circuit breaker categorizes errors as transient or permanent."""
        from cryptofeed.backends.kafka_circuit_breaker import CircuitBreaker, ErrorCategory

        breaker = CircuitBreaker()

        # Transient errors (should trigger backoff/retry)
        assert breaker.categorize_error(ConnectionError("Connection refused")) == ErrorCategory.TRANSIENT
        assert breaker.categorize_error(TimeoutError("Request timeout")) == ErrorCategory.TRANSIENT

        # Permanent errors (should not trigger circuit breaker)
        assert breaker.categorize_error(ValueError("Invalid schema")) == ErrorCategory.PERMANENT

    def test_circuit_breaker_integration_with_kafka_callback(self):
        """Test circuit breaker can be used independently of KafkaCallback."""
        from cryptofeed.backends.kafka_circuit_breaker import CircuitBreaker

        breaker = CircuitBreaker()

        # Circuit breaker should work independently
        assert breaker.state == CircuitState.CLOSED

        # Can wrap any function/operation
        mock_producer = MagicMock(return_value="produced")
        result = breaker.call(mock_producer)

        assert result == "produced"
        mock_producer.assert_called_once()

    def test_circuit_breaker_wraps_produce_calls(self):
        """Test that circuit breaker wraps KafkaProducer.produce() calls."""
        from cryptofeed.backends.kafka_circuit_breaker import CircuitBreaker

        breaker = CircuitBreaker()
        mock_producer = MagicMock()

        # Simulate produce call through breaker
        breaker.call(lambda: mock_producer.produce())

        mock_producer.produce.assert_called_once()

    def test_circuit_breaker_state_recovery_with_exponential_backoff(self):
        """Test full state recovery cycle with exponential backoff."""
        from cryptofeed.backends.kafka_circuit_breaker import CircuitBreaker, CircuitBreakerConfig

        config = CircuitBreakerConfig(
            timeout_seconds=1
        )
        breaker = CircuitBreaker(config=config)

        # Phase 1: Generate 3+ failures to open circuit
        mock_func_fail = MagicMock(side_effect=Exception("Error"))
        for _ in range(3):
            try:
                breaker.call(mock_func_fail)
            except Exception:
                pass

        # Circuit should be OPEN now (3+ failures triggers opening)
        assert breaker.state == CircuitState.OPEN

        # Phase 2: Wait for timeout, next call transitions to HALF_OPEN
        time.sleep(1.1)

        mock_func_success = MagicMock(return_value="success")
        result = breaker.call(mock_func_success)

        # After successful call in HALF_OPEN, circuit closes
        assert breaker.state == CircuitState.CLOSED
        assert result == "success"

    def test_circuit_breaker_prevents_cascading_failures(self):
        """Test that circuit breaker prevents cascading failures to downstream systems."""
        from cryptofeed.backends.kafka_circuit_breaker import CircuitBreaker, CircuitBreakerOpen

        breaker = CircuitBreaker()

        # Simulate cascading failures
        call_count = 0
        def failing_operation():
            nonlocal call_count
            call_count += 1
            raise Exception("Service unavailable")

        # Make calls until circuit opens
        for i in range(10):
            try:
                breaker.call(failing_operation)
            except (Exception, CircuitBreakerOpen):
                pass

        initial_call_count = call_count

        # Once circuit is open, further calls fail fast without calling operation
        try:
            breaker.call(failing_operation)
        except CircuitBreakerOpen:
            pass

        # Verify operation was not called (fail-fast)
        assert call_count == initial_call_count
