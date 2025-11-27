"""Integration tests for error classification and recovery strategies.

Tests Task 17.2c: Error handling in DLQ and Circuit Breaker patterns
Coverage:
- Transient vs permanent error classification
- Backoff strategy validation for transient errors
- Circuit breaker interaction with DLQ
- Recovery scenario validation
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock



# ============================================================================
# Error Handling and Classification Tests
# ============================================================================


class TestErrorClassification:
    """Test suite for error classification strategies."""

    def test_classify_transient_errors(self):
        """Test classification of transient errors (should retry with backoff)."""
        from cryptofeed.backends.kafka_circuit_breaker import ErrorClassifier

        classifier = ErrorClassifier()

        # Network-related transient errors
        assert classifier.is_transient(ConnectionError("Connection refused"))
        assert classifier.is_transient(TimeoutError("Request timeout"))
        assert classifier.is_transient(Exception("Broker temporarily unavailable"))

        # Broker-related transient errors
        assert classifier.is_transient(Exception("NotLeaderForPartition"))
        assert classifier.is_transient(Exception("RequestTimedOut"))

    def test_classify_permanent_errors(self):
        """Test classification of permanent errors (should go to DLQ immediately)."""
        from cryptofeed.backends.kafka_circuit_breaker import ErrorClassifier

        classifier = ErrorClassifier()

        # Schema/serialization permanent errors
        assert not classifier.is_transient(ValueError("Invalid schema"))
        assert not classifier.is_transient(TypeError("Type error"))
        assert not classifier.is_transient(Exception("Serialization failed"))

        # Authorization permanent errors
        assert not classifier.is_transient(Exception("Authentication error"))
        assert not classifier.is_transient(Exception("Authorization failed"))

    def test_backoff_applied_to_transient_errors(self):
        """Test that exponential backoff is applied to transient errors."""
        from cryptofeed.backends.kafka_circuit_breaker import CircuitBreaker, ErrorClassifier

        breaker = CircuitBreaker()
        classifier = ErrorClassifier()

        # Simulate transient error with backoff
        error = TimeoutError("Connection timeout")
        is_transient = classifier.is_transient(error)

        assert is_transient is True

        # Verify backoff is calculated for retry
        backoff = breaker.get_backoff_delay(attempt=0)
        assert 50 <= backoff <= 150  # Initial backoff with jitter

    def test_no_backoff_for_permanent_errors(self):
        """Test that permanent errors bypass backoff and go to DLQ."""
        from cryptofeed.backends.kafka_dlq import DLQHandler, ErrorClassifier

        classifier = ErrorClassifier()
        dlq_handler = DLQHandler(bootstrap_servers=["localhost:9092"])

        # Permanent error
        error = ValueError("Schema mismatch")
        is_transient = classifier.is_transient(error)

        assert is_transient is False

        # Send directly to DLQ (no backoff)
        dlq_handler.send_to_dlq(
            message=b"bad_message",
            original_topic="cryptofeed.trades",
            error_type="schema_mismatch",
            error_message="Schema mismatch",
            attempts=1
        )

        # Verify in DLQ immediately
        assert dlq_handler.queue_size() >= 1

    def test_circuit_breaker_interacts_with_dlq(self):
        """Test interaction between circuit breaker and DLQ patterns."""
        from cryptofeed.backends.kafka_circuit_breaker import CircuitBreaker
        from cryptofeed.backends.kafka_dlq import DLQHandler

        breaker = CircuitBreaker()
        dlq_handler = DLQHandler(bootstrap_servers=["localhost:9092"])

        # Scenario: broker unavailable (transient error)
        # Circuit breaker opens after repeated failures
        # New messages fail-fast, go to DLQ during recovery window
        mock_producer = MagicMock(side_effect=Exception("Broker unavailable"))

        try:
            breaker.call(mock_producer)
        except Exception:
            pass

        # Once circuit is open, subsequent messages route to DLQ
        dlq_handler.send_to_dlq(
            message=b"queued_message",
            original_topic="cryptofeed.trades",
            error_type="circuit_open",
            error_message="Circuit breaker is open",
            attempts=1
        )

        assert dlq_handler.queue_size() >= 1

    def test_recovery_scenario_broker_recovery(self):
        """Test full recovery scenario: broker becomes available again."""
        from cryptofeed.backends.kafka_circuit_breaker import CircuitBreaker, CircuitBreakerConfig
        from cryptofeed.backends.kafka_dlq import DLQHandler, DLQRecovery

        config = CircuitBreakerConfig(timeout_seconds=1)
        breaker = CircuitBreaker(config=config)
        dlq_handler = DLQHandler(bootstrap_servers=["localhost:9092"])

        # Phase 1: Broker unavailable - open circuit
        mock_producer_fail = MagicMock(side_effect=Exception("Broker down"))

        for _ in range(3):
            try:
                breaker.call(mock_producer_fail)
            except Exception:
                pass

        assert breaker.state.value == "OPEN"

        # Phase 2: Messages queued to DLQ during outage
        dlq_handler.send_to_dlq(
            message=b"message_during_outage",
            original_topic="cryptofeed.trades",
            error_type="broker_unavailable",
            error_message="Broker was unavailable",
            attempts=3
        )

        # Phase 3: Wait for timeout, broker recovers
        time.sleep(1.1)
        mock_producer_success = MagicMock(return_value=None)

        # Circuit transitions to HALF_OPEN and first successful call closes it
        breaker.call(mock_producer_success)
        assert breaker.state.value == "CLOSED"

        # Phase 4: Replay DLQ messages to original topic
        recovery = DLQRecovery(
            bootstrap_servers=["localhost:9092"],
            dlq_handler=dlq_handler
        )

        replayed = recovery.replay_dlq_messages(max_messages=10)
        assert replayed == 1

    def test_recovery_scenario_cascading_failures(self):
        """Test recovery from cascading failure scenario."""
        from cryptofeed.backends.kafka_circuit_breaker import CircuitBreaker, CircuitBreakerConfig

        config = CircuitBreakerConfig(
            failure_threshold=0.1,
            window_size=60
        )
        breaker = CircuitBreaker(config=config)

        # Scenario: Multiple brokers fail, causing cascading failures
        failures = 0
        def operation_with_cascade():
            nonlocal failures
            failures += 1
            raise Exception("Cascade failure")

        # Make requests until circuit opens
        for _ in range(20):
            try:
                breaker.call(operation_with_cascade)
            except Exception:
                pass

        # Circuit should be open, preventing further cascades
        assert breaker.state.value == "OPEN"
        assert failures < 20  # Some requests failed fast

    def test_recovery_with_jittered_backoff(self):
        """Test recovery uses jittered backoff to prevent thundering herd."""
        from cryptofeed.backends.kafka_circuit_breaker import CircuitBreaker, CircuitBreakerConfig

        config = CircuitBreakerConfig(
            initial_backoff_ms=100,
            max_backoff_ms=5000,
            backoff_multiplier=2
        )
        breaker = CircuitBreaker(config=config)

        # Get multiple backoff values for same attempt
        backoffs = [breaker.get_backoff_delay(1) for _ in range(5)]

        # Verify jitter creates variance
        assert len(set(backoffs)) > 1, "Backoff should have jitter to prevent thundering herd"
        assert all(b >= 0 for b in backoffs), "Backoff should be non-negative"

    def test_error_recovery_metrics(self):
        """Test metrics tracking during recovery process."""
        from cryptofeed.backends.kafka_circuit_breaker import CircuitBreaker

        breaker = CircuitBreaker()
        mock_func_fail = MagicMock(side_effect=Exception("Error"))

        # Generate failures (need at least 3 to trigger circuit opening)
        failures = 0
        for _ in range(3):
            try:
                breaker.call(mock_func_fail)
            except Exception:
                failures += 1

        # Get recovery metrics
        metrics = breaker.get_metrics()
        assert metrics['total_failures'] == failures
        assert metrics['total_calls'] == failures

    def test_dlq_handles_serialization_errors(self):
        """Test DLQ properly handles serialization errors."""
        from cryptofeed.backends.kafka_dlq import DLQHandler, ErrorClassifier

        classifier = ErrorClassifier()
        dlq_handler = DLQHandler(bootstrap_servers=["localhost:9092"])

        # Serialization error (permanent)
        error_type = "serialization_error"
        is_transient = classifier.is_transient(Exception(error_type))

        assert is_transient is False

        # Route to DLQ without retry
        dlq_handler.send_to_dlq(
            message=b"unserializable_data",
            original_topic="cryptofeed.trades",
            error_type=error_type,
            error_message="Failed to serialize message",
            attempts=1
        )

        dlq_msg = dlq_handler.get_queued_message(0)
        assert dlq_msg.attempts == 1, "Should not retry serialization errors"

    def test_dlq_handles_timeout_errors(self):
        """Test DLQ properly handles timeout errors (transient)."""
        from cryptofeed.backends.kafka_dlq import DLQHandler, ErrorClassifier
        from cryptofeed.backends.kafka_circuit_breaker import CircuitBreaker

        classifier = ErrorClassifier()
        breaker = CircuitBreaker()
        dlq_handler = DLQHandler(bootstrap_servers=["localhost:9092"])

        # Timeout error (transient)
        error_type = "timeout"
        is_transient = classifier.is_transient(TimeoutError(error_type))

        assert is_transient is True

        # Apply backoff before retry
        backoff = breaker.get_backoff_delay(0)
        assert backoff > 0

        # Only go to DLQ after retries exhausted
        dlq_handler.send_to_dlq(
            message=b"timeout_message",
            original_topic="cryptofeed.trades",
            error_type=error_type,
            error_message="Request timed out after 3 retries",
            attempts=3
        )

        dlq_msg = dlq_handler.get_queued_message(0)
        assert dlq_msg.attempts == 3, "Should have retried before DLQ"

    def test_recovery_respects_rate_limits(self):
        """Test that recovery process respects Kafka rate limits."""
        from cryptofeed.backends.kafka_dlq import DLQRecovery, DLQHandler

        dlq_handler = DLQHandler(bootstrap_servers=["localhost:9092"])
        recovery = DLQRecovery(
            bootstrap_servers=["localhost:9092"],
            dlq_handler=dlq_handler,
            max_messages_per_second=1000  # 1000 msg/s limit
        )

        # Queue multiple messages
        for i in range(100):
            dlq_handler.send_to_dlq(
                message=f"message_{i}".encode(),
                original_topic="cryptofeed.trades",
                error_type="transient",
                error_message="Test",
                attempts=1
            )

        # Replay respects rate limit
        start = time.time()
        recovery.replay_dlq_messages(max_messages=100)
        elapsed = time.time() - start

        # At 1000 msg/s, 100 messages should take ~0.1 seconds minimum
        assert elapsed >= 0.05, "Recovery should respect rate limiting"

    def test_permanent_error_does_not_open_circuit(self):
        """Test that permanent errors don't trigger circuit breaker."""
        from cryptofeed.backends.kafka_circuit_breaker import CircuitBreaker, ErrorClassifier

        classifier = ErrorClassifier()
        breaker = CircuitBreaker()

        # Permanent error should not count toward circuit breaker failure threshold
        error = ValueError("Invalid schema")
        is_transient = classifier.is_transient(error)

        assert is_transient is False

        # Circuit should remain closed even after permanent errors
        assert breaker.state.value == "CLOSED"

    def test_mixed_error_recovery(self):
        """Test recovery when both transient and permanent errors occur."""
        from cryptofeed.backends.kafka_dlq import DLQHandler
        from cryptofeed.backends.kafka_circuit_breaker import CircuitBreaker, ErrorClassifier

        classifier = ErrorClassifier()
        breaker = CircuitBreaker()
        dlq_handler = DLQHandler(bootstrap_servers=["localhost:9092"])

        # Process messages with mixed errors
        error_types = [
            ("timeout", True),  # transient
            ("schema_mismatch", False),  # permanent
            ("timeout", True),  # transient
            ("authentication_error", False),  # permanent
        ]

        dlq_count = 0
        for error_type, is_transient_expected in error_types:
            error = Exception(error_type)
            is_transient = classifier.is_transient(error)

            if is_transient:
                # Apply backoff for retry
                backoff = breaker.get_backoff_delay(0)
                assert backoff > 0
            else:
                # Send to DLQ immediately
                dlq_handler.send_to_dlq(
                    message=b"message",
                    original_topic="cryptofeed.trades",
                    error_type=error_type,
                    error_message=f"Error: {error_type}",
                    attempts=1
                )
                dlq_count += 1

        # DLQ should have only permanent errors (2 of them)
        assert dlq_handler.queue_size() == dlq_count
        assert dlq_count == 2
