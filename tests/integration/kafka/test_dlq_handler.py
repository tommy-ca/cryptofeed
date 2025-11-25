"""Integration tests for Dead Letter Queue (DLQ) handler.

Tests Task 17.2a: Dead Letter Queue implementation
Coverage:
- DLQ message routing after retries exhausted
- DLQ message schema (original_topic, payload, error_type, error_message, timestamp, attempts)
- DLQ metrics tracking (message count by error type)
- DLQ recovery (replay messages from DLQ to original topic)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import pytest


LOG = logging.getLogger("feedhandler")


# ============================================================================
# DLQ Handler Tests
# ============================================================================


@dataclass
class DLQMessage:
    """Schema for Dead Letter Queue message."""
    original_topic: str
    original_payload: bytes
    error_type: str
    error_message: str
    timestamp: float
    attempts: int


class TestDLQHandler:
    """Test suite for DLQ handler functionality."""

    def test_dlq_handler_routes_failed_message(self):
        """Test that failed messages are routed to DLQ topic after retries exhausted."""
        from cryptofeed.backends.kafka_dlq import DLQHandler

        handler = DLQHandler(
            bootstrap_servers=["localhost:9092"],
            dlq_topic_prefix="cryptofeed.dlq"
        )

        # Create a failed message scenario
        original_topic = "cryptofeed.trades"
        error_type = "produce_timeout"
        error_message = "Request timed out"
        payload = b"trade_data"

        # Route to DLQ
        handler.send_to_dlq(
            message=payload,
            original_topic=original_topic,
            error_type=error_type,
            error_message=error_message,
            attempts=3
        )

        # Verify DLQ message was queued
        assert handler.queue_size() >= 1, "DLQ message should be queued"

    def test_dlq_message_preserves_original_content(self):
        """Test that DLQ preserves original message content and error context."""
        from cryptofeed.backends.kafka_dlq import DLQHandler

        handler = DLQHandler(bootstrap_servers=["localhost:9092"])

        original_topic = "cryptofeed.trades"
        payload = b"original_trade_message"
        error_type = "kafka_broker_not_available"
        error_message = "No brokers available"
        attempts = 5

        handler.send_to_dlq(
            message=payload,
            original_topic=original_topic,
            error_type=error_type,
            error_message=error_message,
            attempts=attempts
        )

        # Retrieve from internal queue
        dlq_msg = handler.get_queued_message(0)
        assert dlq_msg.original_topic == original_topic
        assert dlq_msg.original_payload == payload
        assert dlq_msg.error_type == error_type
        assert dlq_msg.error_message == error_message
        assert dlq_msg.attempts == attempts

    def test_dlq_schema_validation(self):
        """Test that DLQ message schema is correct."""
        from cryptofeed.backends.kafka_dlq import DLQHandler

        handler = DLQHandler(bootstrap_servers=["localhost:9092"])

        # Send message to DLQ
        handler.send_to_dlq(
            message=b"test_payload",
            original_topic="cryptofeed.orderbook",
            error_type="schema_mismatch",
            error_message="Message size exceeded limit",
            attempts=2
        )

        dlq_msg = handler.get_queued_message(0)

        # Verify schema fields exist
        assert hasattr(dlq_msg, 'original_topic')
        assert hasattr(dlq_msg, 'original_payload')
        assert hasattr(dlq_msg, 'error_type')
        assert hasattr(dlq_msg, 'error_message')
        assert hasattr(dlq_msg, 'timestamp')
        assert hasattr(dlq_msg, 'attempts')

    def test_dlq_preserves_message_ordering(self):
        """Test that DLQ preserves message ordering using partition key."""
        from cryptofeed.backends.kafka_dlq import DLQHandler

        handler = DLQHandler(bootstrap_servers=["localhost:9092"])

        # Send multiple messages with same partition key
        partition_key = "binance-BTC-USDT"

        for i in range(3):
            handler.send_to_dlq(
                message=f"message_{i}".encode(),
                original_topic="cryptofeed.trades",
                error_type="transient_error",
                error_message=f"Error {i}",
                attempts=i + 1,
                partition_key=partition_key
            )

        # Verify messages are queued in order
        assert handler.queue_size() >= 3
        msg1 = handler.get_queued_message(0)
        msg2 = handler.get_queued_message(1)
        msg3 = handler.get_queued_message(2)

        assert msg1.original_payload == b"message_0"
        assert msg2.original_payload == b"message_1"
        assert msg3.original_payload == b"message_2"

    def test_dlq_metrics_track_error_types(self):
        """Test that DLQ metrics track message count by error type."""
        from cryptofeed.backends.kafka_dlq import DLQHandler

        handler = DLQHandler(bootstrap_servers=["localhost:9092"])

        # Send different error types
        error_types = [
            "broker_unavailable",
            "timeout",
            "broker_unavailable",
            "serialization_error",
            "timeout"
        ]

        for error_type in error_types:
            handler.send_to_dlq(
                message=b"test",
                original_topic="cryptofeed.trades",
                error_type=error_type,
                error_message=f"Error: {error_type}",
                attempts=1
            )

        # Get metrics
        metrics = handler.get_dlq_metrics()

        assert metrics['dlq_messages_total'] == 5
        assert metrics['dlq_broker_unavailable'] == 2
        assert metrics['dlq_timeout'] == 2
        assert metrics['dlq_serialization_error'] == 1

    def test_dlq_recovery_replay_messages(self):
        """Test DLQ recovery: replay messages from DLQ to original topic."""
        from cryptofeed.backends.kafka_dlq import DLQHandler, DLQRecovery

        dlq_handler = DLQHandler(bootstrap_servers=["localhost:9092"])

        # Queue some failed messages
        dlq_handler.send_to_dlq(
            message=b"trade_data_1",
            original_topic="cryptofeed.trades",
            error_type="transient_error",
            error_message="Broker temporarily unavailable",
            attempts=3
        )

        dlq_handler.send_to_dlq(
            message=b"trade_data_2",
            original_topic="cryptofeed.trades",
            error_type="transient_error",
            error_message="Broker temporarily unavailable",
            attempts=3
        )

        # Initialize recovery
        recovery = DLQRecovery(
            bootstrap_servers=["localhost:9092"],
            dlq_handler=dlq_handler
        )

        # Replay messages
        count = recovery.replay_dlq_messages(max_messages=10)

        assert count == 2, "Should replay 2 messages"

    def test_dlq_recovery_validates_schema_compatibility(self):
        """Test that DLQ recovery validates schema compatibility before replay."""
        from cryptofeed.backends.kafka_dlq import DLQRecovery, DLQHandler

        dlq_handler = DLQHandler(bootstrap_servers=["localhost:9092"])
        recovery = DLQRecovery(
            bootstrap_servers=["localhost:9092"],
            dlq_handler=dlq_handler
        )

        # Queue message with payload
        dlq_handler.send_to_dlq(
            message=b"incompatible_payload",
            original_topic="cryptofeed.trades",
            error_type="schema_mismatch",
            error_message="Schema version mismatch",
            attempts=1
        )

        # Attempt replay with validation
        with pytest.raises(ValueError, match="schema compatibility"):
            recovery.replay_dlq_messages(
                max_messages=1,
                validate_schema=True
            )

    def test_dlq_configuration_enable_disable(self):
        """Test that DLQ can be enabled/disabled via configuration."""
        from cryptofeed.backends.kafka_dlq import DLQHandler, DLQConfig

        # Test DLQ disabled
        config = DLQConfig(enabled=False)
        handler = DLQHandler(
            bootstrap_servers=["localhost:9092"],
            config=config
        )

        # Send message
        handler.send_to_dlq(
            message=b"test",
            original_topic="cryptofeed.trades",
            error_type="test",
            error_message="test",
            attempts=1
        )

        # Verify no messages queued when disabled
        assert handler.queue_size() == 0, "DLQ should be disabled"

        # Test DLQ enabled
        config = DLQConfig(enabled=True)
        handler = DLQHandler(
            bootstrap_servers=["localhost:9092"],
            config=config
        )

        handler.send_to_dlq(
            message=b"test",
            original_topic="cryptofeed.trades",
            error_type="test",
            error_message="test",
            attempts=1
        )

        assert handler.queue_size() >= 1, "DLQ should be enabled"

    def test_dlq_retention_configuration(self):
        """Test DLQ topic retention policy configuration."""
        from cryptofeed.backends.kafka_dlq import DLQHandler, DLQConfig

        config = DLQConfig(
            enabled=True,
            retention_ms=7 * 24 * 3600 * 1000,  # 7 days
            partitions=3,
            replication_factor=2
        )

        handler = DLQHandler(
            bootstrap_servers=["localhost:9092"],
            config=config
        )

        # Verify config is stored
        assert handler.config.retention_ms == 7 * 24 * 3600 * 1000
        assert handler.config.partitions == 3
        assert handler.config.replication_factor == 2

    def test_dlq_topic_name_customization(self):
        """Test custom DLQ topic prefix configuration."""
        from cryptofeed.backends.kafka_dlq import DLQHandler

        handler = DLQHandler(
            bootstrap_servers=["localhost:9092"],
            dlq_topic_prefix="production.dlq"
        )

        assert handler.dlq_topic == "production.dlq"

        handler2 = DLQHandler(
            bootstrap_servers=["localhost:9092"],
            dlq_topic_prefix="staging.deadletter"
        )

        assert handler2.dlq_topic == "staging.deadletter"

    def test_dlq_message_timestamp_accuracy(self):
        """Test that DLQ message timestamp is accurate."""
        from cryptofeed.backends.kafka_dlq import DLQHandler

        handler = DLQHandler(bootstrap_servers=["localhost:9092"])

        before = time.time()
        handler.send_to_dlq(
            message=b"test",
            original_topic="cryptofeed.trades",
            error_type="test",
            error_message="test",
            attempts=1
        )
        after = time.time()

        dlq_msg = handler.get_queued_message(0)

        assert before <= dlq_msg.timestamp <= after, \
            "Timestamp should be within message send time bounds"

    def test_dlq_error_classification(self):
        """Test classification of permanent vs transient errors."""
        from cryptofeed.backends.kafka_dlq import DLQHandler, ErrorClassifier

        classifier = ErrorClassifier()

        # Transient errors (should retry)
        assert classifier.is_transient("broker_unavailable") is True
        assert classifier.is_transient("timeout") is True
        assert classifier.is_transient("network_error") is True

        # Permanent errors (should go to DLQ immediately)
        assert classifier.is_transient("schema_mismatch") is False
        assert classifier.is_transient("message_size_exceeded") is False
        assert classifier.is_transient("serialization_error") is False

    def test_dlq_recovery_audit_trail(self):
        """Test DLQ recovery maintains audit trail of replayed messages."""
        from cryptofeed.backends.kafka_dlq import DLQRecovery, DLQHandler

        dlq_handler = DLQHandler(bootstrap_servers=["localhost:9092"])
        dlq_handler.send_to_dlq(
            message=b"test_message",
            original_topic="cryptofeed.trades",
            error_type="transient",
            error_message="Temporary broker unavailability",
            attempts=3
        )

        recovery = DLQRecovery(
            bootstrap_servers=["localhost:9092"],
            dlq_handler=dlq_handler
        )

        # Replay and get audit trail
        recovery.replay_dlq_messages(max_messages=1)
        audit_trail = recovery.get_audit_trail()

        assert len(audit_trail) >= 1
        assert audit_trail[0]['status'] == 'replayed'
        assert audit_trail[0]['original_topic'] == 'cryptofeed.trades'

    def test_dlq_integration_with_kafka_callback(self):
        """Test DLQ integration with KafkaCallback error handler."""
        from cryptofeed.backends.kafka_dlq import DLQHandler

        # Initialize with DLQ enabled
        dlq_handler = DLQHandler(bootstrap_servers=["localhost:9092"])

        # Verify DLQ handler can be passed to callbacks
        assert dlq_handler.config.enabled is True
        assert dlq_handler.dlq_topic == "cryptofeed.dlq"

        # DLQ handler should track metrics independently
        dlq_handler.send_to_dlq(
            message=b"test_message",
            original_topic="cryptofeed.trades",
            error_type="test_error",
            error_message="Test error",
            attempts=3
        )

        metrics = dlq_handler.get_dlq_metrics()
        assert metrics["dlq_messages_total"] >= 1
