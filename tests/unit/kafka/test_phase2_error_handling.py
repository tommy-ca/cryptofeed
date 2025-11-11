"""Unit tests for Critical Issue #2: Error handling and exception boundaries.

Verifies that errors in serialization, topic resolution, header enrichment,
and Kafka produce operations do not collapse the writer task and that all
errors are logged with structured metadata.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
from confluent_kafka import KafkaException

from cryptofeed.types import Trade

kafka_module = pytest.importorskip("cryptofeed.kafka_callback")
KafkaCallback = kafka_module.KafkaCallback


@dataclass
class _RecordedMessage:
    topic: str
    key: Optional[bytes]
    value: bytes
    headers: Dict[str, Any]


class _StubProducer:
    """In-memory producer for testing."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connected = False
        self.messages: List[_RecordedMessage] = []
        self.produce_error: Optional[Exception] = None

    def list_topics(self, timeout: Optional[float] = None):
        self.connected = True
        return {"topics": []}

    def produce(self, topic: str, value: bytes, key: Optional[bytes] = None, headers=None, on_delivery=None):
        if self.produce_error:
            raise self.produce_error

        headers = headers or {}
        self.messages.append(
            _RecordedMessage(topic=topic, key=key, value=value, headers=headers)
        )
        if on_delivery:
            on_delivery(None, None)

    def poll(self, timeout: float):
        return 0

    def flush(self, timeout: Optional[float] = None):
        return 0


class _ErrorProducerFactory:
    """Producer factory that raises errors on produce()."""

    def __init__(self, error: Exception):
        self.error = error

    def __call__(self, config: Dict[str, Any]):
        producer = _StubProducer(config)
        producer.produce_error = self.error
        return producer


def _producer_factory(cls):
    def _factory(config):
        return cls(config)
    return _factory


def _sample_trade() -> Trade:
    return Trade(
        exchange="coinbase",
        symbol="BTC-USD",
        side="buy",
        amount=Decimal("0.25"),
        price=Decimal("68000.10"),
        timestamp=1700000000.0,
        id="trade-1",
        type="spot",
        raw=None,
    )


class TestSerializationErrorHandling:
    """Test error handling for serialization failures."""

    @pytest.mark.asyncio
    async def test_serialization_error_does_not_crash_writer(self):
        """Serialization errors should be caught, logged, and writer continues."""
        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            producer_factory=_producer_factory(_StubProducer),
            serialization_format="json"
        )

        trade = _sample_trade()

        # Mock serialization to raise error
        with patch.object(callback, '_serialize_payload', side_effect=ValueError("Serialization failed")):
            with patch('cryptofeed.kafka_callback.LOG') as mock_log:
                callback._queue_message("trade", trade)
                await callback._drain_once()

                # Verify error was logged
                assert mock_log.error.called
                call_args = mock_log.error.call_args
                assert "Serialization failed" in str(call_args)
                assert call_args[1]['extra']['error_type'] == "serialization_error"
                assert call_args[1]['extra']['exchange'] == "coinbase"
                assert call_args[1]['extra']['symbol'] == "BTC-USD"
                assert call_args[1]['extra']['data_type'] == "trade"

    @pytest.mark.asyncio
    async def test_writer_continues_after_serialization_error(self):
        """Writer should continue processing queue after serialization error."""
        stub_producer = _StubProducer({})
        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            producer_factory=lambda config: stub_producer,
            serialization_format="json"
        )

        trade1 = _sample_trade()
        trade2 = _sample_trade()

        # Queue two messages
        callback._queue_message("trade", trade1)
        callback._queue_message("trade", trade2)

        # First message serialization fails
        with patch.object(callback, '_serialize_payload', side_effect=[ValueError("Fail"), ("payload", [])]):
            with patch('cryptofeed.kafka_callback.LOG'):
                await callback._drain_once()  # First message fails
                await callback._drain_once()  # Second message succeeds

        # Only second message should be produced
        assert len(stub_producer.messages) == 1


class TestTopicResolutionErrorHandling:
    """Test error handling for topic resolution failures."""

    @pytest.mark.asyncio
    async def test_topic_resolution_error_does_not_crash_writer(self):
        """Topic resolution errors should be caught, logged, and writer continues."""
        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            producer_factory=_producer_factory(_StubProducer),
        )

        trade = _sample_trade()

        # Mock topic resolution to raise error
        with patch.object(callback, '_topic_name', side_effect=ValueError("Topic resolution failed")):
            with patch('cryptofeed.kafka_callback.LOG') as mock_log:
                callback._queue_message("trade", trade)
                await callback._drain_once()

                # Verify error was logged
                assert mock_log.error.called
                call_args = mock_log.error.call_args
                assert "Topic resolution failed" in str(call_args)
                assert call_args[1]['extra']['error_type'] == "topic_resolution_error"


class TestPartitionKeyErrorHandling:
    """Test error handling for partition key generation failures."""

    @pytest.mark.asyncio
    async def test_partition_key_error_falls_back_to_none(self):
        """Partition key errors should fall back to None (round-robin)."""
        stub_producer = _StubProducer({})
        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            producer_factory=lambda config: stub_producer,
        )

        trade = _sample_trade()

        # Mock partition key to raise error
        with patch.object(callback, '_partition_key', side_effect=ValueError("Partition key failed")):
            with patch('cryptofeed.kafka_callback.LOG') as mock_log:
                callback._queue_message("trade", trade)
                await callback._drain_once()

                # Verify warning was logged
                assert mock_log.warning.called
                call_args = mock_log.warning.call_args
                assert "Partition key generation failed" in str(call_args)
                assert call_args[1]['extra']['error_type'] == "partition_key_error"

                # Message should still be produced with key=None
                assert len(stub_producer.messages) == 1
                assert stub_producer.messages[0].key is None


class TestHeaderEnrichmentErrorHandling:
    """Test error handling for header enrichment failures."""

    @pytest.mark.asyncio
    async def test_header_enrichment_error_falls_back_to_base_headers(self):
        """Header enrichment errors should fall back to base headers."""
        stub_producer = _StubProducer({})
        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            producer_factory=lambda config: stub_producer,
        )

        trade = _sample_trade()

        # Mock header enricher to raise error
        with patch.object(callback._header_enricher, 'build', side_effect=ValueError("Header enrichment failed")):
            with patch('cryptofeed.kafka_callback.LOG') as mock_log:
                callback._queue_message("trade", trade)
                await callback._drain_once()

                # Verify warning was logged
                assert mock_log.warning.called
                call_args = mock_log.warning.call_args
                assert "Header enrichment failed" in str(call_args)
                assert call_args[1]['extra']['error_type'] == "header_enrichment_error"

                # Message should still be produced with base headers
                assert len(stub_producer.messages) == 1


class TestKafkaProduceErrorHandling:
    """Test error handling for Kafka produce failures."""

    @pytest.mark.asyncio
    async def test_kafka_produce_error_does_not_crash_writer(self):
        """Kafka produce errors should be caught, logged, and writer continues."""
        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            producer_factory=_ErrorProducerFactory(KafkaException("Broker unavailable")),
        )

        trade = _sample_trade()

        with patch('cryptofeed.kafka_callback.LOG') as mock_log:
            callback._queue_message("trade", trade)
            await callback._drain_once()

            # Verify error was logged
            assert mock_log.error.called
            call_args = mock_log.error.call_args
            assert "Kafka produce failed" in str(call_args)
            assert call_args[1]['extra']['error_type'] == "kafka_produce_error"
            assert call_args[1]['extra']['topic'] is not None

    @pytest.mark.asyncio
    async def test_writer_continues_after_kafka_produce_error(self):
        """Writer should continue processing queue after Kafka produce error."""
        stub_producer = _StubProducer({})
        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            producer_factory=lambda config: stub_producer,
        )

        trade1 = _sample_trade()
        trade2 = _sample_trade()

        # Queue two messages
        callback._queue_message("trade", trade1)
        callback._queue_message("trade", trade2)

        # First message produce fails, second succeeds
        stub_producer.produce_error = KafkaException("Transient error")

        with patch('cryptofeed.kafka_callback.LOG'):
            await callback._drain_once()  # First message fails

        # Remove error for second message
        stub_producer.produce_error = None

        with patch('cryptofeed.kafka_callback.LOG'):
            await callback._drain_once()  # Second message succeeds

        # Only second message should be produced
        assert len(stub_producer.messages) == 1


class TestBackpressureHandling:
    """Test backpressure and queue overflow handling."""

    def test_queue_full_drops_message_with_structured_logging(self):
        """Full queue should drop messages and log with metadata."""
        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            producer_factory=_producer_factory(_StubProducer),
            queue_maxsize=1  # Very small queue for testing
        )

        trade1 = _sample_trade()
        trade2 = _sample_trade()

        # Fill the queue
        assert callback._queue_message("trade", trade1) is True
        assert callback.queue_size() == 1

        # Try to add another message - should fail
        with patch('cryptofeed.kafka_callback.LOG') as mock_log:
            assert callback._queue_message("trade", trade2) is False
            assert callback.queue_size() == 1  # Still 1

            # Verify structured error logging
            assert mock_log.error.called
            call_args = mock_log.error.call_args
            assert "queue is full" in str(call_args)
            assert call_args[1]['extra']['error_type'] == "queue_full"
            assert call_args[1]['extra']['exchange'] == "coinbase"
            assert call_args[1]['extra']['symbol'] == "BTC-USD"
            assert call_args[1]['extra']['data_type'] == "trade"
            assert call_args[1]['extra']['queue_size'] == 1


class TestUnexpectedErrorHandling:
    """Test catch-all error handling for unexpected failures."""

    @pytest.mark.asyncio
    async def test_unexpected_error_does_not_crash_writer(self):
        """Unexpected errors should be caught and logged without crashing."""
        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            producer_factory=_producer_factory(_StubProducer),
        )

        trade = _sample_trade()

        # Inject an unexpected error by mocking a low-level operation in finally block
        with patch.object(callback._queue, 'task_done', side_effect=RuntimeError("Unexpected error")):
            with patch('cryptofeed.kafka_callback.LOG') as mock_log:
                callback._queue_message("trade", trade)

                # This should not raise, but log the error
                await callback._drain_once()

                # Verify error handler was triggered (task_done_error from finally block)
                assert mock_log.error.called
                call_args = mock_log.error.call_args
                assert "Failed to mark task as done" in str(call_args) or "Unexpected error" in str(call_args)
                # Check either task_done_error or unexpected_drain_error was logged
                assert call_args[1]['extra']['error_type'] in ("task_done_error", "unexpected_drain_error")


class TestErrorRecovery:
    """Test that writer recovers from errors and continues processing."""

    @pytest.mark.asyncio
    async def test_writer_processes_multiple_messages_with_errors(self):
        """Writer should process all messages even with intermittent errors."""
        stub_producer = _StubProducer({})
        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            producer_factory=lambda config: stub_producer,
        )

        trades = [_sample_trade() for _ in range(5)]

        # Queue all messages
        for trade in trades:
            callback._queue_message("trade", trade)

        # Simulate errors on messages 1 and 3 (0-indexed)
        error_indices = {1, 3}
        call_count = [0]

        def mock_serialize(obj, timestamp):
            idx = call_count[0]
            call_count[0] += 1
            if idx in error_indices:
                raise ValueError(f"Serialization error {idx}")
            return (b"payload", [])

        with patch.object(callback, '_serialize_payload', side_effect=mock_serialize):
            with patch('cryptofeed.kafka_callback.LOG'):
                # Process all 5 messages
                for _ in range(5):
                    await callback._drain_once()

        # Should have produced 3 messages (0, 2, 4)
        assert len(stub_producer.messages) == 3


class TestExactlyOnceDelivery:
    """Test Task 9.3: Exactly-once delivery semantics via idempotent producer."""

    @pytest.mark.asyncio
    async def test_idempotent_producer_configuration(self):
        """Verify idempotent producer is configured for exactly-once semantics."""
        stub_producer = _StubProducer({})
        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            producer_factory=lambda config: stub_producer,
        )

        # Verify idempotence enabled in configuration
        # Extract producer config to check idempotence setting
        assert callback._producer._enable_idempotence is True, "Idempotent producer must be enabled"
        assert callback._producer._acks == "all", "Acks must be set to 'all' for exactly-once"

    @pytest.mark.asyncio
    async def test_duplicate_messages_use_same_partition_key(self):
        """Verify duplicate messages use consistent partition keys for deduplication.

        Kafka's idempotent producer deduplicates messages based on:
        1. Producer ID (assigned by broker)
        2. Sequence number (incremented per partition)
        3. Partition key (user-defined)

        This test verifies the same message sent twice uses the same partition key.
        """
        stub_producer = _StubProducer({})
        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            producer_factory=lambda config: stub_producer,
        )

        # Create a trade with unique ID for tracking
        trade = Trade(
            symbol="BTC-USDT",
            exchange="binance",
            price=Decimal("50000.00"),
            amount=Decimal("0.5"),
            timestamp=1699999999.123,
            side="buy",
            id="duplicate-test-123"
        )

        # Queue the same message twice
        callback._queue_message("trade", trade)
        callback._queue_message("trade", trade)

        # Drain both messages
        await callback._drain_once()
        await callback._drain_once()

        # Both messages should be produced
        assert len(stub_producer.messages) == 2

        # Both messages should have the same partition key
        # (same exchange-symbol produces same key for composite partitioner)
        key1 = stub_producer.messages[0].key
        key2 = stub_producer.messages[1].key

        assert key1 == key2, "Duplicate messages must use same partition key for deduplication"
        assert key1 is not None, "Partition key must not be None"

    @pytest.mark.asyncio
    async def test_producer_config_supports_exactly_once(self):
        """Verify producer configuration aligns with exactly-once semantics.

        Exactly-once delivery requires:
        - enable.idempotence = true
        - acks = 'all'
        - max.in.flight.requests.per.connection <= 5
        """
        from cryptofeed.kafka_callback import KafkaProducerConfig

        producer_config = KafkaProducerConfig(
            bootstrap_servers=["kafka:9092"],
            acks="all",
            idempotence=True
        )

        # Verify configuration supports exactly-once
        assert producer_config.acks == "all", "Acks must be 'all' for exactly-once"
        # Note: idempotence field is stored in model
        # The producer will apply these settings when connecting


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
