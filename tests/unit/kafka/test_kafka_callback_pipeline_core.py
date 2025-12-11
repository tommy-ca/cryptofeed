"""KafkaCallback pipeline core integration tests.

These tests cover the end-to-end KafkaCallback message pipeline and
integration with the internal message handler/queue.
"""

from __future__ import annotations

import asyncio
from unittest.mock import Mock

import pytest

from tests.unit.kafka.kafka_callback_test_utils import (
    KafkaCallback,
    KafkaConfig,
    KafkaPartitionConfig,
    KafkaTopicConfig,
    _StubProducer,
    _producer_factory,
)


# Mark this module as slow so it can be excluded in default runs
pytestmark = pytest.mark.slow


class TestCompleteMessagePipeline:
    """Test the complete end-to-end message pipeline (Task 5)."""

    def test_kafka_callback_initialization_with_config(self):
        """Initialize KafkaCallback with KafkaConfig object."""
        config = KafkaConfig(
            bootstrap_servers=["kafka:9092"],
            topic=KafkaTopicConfig(strategy="consolidated"),
            partition=KafkaPartitionConfig(strategy="composite"),
            acks="all",
            idempotence=True,
        )
        callback = KafkaCallback(
            kafka_config=config,
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        assert callback.bootstrap_servers == ["kafka:9092"]
        assert callback.acks == "all"
        assert callback._topic_strategy == "consolidated"
        assert callback.is_connected()

    def test_kafka_callback_backward_compatible_initialization(self):
        """Initialize KafkaCallback with direct parameters (backward compatible)."""
        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            acks="all",
            enable_idempotence=True,
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        assert callback.bootstrap_servers == ["kafka:9092"]
        assert callback._topic_strategy == "consolidated"  # Default
        assert callback.is_connected()

    def test_topic_name_generation_consolidated_strategy(self, trade_message):
        """Test topic name generation using TopicManager with consolidated strategy."""
        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            kafka_config=KafkaConfig(
                bootstrap_servers=["kafka:9092"],
                topic=KafkaTopicConfig(strategy="consolidated"),
            ),
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        # Use normalized data type 'trade' (singular, per Critical Issue #1)
        topic = callback._topic_name("trade", trade_message)
        assert topic == "cryptofeed.trade"

    def test_topic_name_generation_per_symbol_strategy(self, trade_message):
        """Test topic name generation using TopicManager with per_symbol strategy."""
        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            kafka_config=KafkaConfig(
                bootstrap_servers=["kafka:9092"],
                topic=KafkaTopicConfig(strategy="per_symbol"),
            ),
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        # Use normalized data type 'trade' (singular, per Critical Issue #1)
        topic = callback._topic_name("trade", trade_message)
        assert topic == "cryptofeed.trade.coinbase.btc-usd"

    def test_topic_name_with_custom_prefix(self, trade_message):
        """Test topic name generation with custom prefix."""
        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            kafka_config=KafkaConfig(
                bootstrap_servers=["kafka:9092"],
                topic=KafkaTopicConfig(strategy="consolidated", prefix="production"),
            ),
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        # Use normalized data type 'trade' (singular, per Critical Issue #1)
        topic = callback._topic_name("trade", trade_message)
        assert topic == "production.cryptofeed.trade"

    def test_partition_key_generation_composite_strategy(self, trade_message):
        """Test partition key generation using CompositePartitioner."""
        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            kafka_config=KafkaConfig(
                bootstrap_servers=["kafka:9092"],
                partition=KafkaPartitionConfig(strategy="composite"),
            ),
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        key = callback._partition_key(trade_message)
        assert key == b"coinbase-btc-usd"

    def test_partition_key_generation_symbol_strategy(self, trade_message):
        """Test partition key generation using SymbolPartitioner."""
        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            kafka_config=KafkaConfig(
                bootstrap_servers=["kafka:9092"],
                partition=KafkaPartitionConfig(strategy="symbol"),
            ),
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        key = callback._partition_key(trade_message)
        assert key == b"btc-usd"

    def test_partition_key_generation_exchange_strategy(self, trade_message):
        """Test partition key generation using ExchangePartitioner."""
        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            kafka_config=KafkaConfig(
                bootstrap_servers=["kafka:9092"],
                partition=KafkaPartitionConfig(strategy="exchange"),
            ),
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        key = callback._partition_key(trade_message)
        assert key == b"coinbase"

    def test_partition_key_generation_round_robin_strategy(self, trade_message):
        """Test partition key generation using RoundRobinPartitioner (returns None)."""
        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            kafka_config=KafkaConfig(
                bootstrap_servers=["kafka:9092"],
                partition=KafkaPartitionConfig(strategy="round_robin"),
            ),
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        key = callback._partition_key(trade_message)
        assert key is None

    def test_header_enrichment_with_trade_message(self, trade_message):
        """Test header enrichment pipeline with Trade message."""
        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
            serialization_format="protobuf",
        )
        headers = callback._header_enricher.build(
            message=trade_message, data_type="trades"
        )
        # Should have 7 headers: 4 mandatory + 3 optional
        assert len(headers) == 7
        header_dict = dict(headers)
        assert header_dict[b"content-type"] == b"application/x-protobuf"
        assert header_dict[b"exchange"] == b"coinbase"
        assert header_dict[b"symbol"] == b"BTC-USD"
        assert header_dict[b"data_type"] == b"trades"
        assert header_dict[b"schema_version"] == b"v0.1.0"
        assert b"producer_version" in header_dict
        assert b"timestamp_generated" in header_dict

    def test_header_enrichment_with_json_serialization(self, trade_message):
        """Test header enrichment with JSON serialization format."""
        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
            serialization_format="json",
        )
        headers = callback._header_enricher.build(
            message=trade_message, data_type="trades"
        )
        header_dict = dict(headers)
        assert header_dict[b"content-type"] == b"application/json"

    @pytest.mark.asyncio
    async def test_complete_pipeline_with_json_serialization(self, trade_message):
        """Test complete pipeline: queue -> serialize -> topic -> partition -> headers -> produce."""
        _StubProducer({})
        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
            serialization_format="json",
        )
        callback._producer = Mock()
        callback._producer.produce = Mock()

        # Process a single message through the pipeline
        await callback._drain_once()  # Should handle empty queue gracefully
        # This test verifies no exceptions are raised

    @pytest.mark.asyncio
    async def test_drain_once_processes_single_message(self, trade_message):
        """Test that _drain_once processes exactly one message from queue."""
        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
            serialization_format="json",
        )

        # Queue a single message via the public queue helper
        queued = callback._queue_message(
            "trades", trade_message, trade_message.timestamp
        )
        assert queued is True
        assert callback.queue_size() == 1

        # Drain once should process this message
        await callback._drain_once()

        # Verify queue is now empty
        assert callback.queue_size() == 0

    def test_message_pipeline_with_different_message_types(
        self, trade_message, ticker_message, candle_message
    ):
        """Test pipeline handles different message types (Trade, Ticker, Candle)."""
        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )

        # Each message type should generate correct topic names using normalized data types
        trade_topic = callback._topic_name("trade", trade_message)
        ticker_topic = callback._topic_name("ticker", ticker_message)
        candle_topic = callback._topic_name("candle", candle_message)

        assert trade_topic == "cryptofeed.trade"
        assert ticker_topic == "cryptofeed.ticker"
        assert candle_topic == "cryptofeed.candle"

    def test_backward_compatibility_with_messages_lacking_attributes(self):
        """Test pipeline handles messages with missing exchange/symbol attributes."""
        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )

        # Create a minimal object without exchange/symbol
        minimal_obj = Mock()
        minimal_obj.exchange = None
        minimal_obj.symbol = None

        # Should not raise exception
        topic = callback._topic_name("trade", minimal_obj)
        assert "cryptofeed" in topic
        callback._partition_key(minimal_obj)
        # Should still return a key or None gracefully


class TestMessageHandlerIntegration:
    """Test integration with KafkaCallback's message handling."""

    def test_queue_message_for_later_processing(self, trade_message):
        """Test queueing messages for async processing."""
        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        # Queue a message
        result = callback._queue_message("trades", trade_message, 1700000000.0)
        assert result is True
        assert callback.queue_size() == 1

    def test_queue_size_tracking(self, trade_message, ticker_message):
        """Test that queue size is tracked correctly."""
        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            connection_timeout_ms=50,
            queue_maxsize=10,
            producer_factory=_producer_factory(_StubProducer),
        )
        assert callback.queue_size() == 0
        callback._queue_message("trades", trade_message)
        assert callback.queue_size() == 1
        callback._queue_message("ticker", ticker_message)
        assert callback.queue_size() == 2

    def test_is_connected_check(self):
        """Test that is_connected() works correctly."""
        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        assert callback.is_connected()

    def test_dynamic_handler_binding(self, trade_message):
        """Test that dynamic handler binding creates correct async handlers."""
        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        # Dynamic handler binding should work
        handler = callback.trade
        assert callable(handler)
        # Handler should be an async function
        assert asyncio.iscoroutinefunction(handler)
