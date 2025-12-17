"""KafkaCallback performance and scale tests."""

from __future__ import annotations

import pytest

from tests.unit.kafka.kafka_callback_test_utils import (
    KafkaCallback,
    _StubProducer,
    _producer_factory,
)


pytestmark = pytest.mark.slow


class TestPerformanceAndScale:
    """Test performance characteristics and scaling behavior."""

    def test_partition_key_generation_is_fast(self, trade_message):
        """Test that partition key generation is fast (<1ms)."""
        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        import time

        start = time.time()
        for _ in range(1000):
            callback._partition_key(trade_message)
        elapsed = time.time() - start
        # Should be able to generate 1000 keys in <100ms
        assert elapsed < 0.1

    def test_topic_name_generation_is_fast(self, trade_message):
        """Test that topic name generation is fast (<1ms)."""
        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        import time

        start = time.time()
        for _ in range(1000):
            callback._topic_name("trade", trade_message)
        elapsed = time.time() - start
        # Should be able to generate 1000 topic names in <100ms
        assert elapsed < 0.1

    def test_header_enrichment_is_fast(self, trade_message):
        """Test that header enrichment is fast (<1ms)."""
        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        import time

        start = time.time()
        for _ in range(1000):
            callback._header_enricher.build(trade_message, "trades")
        elapsed = time.time() - start
        # Should be able to enrich 1000 messages in <100ms
        assert elapsed < 0.1

    def test_multiple_message_types_in_burst(
        self, trade_message, ticker_message, candle_message, orderbook_message
    ):
        """Test pipeline handles burst of different message types."""
        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        messages = [
            (trade_message, "trades"),
            (ticker_message, "ticker"),
            (candle_message, "candles"),
            (orderbook_message, "orderbook"),
        ]
        # Process all messages without error
        for msg, data_type in messages * 25:  # 100 total messages
            callback._topic_name(data_type, msg)
            callback._partition_key(msg)
            callback._header_enricher.build(msg, data_type)
