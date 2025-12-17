"""KafkaCallback backward compatibility tests."""

from __future__ import annotations

import pytest

from tests.unit.kafka.kafka_callback_test_utils import (
    CompositePartitioner,
    KafkaCallback,
    KafkaConfig,
    KafkaTopicConfig,
    _StubProducer,
    _producer_factory,
)


pytestmark = pytest.mark.slow


class TestBackwardCompatibility:
    """Test backward compatibility with existing deployments."""

    def test_per_symbol_strategy_still_works(self, trade_message):
        """Test that per-symbol strategy still works for legacy deployments."""
        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            kafka_config=KafkaConfig(
                bootstrap_servers=["kafka:9092"],
                topic=KafkaTopicConfig(strategy="per_symbol"),
            ),
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        topic = callback._topic_name("trade", trade_message)
        assert "coinbase" in topic
        assert "btc-usd" in topic

    def test_initialization_without_config_object(self):
        """Test backward-compatible initialization without KafkaConfig."""
        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            acks="all",
            enable_idempotence=True,
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        assert callback.acks == "all"
        assert callback.enable_idempotence is True

    def test_default_strategy_is_consolidated(self):
        """Test that default topic strategy is consolidated."""
        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        assert callback._topic_strategy == "consolidated"

    def test_default_partitioner_is_composite(self):
        """Test that default partitioner is composite."""
        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        assert isinstance(callback._partitioner, CompositePartitioner)

    def test_message_serialization_format_json(self, trade_message):
        """Test that JSON serialization format still works."""
        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
            serialization_format="json",
        )
        headers = callback._header_enricher.build(trade_message, "trades")
        header_dict = dict(headers)
        assert header_dict[b"content-type"] == b"application/json"

    def test_message_serialization_format_protobuf(self, trade_message):
        """Test that protobuf serialization format works."""
        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
            serialization_format="protobuf",
        )
        headers = callback._header_enricher.build(trade_message, "trades")
        header_dict = dict(headers)
        assert header_dict[b"content-type"] == b"application/x-protobuf"
