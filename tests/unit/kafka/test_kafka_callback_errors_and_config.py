"""KafkaCallback configuration scenarios and error-handling tests."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from tests.unit.kafka.kafka_callback_test_utils import (
    KafkaCallback,
    KafkaConfig,
    KafkaPartitionConfig,
    KafkaProducerConfig,
    KafkaTopicConfig,
    _StubProducer,
    _producer_factory,
)


pytestmark = pytest.mark.slow


class TestErrorHandlingAndEdgeCases:
    """Test error handling, edge cases, and graceful degradation."""

    def test_missing_required_parameter_raises_error(self):
        """Test that missing bootstrap_servers raises error."""
        with pytest.raises(TypeError):
            KafkaCallback()  # No bootstrap_servers or kafka_config

    def test_invalid_topic_strategy_raises_error(self):
        """Test that invalid topic strategy raises error."""
        with pytest.raises(ValueError):
            KafkaConfig(
                bootstrap_servers=["kafka:9092"],
                topic=KafkaTopicConfig(strategy="invalid_strategy"),
            )

    def test_invalid_partition_strategy_raises_error(self):
        """Test that invalid partition strategy raises error."""
        with pytest.raises(ValueError):
            KafkaPartitionConfig(strategy="invalid_partitioner")

    def test_invalid_acks_value_raises_error(self):
        """Test that invalid acks value raises error."""
        with pytest.raises(ValueError):
            KafkaProducerConfig(bootstrap_servers=["kafka:9092"], acks="invalid")

    def test_empty_bootstrap_servers_raises_error(self):
        """Test that empty bootstrap_servers raises error."""
        with pytest.raises(ValueError):
            KafkaConfig(bootstrap_servers=[])

    def test_non_positive_partitions_raises_error(self):
        """Test that non-positive partition count raises error."""
        with pytest.raises(ValueError):
            KafkaTopicConfig(partitions_per_topic=0)

    def test_non_positive_replication_factor_raises_error(self):
        """Test that non-positive replication factor raises error."""
        with pytest.raises(ValueError):
            KafkaTopicConfig(replication_factor=0)

    def test_message_with_unknown_exchange_uses_fallback(self):
        """Test that message with unknown exchange uses fallback gracefully."""
        obj = Mock()
        obj.exchange = None
        obj.symbol = "BTC-USD"
        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        headers = callback._header_enricher.build(obj, "trades")
        header_dict = dict(headers)
        assert header_dict[b"exchange"] == b"unknown"

    def test_message_with_unknown_symbol_uses_fallback(self):
        """Test that message with unknown symbol uses fallback gracefully."""
        obj = Mock()
        obj.exchange = "coinbase"
        obj.symbol = None
        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        headers = callback._header_enricher.build(obj, "trades")
        header_dict = dict(headers)
        assert header_dict[b"symbol"] == b"unknown"


class TestConfigurationScenarios:
    """Test various configuration scenarios and combinations."""

    def test_consolidated_topic_with_composite_partitioner(self, trade_message):
        """Test consolidated topics with composite partitioner."""
        config = KafkaConfig(
            bootstrap_servers=["kafka:9092"],
            topic=KafkaTopicConfig(strategy="consolidated"),
            partition=KafkaPartitionConfig(strategy="composite"),
        )
        callback = KafkaCallback(
            kafka_config=config,
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        assert callback._topic_name("trade", trade_message) == "cryptofeed.trade"
        assert callback._partition_key(trade_message) == b"coinbase-btc-usd"

    def test_per_symbol_topic_with_symbol_partitioner(self, trade_message):
        """Test per-symbol topics with symbol partitioner."""
        config = KafkaConfig(
            bootstrap_servers=["kafka:9092"],
            topic=KafkaTopicConfig(strategy="per_symbol"),
            partition=KafkaPartitionConfig(strategy="symbol"),
        )
        callback = KafkaCallback(
            kafka_config=config,
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        assert (
            callback._topic_name("trade", trade_message)
            == "cryptofeed.trade.coinbase.btc-usd"
        )
        assert callback._partition_key(trade_message) == b"btc-usd"

    def test_consolidated_topic_with_prefix_and_exchange_partitioner(
        self, trade_message
    ):
        """Test consolidated topics with prefix and exchange partitioner."""
        config = KafkaConfig(
            bootstrap_servers=["kafka:9092"],
            topic=KafkaTopicConfig(strategy="consolidated", prefix="production"),
            partition=KafkaPartitionConfig(strategy="exchange"),
        )
        callback = KafkaCallback(
            kafka_config=config,
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        assert (
            callback._topic_name("trade", trade_message)
            == "production.cryptofeed.trade"
        )
        assert callback._partition_key(trade_message) == b"coinbase"

    def test_round_robin_partitioner_returns_none(self, trade_message):
        """Test that round-robin partitioner returns None for partition key."""
        config = KafkaConfig(
            bootstrap_servers=["kafka:9092"],
            partition=KafkaPartitionConfig(strategy="round_robin"),
        )
        callback = KafkaCallback(
            kafka_config=config,
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        assert callback._partition_key(trade_message) is None

    def test_load_config_from_dict(self):
        """Test loading KafkaConfig from dictionary."""
        config_dict = {
            "bootstrap_servers": ["kafka:9092", "kafka:9093"],
            "topic": {"strategy": "consolidated", "prefix": "prod"},
            "partition": {"strategy": "composite"},
            "acks": "all",
            "idempotence": True,
        }
        config = KafkaConfig.from_dict(config_dict)
        assert config.bootstrap_servers == ["kafka:9092", "kafka:9093"]
        assert config.topic.strategy == "consolidated"
        assert config.partition.strategy == "composite"

    def test_whitespace_only_prefix_becomes_default(self):
        """Test that whitespace-only prefix is normalized to 'cryptofeed'."""
        config = KafkaTopicConfig(prefix="   ")
        assert config.prefix == "cryptofeed"

    def test_none_prefix_becomes_default(self):
        """Test that None prefix is normalized to 'cryptofeed'."""
        config = KafkaTopicConfig(prefix=None)
        assert config.prefix == "cryptofeed"

    def test_multiple_bootstrap_servers(self):
        """Test configuration with multiple bootstrap servers."""
        config = KafkaConfig(
            bootstrap_servers=["kafka1:9092", "kafka2:9092", "kafka3:9092"]
        )
        assert len(config.bootstrap_servers) == 3
        callback = KafkaCallback(
            kafka_config=config,
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        assert callback.bootstrap_servers == [
            "kafka1:9092",
            "kafka2:9092",
            "kafka3:9092",
        ]
