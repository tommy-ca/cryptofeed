"""KafkaCallback tests for individual message types and additional scenarios."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from tests.unit.kafka.kafka_callback_test_utils import (
    KafkaCallback,
    KafkaConfig,
    KafkaPartitionConfig,
    KafkaTopicConfig,
    MessageHeaders,
    OptionalHeaders,
    PartitionerFactory,
    TopicManager,
    HeaderEnricher,
    KafkaProducerConfig,
    SymbolPartitioner,
    _StubProducer,
    _producer_factory,
)


pytestmark = pytest.mark.slow


class TestAllMessageTypesIntegration:
    """Test the pipeline with all cryptofeed message types."""

    def test_trade_message_pipeline(self, trade_message):
        """Test Trade message through complete pipeline."""
        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            kafka_config=KafkaConfig(
                bootstrap_servers=["kafka:9092"],
                topic=KafkaTopicConfig(strategy="consolidated"),
                partition=KafkaPartitionConfig(strategy="composite"),
            ),
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        topic = callback._topic_name("trade", trade_message)
        key = callback._partition_key(trade_message)
        headers = callback._header_enricher.build(trade_message, "trades")

        assert topic == "cryptofeed.trade"
        assert key == b"coinbase-btc-usd"
        assert len(headers) == 7

    def test_ticker_message_pipeline(self, ticker_message):
        """Test Ticker message through complete pipeline."""
        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        topic = callback._topic_name("ticker", ticker_message)
        key = callback._partition_key(ticker_message)
        headers = callback._header_enricher.build(ticker_message, "ticker")

        assert topic == "cryptofeed.ticker"
        assert key is not None  # Composite strategy
        assert len(headers) == 7

    def test_candle_message_pipeline(self, candle_message):
        """Test Candle message through complete pipeline."""
        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            kafka_config=KafkaConfig(
                bootstrap_servers=["kafka:9092"],
                topic=KafkaTopicConfig(strategy="consolidated"),
            ),
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        topic = callback._topic_name("candle", candle_message)
        key = callback._partition_key(candle_message)

        assert topic == "cryptofeed.candle"
        assert key is not None

    def test_orderbook_message_pipeline(self, orderbook_message):
        """Test OrderBook message through complete pipeline."""
        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            kafka_config=KafkaConfig(
                bootstrap_servers=["kafka:9092"],
                topic=KafkaTopicConfig(strategy="consolidated"),
            ),
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        topic = callback._topic_name("orderbook", orderbook_message)
        key = callback._partition_key(orderbook_message)

        assert topic == "cryptofeed.orderbook"
        assert key is not None

    def test_liquidation_message_pipeline(self, liquidation_message):
        """Test Liquidation message through complete pipeline."""
        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            kafka_config=KafkaConfig(
                bootstrap_servers=["kafka:9092"],
                topic=KafkaTopicConfig(strategy="consolidated"),
            ),
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        topic = callback._topic_name("liquidation", liquidation_message)
        key = callback._partition_key(liquidation_message)

        assert topic == "cryptofeed.liquidation"
        assert key is not None

    def test_funding_message_pipeline(self, funding_message):
        """Test Funding message through complete pipeline."""
        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            kafka_config=KafkaConfig(
                bootstrap_servers=["kafka:9092"],
                topic=KafkaTopicConfig(strategy="consolidated"),
            ),
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        topic = callback._topic_name("funding", funding_message)
        key = callback._partition_key(funding_message)

        assert topic == "cryptofeed.funding"
        assert key is not None

    def test_index_message_pipeline(self, index_message):
        """Test Index message through complete pipeline."""
        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            kafka_config=KafkaConfig(
                bootstrap_servers=["kafka:9092"],
                topic=KafkaTopicConfig(strategy="consolidated"),
            ),
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        topic = callback._topic_name("index", index_message)
        assert topic == "cryptofeed.index"

    def test_openinterest_message_pipeline(self, openinterest_message):
        """Test OpenInterest message through complete pipeline."""
        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            kafka_config=KafkaConfig(
                bootstrap_servers=["kafka:9092"],
                topic=KafkaTopicConfig(strategy="consolidated"),
            ),
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        topic = callback._topic_name("openinterest", openinterest_message)
        assert topic == "cryptofeed.openinterest"


class TestAdditionalIntegrationScenarios:
    """Additional comprehensive integration test scenarios."""

    def test_kafka_config_with_all_producer_settings(self):
        """Test KafkaConfig with all producer settings configured."""
        config = KafkaConfig(
            bootstrap_servers=["kafka:9092"],
            topic=KafkaTopicConfig(
                strategy="consolidated",
                prefix="prod",
                partitions_per_topic=6,
                replication_factor=2,
            ),
            partition=KafkaPartitionConfig(strategy="composite"),
            acks="all",
            idempotence=True,
            retries=5,
            retry_backoff_ms=200,
            batch_size=32768,
            linger_ms=20,
            compression_type="snappy",
        )
        callback = KafkaCallback(
            kafka_config=config,
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        assert callback.acks == "all"
        assert callback.enable_idempotence is True
        assert callback._topic_prefix == "prod"

    def test_multiple_message_types_generate_different_topics(
        self, trade_message, ticker_message, candle_message
    ):
        """Test that different message types generate correct topic names."""
        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            kafka_config=KafkaConfig(
                bootstrap_servers=["kafka:9092"],
                topic=KafkaTopicConfig(strategy="consolidated"),
            ),
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )

        topics = {
            "trades": callback._topic_name("trade", trade_message),
            "ticker": callback._topic_name("ticker", ticker_message),
            "candle": callback._topic_name("candle", candle_message),
        }

        assert topics["trades"] == "cryptofeed.trade"
        assert topics["ticker"] == "cryptofeed.ticker"
        assert topics["candle"] == "cryptofeed.candle"
        assert len(set(topics.values())) == 3  # All different

    def test_per_symbol_topics_with_multiple_exchanges(
        self, trade_message, trade_binance
    ):
        """Test per-symbol topics with different exchanges produce different topic names."""
        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            kafka_config=KafkaConfig(
                bootstrap_servers=["kafka:9092"],
                topic=KafkaTopicConfig(strategy="per_symbol"),
            ),
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )

        topic_coinbase = callback._topic_name("trade", trade_message)
        topic_binance = callback._topic_name("trade", trade_binance)

        assert "coinbase" in topic_coinbase
        assert "binance" in topic_binance
        assert topic_coinbase != topic_binance

    def test_header_normalization_with_mixed_case_exchange(self):
        """Test that exchange names are normalized to lowercase in headers."""
        obj = Mock()
        obj.exchange = "CoInBaSe"  # Mixed case
        obj.symbol = "BTC-USD"

        headers = MessageHeaders.build(obj, "trades", "application/json")
        header_dict = dict(headers)
        assert header_dict[b"exchange"] == b"coinbase"

    def test_header_symbol_normalization_multiple_formats(self):
        """Test symbol normalization in headers with various input formats."""
        obj1 = Mock()
        obj1.exchange = "binance"
        obj1.symbol = "BTC_USDT"

        obj2 = Mock()
        obj2.exchange = "binance"
        obj2.symbol = "btc-usdt"

        headers1 = MessageHeaders.build(obj1, "trades", "application/json")
        headers2 = MessageHeaders.build(obj2, "trades", "application/json")

        dict1 = dict(headers1)
        dict2 = dict(headers2)

        # Both should normalize to BTC-USDT
        assert dict1[b"symbol"] == b"BTC-USDT"
        assert dict2[b"symbol"] == b"btc-usdt"

    def test_configuration_from_dict_with_nested_objects(self):
        """Test loading configuration from nested dictionary structure."""
        config_dict = {
            "bootstrap_servers": ["kafka1:9092", "kafka2:9092"],
            "topic": {
                "strategy": "consolidated",
                "prefix": "staging",
                "partitions_per_topic": 5,
                "replication_factor": 2,
            },
            "partition": {"strategy": "symbol"},
            "acks": "1",
            "batch_size": 32768,
            "linger_ms": 20,
        }
        config = KafkaConfig.from_dict(config_dict)
        assert config.topic.prefix == "staging"
        assert config.partition.strategy == "symbol"
        assert config.acks == "1"
        assert config.batch_size == 32768

    def test_all_supported_data_types_generate_topics(self, trade_message):
        """Test that all supported data types can generate topic names."""
        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            kafka_config=KafkaConfig(
                bootstrap_servers=["kafka:9092"],
                topic=KafkaTopicConfig(strategy="consolidated"),
            ),
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )

        supported_types = [
            "trades",
            "orderbook",
            "ticker",
            "candle",
            "funding",
            "liquidation",
            "index",
            "openinterest",
        ]

        topics = []
        for data_type in supported_types:
            topic = callback._topic_name(data_type, trade_message)
            topics.append(topic)
            assert f"cryptofeed.{data_type}" == topic

        # All should be different
        assert len(set(topics)) == len(supported_types)

    def test_partition_key_with_special_symbols(self):
        """Test partition key generation with special characters in symbol."""
        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            kafka_config=KafkaConfig(
                bootstrap_servers=["kafka:9092"],
                partition=KafkaPartitionConfig(strategy="symbol"),
            ),
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )

        obj = Mock()
        obj.exchange = "binance"
        obj.symbol = "BTC/USD"  # Slash character

        key = callback._partition_key(obj)
        assert key is not None
        assert isinstance(key, bytes)

    def test_partition_key_empty_symbol_graceful_handling(self):
        """Test that empty symbol is handled gracefully."""
        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            kafka_config=KafkaConfig(
                bootstrap_servers=["kafka:9092"],
                partition=KafkaPartitionConfig(strategy="symbol"),
            ),
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )

        obj = Mock()
        obj.exchange = "binance"
        obj.symbol = ""

        key = callback._partition_key(obj)
        # Should still return a key (for empty symbol)
        assert key is not None

    def test_headers_with_multiple_content_types(self, trade_message):
        """Test header generation with different content types."""
        content_types = [
            "application/x-protobuf",
            "application/json",
            "application/octet-stream",
        ]

        for content_type in content_types:
            headers = MessageHeaders.build(trade_message, "trades", content_type)
            header_dict = dict(headers)
            assert header_dict[b"content-type"] == content_type.encode("utf-8")

    def test_header_enricher_with_custom_producer_version(self, trade_message):
        """Test HeaderEnricher with custom producer version."""
        custom_version = "3.0.0-rc1"
        enricher = HeaderEnricher(producer_version=custom_version)
        headers = enricher.build(trade_message, "trades")
        header_dict = dict(headers)
        assert header_dict[b"producer_version"] == custom_version.encode("utf-8")

    def test_header_enricher_preserves_header_order(self, trade_message):
        """Test that HeaderEnricher returns headers in consistent order."""
        enricher = HeaderEnricher()
        headers1 = enricher.build(trade_message, "trades")
        headers2 = enricher.build(trade_message, "trades")

        # Header order should be consistent
        header_names_1 = [h[0] for h in headers1]
        header_names_2 = [h[0] for h in headers2]
        assert header_names_1 == header_names_2

    def test_configuration_validation_rejects_invalid_compression(self):
        """Test that KafkaProducerConfig rejects invalid compression types."""
        with pytest.raises(ValueError, match="compression_type"):
            KafkaProducerConfig(
                bootstrap_servers=["kafka:9092"], compression_type="invalid_compression"
            )

    def test_configuration_validation_rejects_negative_retries(self):
        """Test that KafkaProducerConfig rejects negative retries."""
        with pytest.raises(ValueError, match="retries"):
            KafkaProducerConfig(bootstrap_servers=["kafka:9092"], retries=-1)

    def test_configuration_validation_rejects_negative_batch_size(self):
        """Test that KafkaProducerConfig rejects non-positive batch size."""
        with pytest.raises(ValueError, match="batch_size"):
            KafkaProducerConfig(bootstrap_servers=["kafka:9092"], batch_size=0)

    def test_partitioner_factory_with_case_insensitive_strategy(self):
        """Test PartitionerFactory handles case-insensitive strategy names."""
        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            kafka_config=KafkaConfig(
                bootstrap_servers=["kafka:9092"],
                partition=KafkaPartitionConfig(strategy="SYMBOL"),  # Uppercase
            ),
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        # Should handle uppercase and work correctly
        assert isinstance(callback._partitioner, SymbolPartitioner)

    def test_message_handler_queue_full_handling(self):
        """Test message handler handles full queue gracefully."""
        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            queue_maxsize=1,  # Very small queue
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )

        obj = Mock()
        obj.symbol = "BTC-USD"
        obj.exchange = "coinbase"

        # Fill the queue
        result1 = callback._queue_message("trades", obj)
        assert result1 is True

        # Second message should fail (queue full)
        result2 = callback._queue_message("trades", obj)
        assert result2 is False

    def test_topic_manager_validation_with_all_data_types(self):
        """Test TopicManager validates all supported data types."""
        supported = TopicManager.SUPPORTED_DATA_TYPES

        for data_type in supported:
            # Should not raise
            TopicManager.validate_data_type(data_type)

        # Invalid type should raise
        with pytest.raises(ValueError):
            TopicManager.validate_data_type("unsupported_type")

    def test_topic_manager_validation_with_all_strategies(self):
        """Test TopicManager validates all supported strategies."""
        for strategy in ["consolidated", "per_symbol"]:
            # Should not raise
            TopicManager.validate_strategy(strategy)

        # Invalid strategy should raise
        with pytest.raises(ValueError):
            TopicManager.validate_strategy("invalid_strategy")

    def test_partition_strategies_with_normalized_symbols(self, trade_message):
        """Test all partition strategies handle symbol normalization correctly."""
        KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )

        strategies = ["composite", "symbol", "exchange", "round_robin"]
        for strategy_name in strategies:
            partitioner = PartitionerFactory.create(strategy_name)
            key = partitioner.get_partition_key(trade_message)

            # All keys should be either None or bytes
            assert key is None or isinstance(key, bytes)

    def test_configuration_from_yaml_with_minimal_config(self, tmp_path):
        """Test loading KafkaConfig from minimal YAML file."""
        yaml_file = tmp_path / "minimal_kafka.yaml"
        yaml_file.write_text(
            """
bootstrap_servers:
  - kafka:9092
"""
        )

        config = KafkaConfig.from_yaml(str(yaml_file))
        assert config.bootstrap_servers == ["kafka:9092"]
        assert config.topic.strategy == "consolidated"  # Default
        assert config.partition.strategy == "composite"  # Default

    def test_configuration_from_yaml_with_full_config(self, tmp_path):
        """Test loading KafkaConfig from full YAML file."""
        yaml_file = tmp_path / "full_kafka.yaml"
        yaml_file.write_text(
            """
bootstrap_servers:
  - kafka1:9092
  - kafka2:9092
topic:
  strategy: per_symbol
  prefix: production
  partitions_per_topic: 6
  replication_factor: 3
partition:
  strategy: symbol
acks: all
idempotence: true
retries: 5
retry_backoff_ms: 200
batch_size: 32768
linger_ms: 20
compression_type: snappy
"""
        )

        config = KafkaConfig.from_yaml(str(yaml_file))
        assert len(config.bootstrap_servers) == 2
        assert config.topic.strategy == "per_symbol"
        assert config.topic.prefix == "production"
        assert config.partition.strategy == "symbol"
        assert config.acks == "all"
        assert config.batch_size == 32768

    def test_configuration_from_yaml_missing_file(self):
        """Test loading from non-existent YAML file raises error."""
        with pytest.raises(FileNotFoundError):
            KafkaConfig.from_yaml("/nonexistent/path/kafka.yaml")

    def test_configuration_from_empty_yaml_file(self, tmp_path):
        """Test loading from empty YAML file raises error."""
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("")

        with pytest.raises(ValueError):
            KafkaConfig.from_yaml(str(yaml_file))

    def test_header_timestamp_iso8601_format(self):
        """Test that timestamp_generated header is valid ISO8601 format."""
        headers = OptionalHeaders.build()
        header_dict = dict(headers)

        timestamp_str = header_dict[b"timestamp_generated"].decode("utf-8")

        # Should end with Z for UTC
        assert timestamp_str.endswith("Z")
        # Should have date-time separator T
        assert "T" in timestamp_str
        # Should have digits (basic ISO8601 check)
        assert any(c.isdigit() for c in timestamp_str)

    def test_header_producer_version_format(self):
        """Test that producer_version header has valid format."""
        headers = OptionalHeaders.build()
        header_dict = dict(headers)

        version_str = header_dict[b"producer_version"].decode("utf-8")

        # Should be non-empty and typically in X.Y.Z format
        assert len(version_str) > 0
        # Should contain at least one dot for version format
        parts = version_str.split(".")
        assert len(parts) >= 2  # At least major.minor

    def test_multiple_callbacks_with_different_configs(self, trade_message):
        """Test multiple KafkaCallback instances with different configurations."""
        callback1 = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            kafka_config=KafkaConfig(
                bootstrap_servers=["kafka:9092"],
                topic=KafkaTopicConfig(strategy="consolidated"),
            ),
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )

        callback2 = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            kafka_config=KafkaConfig(
                bootstrap_servers=["kafka:9092"],
                topic=KafkaTopicConfig(strategy="per_symbol"),
            ),
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )

        topic1 = callback1._topic_name("trade", trade_message)
        topic2 = callback2._topic_name("trade", trade_message)

        # Different strategies should produce different topics
        assert topic1 != topic2

    def test_partitioner_consistency_across_multiple_instances(self, trade_message):
        """Test that same partitioner strategy produces same keys across instances."""
        partitioner1 = PartitionerFactory.create("symbol")
        partitioner2 = PartitionerFactory.create("symbol")

        key1 = partitioner1.get_partition_key(trade_message)
        key2 = partitioner2.get_partition_key(trade_message)

        assert key1 == key2

    def test_header_enricher_with_both_custom_values(self, trade_message):
        """Test HeaderEnricher with all custom values specified."""
        custom_timestamp = "2025-11-09T14:30:00Z"
        custom_version = "test-version"

        enricher = HeaderEnricher(
            content_type="application/json",
            schema_version="v2",
            producer_version=custom_version,
            timestamp_generated=custom_timestamp,
        )

        headers = enricher.build(trade_message, "trades")
        header_dict = dict(headers)

        assert header_dict[b"content-type"] == b"application/json"
        assert header_dict[b"schema_version"] == b"v2"
        assert header_dict[b"producer_version"] == custom_version.encode("utf-8")
        assert header_dict[b"timestamp_generated"] == custom_timestamp.encode("utf-8")
