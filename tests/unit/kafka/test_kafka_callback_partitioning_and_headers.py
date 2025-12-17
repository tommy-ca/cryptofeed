"""KafkaCallback partition key consistency and header generation tests."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from tests.unit.kafka.kafka_callback_test_utils import (
    HeaderEnricher,
    KafkaCallback,
    KafkaConfig,
    KafkaPartitionConfig,
    MessageHeaders,
    OptionalHeaders,
    _StubProducer,
    _producer_factory,
)


pytestmark = pytest.mark.slow


class TestPartitionKeyConsistency:
    """Test that partition keys are consistent and deterministic."""

    def test_symbol_partitioner_consistency(self, trade_message):
        """Test that same symbol always generates same partition key."""
        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            kafka_config=KafkaConfig(
                bootstrap_servers=["kafka:9092"],
                partition=KafkaPartitionConfig(strategy="symbol"),
            ),
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        key1 = callback._partition_key(trade_message)
        key2 = callback._partition_key(trade_message)
        assert key1 == key2 == b"btc-usd"

    def test_composite_partitioner_consistency(self, trade_message):
        """Test that same exchange-symbol always generates same partition key."""
        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            kafka_config=KafkaConfig(
                bootstrap_servers=["kafka:9092"],
                partition=KafkaPartitionConfig(strategy="composite"),
            ),
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        key1 = callback._partition_key(trade_message)
        key2 = callback._partition_key(trade_message)
        assert key1 == key2 == b"coinbase-btc-usd"

    def test_exchange_partitioner_consistency(self, trade_message):
        """Test that same exchange always generates same partition key."""
        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            kafka_config=KafkaConfig(
                bootstrap_servers=["kafka:9092"],
                partition=KafkaPartitionConfig(strategy="exchange"),
            ),
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        key1 = callback._partition_key(trade_message)
        key2 = callback._partition_key(trade_message)
        assert key1 == key2 == b"coinbase"

    def test_different_exchanges_different_keys(self, trade_message, trade_binance):
        """Test that different exchanges generate different partition keys."""
        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            kafka_config=KafkaConfig(
                bootstrap_servers=["kafka:9092"],
                partition=KafkaPartitionConfig(strategy="exchange"),
            ),
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        key_coinbase = callback._partition_key(trade_message)
        key_binance = callback._partition_key(trade_binance)
        assert key_coinbase != key_binance
        assert key_coinbase == b"coinbase"
        assert key_binance == b"binance"

    def test_different_symbols_different_composite_keys(
        self, trade_message, trade_binance
    ):
        """Test that different exchange-symbol pairs generate different keys."""
        callback = KafkaCallback(
            bootstrap_servers=["kafka:9092"],
            kafka_config=KafkaConfig(
                bootstrap_servers=["kafka:9092"],
                partition=KafkaPartitionConfig(strategy="composite"),
            ),
            connection_timeout_ms=50,
            producer_factory=_producer_factory(_StubProducer),
        )
        key1 = callback._partition_key(trade_message)
        key2 = callback._partition_key(trade_binance)
        assert key1 != key2


class TestHeaderGeneration:
    """Test header generation for all message types."""

    def test_mandatory_headers_structure(self, trade_message):
        """Test that mandatory headers have correct structure."""
        headers = MessageHeaders.build(
            message=trade_message,
            data_type="trades",
            content_type="application/x-protobuf",
        )
        assert len(headers) == 4
        assert all(isinstance(h, tuple) and len(h) == 2 for h in headers)
        assert all(isinstance(h[0], bytes) and isinstance(h[1], bytes) for h in headers)

    def test_optional_headers_structure(self):
        """Test that optional headers have correct structure."""
        headers = OptionalHeaders.build(
            schema_version="v1",
            producer_version="2.4.1",
            timestamp_generated="2025-11-09T12:34:56Z",
        )
        assert len(headers) == 4
        assert all(isinstance(h, tuple) and len(h) == 2 for h in headers)
        assert all(isinstance(h[0], bytes) and isinstance(h[1], bytes) for h in headers)

    def test_header_enricher_combines_mandatory_and_optional(self, trade_message):
        """Test that HeaderEnricher combines both mandatory and optional headers."""
        enricher = HeaderEnricher(
            content_type="application/x-protobuf", schema_version="v1"
        )
        headers = enricher.build(trade_message, "trades")
        assert len(headers) == 8  # 4 mandatory + 4 optional
        header_names = [h[0] for h in headers]
        assert b"content-type" in header_names
        assert b"exchange" in header_names
        assert b"symbol" in header_names
        assert b"data_type" in header_names
        assert b"schema_version" in header_names
        assert b"producer_version" in header_names
        assert b"timestamp_generated" in header_names

    def test_header_values_are_encoded_as_bytes(self, trade_message):
        """Test that all header values are UTF-8 encoded bytes."""
        enricher = HeaderEnricher()
        headers = enricher.build(trade_message, "trades")
        for name, value in headers:
            assert isinstance(name, bytes)
            assert isinstance(value, bytes)

    def test_headers_with_special_characters_in_symbol(self):
        """Test header generation with special characters in symbol."""
        obj = Mock()
        obj.exchange = "binance"
        obj.symbol = "BTC_USDT"  # Underscore should be converted and normalized to lowercase
        headers = MessageHeaders.build(obj, "trades", "application/json")
        header_dict = dict(headers)
        assert header_dict[b"symbol"] == b"btc-usdt"  # Normalized: underscore→hyphen, lowercase

    def test_headers_with_case_insensitivity(self):
        """Test that exchange names are normalized to lowercase in headers."""
        obj = Mock()
        obj.exchange = "COINBASE"  # Uppercase
        obj.symbol = "BTC-USD"
        headers = MessageHeaders.build(obj, "trades", "application/json")
        header_dict = dict(headers)
        assert header_dict[b"exchange"] == b"coinbase"

    def test_optional_headers_with_custom_timestamp(self):
        """Test optional headers with custom timestamp."""
        custom_timestamp = "2025-11-09T14:30:00Z"
        headers = OptionalHeaders.build(timestamp_generated=custom_timestamp)
        header_dict = dict(headers)
        assert header_dict[b"timestamp_generated"] == b"2025-11-09T14:30:00Z"

    def test_optional_headers_with_default_timestamp(self):
        """Test optional headers generate default timestamp when not provided."""
        headers = OptionalHeaders.build()
        header_dict = dict(headers)
        assert b"timestamp_generated" in header_dict
        # Timestamp should be a valid ISO8601 format ending with Z
        ts_value = header_dict[b"timestamp_generated"].decode("utf-8")
        assert ts_value.endswith("Z")
        assert "T" in ts_value  # Should have date-time separator

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
