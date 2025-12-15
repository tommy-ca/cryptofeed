"""Tests for message header enrichment system (Task 3).

This module tests the message header enrichment pipeline that adds routing
metadata to Kafka messages:
- Mandatory headers: content-type, exchange, symbol, data_type
- Optional headers: schema_version, producer_version, timestamp_generated

All headers must be returned as list of tuples with byte values (UTF-8 encoded).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

import pytest

from cryptofeed.kafka_callback import HeaderEnricher, MessageHeaders, OptionalHeaders


# ============================================================================
# Test Fixtures and Mock Messages
# ============================================================================


@dataclass(slots=True)
class MockTrade:
    """Mock Trade message for testing."""

    exchange: str = "coinbase"
    symbol: str = "BTC-USD"
    price: float = 50000.0
    amount: float = 1.0
    timestamp: float = 1234567890.0


@dataclass(slots=True)
class MockTicker:
    """Mock Ticker message for testing."""

    exchange: str = "binance"
    symbol: str = "ETH-USDT"
    bid: float = 3000.0
    ask: float = 3001.0
    timestamp: float = 1234567890.0


@dataclass(slots=True)
class MockOrderBook:
    """Mock OrderBook message for testing."""

    exchange: str = "kraken"
    symbol: str = "SOL-USD"
    timestamp: float = 1234567890.0


@dataclass(slots=True)
class MockCandle:
    """Mock Candle message for testing."""

    exchange: str = "bitmex"
    symbol: str = "XBT-USD"
    timestamp: float = 1234567890.0


@dataclass(slots=True)
class MockMessageWithoutMetadata:
    """Mock message without exchange or symbol."""

    timestamp: float = 1234567890.0


# ============================================================================
# MessageHeaders (Mandatory Headers) Tests
# ============================================================================


class TestMessageHeadersClass:
    """Test MessageHeaders class for building mandatory headers."""

    def test_mandatory_headers_all_present(self):
        """Test that all mandatory headers are present in output."""
        trade = MockTrade()
        headers = MessageHeaders.build(
            message=trade, data_type="trades", content_type="application/x-protobuf"
        )

        # Convert to dict for easier checking
        header_dict = dict(headers)

        assert b"content-type" in header_dict
        assert b"exchange" in header_dict
        assert b"symbol" in header_dict
        assert b"data_type" in header_dict

    def test_content_type_protobuf(self):
        """Test content-type header for protobuf format."""
        trade = MockTrade()
        headers = MessageHeaders.build(
            message=trade, data_type="trades", content_type="application/x-protobuf"
        )

        header_dict = dict(headers)
        assert header_dict[b"content-type"] == b"application/x-protobuf"

    def test_content_type_json(self):
        """Test content-type header for JSON format."""
        trade = MockTrade()
        headers = MessageHeaders.build(
            message=trade, data_type="trades", content_type="application/json"
        )

        header_dict = dict(headers)
        assert header_dict[b"content-type"] == b"application/json"

    def test_exchange_header_extraction(self):
        """Test exchange header extracted from message."""
        trade = MockTrade(exchange="binance")
        headers = MessageHeaders.build(
            message=trade, data_type="trades", content_type="application/x-protobuf"
        )

        header_dict = dict(headers)
        assert header_dict[b"exchange"] == b"binance"

    def test_exchange_header_case_normalization(self):
        """Test exchange header is lowercase."""
        trade = MockTrade(exchange="Coinbase")
        headers = MessageHeaders.build(
            message=trade, data_type="trades", content_type="application/x-protobuf"
        )

        header_dict = dict(headers)
        assert header_dict[b"exchange"] == b"coinbase"

    def test_symbol_header_extraction(self):
        """Test symbol header extracted from message."""
        trade = MockTrade(symbol="BTC-USD")
        headers = MessageHeaders.build(
            message=trade, data_type="trades", content_type="application/x-protobuf"
        )

        header_dict = dict(headers)
        assert header_dict[b"symbol"] == b"btc-usd"

    def test_symbol_header_normalization(self):
        """Test symbol header normalization (underscores to hyphens)."""
        trade = MockTrade(symbol="BTC_USD")
        headers = MessageHeaders.build(
            message=trade, data_type="trades", content_type="application/x-protobuf"
        )

        header_dict = dict(headers)
        # Symbol normalization: underscores to hyphens
        assert header_dict[b"symbol"] == b"btc-usd"

    def test_data_type_header(self):
        """Test data_type header is set correctly."""
        trade = MockTrade()
        headers = MessageHeaders.build(
            message=trade, data_type="trades", content_type="application/x-protobuf"
        )

        header_dict = dict(headers)
        assert header_dict[b"data_type"] == b"trades"

    def test_data_type_header_various_types(self):
        """Test data_type header with various data types."""
        trade = MockTrade()

        for data_type in ["trades", "orderbook", "ticker", "candle", "funding"]:
            headers = MessageHeaders.build(
                message=trade,
                data_type=data_type,
                content_type="application/x-protobuf",
            )

            header_dict = dict(headers)
            assert header_dict[b"data_type"] == data_type.encode()

    def test_headers_are_bytes(self):
        """Test that all header values are bytes."""
        trade = MockTrade()
        headers = MessageHeaders.build(
            message=trade, data_type="trades", content_type="application/x-protobuf"
        )

        # headers should be list of (bytes, bytes) tuples
        assert isinstance(headers, list)
        for header_name, header_value in headers:
            assert isinstance(header_name, bytes), (
                f"Header name {header_name} is not bytes"
            )
            assert isinstance(header_value, bytes), (
                f"Header value {header_value} is not bytes"
            )

    def test_headers_encoding_utf8(self):
        """Test that headers are UTF-8 encoded."""
        trade = MockTrade(exchange="binance", symbol="BTC-USDT")
        headers = MessageHeaders.build(
            message=trade, data_type="trades", content_type="application/x-protobuf"
        )

        header_dict = dict(headers)

        # Verify encoding by round-trip
        exchange_str = header_dict[b"exchange"].decode("utf-8")
        assert exchange_str == "binance"

        symbol_str = header_dict[b"symbol"].decode("utf-8")
        assert symbol_str == "btc-usdt"

    def test_headers_are_list_of_tuples(self):
        """Test that headers are returned as list of tuples."""
        trade = MockTrade()
        headers = MessageHeaders.build(
            message=trade, data_type="trades", content_type="application/x-protobuf"
        )

        assert isinstance(headers, list)
        assert len(headers) >= 4  # At least 4 mandatory headers
        assert all(isinstance(h, tuple) and len(h) == 2 for h in headers)

    def test_empty_exchange_fallback(self):
        """Test handling of messages without exchange attribute."""
        message = MockMessageWithoutMetadata()
        headers = MessageHeaders.build(
            message=message, data_type="trades", content_type="application/x-protobuf"
        )

        header_dict = dict(headers)
        # Should default to "unknown" or empty
        assert b"exchange" in header_dict

    def test_empty_symbol_fallback(self):
        """Test handling of messages without symbol attribute."""
        message = MockMessageWithoutMetadata()
        headers = MessageHeaders.build(
            message=message, data_type="trades", content_type="application/x-protobuf"
        )

        header_dict = dict(headers)
        # Should default to "unknown" or empty
        assert b"symbol" in header_dict

    def test_special_characters_in_exchange(self):
        """Test handling of special characters in exchange name."""
        trade = MockTrade(exchange="kraken-us")
        headers = MessageHeaders.build(
            message=trade, data_type="trades", content_type="application/x-protobuf"
        )

        header_dict = dict(headers)
        assert header_dict[b"exchange"] == b"kraken-us"

    def test_special_characters_in_symbol(self):
        """Test handling of special characters in symbol."""
        trade = MockTrade(symbol="BTC/USD")
        headers = MessageHeaders.build(
            message=trade, data_type="trades", content_type="application/x-protobuf"
        )

        header_dict = dict(headers)
        # Normalization replaces separators with hyphens and lowercases
        assert header_dict[b"symbol"] == b"btc-usd"

    def test_mandatory_headers_consistency(self):
        """Test that same message always produces same headers."""
        trade1 = MockTrade(exchange="coinbase", symbol="BTC-USD")
        trade2 = MockTrade(exchange="coinbase", symbol="BTC-USD")

        headers1 = MessageHeaders.build(
            message=trade1, data_type="trades", content_type="application/x-protobuf"
        )
        headers2 = MessageHeaders.build(
            message=trade2, data_type="trades", content_type="application/x-protobuf"
        )

        # Should produce identical headers
        dict1 = dict(headers1)
        dict2 = dict(headers2)
        assert dict1 == dict2

    def test_mandatory_headers_with_different_exchanges(self):
        """Test mandatory headers with various exchanges."""
        exchanges = ["coinbase", "binance", "kraken", "bybit", "dydx"]

        for exchange in exchanges:
            trade = MockTrade(exchange=exchange)
            headers = MessageHeaders.build(
                message=trade, data_type="trades", content_type="application/x-protobuf"
            )

            header_dict = dict(headers)
            assert header_dict[b"exchange"] == exchange.lower().encode()


# ============================================================================
# OptionalHeaders Tests
# ============================================================================


class TestOptionalHeadersClass:
    """Test OptionalHeaders class for building optional headers."""

    def test_schema_version_header_default(self):
        """Test schema_version header has default value."""
        headers = OptionalHeaders.build()
        header_dict = dict(headers)

        assert b"schema_version" in header_dict
        assert header_dict[b"schema_version"] == b"v1"

    def test_schema_version_header_custom(self):
        """Test schema_version header can be customized."""
        headers = OptionalHeaders.build(schema_version="v2")
        header_dict = dict(headers)

        assert header_dict[b"schema_version"] == b"v2"

    def test_producer_version_header_present(self):
        """Test producer_version header is present."""
        headers = OptionalHeaders.build()
        header_dict = dict(headers)

        assert b"producer_version" in header_dict
        # Should be non-empty
        assert header_dict[b"producer_version"]

    def test_producer_version_header_format(self):
        """Test producer_version header is valid semantic version."""
        headers = OptionalHeaders.build()
        header_dict = dict(headers)

        version_str = header_dict[b"producer_version"].decode("utf-8")
        # Should match semantic versioning: major.minor.patch
        assert re.match(r"^\d+\.\d+\.\d+", version_str)

    def test_producer_version_header_custom(self):
        """Test producer_version can be customized."""
        headers = OptionalHeaders.build(producer_version="1.2.3")
        header_dict = dict(headers)

        assert header_dict[b"producer_version"] == b"1.2.3"

    def test_timestamp_generated_header_present(self):
        """Test timestamp_generated header is present."""
        headers = OptionalHeaders.build()
        header_dict = dict(headers)

        assert b"timestamp_generated" in header_dict
        assert header_dict[b"timestamp_generated"]

    def test_timestamp_generated_header_iso8601_format(self):
        """Test timestamp_generated is ISO8601 format."""
        headers = OptionalHeaders.build()
        header_dict = dict(headers)

        timestamp_str = header_dict[b"timestamp_generated"].decode("utf-8")

        # Should be parseable as ISO8601
        try:
            datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        except ValueError:
            pytest.fail(f"Timestamp {timestamp_str} is not valid ISO8601")

    def test_timestamp_generated_header_custom(self):
        """Test timestamp_generated can be customized."""
        custom_time = "2025-11-09T12:00:00Z"
        headers = OptionalHeaders.build(timestamp_generated=custom_time)
        header_dict = dict(headers)

        assert header_dict[b"timestamp_generated"] == custom_time.encode()

    def test_optional_headers_are_bytes(self):
        """Test that all optional header values are bytes."""
        headers = OptionalHeaders.build()

        assert isinstance(headers, list)
        for header_name, header_value in headers:
            assert isinstance(header_name, bytes), (
                f"Header name {header_name} is not bytes"
            )
            assert isinstance(header_value, bytes), (
                f"Header value {header_value} is not bytes"
            )

    def test_optional_headers_encoding_utf8(self):
        """Test that optional headers are UTF-8 encoded."""
        headers = OptionalHeaders.build(schema_version="v2", producer_version="2.0.0")
        header_dict = dict(headers)

        # Verify encoding by round-trip
        version_str = header_dict[b"producer_version"].decode("utf-8")
        assert version_str == "2.0.0"

    def test_optional_headers_are_list_of_tuples(self):
        """Test that optional headers are returned as list of tuples."""
        headers = OptionalHeaders.build()

        assert isinstance(headers, list)
        assert len(headers) >= 3  # At least 3 optional headers
        assert all(isinstance(h, tuple) and len(h) == 2 for h in headers)

    def test_optional_headers_consistency(self):
        """Test that same parameters produce same headers (except timestamp)."""
        headers1 = OptionalHeaders.build(schema_version="v1", producer_version="2.4.1")
        headers2 = OptionalHeaders.build(schema_version="v1", producer_version="2.4.1")

        # Extract without timestamp for comparison
        dict1 = {k: v for k, v in headers1 if k != b"timestamp_generated"}
        dict2 = {k: v for k, v in headers2 if k != b"timestamp_generated"}

        # Schema version and producer version should be identical
        assert dict1[b"schema_version"] == dict2[b"schema_version"]
        assert dict1[b"producer_version"] == dict2[b"producer_version"]

    def test_optional_headers_all_present(self):
        """Test that all 3 optional headers are present."""
        headers = OptionalHeaders.build()
        header_dict = dict(headers)

        assert b"schema_version" in header_dict
        assert b"producer_version" in header_dict
        assert b"timestamp_generated" in header_dict

    def test_schema_version_various_values(self):
        """Test schema_version with various values."""
        for version in ["v1", "v2", "v3", "1.0", "2.0"]:
            headers = OptionalHeaders.build(schema_version=version)
            header_dict = dict(headers)

            assert header_dict[b"schema_version"] == version.encode()


# ============================================================================
# HeaderEnricher (Full Pipeline) Tests
# ============================================================================


class TestHeaderEnricherClass:
    """Test HeaderEnricher class for complete enrichment pipeline."""

    def test_enricher_returns_headers_list(self):
        """Test that enricher returns list of header tuples."""
        trade = MockTrade()
        enricher = HeaderEnricher(content_type="application/x-protobuf")

        headers = enricher.build(message=trade, data_type="trades")

        assert isinstance(headers, list)
        assert all(isinstance(h, tuple) and len(h) == 2 for h in headers)

    def test_enricher_includes_mandatory_headers(self):
        """Test that enricher includes all mandatory headers."""
        trade = MockTrade()
        enricher = HeaderEnricher(content_type="application/x-protobuf")

        headers = enricher.build(message=trade, data_type="trades")
        header_dict = dict(headers)

        # Check mandatory headers
        assert b"content-type" in header_dict
        assert b"exchange" in header_dict
        assert b"symbol" in header_dict
        assert b"data_type" in header_dict

    def test_enricher_includes_optional_headers(self):
        """Test that enricher includes all optional headers."""
        trade = MockTrade()
        enricher = HeaderEnricher(content_type="application/x-protobuf")

        headers = enricher.build(message=trade, data_type="trades")
        header_dict = dict(headers)

        # Check optional headers
        assert b"schema_version" in header_dict
        assert b"producer_version" in header_dict
        assert b"timestamp_generated" in header_dict

    def test_enricher_total_header_count(self):
        """Test that enricher produces all expected headers."""
        trade = MockTrade()
        enricher = HeaderEnricher(content_type="application/x-protobuf")

        headers = enricher.build(message=trade, data_type="trades")
        header_dict = dict(headers)

        # Should have: 4 mandatory + 3 optional + 1 serialization header = 8 headers
        assert len(headers) == 8
        assert b"cf.serialization_format" in header_dict

    def test_enricher_with_protobuf_content_type(self):
        """Test enricher with protobuf content type."""
        trade = MockTrade()
        enricher = HeaderEnricher(content_type="application/x-protobuf")

        headers = enricher.build(message=trade, data_type="trades")
        header_dict = dict(headers)

        assert header_dict[b"content-type"] == b"application/x-protobuf"

    def test_enricher_with_json_content_type(self):
        """Test enricher with JSON content type."""
        trade = MockTrade()
        enricher = HeaderEnricher(content_type="application/json")

        headers = enricher.build(message=trade, data_type="trades")
        header_dict = dict(headers)

        assert header_dict[b"content-type"] == b"application/json"

    def test_enricher_with_trade_message(self):
        """Test enricher with Trade message."""
        trade = MockTrade(exchange="coinbase", symbol="BTC-USD")
        enricher = HeaderEnricher(content_type="application/x-protobuf")

        headers = enricher.build(message=trade, data_type="trades")
        header_dict = dict(headers)

        assert header_dict[b"exchange"] == b"coinbase"
        assert header_dict[b"symbol"] == b"btc-usd"
        assert header_dict[b"data_type"] == b"trades"

    def test_enricher_with_ticker_message(self):
        """Test enricher with Ticker message."""
        ticker = MockTicker(exchange="binance", symbol="ETH-USDT")
        enricher = HeaderEnricher(content_type="application/x-protobuf")

        headers = enricher.build(message=ticker, data_type="ticker")
        header_dict = dict(headers)

        assert header_dict[b"exchange"] == b"binance"
        assert header_dict[b"symbol"] == b"eth-usdt"
        assert header_dict[b"data_type"] == b"ticker"

    def test_enricher_with_orderbook_message(self):
        """Test enricher with OrderBook message."""
        orderbook = MockOrderBook(exchange="kraken", symbol="SOL-USD")
        enricher = HeaderEnricher(content_type="application/x-protobuf")

        headers = enricher.build(message=orderbook, data_type="orderbook")
        header_dict = dict(headers)

        assert header_dict[b"exchange"] == b"kraken"
        assert header_dict[b"symbol"] == b"sol-usd"
        assert header_dict[b"data_type"] == b"orderbook"

    def test_enricher_with_candle_message(self):
        """Test enricher with Candle message."""
        candle = MockCandle(exchange="bitmex", symbol="XBT-USD")
        enricher = HeaderEnricher(content_type="application/x-protobuf")

        headers = enricher.build(message=candle, data_type="candles")
        header_dict = dict(headers)

        assert header_dict[b"exchange"] == b"bitmex"
        assert header_dict[b"symbol"] == b"xbt-usd"
        assert header_dict[b"data_type"] == b"candles"

    def test_enricher_consistency_same_message(self):
        """Test that enricher produces consistent headers for same message."""
        trade1 = MockTrade(exchange="coinbase", symbol="BTC-USD")
        trade2 = MockTrade(exchange="coinbase", symbol="BTC-USD")

        enricher = HeaderEnricher(content_type="application/x-protobuf")

        headers1 = enricher.build(message=trade1, data_type="trades")
        headers2 = enricher.build(message=trade2, data_type="trades")

        # Convert to dicts and compare (excluding timestamp which may differ)
        dict1 = {k: v for k, v in headers1 if k != b"timestamp_generated"}
        dict2 = {k: v for k, v in headers2 if k != b"timestamp_generated"}

        assert dict1 == dict2

    def test_enricher_different_messages(self):
        """Test that enricher produces different headers for different messages."""
        trade = MockTrade(exchange="coinbase", symbol="BTC-USD")
        ticker = MockTicker(exchange="binance", symbol="ETH-USDT")

        enricher = HeaderEnricher(content_type="application/x-protobuf")

        headers_trade = enricher.build(message=trade, data_type="trades")
        headers_ticker = enricher.build(message=ticker, data_type="ticker")

        dict_trade = dict(headers_trade)
        dict_ticker = dict(headers_ticker)

        # Exchange should be different
        assert dict_trade[b"exchange"] != dict_ticker[b"exchange"]
        # Symbol should be different
        assert dict_trade[b"symbol"] != dict_ticker[b"symbol"]
        # Data type should be different
        assert dict_trade[b"data_type"] != dict_ticker[b"data_type"]

    def test_enricher_all_headers_are_bytes(self):
        """Test that all enricher headers are bytes."""
        trade = MockTrade()
        enricher = HeaderEnricher(content_type="application/x-protobuf")

        headers = enricher.build(message=trade, data_type="trades")

        for header_name, header_value in headers:
            assert isinstance(header_name, bytes)
            assert isinstance(header_value, bytes)

    def test_enricher_with_custom_schema_version(self):
        """Test enricher with custom schema version."""
        trade = MockTrade()
        enricher = HeaderEnricher(
            content_type="application/x-protobuf", schema_version="v2"
        )

        headers = enricher.build(message=trade, data_type="trades")
        header_dict = dict(headers)

        assert header_dict[b"schema_version"] == b"v2"

    def test_enricher_with_custom_producer_version(self):
        """Test enricher with custom producer version."""
        trade = MockTrade()
        enricher = HeaderEnricher(
            content_type="application/x-protobuf", producer_version="1.0.0"
        )

        headers = enricher.build(message=trade, data_type="trades")
        header_dict = dict(headers)

        assert header_dict[b"producer_version"] == b"1.0.0"

    def test_enricher_initialization_with_all_params(self):
        """Test enricher initialization with all parameters."""
        enricher = HeaderEnricher(
            content_type="application/x-protobuf",
            schema_version="v2",
            producer_version="3.0.0",
        )

        trade = MockTrade()
        headers = enricher.build(message=trade, data_type="trades")
        header_dict = dict(headers)

        assert header_dict[b"content-type"] == b"application/x-protobuf"
        assert header_dict[b"schema_version"] == b"v2"
        assert header_dict[b"producer_version"] == b"3.0.0"

    def test_enricher_handles_message_with_special_characters(self):
        """Test enricher handles special characters in exchange/symbol."""
        trade = MockTrade(exchange="kraken-us", symbol="BTC/USD")
        enricher = HeaderEnricher(content_type="application/x-protobuf")

        headers = enricher.build(message=trade, data_type="trades")
        header_dict = dict(headers)

        # Normalization preserves exchange hyphen and normalizes symbol separators
        assert header_dict[b"exchange"] == b"kraken-us"
        assert header_dict[b"symbol"] == b"btc-usd"

    def test_enricher_with_all_data_types(self):
        """Test enricher with all supported data types."""
        data_types = [
            "trades",
            "orderbook",
            "ticker",
            "candles",
            "funding",
            "liquidation",
            "index",
            "openinterest",
            "fills",
            "balances",
            "positions",
            "margin",
            "orders",
            "transactions",
        ]

        trade = MockTrade()
        enricher = HeaderEnricher(content_type="application/x-protobuf")

        for data_type in data_types:
            headers = enricher.build(message=trade, data_type=data_type)
            header_dict = dict(headers)

            assert header_dict[b"data_type"] == data_type.encode()

    def test_enricher_header_order_consistency(self):
        """Test that enricher produces headers in consistent order."""
        trade = MockTrade()
        enricher = HeaderEnricher(content_type="application/x-protobuf")

        headers1 = enricher.build(message=trade, data_type="trades")
        headers2 = enricher.build(message=trade, data_type="trades")

        # Extract header names
        names1 = [name for name, _ in headers1]
        names2 = [name for name, _ in headers2]

        # Order should be consistent (excluding timestamps)
        assert names1 == names2

    def test_enricher_no_duplicate_headers(self):
        """Test that enricher doesn't produce duplicate headers."""
        trade = MockTrade()
        enricher = HeaderEnricher(content_type="application/x-protobuf")

        headers = enricher.build(message=trade, data_type="trades")
        header_names = [name for name, _ in headers]

        # No duplicates
        assert len(header_names) == len(set(header_names))

    def test_enricher_iso8601_timestamp_format(self):
        """Test that timestamp_generated is proper ISO8601."""
        from datetime import timezone

        trade = MockTrade()
        enricher = HeaderEnricher(content_type="application/x-protobuf")

        headers = enricher.build(message=trade, data_type="trades")
        header_dict = dict(headers)

        timestamp_str = header_dict[b"timestamp_generated"].decode("utf-8")

        # Should be valid ISO8601
        try:
            parsed = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            # Should be recent (within last hour)
            now_utc = datetime.now(timezone.utc)
            assert (now_utc - parsed).total_seconds() < 3600
        except ValueError:
            pytest.fail(f"Invalid ISO8601 timestamp: {timestamp_str}")

    def test_enricher_header_values_not_empty(self):
        """Test that no header value is empty."""
        trade = MockTrade()
        enricher = HeaderEnricher(content_type="application/x-protobuf")

        headers = enricher.build(message=trade, data_type="trades")

        for header_name, header_value in headers:
            assert header_value, f"Empty value for header {header_name}"

    def test_enricher_symbol_normalization_consistency(self):
        """Test symbol normalization is consistent across multiple calls."""
        trade1 = MockTrade(symbol="BTC_USD")
        trade2 = MockTrade(symbol="BTC-USD")

        enricher = HeaderEnricher(content_type="application/x-protobuf")

        headers1 = enricher.build(message=trade1, data_type="trades")
        headers2 = enricher.build(message=trade2, data_type="trades")

        dict1 = dict(headers1)
        dict2 = dict(headers2)

        # Both should normalize to same value (underscores to hyphens)
        assert dict1[b"symbol"] == dict2[b"symbol"]

    def test_enricher_exchange_case_insensitivity(self):
        """Test exchange names are normalized to lowercase."""
        for exchange in ["Coinbase", "BINANCE", "Kraken", "bybit"]:
            trade = MockTrade(exchange=exchange)
            enricher = HeaderEnricher(content_type="application/x-protobuf")

            headers = enricher.build(message=trade, data_type="trades")
            header_dict = dict(headers)

            assert header_dict[b"exchange"] == exchange.lower().encode()


# ============================================================================
# Edge Cases and Integration Tests
# ============================================================================


class TestHeadersEdgeCases:
    """Test edge cases and special scenarios for headers."""

    def test_message_with_whitespace_in_exchange(self):
        """Test handling of whitespace in exchange name."""
        trade = MockTrade(exchange=" coinbase ")
        enricher = HeaderEnricher(content_type="application/x-protobuf")

        headers = enricher.build(message=trade, data_type="trades")
        header_dict = dict(headers)

        # Should strip and normalize
        assert header_dict[b"exchange"] in [b"coinbase", b" coinbase "]

    def test_message_with_whitespace_in_symbol(self):
        """Test handling of whitespace in symbol."""
        trade = MockTrade(symbol=" BTC-USD ")
        enricher = HeaderEnricher(content_type="application/x-protobuf")

        headers = enricher.build(message=trade, data_type="trades")
        header_dict = dict(headers)

        # Should handle whitespace
        assert b"symbol" in header_dict

    def test_very_long_symbol_name(self):
        """Test handling of very long symbol names."""
        long_symbol = "A" * 100 + "-" + "B" * 100
        trade = MockTrade(symbol=long_symbol)
        enricher = HeaderEnricher(content_type="application/x-protobuf")

        headers = enricher.build(message=trade, data_type="trades")
        header_dict = dict(headers)

        # Should encode without errors
        assert b"symbol" in header_dict
        assert len(header_dict[b"symbol"]) > 0

    def test_very_long_exchange_name(self):
        """Test handling of very long exchange names."""
        long_exchange = "exchange-" + "x" * 100
        trade = MockTrade(exchange=long_exchange)
        enricher = HeaderEnricher(content_type="application/x-protobuf")

        headers = enricher.build(message=trade, data_type="trades")
        header_dict = dict(headers)

        # Should encode without errors
        assert b"exchange" in header_dict
        assert len(header_dict[b"exchange"]) > 0

    def test_unicode_characters_in_exchange(self):
        """Test handling of unicode characters."""
        trade = MockTrade(exchange="交易所")  # Chinese characters for "exchange"
        enricher = HeaderEnricher(content_type="application/x-protobuf")

        headers = enricher.build(message=trade, data_type="trades")
        header_dict = dict(headers)

        # Should encode unicode properly
        assert header_dict[b"exchange"] == "交易所".encode("utf-8")

    def test_unicode_characters_in_symbol(self):
        """Test handling of unicode characters in symbol."""
        trade = MockTrade(symbol="BTC-元")  # Mixed ASCII and Chinese
        enricher = HeaderEnricher(content_type="application/x-protobuf")

        headers = enricher.build(message=trade, data_type="trades")
        header_dict = dict(headers)

        # Should encode unicode properly
        assert header_dict[b"symbol"] == "btc-元".encode("utf-8")

    def test_numeric_exchange_values(self):
        """Test handling of numeric exchange values."""
        # Create mock with numeric exchange
        trade = MockTrade()
        trade.exchange = 123  # Numeric instead of string

        enricher = HeaderEnricher(content_type="application/x-protobuf")

        # Should convert to string and encode
        headers = enricher.build(message=trade, data_type="trades")
        header_dict = dict(headers)

        assert b"exchange" in header_dict

    def test_numeric_symbol_values(self):
        """Test handling of numeric symbol values."""
        trade = MockTrade()
        trade.symbol = 456  # Numeric instead of string

        enricher = HeaderEnricher(content_type="application/x-protobuf")

        # Should convert to string and encode
        headers = enricher.build(message=trade, data_type="trades")
        header_dict = dict(headers)

        assert b"symbol" in header_dict

    def test_none_exchange_value(self):
        """Test handling of None exchange value."""
        trade = MockTrade(exchange=None)
        enricher = HeaderEnricher(content_type="application/x-protobuf")

        # Should handle gracefully
        headers = enricher.build(message=trade, data_type="trades")
        header_dict = dict(headers)

        assert b"exchange" in header_dict

    def test_none_symbol_value(self):
        """Test handling of None symbol value."""
        trade = MockTrade(symbol=None)
        enricher = HeaderEnricher(content_type="application/x-protobuf")

        # Should handle gracefully
        headers = enricher.build(message=trade, data_type="trades")
        header_dict = dict(headers)

        assert b"symbol" in header_dict

    def test_empty_string_exchange(self):
        """Test handling of empty string exchange."""
        trade = MockTrade(exchange="")
        enricher = HeaderEnricher(content_type="application/x-protobuf")

        headers = enricher.build(message=trade, data_type="trades")
        header_dict = dict(headers)

        assert b"exchange" in header_dict

    def test_empty_string_symbol(self):
        """Test handling of empty string symbol."""
        trade = MockTrade(symbol="")
        enricher = HeaderEnricher(content_type="application/x-protobuf")

        headers = enricher.build(message=trade, data_type="trades")
        header_dict = dict(headers)

        assert b"symbol" in header_dict

    def test_mixed_case_content_type(self):
        """Test that content-type is passed through correctly."""
        trade = MockTrade()
        enricher = HeaderEnricher(content_type="application/x-protobuf")

        headers = enricher.build(message=trade, data_type="trades")
        header_dict = dict(headers)

        # Should be lowercase header name
        assert b"content-type" in header_dict
        assert b"content-Type" not in header_dict


# ============================================================================
# Performance and Scalability Tests
# ============================================================================


class TestHeadersPerformance:
    """Test performance characteristics of header generation."""

    def test_header_generation_speed(self):
        """Test that header generation is fast (<1ms per message)."""
        import time

        trade = MockTrade()
        enricher = HeaderEnricher(content_type="application/x-protobuf")

        start = time.time()
        for _ in range(1000):
            enricher.build(message=trade, data_type="trades")
        elapsed = time.time() - start

        avg_time_ms = (elapsed / 1000) * 1000
        # Should be very fast - less than 1ms per header generation
        assert avg_time_ms < 1.0, f"Header generation too slow: {avg_time_ms}ms"

    def test_header_generation_memory_efficiency(self):
        """Test that header generation doesn't leak memory."""
        trade = MockTrade()
        enricher = HeaderEnricher(content_type="application/x-protobuf")

        # Generate many headers and verify no memory issues
        headers_list = []
        for _ in range(10000):
            headers = enricher.build(message=trade, data_type="trades")
            headers_list.append(headers)

        # All headers should be generated without error
        assert len(headers_list) == 10000

    def test_multiple_enricher_instances(self):
        """Test multiple enricher instances work independently."""
        enricher1 = HeaderEnricher(
            content_type="application/x-protobuf", schema_version="v1"
        )
        enricher2 = HeaderEnricher(content_type="application/json", schema_version="v2")

        trade = MockTrade()

        headers1 = enricher1.build(message=trade, data_type="trades")
        headers2 = enricher2.build(message=trade, data_type="trades")

        dict1 = dict(headers1)
        dict2 = dict(headers2)

        assert dict1[b"content-type"] == b"application/x-protobuf"
        assert dict2[b"content-type"] == b"application/json"
        assert dict1[b"schema_version"] == b"v1"
        assert dict2[b"schema_version"] == b"v2"
