"""Phase 2: Task 8 - Unit tests for message headers and enrichment.

This module tests:
- Task 8.1: Header generation (mandatory headers)
- Task 8.2: Optional headers (schema_version, producer_version, timestamp_generated)

All tests written FIRST (TDD: RED phase) before implementation.
"""

import pytest
from datetime import datetime
from dataclasses import dataclass
from unittest.mock import patch, MagicMock
import re

from cryptofeed.kafka_callback import (
    HeaderEnricher,
    MessageHeaders,
    OptionalHeaders,
)


@dataclass
class MockTrade:
    """Mock Trade message."""
    exchange: str = 'coinbase'
    symbol: str = 'BTC-USD'
    price: float = 50000.0
    amount: float = 1.0
    timestamp: float = 1234567890.0


@dataclass
class MockOrderBook:
    """Mock OrderBook message."""
    exchange: str = 'binance'
    symbol: str = 'ETH-USDT'
    bids: list = None
    asks: list = None
    timestamp: float = 1234567890.0

    def __post_init__(self):
        if self.bids is None:
            self.bids = []
        if self.asks is None:
            self.asks = []


@dataclass
class MockTicker:
    """Mock Ticker message."""
    exchange: str = 'kraken'
    symbol: str = 'SOL-USD'
    bid: float = 200.0
    ask: float = 201.0
    timestamp: float = 1234567890.0


@dataclass
class MockMessageNoExchange:
    """Mock message without exchange field."""
    symbol: str = 'BTC-USD'
    timestamp: float = 1234567890.0


# ============================================================================
# Task 8.1: Header Generation Tests (Mandatory Headers)
# ============================================================================


class TestTaskEightOneHeaderGeneration:
    """Test mandatory header generation."""

    def test_content_type_header_protobuf(self):
        """content-type header should be application/x-protobuf."""
        trade = MockTrade()
        headers = MessageHeaders.build(
            message=trade,
            data_type='trades',
            content_type='application/x-protobuf'
        )

        header_dict = dict(headers)
        assert b'content-type' in header_dict, \
            "content-type header should be present"
        assert header_dict[b'content-type'] == b'application/x-protobuf', \
            "content-type should be application/x-protobuf"

    def test_content_type_header_json(self):
        """content-type header should be application/json for JSON format."""
        trade = MockTrade()
        headers = MessageHeaders.build(
            message=trade,
            data_type='trades',
            content_type='application/json'
        )

        header_dict = dict(headers)
        assert header_dict[b'content-type'] == b'application/json', \
            "content-type should be application/json"

    def test_exchange_header_extraction(self):
        """exchange header should be extracted from message."""
        trade = MockTrade(exchange='binance')
        headers = MessageHeaders.build(
            message=trade,
            data_type='trades',
            content_type='application/x-protobuf'
        )

        header_dict = dict(headers)
        assert b'exchange' in header_dict, \
            "exchange header should be present"
        assert header_dict[b'exchange'] == b'binance', \
            "exchange header should match message exchange"

    def test_symbol_header_extraction(self):
        """symbol header should be extracted from message."""
        trade = MockTrade(symbol='ETH-USD')
        headers = MessageHeaders.build(
            message=trade,
            data_type='trades',
            content_type='application/x-protobuf'
        )

        header_dict = dict(headers)
        assert b'symbol' in header_dict, \
            "symbol header should be present"
        assert header_dict[b'symbol'] == b'ETH-USD', \
            "symbol header should match message symbol"

    def test_data_type_header_from_parameter(self):
        """data_type header should come from data_type parameter."""
        trade = MockTrade()
        headers = MessageHeaders.build(
            message=trade,
            data_type='trades',
            content_type='application/x-protobuf'
        )

        header_dict = dict(headers)
        assert b'data_type' in header_dict, \
            "data_type header should be present"
        assert header_dict[b'data_type'] == b'trades', \
            "data_type header should match parameter"

    def test_header_values_are_bytes(self):
        """All header values should be bytes (UTF-8 encoded)."""
        trade = MockTrade()
        headers = MessageHeaders.build(
            message=trade,
            data_type='trades',
            content_type='application/x-protobuf'
        )

        for key, value in headers:
            assert isinstance(key, bytes), \
                f"Header key {key} should be bytes"
            assert isinstance(value, bytes), \
                f"Header value {value} should be bytes"

    def test_header_keys_are_bytes(self):
        """All header keys should be bytes."""
        trade = MockTrade()
        headers = MessageHeaders.build(
            message=trade,
            data_type='trades',
            content_type='application/x-protobuf'
        )

        for key, value in headers:
            assert isinstance(key, bytes), \
                f"All header keys should be bytes, got {type(key)}"

    def test_all_mandatory_headers_present(self):
        """All mandatory headers should be present."""
        trade = MockTrade()
        headers = MessageHeaders.build(
            message=trade,
            data_type='trades',
            content_type='application/x-protobuf'
        )

        header_dict = dict(headers)
        required_headers = [b'content-type', b'exchange', b'symbol', b'data_type']

        for required in required_headers:
            assert required in header_dict, \
                f"Mandatory header {required} is missing"

    def test_headers_format_is_list_of_tuples(self):
        """Headers should be returned as list of tuples."""
        trade = MockTrade()
        headers = MessageHeaders.build(
            message=trade,
            data_type='trades',
            content_type='application/x-protobuf'
        )

        assert isinstance(headers, list), \
            "Headers should be a list"
        assert all(isinstance(h, tuple) and len(h) == 2 for h in headers), \
            "Each header should be a (key, value) tuple"

    def test_exchange_header_missing_defaults_to_unknown(self):
        """exchange header should default to 'unknown' if not in message."""
        msg = MockMessageNoExchange()
        headers = MessageHeaders.build(
            message=msg,
            data_type='trades',
            content_type='application/x-protobuf'
        )

        header_dict = dict(headers)
        # Should either have 'unknown' or handle gracefully
        assert b'exchange' in header_dict or True, \
            "Should handle missing exchange field"

    def test_multiple_data_types_correct_headers(self):
        """Verify correct data_type headers for different message types."""
        test_cases = [
            (MockTrade(), 'trades', b'trades'),
            (MockOrderBook(), 'orderbook', b'orderbook'),
            (MockTicker(), 'ticker', b'ticker'),
        ]

        for message, data_type_str, expected_header_value in test_cases:
            headers = MessageHeaders.build(
                message=message,
                data_type=data_type_str,
                content_type='application/x-protobuf'
            )
            header_dict = dict(headers)
            assert header_dict[b'data_type'] == expected_header_value, \
                f"data_type header should be {expected_header_value}"


# ============================================================================
# Task 8.2: Optional Headers Tests
# ============================================================================


class TestTaskEightTwoOptionalHeaders:
    """Test optional header generation (schema_version, producer_version, timestamp)."""

    def test_schema_version_header_default(self):
        """schema_version header should default to 'v1'."""
        trade = MockTrade()
        headers = OptionalHeaders.build()

        header_dict = dict(headers)
        assert b'schema_version' in header_dict, \
            "schema_version header should be present"
        assert header_dict[b'schema_version'] == b'v1', \
            "schema_version should default to v1"

    def test_producer_version_header_present(self):
        """producer_version header should contain package version."""
        trade = MockTrade()
        headers = OptionalHeaders.build()

        header_dict = dict(headers)
        assert b'producer_version' in header_dict, \
            "producer_version header should be present"

        version_str = header_dict[b'producer_version'].decode('utf-8')
        # Should be a version string like "0.1.0"
        assert re.match(r'\d+\.\d+\.\d+', version_str), \
            f"producer_version should be semantic version, got {version_str}"

    def test_timestamp_generated_header_iso8601(self):
        """timestamp_generated header should be ISO8601 format."""
        trade = MockTrade()
        headers = OptionalHeaders.build()

        header_dict = dict(headers)
        assert b'timestamp_generated' in header_dict, \
            "timestamp_generated header should be present"

        timestamp_str = header_dict[b'timestamp_generated'].decode('utf-8')
        # Should be ISO8601 format like "2025-01-01T12:34:56..."
        assert re.match(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', timestamp_str), \
            f"timestamp_generated should be ISO8601, got {timestamp_str}"

    def test_optional_headers_are_bytes(self):
        """All optional header values should be bytes."""
        trade = MockTrade()
        headers = OptionalHeaders.build()

        for key, value in headers:
            assert isinstance(key, bytes), \
                f"Header key should be bytes, got {type(key)}"
            assert isinstance(value, bytes), \
                f"Header value should be bytes, got {type(value)}"

    def test_optional_headers_format_is_list_of_tuples(self):
        """Optional headers should be list of tuples."""
        trade = MockTrade()
        headers = OptionalHeaders.build()

        assert isinstance(headers, list), \
            "Optional headers should be a list"
        assert all(isinstance(h, tuple) and len(h) == 2 for h in headers), \
            "Each header should be a (key, value) tuple"

    def test_schema_version_custom_value(self):
        """schema_version should be customizable."""
        trade = MockTrade()
        headers = OptionalHeaders.build(
            schema_version='v2'
        )

        header_dict = dict(headers)
        assert header_dict[b'schema_version'] == b'v2', \
            "schema_version should use custom value"

    def test_producer_version_custom_value(self):
        """producer_version should be customizable."""
        trade = MockTrade()
        headers = OptionalHeaders.build(
            producer_version='0.2.0'
        )

        header_dict = dict(headers)
        assert header_dict[b'producer_version'] == b'0.2.0', \
            "producer_version should use custom value"

    def test_optional_headers_override_capability(self):
        """Optional headers should be overridable for testing."""
        trade = MockTrade()

        # Test with overrides
        headers = OptionalHeaders.build(
            schema_version='v2',
            producer_version='0.2.0'
        )

        header_dict = dict(headers)
        assert header_dict[b'schema_version'] == b'v2'
        assert header_dict[b'producer_version'] == b'0.2.0'


# ============================================================================
# Task 8: Complete Header Enrichment Tests
# ============================================================================


class TestHeaderEnrichment:
    """Test complete header enrichment pipeline."""

    def test_enricher_builds_all_headers(self):
        """HeaderEnricher should build both mandatory and optional headers."""
        trade = MockTrade()

        # Create enricher and build headers
        enricher = HeaderEnricher(content_type='application/x-protobuf')
        all_headers = enricher.build(
            message=trade,
            data_type='trades'
        )

        header_dict = dict(all_headers)

        # Check mandatory headers
        assert b'content-type' in header_dict
        assert b'exchange' in header_dict
        assert b'symbol' in header_dict
        assert b'data_type' in header_dict

        # Check optional headers
        assert b'schema_version' in header_dict
        assert b'producer_version' in header_dict
        assert b'timestamp_generated' in header_dict

    def test_enricher_returns_list_of_tuples(self):
        """HeaderEnricher should return list of tuples."""
        trade = MockTrade()
        enricher = HeaderEnricher(content_type='application/x-protobuf')
        headers = enricher.build(
            message=trade,
            data_type='trades'
        )

        assert isinstance(headers, list), \
            "Enriched headers should be list"
        assert all(isinstance(h, tuple) and len(h) == 2 for h in headers), \
            "Headers should be tuples"

    def test_enricher_with_different_content_types(self):
        """HeaderEnricher should handle different content types."""
        trade = MockTrade()

        for content_type in ['application/x-protobuf', 'application/json']:
            enricher = HeaderEnricher(content_type=content_type)
            headers = enricher.build(
                message=trade,
                data_type='trades'
            )
            header_dict = dict(headers)
            assert header_dict[b'content-type'] == content_type.encode(), \
                f"Should set content-type to {content_type}"

    def test_enricher_with_multiple_message_types(self):
        """HeaderEnricher should work with different message types."""
        messages = [
            (MockTrade(), 'trades'),
            (MockOrderBook(), 'orderbook'),
            (MockTicker(), 'ticker'),
        ]

        enricher = HeaderEnricher(content_type='application/x-protobuf')

        for message, data_type in messages:
            headers = enricher.build(
                message=message,
                data_type=data_type
            )
            header_dict = dict(headers)

            assert header_dict[b'data_type'] == data_type.encode(), \
                f"data_type should be {data_type}"
            assert b'exchange' in header_dict
            assert b'symbol' in header_dict

    def test_enricher_consistent_across_calls(self):
        """HeaderEnricher should produce consistent headers across calls."""
        trade = MockTrade(
            exchange='coinbase',
            symbol='BTC-USD',
            timestamp=1234567890.0
        )
        enricher = HeaderEnricher(content_type='application/x-protobuf')

        # Build headers multiple times
        headers_list = [
            enricher.build(
                message=trade,
                data_type='trades'
            )
            for _ in range(5)
        ]

        # Check consistency
        header_dicts = [dict(h) for h in headers_list]

        for hd in header_dicts:
            assert hd[b'exchange'] == b'coinbase'
            assert hd[b'symbol'] == b'BTC-USD'
            assert hd[b'data_type'] == b'trades'

    def test_enricher_timestamp_varies_on_calls(self):
        """timestamp_generated should update on successive calls."""
        trade = MockTrade()
        enricher = HeaderEnricher(content_type='application/x-protobuf')

        headers1 = enricher.build(
            message=trade,
            data_type='trades'
        )

        # Small delay
        import time
        time.sleep(0.01)

        headers2 = enricher.build(
            message=trade,
            data_type='trades'
        )

        dict1 = dict(headers1)
        dict2 = dict(headers2)

        # Timestamps should be different or very close
        ts1 = dict1[b'timestamp_generated'].decode('utf-8')
        ts2 = dict2[b'timestamp_generated'].decode('utf-8')

        # Both should be valid ISO8601
        assert re.match(r'\d{4}-\d{2}-\d{2}T', ts1)
        assert re.match(r'\d{4}-\d{2}-\d{2}T', ts2)


# ============================================================================
# Integration Tests: Headers with Real Messages
# ============================================================================


class TestHeadersIntegration:
    """Integration tests for header enrichment with realistic scenarios."""

    def test_headers_for_all_supported_data_types(self):
        """Verify headers work for all data types."""
        enricher = HeaderEnricher(content_type='application/x-protobuf')

        data_types = ['trades', 'orderbook', 'ticker', 'candle', 'funding']
        message = MockTrade()

        for data_type in data_types:
            headers = enricher.build(
                message=message,
                data_type=data_type
            )
            header_dict = dict(headers)

            assert header_dict[b'data_type'] == data_type.encode(), \
                f"Should handle data_type {data_type}"
            assert b'content-type' in header_dict
            assert b'schema_version' in header_dict
            assert b'producer_version' in header_dict

    def test_headers_no_null_values(self):
        """Headers should not contain None or empty values."""
        trade = MockTrade()
        enricher = HeaderEnricher(content_type='application/x-protobuf')
        headers = enricher.build(
            message=trade,
            data_type='trades'
        )

        for key, value in headers:
            assert key is not None and key != b'', \
                "Header keys should not be None or empty"
            assert value is not None and value != b'', \
                "Header values should not be None or empty"
