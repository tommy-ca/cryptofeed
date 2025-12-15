"""Unit tests for protobuf converter v2beta1 field population."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import pytest

from cryptofeed.backends.protobuf.converters import trade_to_proto, orderbook_to_proto
from cryptofeed.backends.protobuf.bindings import trade_pb2, order_book_pb2


# Test dataclasses mimicking Trade and OrderBook with new fields
@dataclass
class Trade:
    """Trade object with v2beta1 optional fields."""
    exchange: str
    symbol: str
    side: str
    price: Decimal
    amount: Decimal
    timestamp: float
    id: str | None = None
    type: str | None = None
    maker: bool | None = None
    event_time: float | None = None
    match_id: str | None = None
    liquidity_flag: str | None = None


@dataclass
class OrderBook:
    """OrderBook object with v2beta1 optional fields."""
    exchange: str
    symbol: str
    bids: dict
    asks: dict
    timestamp: float
    sequence_number: int | None = None
    checksum: str | None = None
    event_time: float | None = None
    last_update_id: int | None = None


class TestTradeToProtoNewFields:
    """Test trade_to_proto() converter with v2beta1 field population."""

    def test_trade_with_all_new_fields_populated(self):
        """Verify all new fields are populated when present in Trade object."""
        trade = Trade(
            exchange="binance",
            symbol="BTC-USD",
            side="buy",
            price=Decimal("50000.00"),
            amount=Decimal("1.0"),
            timestamp=123.456,
            id="12345",
            type="spot",
            maker=True,
            event_time=123.789,
            match_id="67890",
            liquidity_flag="maker",
        )

        proto = trade_to_proto(trade)

        # Verify existing fields still work
        assert proto.exchange == "binance"
        assert proto.symbol == "BTC-USD"
        assert proto.price == "50000.00"
        assert proto.amount == "1.0"
        assert proto.timestamp == 123_456_000  # microseconds

        # Verify new fields are populated
        assert proto.HasField("maker")
        assert proto.maker is True

        assert proto.HasField("event_time")
        assert proto.event_time == 123_789_000  # microseconds (123.789 * 1_000_000)

        assert proto.HasField("match_id")
        assert proto.match_id == "67890"

        assert proto.HasField("liquidity_flag")
        assert proto.liquidity_flag == "maker"

    def test_trade_with_maker_false(self):
        """Verify maker field correctly handles False value (not just True)."""
        trade = Trade(
            exchange="binance",
            symbol="BTC-USD",
            side="sell",
            price=Decimal("50000.00"),
            amount=Decimal("1.0"),
            timestamp=123.456,
            maker=False,  # Explicitly False (taker side)
        )

        proto = trade_to_proto(trade)

        assert proto.HasField("maker")
        assert proto.maker is False

    def test_trade_with_no_new_fields(self):
        """Verify protobuf fields remain unset when Trade object doesn't have new attributes."""
        trade = Trade(
            exchange="binance",
            symbol="BTC-USD",
            side="buy",
            price=Decimal("50000.00"),
            amount=Decimal("1.0"),
            timestamp=123.456,
            # No new fields: maker, event_time, match_id, liquidity_flag all None
        )

        proto = trade_to_proto(trade)

        # Verify existing fields work
        assert proto.exchange == "binance"

        # Verify new fields are NOT set in protobuf message
        assert not proto.HasField("maker")
        assert not proto.HasField("event_time")
        assert not proto.HasField("match_id")
        assert not proto.HasField("liquidity_flag")

    def test_trade_with_none_new_fields(self):
        """Verify None values for new fields leave protobuf fields unset."""
        trade = Trade(
            exchange="binance",
            symbol="BTC-USD",
            side="buy",
            price=Decimal("50000.00"),
            amount=Decimal("1.0"),
            timestamp=123.456,
            maker=None,
            event_time=None,
            match_id=None,
            liquidity_flag=None,
        )

        proto = trade_to_proto(trade)

        # Fields should not be populated when values are None
        assert not proto.HasField("maker")
        assert not proto.HasField("event_time")
        assert not proto.HasField("match_id")
        assert not proto.HasField("liquidity_flag")

    def test_trade_event_time_conversion_to_microseconds(self):
        """Verify event_time is converted from seconds to microseconds."""
        trade = Trade(
            exchange="binance",
            symbol="BTC-USD",
            side="buy",
            price=Decimal("50000.00"),
            amount=Decimal("1.0"),
            timestamp=123.456,
            event_time=987.654321,  # seconds with microsecond precision
        )

        proto = trade_to_proto(trade)

        # Verify conversion to microseconds
        assert proto.event_time == 987_654_321  # 987.654321 * 1_000_000

    def test_trade_partial_new_fields_populated(self):
        """Verify converter handles partial field population correctly."""
        trade = Trade(
            exchange="binance",
            symbol="BTC-USD",
            side="buy",
            price=Decimal("50000.00"),
            amount=Decimal("1.0"),
            timestamp=123.456,
            maker=True,
            match_id="12345",
            # event_time and liquidity_flag are None
        )

        proto = trade_to_proto(trade)

        # Populated fields should be set
        assert proto.HasField("maker")
        assert proto.maker is True
        assert proto.HasField("match_id")
        assert proto.match_id == "12345"

        # Unpopulated fields should remain unset
        assert not proto.HasField("event_time")
        assert not proto.HasField("liquidity_flag")

    def test_trade_liquidity_flag_variations(self):
        """Verify liquidity_flag handles different string values."""
        for liquidity_value in ["maker", "taker", "MAKER", "TAKER", "unknown"]:
            trade = Trade(
                exchange="binance",
                symbol="BTC-USD",
                side="buy",
                price=Decimal("50000.00"),
                amount=Decimal("1.0"),
                timestamp=123.456,
                liquidity_flag=liquidity_value,
            )

            proto = trade_to_proto(trade)
            assert proto.liquidity_flag == liquidity_value


class TestOrderBookToProtoNewFields:
    """Test orderbook_to_proto() converter with v2beta1 field population."""

    def test_orderbook_with_all_new_fields_populated(self):
        """Verify all new fields are populated when present in OrderBook object."""
        orderbook = OrderBook(
            exchange="binance",
            symbol="BTC-USD",
            bids={Decimal("49999.00"): Decimal("1.0"), Decimal("49998.00"): Decimal("2.0")},
            asks={Decimal("50001.00"): Decimal("1.5"), Decimal("50002.00"): Decimal("2.5")},
            timestamp=123.456,
            sequence_number=100,
            checksum="abc123",
            event_time=123.789,
            last_update_id=200,
        )

        proto = orderbook_to_proto(orderbook)

        # Verify existing fields still work
        assert proto.exchange == "binance"
        assert proto.symbol == "BTC-USD"
        assert proto.timestamp == 123_456_000  # microseconds
        assert proto.sequence == 100
        assert proto.checksum == "abc123"

        # Verify new fields are populated
        assert proto.HasField("event_time")
        assert proto.event_time == 123_789_000  # microseconds (123.789 * 1_000_000)

        assert proto.HasField("last_update_id")
        assert proto.last_update_id == "200"  # String in protobuf schema

    def test_orderbook_with_no_new_fields(self):
        """Verify protobuf fields remain unset when OrderBook doesn't have new attributes."""
        orderbook = OrderBook(
            exchange="binance",
            symbol="BTC-USD",
            bids={Decimal("49999.00"): Decimal("1.0")},
            asks={Decimal("50001.00"): Decimal("1.5")},
            timestamp=123.456,
            # No new fields: event_time, last_update_id all None
        )

        proto = orderbook_to_proto(orderbook)

        # Verify existing fields work
        assert proto.exchange == "binance"

        # Verify new fields are NOT set in protobuf message
        assert not proto.HasField("event_time")
        assert not proto.HasField("last_update_id")

    def test_orderbook_with_none_new_fields(self):
        """Verify None values for new fields leave protobuf fields unset."""
        orderbook = OrderBook(
            exchange="binance",
            symbol="BTC-USD",
            bids={Decimal("49999.00"): Decimal("1.0")},
            asks={Decimal("50001.00"): Decimal("1.5")},
            timestamp=123.456,
            event_time=None,
            last_update_id=None,
        )

        proto = orderbook_to_proto(orderbook)

        # Fields should not be populated when values are None
        assert not proto.HasField("event_time")
        assert not proto.HasField("last_update_id")

    def test_orderbook_event_time_conversion_to_microseconds(self):
        """Verify event_time is converted from seconds to microseconds."""
        orderbook = OrderBook(
            exchange="binance",
            symbol="BTC-USD",
            bids={Decimal("49999.00"): Decimal("1.0")},
            asks={Decimal("50001.00"): Decimal("1.5")},
            timestamp=123.456,
            event_time=987.654321,  # seconds with microsecond precision
        )

        proto = orderbook_to_proto(orderbook)

        # Verify conversion to microseconds
        assert proto.event_time == 987_654_321  # 987.654321 * 1_000_000

    def test_orderbook_partial_new_fields_populated(self):
        """Verify converter handles partial field population correctly."""
        orderbook = OrderBook(
            exchange="binance",
            symbol="BTC-USD",
            bids={Decimal("49999.00"): Decimal("1.0")},
            asks={Decimal("50001.00"): Decimal("1.5")},
            timestamp=123.456,
            event_time=123.789,
            # last_update_id is None
        )

        proto = orderbook_to_proto(orderbook)

        # Populated fields should be set
        assert proto.HasField("event_time")
        assert proto.event_time == 123_789_000

        # Unpopulated fields should remain unset
        assert not proto.HasField("last_update_id")

    def test_orderbook_last_update_id_large_value(self):
        """Verify last_update_id handles large integer values correctly (as string)."""
        orderbook = OrderBook(
            exchange="binance",
            symbol="BTC-USD",
            bids={Decimal("49999.00"): Decimal("1.0")},
            asks={Decimal("50001.00"): Decimal("1.5")},
            timestamp=123.456,
            last_update_id=9_223_372_036_854_775_807,  # Max int64
        )

        proto = orderbook_to_proto(orderbook)

        # last_update_id is a string in protobuf schema
        assert proto.last_update_id == "9223372036854775807"


class TestBackwardCompatibility:
    """Verify backward compatibility with objects missing new fields."""

    def test_trade_without_new_attributes(self):
        """Verify converter works with Trade objects that don't have new attributes at all."""
        # Create a minimal Trade object (simulating old code)
        @dataclass
        class OldTrade:
            exchange: str
            symbol: str
            side: str
            price: Decimal
            amount: Decimal
            timestamp: float
            id: str | None = None
            type: str | None = None

        old_trade = OldTrade(
            exchange="binance",
            symbol="BTC-USD",
            side="buy",
            price=Decimal("50000.00"),
            amount=Decimal("1.0"),
            timestamp=123.456,
        )

        # Converter should handle missing attributes gracefully via hasattr() checks
        proto = trade_to_proto(old_trade)

        # Existing fields should work
        assert proto.exchange == "binance"
        assert proto.symbol == "BTC-USD"

        # New fields should not be set
        assert not proto.HasField("maker")
        assert not proto.HasField("event_time")
        assert not proto.HasField("match_id")
        assert not proto.HasField("liquidity_flag")

    def test_orderbook_without_new_attributes(self):
        """Verify converter works with OrderBook objects that don't have new attributes."""
        @dataclass
        class OldOrderBook:
            exchange: str
            symbol: str
            bids: dict
            asks: dict
            timestamp: float

        old_orderbook = OldOrderBook(
            exchange="binance",
            symbol="BTC-USD",
            bids={Decimal("49999.00"): Decimal("1.0")},
            asks={Decimal("50001.00"): Decimal("1.5")},
            timestamp=123.456,
        )

        # Converter should handle missing attributes gracefully
        proto = orderbook_to_proto(old_orderbook)

        # Existing fields should work
        assert proto.exchange == "binance"
        assert proto.symbol == "BTC-USD"

        # New fields should not be set
        assert not proto.HasField("event_time")
        assert not proto.HasField("last_update_id")
