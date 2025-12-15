"""
Unit tests for Trade and OrderBook type extensions with protobuf v2beta1 fields.

Tests verify new optional attributes (maker, event_time, match_id, liquidity_flag
for Trade; event_time, last_update_id for OrderBook) work correctly and maintain
backward compatibility.

Related spec: pr16-code-review-remediation
Task: 6.3
Requirements: REQ-1.1, REQ-1.2, REQ-1.16
"""
from decimal import Decimal

import pytest

from cryptofeed.types import OrderBook, Trade


class TestTradeSchemaFields:
    """Test Trade type extensions for v2beta1 protobuf schema fields."""

    def test_trade_maker_field_true(self):
        """Verify maker field accepts True (buyer is maker)."""
        trade = Trade(
            exchange="binance",
            symbol="BTC-USD",
            side="buy",
            amount=Decimal("1.0"),
            price=Decimal("50000"),
            timestamp=123.456,
            maker=True,
        )
        assert trade.maker is True

    def test_trade_maker_field_false(self):
        """Verify maker field accepts False (taker side)."""
        trade = Trade(
            exchange="binance",
            symbol="BTC-USD",
            side="sell",
            amount=Decimal("1.0"),
            price=Decimal("50000"),
            timestamp=123.456,
            maker=False,
        )
        assert trade.maker is False

    def test_trade_event_time_float(self):
        """Verify event_time field stores float timestamp."""
        trade = Trade(
            exchange="binance",
            symbol="BTC-USD",
            side="buy",
            amount=Decimal("1.0"),
            price=Decimal("50000"),
            timestamp=123.456,
            event_time=123.789,
        )
        assert trade.event_time == 123.789

    def test_trade_match_id_string(self):
        """Verify match_id field stores exchange match identifier."""
        trade = Trade(
            exchange="binance",
            symbol="BTC-USD",
            side="buy",
            amount=Decimal("1.0"),
            price=Decimal("50000"),
            timestamp=123.456,
            match_id="12345",
        )
        assert trade.match_id == "12345"

    def test_trade_liquidity_flag_string(self):
        """Verify liquidity_flag field stores maker/taker designation."""
        trade = Trade(
            exchange="binance",
            symbol="BTC-USD",
            side="buy",
            amount=Decimal("1.0"),
            price=Decimal("50000"),
            timestamp=123.456,
            liquidity_flag="maker",
        )
        assert trade.liquidity_flag == "maker"

    def test_trade_all_new_fields_populated(self):
        """Verify all new fields can be populated simultaneously."""
        trade = Trade(
            exchange="binance",
            symbol="BTC-USD",
            side="buy",
            amount=Decimal("1.0"),
            price=Decimal("50000"),
            timestamp=123.456,
            maker=True,
            event_time=123.789,
            match_id="67890",
            liquidity_flag="maker",
        )
        assert trade.maker is True
        assert trade.event_time == 123.789
        assert trade.match_id == "67890"
        assert trade.liquidity_flag == "maker"

    def test_trade_optional_fields_default_none(self):
        """Verify optional fields default to None when not provided (backward compatibility)."""
        trade = Trade(
            exchange="binance",
            symbol="BTC-USD",
            side="buy",
            amount=Decimal("1.0"),
            price=Decimal("50000"),
            timestamp=123.456,
        )
        assert trade.maker is None
        assert trade.event_time is None
        assert trade.match_id is None
        assert trade.liquidity_flag is None

    def test_trade_existing_fields_preserved(self):
        """Verify existing Trade fields continue to work (backward compatibility)."""
        trade = Trade(
            exchange="binance",
            symbol="BTC-USD",
            side="buy",
            amount=Decimal("1.0"),
            price=Decimal("50000"),
            timestamp=123.456,
            id="trade-123",
            type="limit",
        )
        assert trade.exchange == "binance"
        assert trade.symbol == "BTC-USD"
        assert trade.side == "buy"
        assert trade.amount == Decimal("1.0")
        assert trade.price == Decimal("50000")
        assert trade.timestamp == 123.456
        assert trade.id == "trade-123"
        assert trade.type == "limit"

    def test_trade_realistic_binance_data(self):
        """Verify new fields with realistic Binance WebSocket data values."""
        # Binance provides: m (maker), E (event time ms), a (aggregate trade ID)
        trade = Trade(
            exchange="binance",
            symbol="BTC-USDT",
            side="sell",  # m=true means buyer is maker, so this trade is sell
            amount=Decimal("0.5"),
            price=Decimal("50123.45"),
            timestamp=1234567890.123,
            maker=True,
            event_time=1234567890.456,
            match_id="987654321",
        )
        assert trade.maker is True
        assert trade.event_time == 1234567890.456
        assert trade.match_id == "987654321"


class TestOrderBookSchemaFields:
    """Test OrderBook type extensions for v2beta1 protobuf schema fields."""

    def test_orderbook_event_time_float(self):
        """Verify event_time field stores exchange event timestamp."""
        ob = OrderBook(exchange="binance", symbol="BTC-USD")
        ob.event_time = 123.456
        assert ob.event_time == 123.456

    def test_orderbook_last_update_id_int(self):
        """Verify last_update_id field stores final update ID."""
        ob = OrderBook(exchange="binance", symbol="BTC-USD")
        ob.last_update_id = 9876543210
        assert ob.last_update_id == 9876543210

    def test_orderbook_all_new_fields_populated(self):
        """Verify both new fields can be set simultaneously."""
        ob = OrderBook(exchange="binance", symbol="BTC-USD")
        ob.event_time = 123.789
        ob.last_update_id = 555555
        assert ob.event_time == 123.789
        assert ob.last_update_id == 555555

    def test_orderbook_optional_fields_default_none(self):
        """Verify new fields default to None (backward compatibility)."""
        ob = OrderBook(exchange="binance", symbol="BTC-USD")
        # New fields should be None by default (not yet set)
        assert not hasattr(ob, "event_time") or ob.event_time is None
        assert not hasattr(ob, "last_update_id") or ob.last_update_id is None

    def test_orderbook_existing_fields_preserved(self):
        """Verify existing OrderBook fields continue to work (backward compatibility)."""
        ob = OrderBook(
            exchange="binance",
            symbol="BTC-USD",
            bids={Decimal("50000"): Decimal("1.0")},
            asks={Decimal("50001"): Decimal("2.0")},
        )
        ob.timestamp = 123.456
        ob.sequence_number = 12345

        assert ob.exchange == "binance"
        assert ob.symbol == "BTC-USD"
        assert ob.timestamp == 123.456
        assert ob.sequence_number == 12345
        assert Decimal("50000") in ob.book.bids
        assert Decimal("50001") in ob.book.asks

    def test_orderbook_realistic_binance_data(self):
        """Verify new fields with realistic Binance order book update values."""
        # Binance provides: E (event time ms), u (final update ID)
        ob = OrderBook(exchange="binance", symbol="BTC-USDT")
        ob.event_time = 1234567890.789  # converted from milliseconds
        ob.last_update_id = 123456789012  # Binance 'u' field
        ob.timestamp = 1234567890.789

        assert ob.event_time == 1234567890.789
        assert ob.last_update_id == 123456789012
