"""
Copyright (C) 2017-2025 Bryant Moscon - bmoscon@gmail.com

Please see the LICENSE file for the terms and conditions
associated with this software.

Tests for protobuf bindings import and usage.
"""

from google.protobuf.message import Message


def test_protobuf_bindings_importable():
    """Verify all protobuf bindings can be imported."""
    from cryptofeed.backends.protobuf.bindings import (
        trade_pb2,
        order_book_pb2,
        ticker_pb2,
        candle_pb2,
        funding_pb2,
    )

    # Verify message classes exist
    assert hasattr(trade_pb2, "Trade")
    assert hasattr(order_book_pb2, "Level2Book")
    assert hasattr(ticker_pb2, "Ticker")
    assert hasattr(candle_pb2, "Candle")
    assert hasattr(funding_pb2, "Funding")


def test_protobuf_trade_message_instantiation():
    """Verify protobuf Trade message can be instantiated."""
    from cryptofeed.backends.protobuf.bindings import trade_pb2

    trade = trade_pb2.Trade()
    trade.symbol = "BTC-USD"
    trade.price = "50000.12"
    trade.amount = "1.5"
    trade.exchange = "coinbase"
    trade.timestamp = 1700000000123000

    assert trade.symbol == "BTC-USD"
    assert trade.price == "50000.12"
    assert trade.amount == "1.5"
    assert isinstance(trade, Message)


def test_protobuf_orderbook_message_instantiation():
    """Verify protobuf Level2Book message can be instantiated."""
    from cryptofeed.backends.protobuf.bindings import order_book_pb2

    book = order_book_pb2.Level2Book()
    book.exchange = "binance"
    book.symbol = "BTC-USDT"
    book.timestamp = 1700000000123000

    # Add bid
    bid = book.bids.add()
    bid.price = "50000"
    bid.quantity = "1.5"

    # Add ask
    ask = book.asks.add()
    ask.price = "50001"
    ask.quantity = "2.0"

    assert book.exchange == "binance"
    assert len(book.bids) == 1
    assert len(book.asks) == 1
    assert book.bids[0].price == "50000"
    assert isinstance(book, Message)


def test_protobuf_serialization_roundtrip():
    """Verify protobuf messages can serialize/deserialize."""
    from cryptofeed.backends.protobuf.bindings import trade_pb2, trade_side_pb2

    # Create original message
    original = trade_pb2.Trade()
    original.symbol = "BTC-USD"
    original.price = "50000.12345678"
    original.amount = "1.5"
    original.exchange = "coinbase"
    original.timestamp = 1700000000123000
    original.side = trade_side_pb2.TRADE_SIDE_BUY

    # Serialize to bytes
    bytes_data = original.SerializeToString()
    assert isinstance(bytes_data, bytes)
    assert len(bytes_data) > 0

    # Deserialize from bytes
    restored = trade_pb2.Trade()
    restored.ParseFromString(bytes_data)

    # Verify all fields match
    assert restored.symbol == original.symbol
    assert restored.price == original.price
    assert restored.amount == original.amount
    assert restored.exchange == original.exchange
    assert restored.timestamp == original.timestamp
    assert restored.side == original.side


def test_protobuf_decimal_precision():
    """Verify protobuf preserves decimal precision as string."""
    from cryptofeed.backends.protobuf.bindings import trade_pb2

    trade = trade_pb2.Trade()
    # Store high-precision decimal as string
    trade.price = "123.456789012345678901234567890"

    # Serialize and deserialize
    bytes_data = trade.SerializeToString()
    restored = trade_pb2.Trade()
    restored.ParseFromString(bytes_data)

    # Verify full precision preserved
    assert restored.price == "123.456789012345678901234567890"


def test_protobuf_timestamp_microseconds():
    """Verify protobuf timestamps are int64 microseconds."""
    from cryptofeed.backends.protobuf.bindings import trade_pb2

    trade = trade_pb2.Trade()
    # Timestamp in microseconds
    trade.timestamp = 1700000000123456

    assert isinstance(trade.timestamp, int)
    assert trade.timestamp == 1700000000123456

    # Serialize and verify
    bytes_data = trade.SerializeToString()
    restored = trade_pb2.Trade()
    restored.ParseFromString(bytes_data)

    assert restored.timestamp == 1700000000123456


def test_protobuf_enum_side():
    """Verify TradeSide enum works correctly."""
    from cryptofeed.backends.protobuf.bindings import trade_pb2, trade_side_pb2

    trade = trade_pb2.Trade()
    trade.side = trade_side_pb2.TRADE_SIDE_BUY

    assert trade.side == trade_side_pb2.TRADE_SIDE_BUY
    assert trade.side != trade_side_pb2.TRADE_SIDE_SELL

    # Test SELL side
    trade.side = trade_side_pb2.TRADE_SIDE_SELL
    assert trade.side == trade_side_pb2.TRADE_SIDE_SELL
