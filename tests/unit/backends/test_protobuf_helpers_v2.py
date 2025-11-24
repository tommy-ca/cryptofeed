import math
from decimal import Decimal

import pytest

from cryptofeed.backends.protobuf_helpers_v2 import (
    candle_to_proto_v2,
    orderbook_to_proto_v2,
    serialize_to_protobuf_v2,
    ticker_to_proto_v2,
    trade_to_proto_v2,
)


class _Trade:
    def __init__(self):
        self.exchange = "coinbase"
        self.symbol = "btc-usd"
        self.side = "buy"
        self.id = "1234"
        self.price = Decimal("35000.1234")
        self.amount = Decimal("0.25")
        self.timestamp = 1700000000.1234567
        self.sequence_number = 42


class _Ticker:
    def __init__(self):
        self.exchange = "binance"
        self.symbol = "eth-usdt"
        self.bid = Decimal("2000.5")
        self.ask = Decimal("2000.6")
        self.bid_size = Decimal("10")
        self.ask_size = Decimal("11")
        self.timestamp = 1700000001.9
        self.sequence_number = 7


class _OrderBook:
    def __init__(self):
        self.exchange = "kraken"
        self.symbol = "ada-usd"
        self.bids = {Decimal("0.25"): Decimal("100")}
        self.asks = {Decimal("0.26"): Decimal("120")}
        self.timestamp = 1700000002.5
        self.sequence_number = 88
        self.checksum = "abc123"


class _Candle:
    def __init__(self):
        self.exchange = "okx"
        self.symbol = "sol-usdt"
        self.start = 1700000000.0
        self.end = 1700000060.0
        self.interval = "1m"
        self.trades = 120
        self.open = Decimal("54.1")
        self.close = Decimal("55.2")
        self.high = Decimal("55.5")
        self.low = Decimal("53.9")
        self.volume = Decimal("1000.123")
        self.closed = True
        self.timestamp = 1700000060.0
        self.sequence_number = 9


def _assert_ts(proto_ts, expected):
    seconds = int(math.floor(expected))
    nanos = int(round((expected - seconds) * 1_000_000_000))
    assert proto_ts.seconds == seconds
    assert proto_ts.nanos == nanos


def test_trade_conversion_to_proto_v2():
    trade = _Trade()

    proto = trade_to_proto_v2(trade)

    assert proto.exchange == "coinbase"
    assert proto.symbol == "btc-usd"
    assert proto.side == proto.SIDE_BUY
    assert proto.trade_id == "1234"
    assert proto.price == pytest.approx(35000.1234)
    assert proto.amount == pytest.approx(0.25)
    _assert_ts(proto.timestamp, trade.timestamp)
    assert proto.sequence_number == 42


def test_ticker_conversion_to_proto_v2():
    ticker = _Ticker()

    proto = ticker_to_proto_v2(ticker)

    assert proto.best_bid_price == pytest.approx(2000.5)
    assert proto.best_ask_price == pytest.approx(2000.6)
    assert proto.best_bid_size == pytest.approx(10.0)
    assert proto.best_ask_size == pytest.approx(11.0)
    _assert_ts(proto.timestamp, ticker.timestamp)
    assert proto.sequence_number == 7


def test_orderbook_conversion_to_proto_v2():
    book = _OrderBook()

    proto = orderbook_to_proto_v2(book)

    assert proto.exchange == "kraken"
    assert proto.symbol == "ada-usd"
    assert len(proto.bids) == 1
    assert len(proto.asks) == 1
    assert proto.bids[0].price == pytest.approx(0.25)
    assert proto.bids[0].quantity == pytest.approx(100.0)
    assert proto.asks[0].price == pytest.approx(0.26)
    assert proto.asks[0].quantity == pytest.approx(120.0)
    _assert_ts(proto.timestamp, book.timestamp)
    assert proto.sequence_number == 88
    assert proto.checksum == "abc123"


def test_candle_conversion_to_proto_v2():
    candle = _Candle()

    proto = candle_to_proto_v2(candle)

    assert proto.interval == "1m"
    assert proto.trades == 120
    assert proto.open == pytest.approx(54.1)
    assert proto.close == pytest.approx(55.2)
    assert proto.high == pytest.approx(55.5)
    assert proto.low == pytest.approx(53.9)
    assert proto.volume == pytest.approx(1000.123)
    assert proto.closed is True
    _assert_ts(proto.start, candle.start)
    _assert_ts(proto.end, candle.end)
    _assert_ts(proto.timestamp, candle.timestamp)
    assert proto.sequence_number == 9


def test_serialize_to_protobuf_v2_returns_bytes():
    trade = _Trade()
    encoded = serialize_to_protobuf_v2(trade)
    assert isinstance(encoded, (bytes, bytearray))
    assert len(encoded) > 0
