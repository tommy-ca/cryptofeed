from __future__ import annotations

from decimal import Decimal

import pytest

from cryptofeed.defines import ASK, BID, BUY, SELL
from cryptofeed.exchanges.backpack.adapters import (
    BackpackCandleAdapter,
    BackpackOrderAdapter,
    BackpackOrderBookAdapter,
    BackpackPositionAdapter,
    BackpackTickerAdapter,
    BackpackTradeAdapter,
)
from cryptofeed.exchanges.backpack.errors import (
    BackpackPayloadError,
    BackpackOrderBookGap,
    BackpackOrderBookMissingSnapshot,
)


def test_trade_adapter_parses_payload():
    adapter = BackpackTradeAdapter(exchange="BACKPACK")
    payload = {
        "p": "30000",
        "q": "0.5",
        "side": "buy",
        "t": "trade-id",
        "ts": 1_700_000_000_000,
    }

    trade = adapter.parse(payload, normalized_symbol="BTC-USDT")

    assert trade.exchange == "BACKPACK"
    assert trade.symbol == "BTC-USDT"
    assert trade.amount == Decimal("0.5")
    assert trade.price == Decimal("30000")
    assert trade.id == "trade-id"
    assert trade.timestamp == pytest.approx(1_700_000.0)
    assert trade.side == BUY


def test_order_book_adapter_snapshot_and_delta():
    adapter = BackpackOrderBookAdapter(exchange="BACKPACK")

    snapshot = adapter.apply_snapshot(
        normalized_symbol="BTC-USDT",
        bids=[["30000", "1"]],
        asks=[["30010", "2"]],
        timestamp=1_700_000_000_000,
        sequence=100,
        raw={"type": "snapshot"},
    )

    assert snapshot.sequence_number == 100
    assert snapshot.book.bids[Decimal("30000")] == Decimal("1")
    assert snapshot.book.asks[Decimal("30010")] == Decimal("2")

    delta = adapter.apply_delta(
        normalized_symbol="BTC-USDT",
        bids=[["30000", "0"], ["29990", "1.5"]],
        asks=None,
        timestamp=1_700_000_000_500,
        sequence=101,
        raw={"type": "delta"},
    )

    assert Decimal("30000") not in delta.book.bids
    assert delta.book.bids[Decimal("29990")] == Decimal("1.5")
    assert delta.sequence_number == 101
    assert delta.delta[BID][0][0] == Decimal("30000")
    assert delta.delta[ASK] == []


def test_ticker_adapter_parses_payload():
    adapter = BackpackTickerAdapter(exchange="BACKPACK")
    payload = {
        "symbol": "BTC_USDT",
        "last": "30055",
        "bestBid": "30050",
        "bestAsk": "30060",
        "volume": "15",
        "timestamp": 1_700_000_000_300,
    }

    ticker = adapter.parse(payload, normalized_symbol="BTC-USDT")
    assert ticker.bid == Decimal("30050")
    assert ticker.ask == Decimal("30060")


def test_candle_adapter_parses_payload():
    adapter = BackpackCandleAdapter(exchange="BACKPACK")
    payload = {
        "symbol": "BTC_USDT",
        "interval": "1m",
        "startTime": 1_700_000_000_000,
        "endTime": 1_700_000_060_000,
        "open": "30000",
        "close": "30010",
        "high": "30020",
        "low": "29990",
        "volume": "12.5",
        "trades": 42,
        "timestamp": 1_700_000_060_000,
        "closed": True,
    }

    candle = adapter.parse(payload, normalized_symbol="BTC-USDT")
    assert candle.interval == "1m"
    assert candle.open == Decimal("30000")
    assert candle.closed is True


def test_order_adapter_parses_payload():
    adapter = BackpackOrderAdapter(exchange="BACKPACK")
    payload = {
        "orderId": 123,
        "symbol": "BTC_USDT",
        "side": "sell",
        "status": "filled",
        "type": "limit",
        "price": "30100",
        "size": "2",
        "filledSize": "2",
        "clientOrderId": "abc",
        "timestamp": 1_700_000_000_500,
    }

    order = adapter.parse(payload, normalized_symbol="BTC-USDT")
    assert order.id == "123"
    assert order.side == SELL
    assert order.remaining == Decimal("0")
    assert order.timestamp == pytest.approx(1_700_000.0005)


def test_position_adapter_parses_payload():
    adapter = BackpackPositionAdapter(exchange="BACKPACK")
    payload = {
        "symbol": "BTC_USDT",
        "size": "1.5",
        "entryPrice": "29900",
        "side": "long",
        "unrealizedPnl": "125.5",
        "timestamp": 1_700_000_100_000,
    }

    position = adapter.parse(payload, normalized_symbol="BTC-USDT")
    assert position.position == Decimal("1.5")
    assert position.entry_price == Decimal("29900")
    assert position.side == "LONG"


def test_order_adapter_missing_id_raises():
    adapter = BackpackOrderAdapter(exchange="BACKPACK")
    with pytest.raises(BackpackPayloadError):
        adapter.parse({}, normalized_symbol="BTC-USDT")


def test_order_book_delta_gap_detection():
    adapter = BackpackOrderBookAdapter(exchange="BACKPACK")
    adapter.apply_snapshot(
        normalized_symbol="BTC-USDT",
        bids=[["30000", "1"]],
        asks=[["30010", "2"]],
        sequence=10,
    )

    with pytest.raises(BackpackOrderBookGap):
        adapter.apply_delta(
            normalized_symbol="BTC-USDT",
            bids=[["30020", "1"]],
            asks=None,
            timestamp=1_700_000_001_000,
            sequence=25,
            raw={"type": "delta"},
        )


def test_order_book_delta_without_snapshot_triggers_resync():
    adapter = BackpackOrderBookAdapter(exchange="BACKPACK")
    with pytest.raises(BackpackOrderBookMissingSnapshot):
        adapter.apply_delta(
            normalized_symbol="BTC-USDT",
            bids=[["30000", "1"]],
            asks=None,
            timestamp=1_700_000_001_000,
            sequence=1,
            raw={"type": "delta"},
        )


def test_trade_adapter_invalid_payload_raises():
    adapter = BackpackTradeAdapter(exchange="BACKPACK")
    with pytest.raises(BackpackPayloadError):
        adapter.parse({"side": "buy"}, normalized_symbol="BTC-USDT")
