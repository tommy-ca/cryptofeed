from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest

from cryptofeed.exchanges.backpack.adapters import (
    BackpackCandleAdapter,
    BackpackOrderAdapter,
    BackpackOrderBookAdapter,
    BackpackPositionAdapter,
    BackpackTickerAdapter,
    BackpackTradeAdapter,
)
from cryptofeed.exchanges.backpack.metrics import BackpackMetrics
from cryptofeed.exchanges.backpack.router import BackpackMessageRouter


class CallbackCollector:
    def __init__(self):
        self.items = []

    async def __call__(self, item, timestamp):
        self.items.append((item, timestamp))


@pytest.mark.asyncio
async def test_router_dispatches_trade():
    trade_adapter = BackpackTradeAdapter(exchange="BACKPACK")
    orderbook_adapter = BackpackOrderBookAdapter(exchange="BACKPACK")
    collector = CallbackCollector()

    router = BackpackMessageRouter(
        trade_adapter=trade_adapter,
        order_book_adapter=orderbook_adapter,
        ticker_adapter=BackpackTickerAdapter(exchange="BACKPACK"),
        trade_callback=collector,
        order_book_callback=None,
        ticker_callback=None,
        metrics=BackpackMetrics(),
    )

    await router.dispatch(
        {
            "type": "trade",
            "symbol": "BTC_USDT",
            "price": "30000",
            "size": "1",
            "side": "buy",
            "ts": 1_700_000_000_000,
        }
    )

    assert collector.items
    trade, timestamp = collector.items[0]
    assert trade.symbol == "BTC-USDT"
    assert timestamp == pytest.approx(1_700_000.0)


@pytest.mark.asyncio
async def test_router_dispatches_order_book_snapshot():
    trade_adapter = BackpackTradeAdapter(exchange="BACKPACK")
    orderbook_adapter = BackpackOrderBookAdapter(exchange="BACKPACK")
    collector = CallbackCollector()

    router = BackpackMessageRouter(
        trade_adapter=trade_adapter,
        order_book_adapter=orderbook_adapter,
        ticker_adapter=BackpackTickerAdapter(exchange="BACKPACK"),
        trade_callback=None,
        order_book_callback=collector,
        ticker_callback=None,
        metrics=BackpackMetrics(),
    )

    await router.dispatch(
        {
            "type": "l2_snapshot",
            "symbol": "BTC_USDT",
            "bids": [["30000", "1"]],
            "asks": [["30010", "2"]],
            "timestamp": 1_700_000_000_000,
            "sequence": 42,
        }
    )

    assert collector.items
    book, timestamp = collector.items[0]
    assert book.symbol == "BTC-USDT"
    assert book.sequence_number == 42
    assert timestamp == pytest.approx(1_700_000.0)


@pytest.mark.asyncio
async def test_router_dispatches_ticker():
    trade_adapter = BackpackTradeAdapter(exchange="BACKPACK")
    orderbook_adapter = BackpackOrderBookAdapter(exchange="BACKPACK")
    ticker_adapter = BackpackTickerAdapter(exchange="BACKPACK")
    collector = CallbackCollector()

    router = BackpackMessageRouter(
        trade_adapter=trade_adapter,
        order_book_adapter=orderbook_adapter,
        ticker_adapter=ticker_adapter,
        trade_callback=None,
        order_book_callback=None,
        ticker_callback=collector,
        metrics=BackpackMetrics(),
    )

    await router.dispatch(
        {
            "type": "ticker",
            "symbol": "BTC_USDT",
            "last": "30050",
            "bestBid": "30040",
            "bestAsk": "30060",
            "volume": "10",
            "timestamp": 1_700_000_000_500,
        }
    )

    assert collector.items
    ticker, timestamp = collector.items[0]
    assert ticker.symbol == "BTC-USDT"
    assert float(ticker.bid) == pytest.approx(30040)


@pytest.mark.asyncio
async def test_router_resyncs_on_gap():
    trade_adapter = BackpackTradeAdapter(exchange="BACKPACK")
    orderbook_adapter = BackpackOrderBookAdapter(exchange="BACKPACK")
    metrics = BackpackMetrics()
    collector = CallbackCollector()
    resync_calls = []

    async def resync(symbol, gap):
        resync_calls.append((symbol, gap.actual if gap else None))
        return orderbook_adapter.apply_snapshot(
            normalized_symbol=symbol,
            bids=[["30100", "1.0"]],
            asks=[["30110", "2.0"]],
            sequence=200,
            timestamp=1_700_000_002_000,
        )

    router = BackpackMessageRouter(
        trade_adapter=trade_adapter,
        order_book_adapter=orderbook_adapter,
        ticker_adapter=None,
        trade_callback=None,
        order_book_callback=collector,
        ticker_callback=None,
        metrics=metrics,
        resync_callback=resync,
    )

    await router.dispatch(
        {
            "type": "l2_snapshot",
            "symbol": "BTC_USDT",
            "bids": [["30000", "1"]],
            "asks": [["30010", "2"]],
            "timestamp": 1_700_000_000_000,
            "sequence": 10,
        }
    )

    await router.dispatch(
        {
            "type": "l2_update",
            "symbol": "BTC_USDT",
            "bids": [["29990", "1"]],
            "sequence": 25,
            "timestamp": 1_700_000_001_000,
        }
    )

    assert resync_calls == [("BTC-USDT", 25)]
    assert metrics.orderbook_resyncs == 1
    # Resync should emit a new callback entry
    assert len(collector.items) == 2
    resynced_book, ts = collector.items[-1]
    assert resynced_book.sequence_number == 200
    assert resynced_book.book.bids[Decimal("30100")] == Decimal("1.0")
    assert ts == pytest.approx(1_700_000.002)


@pytest.mark.asyncio
async def test_router_handles_enveloped_payloads():
    trade_adapter = BackpackTradeAdapter(exchange="BACKPACK")
    orderbook_adapter = BackpackOrderBookAdapter(exchange="BACKPACK")
    collector = CallbackCollector()

    router = BackpackMessageRouter(
        trade_adapter=trade_adapter,
        order_book_adapter=orderbook_adapter,
        ticker_adapter=None,
        trade_callback=collector,
        order_book_callback=None,
        ticker_callback=None,
        metrics=BackpackMetrics(),
    )

    await router.dispatch(
        {
            "channel": "trades",
            "symbol": "BTC_USDT",
            "data": {
                "price": "30010",
                "size": "0.1",
                "side": "buy",
                "ts": 1_700_000_003_000,
            },
        }
    )

    assert collector.items
    trade, timestamp = collector.items[0]
    assert trade.symbol == "BTC-USDT"
    assert trade.price == Decimal("30010")
    assert timestamp == pytest.approx(1_700_000.003)


@pytest.mark.asyncio
async def test_router_dispatches_candles_orders_positions():
    trade_adapter = BackpackTradeAdapter(exchange="BACKPACK")
    orderbook_adapter = BackpackOrderBookAdapter(exchange="BACKPACK")
    metrics = BackpackMetrics()
    candle_collector = CallbackCollector()
    order_collector = CallbackCollector()
    position_collector = CallbackCollector()

    router = BackpackMessageRouter(
        trade_adapter=trade_adapter,
        order_book_adapter=orderbook_adapter,
        ticker_adapter=None,
        candle_adapter=BackpackCandleAdapter(exchange="BACKPACK"),
        order_adapter=BackpackOrderAdapter(exchange="BACKPACK"),
        position_adapter=BackpackPositionAdapter(exchange="BACKPACK"),
        trade_callback=None,
        order_book_callback=None,
        ticker_callback=None,
        candle_callback=candle_collector,
        order_callback=order_collector,
        position_callback=position_collector,
        metrics=metrics,
    )

    await router.dispatch(
        {
            "channel": "candles.1m",
            "symbol": "BTC_USDT",
            "data": {
                "interval": "1m",
                "open": "30000",
                "close": "30010",
                "high": "30020",
                "low": "29990",
                "volume": "10",
                "startTime": 1_700_000_000_000,
                "endTime": 1_700_000_060_000,
                "timestamp": 1_700_000_060_000,
            },
        }
    )

    await router.dispatch(
        {
            "channel": "orders.update",
            "symbol": "BTC_USDT",
            "data": {
                "orderId": 1,
                "side": "buy",
                "status": "open",
                "type": "limit",
                "price": "29900",
                "size": "1",
                "timestamp": 1_700_000_100_000,
            },
        }
    )

    await router.dispatch(
        {
            "channel": "positions",
            "symbol": "BTC_USDT",
            "data": {
                "size": "0.5",
                "entryPrice": "29800",
                "side": "short",
                "timestamp": 1_700_000_200_000,
            },
        }
    )

    assert candle_collector.items
    assert order_collector.items
    assert position_collector.items
    candle, _ = candle_collector.items[0]
    assert candle.interval == "1m"
    order, _ = order_collector.items[0]
    assert order.price == Decimal("29900")
    position, _ = position_collector.items[0]
    assert position.position == Decimal("0.5")
    assert metrics.candle_updates == 1
    assert metrics.order_updates == 1
    assert metrics.position_updates == 1
