from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest

from cryptofeed.exchanges.backpack.adapters import BackpackOrderBookAdapter, BackpackTickerAdapter, BackpackTradeAdapter
from cryptofeed.exchanges.backpack.router import (
    BackpackMessageRouter,
    BackpackRouterAdapters,
    BackpackRouterCallbacks,
)
from cryptofeed.exchanges.backpack.metrics import BackpackMetrics


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

    adapters = BackpackRouterAdapters(
        trade=trade_adapter,
        order_book=orderbook_adapter,
        ticker=BackpackTickerAdapter(exchange="BACKPACK"),
    )
    callbacks = BackpackRouterCallbacks(trade=collector)
    router = BackpackMessageRouter(adapters=adapters, callbacks=callbacks)

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

    adapters = BackpackRouterAdapters(
        trade=trade_adapter,
        order_book=orderbook_adapter,
        ticker=BackpackTickerAdapter(exchange="BACKPACK"),
    )
    callbacks = BackpackRouterCallbacks(order_book=collector)
    router = BackpackMessageRouter(adapters=adapters, callbacks=callbacks)

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

    adapters = BackpackRouterAdapters(trade=trade_adapter, order_book=orderbook_adapter, ticker=ticker_adapter)
    callbacks = BackpackRouterCallbacks(ticker=collector)
    router = BackpackMessageRouter(adapters=adapters, callbacks=callbacks)

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
async def test_router_drops_invalid_payload_and_records_metrics():
    metrics = BackpackMetrics()
    adapters = BackpackRouterAdapters(
        trade=BackpackTradeAdapter(exchange="BACKPACK"),
        order_book=BackpackOrderBookAdapter(exchange="BACKPACK"),
        ticker=None,
    )
    router = BackpackMessageRouter(adapters=adapters, callbacks=None, metrics=metrics)

    await router.dispatch({"type": "trade", "price": "100"})

    snapshot = metrics.snapshot()
    assert snapshot["parser_errors"] == 1
    assert snapshot["dropped_messages"] == 1
