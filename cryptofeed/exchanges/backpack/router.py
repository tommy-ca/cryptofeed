"""Backpack message router for translating websocket frames into callbacks."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from cryptofeed.exchanges.native.router import NativeMessageRouter
from .adapters import OrderBookDelta, OrderBookSnapshot
from .metrics import BackpackMetrics


@dataclass(frozen=True)
class BackpackRouterAdapters:
    trade: Any
    order_book: Any
    ticker: Optional[Any] = None


@dataclass(frozen=True)
class BackpackRouterCallbacks:
    trade: Optional[Callable[[Any, float], Awaitable[None]]] = None
    order_book: Optional[Callable[[Any, float], Awaitable[None]]] = None
    ticker: Optional[Callable[[Any, float], Awaitable[None]]] = None

LOG = logging.getLogger("feedhandler")


class BackpackMessageRouter(NativeMessageRouter):
    """Dispatch Backpack websocket messages to registered adapters and callbacks."""

    def __init__(
        self,
        *,
        adapters: BackpackRouterAdapters,
        callbacks: Optional[BackpackRouterCallbacks] = None,
        metrics: Optional[BackpackMetrics] = None,
    ) -> None:
        super().__init__(metrics=metrics, logger=LOG)
        callbacks = callbacks or BackpackRouterCallbacks()
        self._trade_adapter = adapters.trade
        self._order_book_adapter = adapters.order_book
        self._ticker_adapter = adapters.ticker
        self._trade_callback = callbacks.trade
        self._order_book_callback = callbacks.order_book
        self._ticker_callback = callbacks.ticker
        self._metrics = metrics
        self.register_handlers(["trade", "trades"], self._handle_trade)
        self.register_handlers(
            ["l2", "orderbook", "l2_snapshot", "l2_update"],
            self._handle_order_book,
        )
        self.register_handler("ticker", self._handle_ticker)

    async def _handle_trade(self, payload: dict) -> None:
        symbol = payload.get("symbol") or payload.get("topic")
        if not symbol:
            self._drop_payload("trade payload missing symbol", payload)
            return
        normalized_symbol = symbol.replace("_", "-")
        try:
            trade = self._trade_adapter.parse(payload, normalized_symbol=normalized_symbol)
        except (ValueError, KeyError, TypeError) as exc:
            self._drop_payload(f"trade parse error: {exc}", payload)
            return
        timestamp = getattr(trade, "timestamp", None) or 0.0
        if self._metrics:
            self._metrics.record_trade(timestamp)
        if not self._trade_callback:
            return
        await self._trade_callback(trade, timestamp)

    async def _handle_order_book(self, payload: dict) -> None:
        symbol = payload.get("symbol")
        if not symbol:
            self._drop_payload("orderbook payload missing symbol", payload)
            return
        normalized_symbol = symbol.replace("_", "-")

        if self._is_snapshot(payload):
            book = self._handle_snapshot(normalized_symbol, payload)
        else:
            book = self._handle_delta(normalized_symbol, payload)

        if book is None:
            return

        timestamp = getattr(book, "timestamp", None) or 0.0
        self._record_orderbook_metrics(normalized_symbol, timestamp, getattr(book, "sequence_number", None))
        if not self._order_book_callback:
            return
        await self._order_book_callback(book, timestamp)

    def _is_snapshot(self, payload: dict) -> bool:
        return payload.get("snapshot", False) or payload.get("type") == "l2_snapshot"

    def _handle_snapshot(self, symbol: str, payload: dict) -> Optional[Any]:
        snapshot_payload = dict(payload)
        snapshot_payload.pop("symbol", None)
        snapshot = OrderBookSnapshot.from_payload(symbol=symbol, **snapshot_payload)
        try:
            return self._order_book_adapter.apply_snapshot(snapshot)
        except (ValueError, KeyError, TypeError) as exc:
            self._drop_payload(f"orderbook snapshot parse error: {exc}", payload)
            return None

    def _handle_delta(self, symbol: str, payload: dict) -> Optional[Any]:
        delta_payload = dict(payload)
        delta_payload.pop("symbol", None)
        delta = OrderBookDelta.from_payload(symbol=symbol, **delta_payload)
        try:
            return self._order_book_adapter.apply_delta(delta)
        except KeyError:
            self._drop_payload("order book delta received before snapshot", payload)
        except (ValueError, TypeError) as exc:
            self._drop_payload(f"orderbook delta parse error: {exc}", payload)
        return None

    def _record_orderbook_metrics(self, symbol: str, timestamp: float, sequence: Optional[int]) -> None:
        if not self._metrics:
            return
        recorded_timestamp = timestamp if timestamp else None
        self._metrics.record_orderbook(symbol, recorded_timestamp, sequence)

    async def _handle_ticker(self, payload: dict) -> None:
        if not self._ticker_adapter:
            return
        symbol = payload.get("symbol")
        if not symbol:
            self._drop_payload("ticker payload missing symbol", payload)
            return
        normalized_symbol = symbol.replace("_", "-")
        try:
            ticker = self._ticker_adapter.parse(payload, normalized_symbol=normalized_symbol)
        except (ValueError, KeyError, TypeError) as exc:
            self._drop_payload(f"ticker parse error: {exc}", payload)
            return
        timestamp = getattr(ticker, "timestamp", None) or 0.0
        if self._metrics:
            self._metrics.record_ticker(timestamp)
        if not self._ticker_callback:
            return
        await self._ticker_callback(ticker, timestamp)
