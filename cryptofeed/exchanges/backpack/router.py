"""Backpack message router for translating websocket frames into callbacks."""
from __future__ import annotations

import json
import logging
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional

from cryptofeed.types import OrderBook

from .errors import (
    BackpackOrderBookGap,
    BackpackOrderBookMissingSnapshot,
    BackpackPayloadError,
    BackpackRouterError,
    BackpackUnknownChannel,
)
from .metrics import BackpackMetrics


LOG = logging.getLogger("feedhandler")


class BackpackMessageRouter:
    """Dispatch Backpack websocket messages to registered adapters and callbacks."""

    def __init__(
        self,
        *,
        trade_adapter,
        order_book_adapter,
        ticker_adapter=None,
        candle_adapter=None,
        order_adapter=None,
        position_adapter=None,
        trade_callback: Optional[Callable[[Any, float], Awaitable[None]]] = None,
        order_book_callback: Optional[Callable[[Any, float], Awaitable[None]]] = None,
        ticker_callback: Optional[Callable[[Any, float], Awaitable[None]]] = None,
        candle_callback: Optional[Callable[[Any, float], Awaitable[None]]] = None,
        order_callback: Optional[Callable[[Any, float], Awaitable[None]]] = None,
        position_callback: Optional[Callable[[Any, float], Awaitable[None]]] = None,
        metrics: Optional[BackpackMetrics] = None,
        resync_callback: Optional[
            Callable[[str, Optional[BackpackOrderBookGap]], Awaitable[Optional[OrderBook]]]
        ] = None,
    ) -> None:
        self._trade_adapter = trade_adapter
        self._order_book_adapter = order_book_adapter
        self._ticker_adapter = ticker_adapter
        self._candle_adapter = candle_adapter
        self._order_adapter = order_adapter
        self._position_adapter = position_adapter
        self._trade_callback = trade_callback
        self._order_book_callback = order_book_callback
        self._ticker_callback = ticker_callback
        self._candle_callback = candle_callback
        self._order_callback = order_callback
        self._position_callback = position_callback
        self._metrics = metrics
        self._resync_callback = resync_callback
        self._handlers: Dict[str, Callable[[dict], Awaitable[None]]] = {
            "trade": self._handle_trade,
            "trades": self._handle_trade,
            "l2": self._handle_order_book,
            "orderbook": self._handle_order_book,
            "l2_snapshot": self._handle_order_book,
            "l2_update": self._handle_order_book,
            "ticker": self._handle_ticker,
            "candles": self._handle_candle,
            "candle": self._handle_candle,
            "kline": self._handle_candle,
            "orders": self._handle_order,
            "order": self._handle_order,
            "positions": self._handle_position,
            "position": self._handle_position,
        }

    async def dispatch(self, message: str | dict | list) -> None:
        payloads = self._coerce_payloads(message)
        for payload in payloads:
            try:
                handler, channel = self._resolve_handler(payload)
            except BackpackUnknownChannel as exc:
                self._record_dropped(payload, str(exc))
                continue

            try:
                await handler(payload)
            except BackpackOrderBookGap as gap:
                await self._handle_order_book_gap(payload, gap)
            except BackpackOrderBookMissingSnapshot:
                await self._handle_missing_snapshot(payload)
            except BackpackPayloadError as exc:
                self._record_parser_error(payload, exc)
            except Exception as exc:  # pragma: no cover - defensive
                LOG.exception("Backpack router handler error for channel %s", channel)
                raise BackpackRouterError(str(exc)) from exc

    def register_handler(self, channel: str, handler: Callable[[dict], Awaitable[None]]) -> None:
        self._handlers[channel.lower()] = handler

    # ------------------------------------------------------------------
    # Payload coercion helpers
    # ------------------------------------------------------------------
    def _coerce_payloads(self, message: str | dict | list) -> List[dict]:
        decoded = self._decode(message)
        payloads: List[dict] = []
        self._collect_payloads(decoded, payloads)
        return payloads

    @staticmethod
    def _decode(message: str | dict | list) -> Any:
        if isinstance(message, str):
            return json.loads(message)
        return message

    def _collect_payloads(
        self,
        value: Any,
        collection: List[dict],
        channel_hint: Optional[str] = None,
        symbol_hint: Optional[str] = None,
    ) -> None:
        if isinstance(value, list):
            for item in value:
                self._collect_payloads(item, collection, channel_hint, symbol_hint)
            return

        if not isinstance(value, dict):
            return

        local_channel = value.get("channel") or value.get("topic") or value.get("type") or channel_hint
        local_symbol = value.get("symbol") or value.get("market") or symbol_hint
        data = value.get("data")

        if isinstance(data, list):
            for item in data:
                self._collect_payloads(item, collection, local_channel, local_symbol)
            return

        if isinstance(data, dict):
            merged = {**data}
            if local_channel and "channel" not in merged:
                merged["channel"] = local_channel
            if local_symbol and "symbol" not in merged:
                merged["symbol"] = local_symbol
            merged.pop("data", None)
            collection.append(merged)
            return

        payload = {**value}
        payload.pop("data", None)
        if local_channel and "channel" not in payload:
            payload["channel"] = local_channel
        if local_symbol and "symbol" not in payload:
            payload["symbol"] = local_symbol
        collection.append(payload)

    def _resolve_handler(self, payload: dict) -> tuple[Callable[[dict], Awaitable[None]], str]:
        channel_raw = payload.get("channel") or payload.get("type")
        if not channel_raw:
            raise BackpackUnknownChannel("Missing channel/type field")
        channel = str(channel_raw).lower()
        base_channel = channel.split(".", 1)[0]
        handler = self._handlers.get(base_channel)
        if not handler:
            raise BackpackUnknownChannel(f"Unsupported channel '{channel_raw}'")
        return handler, base_channel

    # ------------------------------------------------------------------
    # Handler implementations
    # ------------------------------------------------------------------
    async def _handle_trade(self, payload: dict) -> None:
        if not self._trade_callback:
            return
        symbol = self._normalize_symbol(payload.get("symbol") or payload.get("topic"))
        trade = self._trade_adapter.parse(payload, normalized_symbol=symbol)
        timestamp = getattr(trade, "timestamp", None) or 0.0
        if self._metrics:
            self._metrics.record_trade(timestamp)
        await self._trade_callback(trade, timestamp)

    async def _handle_order_book(self, payload: dict) -> None:
        if not self._order_book_callback:
            return
        symbol = self._normalize_symbol(payload.get("symbol"))
        is_snapshot = payload.get("snapshot") or payload.get("type") == "l2_snapshot"

        try:
            if is_snapshot:
                book = self._order_book_adapter.apply_snapshot(
                    normalized_symbol=symbol,
                    bids=payload.get("bids", []),
                    asks=payload.get("asks", []),
                    timestamp=payload.get("timestamp"),
                    sequence=payload.get("sequence"),
                    raw=payload,
                )
            else:
                book = self._order_book_adapter.apply_delta(
                    normalized_symbol=symbol,
                    bids=payload.get("bids"),
                    asks=payload.get("asks"),
                    timestamp=payload.get("timestamp"),
                    sequence=payload.get("sequence"),
                    raw=payload,
                )
        except BackpackOrderBookGap as gap:
            await self._handle_order_book_gap(payload, gap)
            return
        except BackpackOrderBookMissingSnapshot:
            await self._handle_missing_snapshot(payload)
            return

        timestamp = getattr(book, "timestamp", None) or 0.0
        if self._metrics:
            self._metrics.record_orderbook(
                symbol,
                timestamp if timestamp else None,
                getattr(book, "sequence_number", None),
            )
        await self._order_book_callback(book, timestamp)

    async def _handle_ticker(self, payload: dict) -> None:
        if not self._ticker_callback or not self._ticker_adapter:
            return
        symbol = self._normalize_symbol(payload.get("symbol"))
        ticker = self._ticker_adapter.parse(payload, normalized_symbol=symbol)
        timestamp = getattr(ticker, "timestamp", None) or 0.0
        if self._metrics:
            self._metrics.record_ticker(timestamp)
        await self._ticker_callback(ticker, timestamp)

    async def _handle_candle(self, payload: dict) -> None:
        if not self._candle_callback or not self._candle_adapter:
            return
        symbol = self._normalize_symbol(payload.get("symbol"))
        candle = self._candle_adapter.parse(payload, normalized_symbol=symbol)
        timestamp = getattr(candle, "timestamp", None) or 0.0
        if self._metrics:
            self._metrics.record_candle(timestamp)
        await self._candle_callback(candle, timestamp)

    async def _handle_order(self, payload: dict) -> None:
        if not self._order_callback or not self._order_adapter:
            return
        symbol = self._normalize_symbol(payload.get("symbol"))
        order = self._order_adapter.parse(payload, normalized_symbol=symbol)
        if self._metrics:
            self._metrics.record_order()
        await self._order_callback(order, getattr(order, "timestamp", None) or 0.0)

    async def _handle_position(self, payload: dict) -> None:
        if not self._position_callback or not self._position_adapter:
            return
        symbol = self._normalize_symbol(payload.get("symbol"))
        position = self._position_adapter.parse(payload, normalized_symbol=symbol)
        if self._metrics:
            self._metrics.record_position()
        await self._position_callback(position, getattr(position, "timestamp", None) or 0.0)

    # ------------------------------------------------------------------
    # Error handling helpers
    # ------------------------------------------------------------------
    async def _handle_order_book_gap(self, payload: dict, gap: BackpackOrderBookGap) -> None:
        symbol = self._normalize_symbol(payload.get("symbol"))
        LOG.warning("Backpack detected order book gap for %s: %s", symbol, gap)
        if self._metrics:
            self._metrics.record_dropped_message()

        if not self._resync_callback or not symbol:
            raise gap

        if self._metrics:
            self._metrics.record_orderbook_resync(symbol)

        try:
            book = await self._resync_callback(symbol, gap)
        except Exception as exc:  # pragma: no cover - defensive
            LOG.exception("Backpack order book resync failed for %s", symbol)
            raise BackpackRouterError(str(exc)) from exc

        if not book or not self._order_book_callback:
            return

        timestamp = getattr(book, "timestamp", None) or 0.0
        if self._metrics:
            self._metrics.record_orderbook(
                symbol,
                timestamp if timestamp else None,
                getattr(book, "sequence_number", None),
            )
        await self._order_book_callback(book, timestamp)

    async def _handle_missing_snapshot(self, payload: dict) -> None:
        symbol = self._normalize_symbol(payload.get("symbol"))
        LOG.warning("Backpack received delta before snapshot for %s", symbol)
        if self._metrics:
            self._metrics.record_dropped_message()

        if not self._resync_callback or not symbol:
            raise BackpackOrderBookMissingSnapshot(f"Snapshot missing for {symbol}")

        if self._metrics:
            self._metrics.record_orderbook_resync(symbol)

        book = await self._resync_callback(symbol, None)
        if not book or not self._order_book_callback:
            return

        timestamp = getattr(book, "timestamp", None) or 0.0
        if self._metrics:
            self._metrics.record_orderbook(
                symbol,
                timestamp if timestamp else None,
                getattr(book, "sequence_number", None),
            )
        await self._order_book_callback(book, timestamp)

    def _record_dropped(self, payload: dict, reason: str) -> None:
        if self._metrics:
            self._metrics.record_dropped_message()
        LOG.debug("Backpack router dropped payload %s: %s", payload, reason)

    def _record_parser_error(self, payload: dict, exc: Exception) -> None:
        if self._metrics:
            self._metrics.record_parser_error()
        LOG.warning("Backpack parser error: %s payload=%s", exc, payload)

    @staticmethod
    def _normalize_symbol(symbol: Optional[str]) -> Optional[str]:
        if symbol is None:
            return None
        return symbol.replace("_", "-")
