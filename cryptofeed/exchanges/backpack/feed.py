"""Native Backpack feed integrating configuration, transports, and adapters."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Dict, List, Optional, Tuple

from cryptofeed.connection import AsyncConnection
from cryptofeed.defines import BACKPACK, CANDLES, L2_BOOK, ORDER_INFO, POSITIONS, TRADES, TICKER
from cryptofeed.feed import Feed
from cryptofeed.symbols import Symbol, Symbols
from cryptofeed.types import OrderBook

from .adapters import (
    BackpackCandleAdapter,
    BackpackOrderAdapter,
    BackpackOrderBookAdapter,
    BackpackPositionAdapter,
    BackpackTickerAdapter,
    BackpackTradeAdapter,
)
from .auth import BackpackAuthHelper
from collections.abc import Mapping

from .errors import BackpackOrderBookGap
from .config import BackpackConfig
from .health import BackpackHealthReport, evaluate_health
from .metrics import BackpackMetrics
from .rest import BackpackRestClient
from .router import BackpackMessageRouter
from .symbols import BackpackSymbolService
from .ws import BackpackSubscription, BackpackWsSession


LOG = logging.getLogger("feedhandler")


class BackpackFeed(Feed):
    """Backpack exchange feed built on native cryptofeed abstractions."""

    id = BACKPACK
    rest_endpoints: List = []
    websocket_endpoints: List = []
    websocket_channels = {
        TRADES: "trades",
        L2_BOOK: "l2",
        TICKER: "ticker",
        CANDLES: "candles",
        ORDER_INFO: "orders",
        POSITIONS: "positions",
    }

    def __init__(
        self,
        *,
        config: BackpackConfig | Mapping[str, object] | None = None,
        feature_flag_enabled: bool = True,
        rest_client_factory=None,
        ws_session_factory=None,
        symbol_service: Optional[BackpackSymbolService] = None,
        **kwargs,
    ) -> None:
        if not feature_flag_enabled:
            raise RuntimeError("Native Backpack feed is disabled. Enable the feature flag to opt-in.")

        self.config = BackpackConfig.coerce(config)
        if self.config.requires_auth and self.config.auth:
            self.key_id = self.config.auth.api_key
            self.key_secret = self.config.auth.private_key_b64
            self.requires_authentication = True
        Symbols.set(self.id, {}, {})
        self.metrics = BackpackMetrics()
        self._rest_client_factory = rest_client_factory or (lambda cfg: BackpackRestClient(cfg))
        self._ws_session_factory = ws_session_factory or (lambda cfg: BackpackWsSession(cfg, metrics=self.metrics))
        self._rest_client = self._rest_client_factory(self.config)
        self._symbol_service = symbol_service or BackpackSymbolService(rest_client=self._rest_client)
        self._trade_adapter = BackpackTradeAdapter(exchange=self.id)
        self._order_book_adapter = BackpackOrderBookAdapter(exchange=self.id, max_depth=kwargs.get("max_depth", 0))
        self._ticker_adapter = BackpackTickerAdapter(exchange=self.id)
        self._candle_adapter = BackpackCandleAdapter(exchange=self.id)
        self._order_adapter = BackpackOrderAdapter(exchange=self.id)
        self._position_adapter = BackpackPositionAdapter(exchange=self.id)
        self._router: Optional[BackpackMessageRouter] = None
        self._ws_session: Optional[BackpackWsSession] = None
        self._connection: Optional["BackpackWsConnection"] = None
        self._resync_locks: Dict[str, asyncio.Lock] = {}

        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # Symbol handling
    # ------------------------------------------------------------------
    def std_symbol_to_exchange_symbol(self, symbol):
        if isinstance(symbol, Symbol):
            normalized = symbol.normalized
        else:
            normalized = str(symbol)

        try:
            return self._symbol_service.native_symbol(normalized)
        except KeyError:
            return normalized.replace("-", "_")

    def exchange_symbol_to_std_symbol(self, symbol):
        if isinstance(symbol, Symbol):
            symbol = symbol.normalized
        return symbol.replace("_", "-")

    # ------------------------------------------------------------------
    # Feed lifecycle helpers
    # ------------------------------------------------------------------
    async def _initialize_router(self) -> None:
        if self._router is None:
            self._router = BackpackMessageRouter(
                trade_adapter=self._trade_adapter,
                order_book_adapter=self._order_book_adapter,
                ticker_adapter=self._ticker_adapter,
                 candle_adapter=self._candle_adapter,
                 order_adapter=self._order_adapter,
                 position_adapter=self._position_adapter,
                trade_callback=self._callback(TRADES),
                order_book_callback=self._callback(L2_BOOK),
                ticker_callback=self._callback(TICKER),
                 candle_callback=self._callback(CANDLES),
                 order_callback=self._callback(ORDER_INFO),
                 position_callback=self._callback(POSITIONS),
                metrics=self.metrics,
                resync_callback=self._resync_order_book,
            )

    def _callback(self, channel):
        callbacks = self.callbacks.get(channel)
        if not callbacks:
            return None

        async def handler(message, timestamp):
            for cb in callbacks:
                await cb(message, timestamp)

        return handler

    def _get_resync_lock(self, normalized_symbol: str) -> asyncio.Lock:
        lock = self._resync_locks.get(normalized_symbol)
        if lock is None:
            lock = asyncio.Lock()
            self._resync_locks[normalized_symbol] = lock
        return lock

    def _snapshot_depth(self) -> int:
        return self.max_depth if getattr(self, "max_depth", 0) else 50

    async def _ensure_symbol_metadata(self) -> None:
        await self._symbol_service.ensure()
        markets = list(self._symbol_service.all_markets())
        mapping = {market.normalized_symbol: market.native_symbol for market in markets}
        if mapping:
            info = {
                "symbols": list(mapping.keys()),
                "instrument_type": {market.normalized_symbol: market.instrument_type for market in markets},
                "price_precision": {market.normalized_symbol: market.price_precision for market in markets if market.price_precision is not None},
                "amount_precision": {market.normalized_symbol: market.amount_precision for market in markets if market.amount_precision is not None},
                "minimum_order_size": {market.normalized_symbol: str(market.min_amount) for market in markets if market.min_amount is not None},
            }
            Symbols.set(self.id, mapping, info)
            self.normalized_symbol_mapping = mapping
            self.exchange_symbol_mapping = {value: key for key, value in mapping.items()}

    def _build_ws_session(self) -> BackpackWsSession:
        auth_helper = BackpackAuthHelper(self.config) if self.config.requires_auth else None
        session = self._ws_session_factory(self.config)
        if auth_helper and getattr(session, "_auth_helper", None) is None:
            session._auth_helper = auth_helper
        return session

    async def _resync_order_book(
        self,
        normalized_symbol: str,
        gap: Optional[BackpackOrderBookGap],
    ) -> Optional[OrderBook]:
        lock = self._get_resync_lock(normalized_symbol)
        async with lock:
            try:
                await self._symbol_service.ensure()
                native_symbol = self._symbol_service.native_symbol(normalized_symbol)
            except KeyError:
                native_symbol = normalized_symbol.replace("-", "_")

            depth = self._snapshot_depth()
            try:
                snapshot = await self._rest_client.fetch_order_book(
                    native_symbol=native_symbol,
                    depth=depth,
                )
            except Exception as exc:  # pragma: no cover - network/runtime failure
                LOG.warning("Backpack snapshot fetch failed for %s: %s", normalized_symbol, exc)
                raise

            self._order_book_adapter.clear(normalized_symbol)
            book = self._order_book_adapter.apply_snapshot(
                normalized_symbol=normalized_symbol,
                bids=snapshot.bids,
                asks=snapshot.asks,
                timestamp=snapshot.timestamp_ms,
                sequence=snapshot.sequence,
                raw={
                    "type": "l2_snapshot",
                    "symbol": native_symbol,
                    "sequence": snapshot.sequence,
                    "timestamp": snapshot.timestamp_ms,
                    "bids": snapshot.bids,
                    "asks": snapshot.asks,
                    "resync": True,
                    "gap_actual": getattr(gap, "actual", None),
                },
            )

            if self.metrics:
                self.metrics.record_orderbook(
                    normalized_symbol,
                    getattr(book, "timestamp", None),
                    getattr(book, "sequence_number", None),
                )

            LOG.info(
                "%s: resynced order book for %s (sequence=%s)",
                self.id,
                normalized_symbol,
                getattr(book, "sequence_number", None),
            )
            return book

    async def _bootstrap_order_books(self, subscriptions: List[BackpackSubscription]) -> None:
        if not subscriptions:
            return

        for sub in subscriptions:
            if sub.channel not in {"l2", "orderbook"}:
                continue
            for native_symbol in sub.symbols:
                normalized = self.exchange_symbol_to_std_symbol(native_symbol)
                if self._order_book_adapter.has_snapshot(normalized):
                    continue
                try:
                    await self._resync_order_book(normalized, None)
                except Exception as exc:  # pragma: no cover - defensive logging
                    LOG.warning(
                        "%s: failed to bootstrap snapshot for %s: %s",
                        self.id,
                        normalized,
                        exc,
                    )

    async def subscribe(self, connection: AsyncConnection):
        await self._ensure_symbol_metadata()
        await self._initialize_router()

        if isinstance(connection, BackpackWsConnection):
            self._ws_session = connection.session

        if not self._ws_session:
            raise RuntimeError("Backpack websocket session unavailable during subscribe")

        subscriptions = []
        for std_channel, exchange_channel in self.websocket_channels.items():
            if exchange_channel not in self.subscription:
                continue
            symbols = list(self.subscription[exchange_channel])
            subscriptions.append(
                BackpackSubscription(
                    channel=exchange_channel,
                    symbols=symbols,
                    private=self.is_authenticated_channel(std_channel),
                )
            )

        if subscriptions:
            await self._bootstrap_order_books(subscriptions)
            await self._ws_session.subscribe(subscriptions)
            LOG.info("%s: subscribed to %s", self.id, ",".join(sub.channel for sub in subscriptions))

    async def message_handler(self, msg: str, conn: AsyncConnection, timestamp: float):
        if self._router:
            await self._router.dispatch(msg)

    async def shutdown(self) -> None:
        if self._ws_session:
            await self._ws_session.close()
        await self._rest_client.close()

    def metrics_snapshot(self) -> dict:
        """Return current metrics snapshot."""
        return self.metrics.snapshot()

    def health(self, *, max_snapshot_age: float = 60.0) -> BackpackHealthReport:
        """Evaluate feed health based on current metrics."""
        return evaluate_health(self.metrics, max_snapshot_age=max_snapshot_age)

    # ------------------------------------------------------------------
    # Override connect to use Backpack session
    # ------------------------------------------------------------------
    def connect(self) -> List[Tuple[AsyncConnection, callable, callable]]:
        if not self._connection:
            self._connection = BackpackWsConnection(self)
        return [(self._connection, self.subscribe, self.message_handler)]


class BackpackWsConnection(AsyncConnection):
    def __init__(self, feed: BackpackFeed):
        super().__init__(f"{feed.id}.native")
        self.feed = feed
        self.session: Optional[BackpackWsSession] = None

    async def _open(self):
        if self.session is None:
            self.session = self.feed._build_ws_session()
            await self.session.open()
            self.feed._ws_session = self.session

    @property
    def is_open(self) -> bool:
        return self.session is not None and self.feed._ws_session is not None

    async def read(self):
        if self.session is None:
            await self._open()
        while True:
            message = await self.session.read()
            yield message

    async def write(self, msg: str):
        if self.session is None:
            await self._open()
        await self.session.send(json.loads(msg))

    async def close(self):
        if self.session:
            await self.session.close()
            self.session = None
