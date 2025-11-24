"""Native Backpack feed integrating configuration, transports, and adapters."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from contextlib import suppress
from typing import Callable, List, Optional, Tuple

from cryptofeed.connection import AsyncConnection
from cryptofeed.defines import BACKPACK, L2_BOOK, TRADES, TICKER
from cryptofeed.feed import Feed
from cryptofeed.symbols import Symbol, Symbols
from cryptofeed.proxy import (
    ConnectionProxies,
    ProxySettings,
    get_proxy_injector,
    init_proxy_system,
)

from .adapters import (
    BackpackOrderBookAdapter,
    BackpackTickerAdapter,
    BackpackTradeAdapter,
)
from .auth import BackpackAuthHelper
from .config import BackpackConfig
from .health import BackpackHealthReport, evaluate_health
from .metrics import BackpackMetrics
from .rest import BackpackRestClient
from .router import (
    BackpackMessageRouter,
    BackpackRouterAdapters,
    BackpackRouterCallbacks,
)
from .symbols import BackpackSymbolService
from .ws import BackpackSubscription, BackpackWsSession, BackpackWsDependencies


LOG = logging.getLogger("feedhandler")


@dataclass(frozen=True)
class BackpackFeedDependencies:
    rest_client_factory: Callable[[BackpackConfig], BackpackRestClient]
    ws_session_factory: Callable[[BackpackConfig], BackpackWsSession]
    symbol_service: Optional[BackpackSymbolService] = None


class BackpackFeed(Feed):
    """Backpack exchange feed built on native cryptofeed abstractions."""

    id = BACKPACK
    rest_endpoints: List = []
    websocket_endpoints: List = []
    websocket_channels = {
        TRADES: "trades",
        L2_BOOK: "l2",
        TICKER: "ticker",
    }

    def __init__(
        self,
        *,
        config: Optional[object] = None,
        backpack_config: Optional[BackpackConfig] = None,
        dependencies: Optional[BackpackFeedDependencies] = None,
        max_depth: int = 0,
        **kwargs,
    ) -> None:
        handler_config = config
        if backpack_config is None and isinstance(handler_config, BackpackConfig):
            backpack_config = handler_config
            handler_config = None

        if backpack_config is None:
            backpack_config = BackpackConfig()

        self.exchange_config = backpack_config
        self.metrics = BackpackMetrics()

        self._apply_proxy_override()
        Symbols.set(self.id, {}, {})

        deps = dependencies or BackpackFeedDependencies(
            rest_client_factory=lambda cfg: BackpackRestClient(cfg),
            ws_session_factory=lambda cfg, metrics=self.metrics: BackpackWsSession(
                cfg,
                dependencies=BackpackWsDependencies(metrics=metrics),
            ),
        )

        self._rest_client_factory = deps.rest_client_factory
        self._ws_session_factory = deps.ws_session_factory
        self._rest_client = self._rest_client_factory(self.exchange_config)
        self._symbol_service = deps.symbol_service or BackpackSymbolService(
            rest_client=self._rest_client
        )

        depth = kwargs.pop("max_depth", max_depth)
        self._trade_adapter = BackpackTradeAdapter(exchange=self.id)
        self._order_book_adapter = BackpackOrderBookAdapter(
            exchange=self.id, max_depth=depth
        )
        self._ticker_adapter = BackpackTickerAdapter(exchange=self.id)

        self._router: Optional[BackpackMessageRouter] = None
        self._ws_session: Optional[BackpackWsSession] = None
        self._connection: Optional["BackpackWsConnection"] = None

        super().__init__(config=handler_config, max_depth=depth, **kwargs)

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
            adapters = BackpackRouterAdapters(
                trade=self._trade_adapter,
                order_book=self._order_book_adapter,
                ticker=self._ticker_adapter,
            )
            callbacks = BackpackRouterCallbacks(
                trade=self._callback(TRADES),
                order_book=self._callback(L2_BOOK),
                ticker=self._callback(TICKER),
            )
            self._router = BackpackMessageRouter(
                adapters=adapters,
                callbacks=callbacks,
                metrics=self.metrics,
            )

    def _callback(self, channel):
        callbacks = self.callbacks.get(channel)
        if not callbacks:
            return None

        async def handler(message, timestamp):
            for cb in callbacks:
                await cb(message, timestamp)

        return handler

    async def _ensure_symbol_metadata(self) -> None:
        await self._symbol_service.ensure()
        mapping = {
            market.normalized_symbol: market.native_symbol
            for market in self._symbol_service.all_markets()
        }
        if mapping:
            info = {
                "symbols": list(mapping.keys()),
            }
            Symbols.set(self.id, mapping, info)
            self.normalized_symbol_mapping = mapping
            self.exchange_symbol_mapping = {
                value: key for key, value in mapping.items()
            }

    def _build_ws_session(self) -> BackpackWsSession:
        auth_helper = (
            BackpackAuthHelper(self.exchange_config)
            if self.exchange_config.requires_auth
            else None
        )
        session = self._ws_session_factory(self.exchange_config)
        if auth_helper and getattr(session, "_auth_helper", None) is None:
            session._auth_helper = auth_helper
        return session

    async def subscribe(self, connection: AsyncConnection):
        await self._ensure_symbol_metadata()
        await self._initialize_router()

        if isinstance(connection, BackpackWsConnection):
            self._ws_session = connection.session

        if not self._ws_session:
            raise RuntimeError(
                "Backpack websocket session unavailable during subscribe"
            )

        subscriptions = []
        for std_channel, exchange_channel in self.websocket_channels.items():
            if exchange_channel not in self.subscription:
                continue
            symbols = [
                self.std_symbol_to_exchange_symbol(symbol)
                for symbol in self.subscription[exchange_channel]
            ]
            subscriptions.append(
                BackpackSubscription(
                    channel=exchange_channel,
                    symbols=symbols,
                    private=self.is_authenticated_channel(std_channel),
                )
            )

        if subscriptions:
            await self._ws_session.subscribe(subscriptions)
            LOG.info(
                "%s: subscribed to %s",
                self.id,
                ",".join(sub.channel for sub in subscriptions),
            )

    async def message_handler(self, msg: str, conn: AsyncConnection, timestamp: float):
        if self._router:
            await self._router.dispatch(msg)

    async def shutdown(self) -> None:
        if self._ws_session:
            await self._ws_session.close()
        await self._rest_client.close()
        await super().shutdown()

    def metrics_snapshot(self) -> dict:
        """Return current metrics snapshot."""
        return self.metrics.snapshot()

    def health(self, *, max_snapshot_age: float = 60.0) -> BackpackHealthReport:
        """Evaluate feed health based on current metrics."""
        return evaluate_health(self.metrics, max_snapshot_age=max_snapshot_age)

    def _apply_proxy_override(self) -> None:
        proxies = self.exchange_config.proxies
        if not proxies:
            return

        injector = get_proxy_injector()
        if injector is None:
            init_proxy_system(ProxySettings())
            injector = get_proxy_injector()

        if injector is None:
            return

        if not injector.settings.enabled:
            LOG.info("proxy: enabling proxy system for Backpack override")
            updated = injector.settings.model_copy(update={"enabled": True})
            init_proxy_system(updated)
            injector = get_proxy_injector()
            if injector is None or not injector.settings.enabled:
                LOG.warning(
                    "proxy: unable to enable proxy system for Backpack override"
                )
                return

        key = self.exchange_config.exchange_id.casefold()
        exchanges = dict(injector.settings.exchanges)
        new_entry = ConnectionProxies(http=proxies, websocket=proxies)
        if exchanges.get(key) != new_entry:
            self.metrics.record_proxy_rotation()
        exchanges[key] = new_entry
        injector.settings.exchanges = {k.casefold(): v for k, v in exchanges.items()}

    # ------------------------------------------------------------------
    # Override connect to use Backpack session
    # ------------------------------------------------------------------
    def connect(self) -> List[Tuple[AsyncConnection, callable, callable, callable]]:
        if not self._connection:
            self._connection = BackpackWsConnection(self)
        return [
            (self._connection, self.subscribe, self.message_handler, self.authenticate)
        ]


class BackpackWsConnection(AsyncConnection):
    def __init__(self, feed: BackpackFeed):
        super().__init__(f"{feed.id}.native")
        self.feed = feed
        self.session: Optional[BackpackWsSession] = None

    async def _open(self):
        if self.session is None:
            candidate = self.feed._build_ws_session()
            try:
                await candidate.open()
            except Exception:
                with suppress(Exception):
                    await candidate.close()
                self.feed._ws_session = None
                raise

            self.session = candidate
            self.feed._ws_session = candidate

    @property
    def is_open(self) -> bool:
        return self.session is not None and self.feed._ws_session is not None

    async def read(self):
        if self.session is None:
            await self._open()
        assert self.session is not None
        while True:
            message = await self.session.read()
            yield message

    async def write(self, msg: str):
        if self.session is None:
            await self._open()
        assert self.session is not None
        await self.session.send(json.loads(msg))

    async def close(self):
        if self.session:
            await self.session.close()
            self.session = None
            self.feed._ws_session = None
