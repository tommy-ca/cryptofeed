"""Backpack native exchange integration scaffolding."""
from __future__ import annotations

from .config import BackpackConfig, BackpackAuthSettings
from .auth import BackpackAuthHelper, BackpackAuthError
from .symbols import BackpackSymbolService, BackpackMarket
from .rest import BackpackRestClient, BackpackOrderBookSnapshot, BackpackRestError
from .ws import BackpackWsSession, BackpackSubscription, BackpackWebsocketError
from .errors import (
    BackpackOrderBookGap,
    BackpackOrderBookMissingSnapshot,
    BackpackPayloadError,
    BackpackRouterError,
    BackpackUnknownChannel,
)
from .metrics import BackpackMetrics
from .health import BackpackHealthReport, evaluate_health
from .feed import BackpackFeed
from .adapters import (
    BackpackTradeAdapter,
    BackpackOrderBookAdapter,
    BackpackTickerAdapter,
    BackpackCandleAdapter,
    BackpackOrderAdapter,
    BackpackPositionAdapter,
)

__all__ = [
    "BackpackConfig",
    "BackpackAuthSettings",
    "BackpackAuthHelper",
    "BackpackAuthError",
    "BackpackSymbolService",
    "BackpackMarket",
    "BackpackRestClient",
    "BackpackOrderBookSnapshot",
    "BackpackRestError",
    "BackpackWsSession",
    "BackpackSubscription",
    "BackpackWebsocketError",
    "BackpackPayloadError",
    "BackpackUnknownChannel",
    "BackpackOrderBookMissingSnapshot",
    "BackpackOrderBookGap",
    "BackpackRouterError",
    "BackpackMetrics",
    "BackpackHealthReport",
    "evaluate_health",
    "BackpackFeed",
    "BackpackTradeAdapter",
    "BackpackOrderBookAdapter",
    "BackpackTickerAdapter",
    "BackpackCandleAdapter",
    "BackpackOrderAdapter",
    "BackpackPositionAdapter",
]
