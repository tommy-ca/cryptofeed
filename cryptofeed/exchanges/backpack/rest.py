"""Backpack REST client built on cryptofeed HTTPAsyncConn."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

from cryptofeed.json_utils import json

from cryptofeed.connection import HTTPAsyncConn
from cryptofeed.exchanges.backpack.config import BackpackConfig


class BackpackRestError(RuntimeError):
    """Raised when Backpack REST operations fail."""


@dataclass(slots=True)
class BackpackOrderBookSnapshot:
    symbol: str
    bids: list[list[str | float]]
    asks: list[list[str | float]]
    sequence: Optional[int]
    timestamp_ms: Optional[int]


class BackpackRestClient:
    """Thin async wrapper around HTTPAsyncConn with Backpack-specific helpers."""

    MARKETS_PATH = "/api/v1/markets"
    L2_DEPTH_PATH = "/api/v1/depth"
    TRADES_PATH = "/api/v1/trades"
    KLINES_PATH = "/api/v1/klines"

    def __init__(self, config: BackpackConfig, *, http_conn_factory=None) -> None:
        self._config = config
        factory = http_conn_factory or (lambda: HTTPAsyncConn("backpack", exchange_id=config.exchange_id))
        self._conn: HTTPAsyncConn = factory()
        self._closed = False

    async def close(self) -> None:
        if not self._closed:
            await self._conn.close()
            self._closed = True

    async def fetch_markets(self) -> Iterable[Dict[str, Any]]:
        """Return Backpack market metadata list."""
        url = f"{self._config.rest_endpoint}{self.MARKETS_PATH}"
        text = await self._conn.read(url)
        try:
            data = json.loads(text)
        except Exception as exc:  # pragma: no cover - JSON backend may raise generic Exception types
            raise BackpackRestError(f"Unable to parse markets payload: {exc}") from exc
        if not isinstance(data, (list, tuple)):
            raise BackpackRestError("Markets endpoint returned unexpected payload")
        return data

    async def fetch_order_book(self, *, native_symbol: str, depth: int = 50) -> BackpackOrderBookSnapshot:
        """Fetch an order book snapshot for the provided native Backpack symbol."""
        url = f"{self._config.rest_endpoint}{self.L2_DEPTH_PATH}"
        params = {"symbol": native_symbol, "limit": depth}
        text = await self._conn.read(url, params=params)
        try:
            data = json.loads(text)
        except Exception as exc:  # pragma: no cover
            raise BackpackRestError(f"Unable to parse order book payload: {exc}") from exc

        if not isinstance(data, dict) or "bids" not in data or "asks" not in data:
            raise BackpackRestError("Malformed order book payload")

        return BackpackOrderBookSnapshot(
            symbol=native_symbol,
            bids=data.get("bids", []),
            asks=data.get("asks", []),
            sequence=data.get("sequence"),
            timestamp_ms=data.get("timestamp"),
        )

    async def fetch_trades(self, *, native_symbol: str, limit: int = 100) -> Iterable[Dict[str, Any]]:
        """Fetch recent trades for the provided native Backpack symbol.
        
        Args:
            native_symbol: Native Backpack symbol (e.g., "BTC_USDC")
            limit: Maximum number of trades to fetch (default: 100, max: 1000)
            
        Returns:
            List of recent trades
        """
        url = f"{self._config.rest_endpoint}{self.TRADES_PATH}"
        params = {"symbol": native_symbol, "limit": min(limit, 1000)}
        text = await self._conn.read(url, params=params)
        try:
            data = json.loads(text)
        except Exception as exc:  # pragma: no cover
            raise BackpackRestError(f"Unable to parse trades payload: {exc}") from exc

        if not isinstance(data, (list, tuple)):
            raise BackpackRestError("Trades endpoint returned unexpected payload")

        return data

    async def fetch_klines(
        self,
        *,
        native_symbol: str,
        interval: str = "1m",
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: Optional[int] = None
    ) -> Iterable[Dict[str, Any]]:
        """Fetch K-line/candle data for the provided native Backpack symbol.
        
        Args:
            native_symbol: Native Backpack symbol (e.g., "BTC_USDC")
            interval: Candle interval (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1month)
            start_time: Start timestamp in seconds (UTC)
            end_time: End timestamp in seconds (UTC), defaults to current time if not provided
            limit: Maximum number of candles to fetch
            
        Returns:
            List of K-line data
        """
        url = f"{self._config.rest_endpoint}{self.KLINES_PATH}"
        params = {
            "symbol": native_symbol,
            "interval": interval
        }
        
        # API requires startTime to be present
        if start_time is None:
            # Default to 24 hours ago if not specified
            import time
            start_time = int(time.time()) - 86400
        
        params["startTime"] = start_time
        
        if end_time is not None:
            params["endTime"] = end_time
            
        text = await self._conn.read(url, params=params)
        try:
            data = json.loads(text)
        except Exception as exc:  # pragma: no cover
            raise BackpackRestError(f"Unable to parse klines payload: {exc}") from exc

        if not isinstance(data, (list, tuple)):
            raise BackpackRestError("Klines endpoint returned unexpected payload")

        # Apply limit if specified
        if limit is not None and len(data) > limit:
            data = data[:limit]

        return data

    async def __aenter__(self) -> "BackpackRestClient":
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()
