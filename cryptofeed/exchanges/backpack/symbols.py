from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, Iterable, Optional

from cryptofeed.defines import FUTURES, PERPETUAL, SPOT
from cryptofeed.symbols import Symbol


@dataclass(frozen=True, slots=True)
class BackpackMarket:
    """Normalized Backpack market metadata."""

    normalized_symbol: str
    native_symbol: str
    instrument_type: str
    base_asset: str
    quote_asset: str
    price_precision: Optional[int]
    amount_precision: Optional[int]
    min_amount: Optional[Decimal]


class BackpackSymbolService:
    """Loads and caches Backpack market metadata for symbol normalization."""

    def __init__(self, *, rest_client, ttl_seconds: int = 900):
        self._rest_client = rest_client
        self._ttl = timedelta(seconds=ttl_seconds)
        self._lock = asyncio.Lock()
        self._markets: Dict[str, BackpackMarket] = {}
        self._expires_at: Optional[datetime] = None

    async def ensure(self, *, force: bool = False) -> None:
        async with self._lock:
            now = datetime.now(timezone.utc)
            if not force and self._expires_at and now < self._expires_at and self._markets:
                return

            raw_markets = await self._rest_client.fetch_markets()
            self._markets = self._parse_markets(raw_markets)
            self._expires_at = now + self._ttl

    def get_market(self, symbol: str) -> BackpackMarket:
        try:
            return self._markets[symbol]
        except KeyError as exc:
            raise KeyError(f"Unknown Backpack symbol: {symbol}") from exc

    def native_symbol(self, symbol: str) -> str:
        return self.get_market(symbol).native_symbol

    def all_markets(self) -> Iterable[BackpackMarket]:
        return self._markets.values()

    def clear(self) -> None:
        self._markets = {}
        self._expires_at = None

    @staticmethod
    def _parse_markets(markets: Iterable[dict]) -> Dict[str, BackpackMarket]:
        parsed: Dict[str, BackpackMarket] = {}
        for entry in markets:
            status = str(entry.get('status', '')).upper()
            if status and status not in {'TRADING', 'ENABLED'}:
                continue

            native_symbol = entry.get('symbol') or entry.get('market')
            base_asset = entry.get('baseSymbol') or entry.get('baseAsset') or entry.get('base')
            quote_asset = entry.get('quoteSymbol') or entry.get('quoteAsset') or entry.get('quote')
            if not native_symbol or not base_asset or not quote_asset:
                continue

            instrument_type_raw = str(entry.get('type', 'spot')).upper()
            if instrument_type_raw == 'PERPETUAL':
                instrument_type = PERPETUAL
            elif instrument_type_raw in {'FUTURE', 'FUTURES'}:
                instrument_type = FUTURES
            else:
                instrument_type = SPOT

            expiry = entry.get('expiry') or entry.get('expiryDate')
            if instrument_type == FUTURES and not expiry:
                # Futures instruments require an expiry to normalize; skip otherwise
                continue

            if instrument_type == FUTURES:
                symbol_obj = Symbol(base_asset, quote_asset, type=instrument_type, expiry_date=expiry)
            else:
                symbol_obj = Symbol(base_asset, quote_asset, type=instrument_type)

            normalized = symbol_obj.normalized

            precision = entry.get('precision') or entry.get('tickSize') or {}
            amount_precision = None
            price_precision = None
            if isinstance(precision, dict):
                price_precision = precision.get('price')
                amount_precision = precision.get('amount')
            elif isinstance(precision, (int, float)):
                price_precision = precision

            limits = entry.get('limits', {}) if isinstance(entry.get('limits'), dict) else {}
            amount_limits = limits.get('amount', {}) if isinstance(limits, dict) else {}
            min_amount_raw = amount_limits.get('min')
            min_amount = Decimal(str(min_amount_raw)) if min_amount_raw is not None else None

            market = BackpackMarket(
                normalized_symbol=normalized,
                native_symbol=native_symbol,
                instrument_type=instrument_type,
                base_asset=base_asset,
                quote_asset=quote_asset,
                price_precision=price_precision,
                amount_precision=amount_precision,
                min_amount=min_amount,
            )
            parsed[normalized] = market
        return parsed


__all__ = [
    "BackpackMarket",
    "BackpackSymbolService",
]
