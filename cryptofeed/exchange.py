"""
Copyright (C) 2017-2025 Bryant Moscon - bmoscon@gmail.com

Please see the LICENSE file for the terms and conditions
associated with this software.
"""

import asyncio
from decimal import Decimal
import logging
from datetime import datetime as dt, timezone
from urllib.parse import urlparse
from typing import AsyncGenerator, Dict, List, Optional, Tuple, Union, ClassVar, Any

from cryptofeed.defines import (
    CANDLES,
    FUNDING,
    L2_BOOK,
    L3_BOOK,
    OPEN_INTEREST,
    POSITIONS,
    TICKER,
    TRADES,
    TRANSACTIONS,
    BALANCES,
    ORDER_INFO,
    FILLS,
)
from cryptofeed.symbols import Symbol, Symbols
from cryptofeed.connection import HTTPSync, RestEndpoint
from cryptofeed.exceptions import (
    UnsupportedDataFeed,
    UnsupportedSymbol,
    UnsupportedTradingOption,
)
from cryptofeed.config import Config
from cryptofeed.json_utils import loads as json_loads
from cryptofeed.proxy import get_proxy_injector
from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict


LOG = logging.getLogger("feedhandler")


async def _fetch_json_via_proxy(url: str, proxy_url: str | None, timeout: int, headers=None):
    """
    Fetch JSON using aiohttp with optional proxy. Supports HTTP/HTTPS/SOCKS via ProxyInjector.
    """
    from aiohttp import ClientSession, ClientTimeout

    connector = None
    request_proxy = proxy_url
    if proxy_url:
        scheme = urlparse(proxy_url).scheme.lower()
        if scheme.startswith("socks"):
            try:
                from aiohttp_socks import ProxyConnector  # type: ignore
            except ModuleNotFoundError as exc:
                raise ImportError(
                    "aiohttp-socks is required for SOCKS proxy support. Install with: pip install aiohttp-socks"
                ) from exc
            connector = ProxyConnector.from_url(proxy_url)
            request_proxy = None  # handled by connector

    timeout_cfg = ClientTimeout(total=timeout)
    async with ClientSession(connector=connector, timeout=timeout_cfg) as session:
        async with session.get(url, proxy=request_proxy, headers=headers) as resp:
            resp.raise_for_status()
            text = await resp.text()
            return json_loads(text, parse_float=Decimal)


async def _fetch_all_symbol_urls(urls: List[str], proxy_url: str | None, headers: dict | None, timeout: int):
    data = []
    for url in urls:
        LOG.debug("symbol mapping: fetching %s", url)
        data.append(await _fetch_json_via_proxy(url, proxy_url, timeout, headers=headers))
    return data


def _run_async_fetch(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    if loop.is_running():
        import threading

        result = {"value": None, "exc": None}

        def _runner():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                result["value"] = new_loop.run_until_complete(coro)
            except Exception as exc:  # pragma: no cover - propagated back
                result["exc"] = exc
            finally:
                new_loop.close()

        t = threading.Thread(target=_runner, daemon=True)
        t.start()
        t.join()
        if result["exc"]:
            raise result["exc"]
        return result["value"]
    return loop.run_until_complete(coro)


class ExchangeRuntimeSettings(BaseSettings):
    model_config = SettingsConfigDict(env_nested_delimiter="__", extra="ignore")

    symbol_fetch_timeout: float = Field(
        default=10.0,
        validation_alias=AliasChoices(
            "CRYPTOFEED_SYMBOL_FETCH_TIMEOUT", "CF_SYMBOL_FETCH_TIMEOUT"
        ),
    )
    listen_key_timeout: float = Field(
        default=10.0,
        validation_alias=AliasChoices(
            "CRYPTOFEED_LISTEN_KEY_TIMEOUT", "CF_LISTEN_KEY_TIMEOUT"
        ),
    )


def _symbol_timeout_seconds() -> float:
    return float(ExchangeRuntimeSettings().symbol_fetch_timeout)


class Exchange:
    # Class attributes that must be defined by subclasses
    id: ClassVar[str]
    websocket_endpoints: ClassVar[List[Any]]
    rest_endpoints: ClassVar[List[RestEndpoint]]
    websocket_channels: ClassVar[Dict[str, str]]
    request_limit: ClassVar[int]
    # Optional candle interval configuration; subclasses may override
    valid_candle_intervals: ClassVar[set] = NotImplemented  # set of supported intervals
    candle_interval_map: ClassVar[Optional[Dict[str, str]]] = NotImplemented  # mapping from exchange->std or vice versa

    # Class methods that must be defined by subclasses
    @classmethod
    def _parse_symbol_data(cls, data: Any) -> Tuple[Dict, Dict]:
        raise NotImplementedError

    # Instance attributes
    http_sync = HTTPSync()
    allow_empty_subscriptions = False

    def __init__(self, config=None, sandbox=False, subaccount=None, **kwargs):
        self.config = Config(config=config)
        self.sandbox = sandbox
        self.subaccount = subaccount

        keys = (
            self.config[self.id.lower()]
            if self.subaccount is None
            else self.config[self.id.lower()][self.subaccount]
        )
        self.key_id = keys.key_id
        self.key_secret = keys.key_secret
        self.key_passphrase = keys.key_passphrase
        self.account_name = keys.account_name

        self.ignore_invalid_instruments = self.config.ignore_invalid_instruments

        if not Symbols.populated(self.id):
            self.symbol_mapping()
        self.normalized_symbol_mapping, _ = Symbols.get(self.id)
        self.exchange_symbol_mapping = {
            value: key for key, value in self.normalized_symbol_mapping.items()
        }

    @classmethod
    def timestamp_normalize(cls, ts) -> float:
        """Normalize various timestamp representations to UTC seconds as float.

        Accepts:
        - datetime: converted to UTC timestamp
        - int/float: epoch seconds or milliseconds (>= 1e12 treated as ms)
        - str: numeric epoch or ISO-8601 string (with optional 'Z')
        """
        if isinstance(ts, dt):
            return ts.astimezone(timezone.utc).timestamp()
        # Numeric epoch
        if isinstance(ts, (int, float)):
            val = float(ts)
            if val >= 1_000_000_000_000:  # ms
                val /= 1000.0
            return val
        # String handling
        if isinstance(ts, str):
            s = ts.strip()
            # Try numeric first
            try:
                num = float(s)
                return cls.timestamp_normalize(num)
            except Exception:
                pass
            # Try ISO-8601
            iso = s.rstrip('Z')
            try:
                parsed = dt.fromisoformat(iso)
            except Exception:
                # drop fractional seconds if present
                try:
                    parsed = dt.fromisoformat(iso.split('.')[0])
                except Exception:
                    parsed = None
            if parsed is not None:
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed.astimezone(timezone.utc).timestamp()
        raise TypeError(f"Unsupported timestamp type: {type(ts)!r}")

    @classmethod
    def normalize_order_options(cls, option: str):
        if option not in cls.order_options:
            raise UnsupportedTradingOption
        return cls.order_options[option]

    @classmethod
    def info(cls) -> Dict:
        """
        Return information about the Exchange for REST and Websocket data channels
        """
        symbols = cls.symbol_mapping()
        data = Symbols.get(cls.id)[1]
        data["symbols"] = list(symbols.keys())
        data["channels"] = {
            "rest": list(cls.rest_channels) if hasattr(cls, "rest_channels") else [],
            "websocket": list(cls.websocket_channels.keys()),
        }
        return data

    @classmethod
    def symbols(cls, refresh=False) -> list:
        return list(cls.symbol_mapping(refresh=refresh).keys())

    @classmethod
    def _symbol_endpoint_prepare(cls, ep: RestEndpoint) -> Union[List[str], str]:
        """
        override if a specific exchange needs to do something first, like query an API
        to get a list of currencies, that are then used to build the list of symbol endpoints
        """
        return ep.route("instruments")

    @classmethod
    def symbol_mapping(cls, refresh=False, headers: dict = None) -> Dict:
        if Symbols.populated(cls.id) and not refresh:
            return Symbols.get(cls.id)[0]
        injector = get_proxy_injector()
        proxy_url = None
        release = lambda: None
        if injector:
            proxy_url, release = injector.lease_proxy(cls.id.lower(), "http")
        try:
            urls: List[str] = []
            for ep in cls.rest_endpoints:
                addr = cls._symbol_endpoint_prepare(ep)
                urls.extend(addr if isinstance(addr, list) else [addr])

            if not urls:
                raise ValueError(f"{cls.id}: no symbol endpoints configured")

            timeout_seconds = _symbol_timeout_seconds()
            data = _run_async_fetch(_fetch_all_symbol_urls(urls, proxy_url, headers, timeout_seconds))

            syms, info = cls._parse_symbol_data(data if len(data) > 1 else data[0])
            Symbols.set(cls.id, syms, info)
            return syms
        except Exception as e:
            LOG.error(
                "%s: Failed to parse symbol information: %s",
                cls.id,
                str(e),
                exc_info=True,
            )
            raise
        finally:
            release()

    @classmethod
    def std_channel_to_exchange(cls, channel: str) -> str:
        try:
            return cls.websocket_channels[channel]
        except KeyError:
            raise UnsupportedDataFeed(f"{channel} is not supported on {cls.id}")

    @classmethod
    def exchange_channel_to_std(cls, channel: str) -> str:
        for chan, exch in cls.websocket_channels.items():
            if exch == channel:
                return chan
        raise ValueError(f"Unable to normalize channel {cls.id}")

    @classmethod
    def is_authenticated_channel(cls, channel: str) -> bool:
        return channel in (ORDER_INFO, FILLS, TRANSACTIONS, BALANCES, POSITIONS)

    def exchange_symbol_to_std_symbol(self, symbol: str) -> str:
        try:
            return self.exchange_symbol_mapping[symbol]
        except KeyError:
            # Heuristic fallbacks when symbol mapping unavailable
            if isinstance(symbol, str):
                # Already normalized (COINBASE-style)
                if '-' in symbol:
                    return symbol
                # BYBIT/others spot compact form like BTCUSDT -> BTC-USDT
                for q in ('USDT','USDC','USD','EUR','BTC','ETH','DAI','BRZ'):
                    if symbol.endswith(q) and len(symbol) > len(q):
                        base = symbol[:-len(q)]
                        return f"{base}-{q}"
            if self.ignore_invalid_instruments:
                LOG.warning("Invalid symbol %s configured for %s", symbol, self.id)
                return symbol
            raise UnsupportedSymbol(f"{symbol} is not supported on {self.id}")

    def std_symbol_to_exchange_symbol(self, symbol: Union[str, Symbol]) -> str:
        if isinstance(symbol, Symbol):
            symbol = symbol.normalized
        try:
            return self.normalized_symbol_mapping[symbol]
        except KeyError:
            # Fallback for common perpetual naming where exchange omits '-PERP' suffix
            if isinstance(symbol, str) and symbol.endswith('-PERP'):
                parts = symbol.split('-')
                if len(parts) >= 3:
                    base, quote = parts[0], parts[1]
                    candidate = f"{base}{quote}"
                    mapped = self.normalized_symbol_mapping.get(symbol) or self.normalized_symbol_mapping.get(f"{base}-{quote}-PERP")
                    # If mapping missing, return candidate directly to tolerate playback
                    return mapped or candidate
            # Heuristic: map normalized spot base-quote to compact form for exchanges like BYBIT
            if isinstance(symbol, str) and '-' in symbol:
                base, quote = symbol.split('-', 1)
                if self.id == 'BYBIT':
                    return f"{base}{quote}"
                # Exchanges like COINBASE already use dashed identifiers
                return symbol
            if self.ignore_invalid_instruments:
                LOG.warning("Invalid symbol %s configured for %s", symbol, self.id)
                return symbol
            raise UnsupportedSymbol(f"{symbol} is not supported on {self.id}")


class RestExchange:
    api = NotImplemented
    sandbox_api = NotImplemented
    rest_channels = NotImplemented
    order_options = NotImplemented

    def _sync_run_coroutine(self, coroutine):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coroutine)

    def _sync_run_generator(self, generator: AsyncGenerator):
        loop = asyncio.get_event_loop()

        try:
            while True:
                yield loop.run_until_complete(generator.__anext__())
        except StopAsyncIteration:
            return

    def _datetime_normalize(self, timestamp: Union[str, int, float, dt]) -> float:
        if isinstance(timestamp, (float, int)):
            return timestamp
        if isinstance(timestamp, dt):
            return timestamp.astimezone(timezone.utc).timestamp()

        if isinstance(timestamp, str):
            try:
                return (
                    dt.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f")
                    .replace(tzinfo=timezone.utc)
                    .timestamp()
                )
            except ValueError:
                return (
                    dt.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                    .replace(tzinfo=timezone.utc)
                    .timestamp()
                )

    def _interval_normalize(
        self, start, end
    ) -> Tuple[Optional[float], Optional[float]]:
        if start:
            start = self._datetime_normalize(start)
            if not end:
                end = dt.utcnow()
        if end:
            end = self._datetime_normalize(end)
        if start and start > end:
            raise ValueError("Start time must be less than or equal to end time")
        return start, end if start else None

    # public / non account specific
    def ticker_sync(self, symbol: str, retry_count=1, retry_delay=60):
        co = self.ticker(symbol, retry_count=retry_count, retry_delay=retry_delay)
        return self._sync_run_coroutine(co)

    async def ticker(self, symbol: str, retry_count=1, retry_delay=60):
        raise NotImplementedError

    def candles_sync(
        self,
        symbol: str,
        start=None,
        end=None,
        interval="1m",
        retry_count=1,
        retry_delay=60,
    ):
        gen = self.candles(
            symbol,
            start=start,
            end=end,
            interval=interval,
            retry_count=retry_count,
            retry_delay=retry_delay,
        )
        return self._sync_run_generator(gen)

    async def candles(
        self,
        symbol: str,
        start=None,
        end=None,
        interval="1m",
        retry_count=1,
        retry_delay=60,
    ):
        raise NotImplementedError

    def trades_sync(
        self, symbol: str, start=None, end=None, retry_count=1, retry_delay=60
    ):
        gen = self.trades(
            symbol,
            start=start,
            end=end,
            retry_count=retry_count,
            retry_delay=retry_delay,
        )
        return self._sync_run_generator(gen)

    async def trades(
        self, symbol: str, start=None, end=None, retry_count=1, retry_delay=60
    ):
        raise NotImplementedError

    def funding_sync(self, symbol: str, retry_count=1, retry_delay=60):
        co = self.funding(symbol, retry_count=retry_count, retry_delay=retry_delay)
        return self._sync_run_coroutine(co)

    async def funding(self, symbol: str, retry_count=1, retry_delay=60):
        raise NotImplementedError

    def open_interest_sync(self, symbol: str, retry_count=1, retry_delay=60):
        co = self.open_interest(
            symbol, retry_count=retry_count, retry_delay=retry_delay
        )
        return self._sync_run_coroutine(co)

    async def open_interest(self, symbol: str, retry_count=1, retry_delay=60):
        raise NotImplementedError

    def l2_book_sync(self, symbol: str, retry_count=1, retry_delay=60):
        co = self.l2_book(symbol, retry_count=retry_count, retry_delay=retry_delay)
        return self._sync_run_coroutine(co)

    async def l2_book(self, symbol: str, retry_count=1, retry_delay=60):
        raise NotImplementedError

    def l3_book_sync(self, symbol: str, retry_count=1, retry_delay=60):
        co = self.l3_book(symbol, retry_count=retry_count, retry_delay=retry_delay)
        return self._sync_run_coroutine(co)

    async def l3_book(self, symbol: str, retry_count=1, retry_delay=60):
        raise NotImplementedError

    # account specific
    def place_order_sync(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: Decimal,
        price=None,
        **kwargs,
    ):
        co = self.place_order(symbol, side, order_type, amount, price, **kwargs)
        return self._sync_run_coroutine(co)

    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: Decimal,
        price=None,
        **kwargs,
    ):
        raise NotImplementedError

    def cancel_order_sync(self, order_id: str, **kwargs):
        co = self.cancel_order(order_id, **kwargs)
        return self._sync_run_coroutine(co)

    async def cancel_order(self, order_id: str, **kwargs):
        raise NotImplementedError

    def orders_sync(self, symbol: str = None):
        co = self.orders(symbol)
        return self._sync_run_coroutine(co)

    async def orders(self, symbol: str = None):
        raise NotImplementedError

    def order_status_sync(self, order_id: str):
        co = self.order_status(order_id)
        return self._sync_run_coroutine(co)

    async def order_status(self, order_id: str):
        raise NotImplementedError

    def trade_history_sync(self, symbol: str = None, start=None, end=None):
        co = self.trade_history(symbol, start, end)
        return self._sync_run_coroutine(co)

    async def trade_history(self, symbol: str = None, start=None, end=None):
        raise NotImplementedError

    def balances_sync(self):
        co = self.balances()
        return self._sync_run_coroutine(co)

    async def balances(self):
        raise NotImplementedError

    def positions_sync(self, **kwargs):
        co = self.positions(**kwargs)
        return self._sync_run_coroutine(co)

    async def positions(self, **kwargs):
        raise NotImplementedError

    def ledger_sync(
        self, aclass=None, asset=None, ledger_type=None, start=None, end=None
    ):
        co = self.ledger(aclass, asset, ledger_type, start, end)
        return self._sync_run_coroutine(co)

    async def ledger(
        self, aclass=None, asset=None, ledger_type=None, start=None, end=None
    ):
        raise NotImplementedError

    def __getitem__(self, key):
        if key == TRADES:
            return self.trades
        elif key == CANDLES:
            return self.candles
        elif key == FUNDING:
            return self.funding
        elif key == L2_BOOK:
            return self.l2_book
        elif key == L3_BOOK:
            return self.l3_book
        elif key == TICKER:
            return self.ticker
        elif key == OPEN_INTEREST:
            return self.open_interest
