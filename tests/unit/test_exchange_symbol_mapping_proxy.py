import pytest
import asyncio

from cryptofeed.exchange import Exchange, RestEndpoint
from cryptofeed.connection import Routes
from cryptofeed.defines import TRADES
from cryptofeed.proxy import (
    ProxySettings,
    ProxyConfig,
    ConnectionProxies,
    init_proxy_system,
)


class _DummyExchange(Exchange):
    id = "DUMMY"
    websocket_endpoints = []
    rest_endpoints = [
        RestEndpoint("https://api.example.com", routes=Routes("/api/v3/exchangeInfo"))
    ]
    websocket_channels = {TRADES: "trades"}
    request_limit = 1

    @classmethod
    def _parse_symbol_data(cls, data):
        # Minimal shape expected by Exchange.symbol_mapping consumers
        if isinstance(data, dict):
            symbols = data.get("symbols", [])
        else:
            symbols = data["symbols"]
        ret = {}
        info = {"tick_size": {}, "instrument_type": {}}
        for symbol in symbols:
            std = f"{symbol['baseAsset']}-{symbol['quoteAsset']}"
            ret[std] = symbol["symbol"]
            info["tick_size"][std] = symbol["filters"][0]["tickSize"]
            info["instrument_type"][std] = symbol.get("contractType", "SPOT")
        return ret, info


@pytest.mark.asyncio
async def test_symbol_mapping_uses_proxy(monkeypatch):
    calls = []

    async def fake_fetch(url, proxy_url, timeout, headers=None):
        calls.append((url, proxy_url, timeout))
        return {
            "symbols": [
                {
                    "symbol": "BTCUSDT",
                    "status": "TRADING",
                    "contractStatus": "TRADING",
                    "baseAsset": "BTC",
                    "quoteAsset": "USDT",
                    "filters": [{"tickSize": "0.1"}],
                }
            ]
        }

    monkeypatch.setattr("cryptofeed.exchange._fetch_json_via_proxy", fake_fetch)
    settings = ProxySettings(
        enabled=True,
        default=ConnectionProxies(http=ProxyConfig(url="http://proxy:8080")),
    )
    init_proxy_system(settings)
    monkeypatch.setenv("CF_SYMBOL_FETCH_TIMEOUT", "3")

    # Trigger symbol mapping with refresh to force fetch
    _DummyExchange.symbol_mapping(refresh=True)

    init_proxy_system(ProxySettings(enabled=False))

    assert calls, "fetch should be invoked"
    assert calls[0][1] == "http://proxy:8080"
    assert calls[0][2] == 3.0


@pytest.mark.asyncio
async def test_symbol_mapping_respects_default_timeout(monkeypatch):
    """Test that symbol mapping uses default 10s timeout when not configured"""
    calls = []

    async def fake_fetch(url, proxy_url, timeout, headers=None):
        calls.append((url, proxy_url, timeout))
        return {
            "symbols": [
                {
                    "symbol": "BTCUSDT",
                    "status": "TRADING",
                    "contractStatus": "TRADING",
                    "baseAsset": "BTC",
                    "quoteAsset": "USDT",
                    "filters": [{"tickSize": "0.1"}],
                }
            ]
        }

    monkeypatch.setattr("cryptofeed.exchange._fetch_json_via_proxy", fake_fetch)
    settings = ProxySettings(
        enabled=True,
        default=ConnectionProxies(http=ProxyConfig(url="http://proxy:8080")),
    )
    init_proxy_system(settings)

    # Don't set timeout env var - should use default 10.0
    _DummyExchange.symbol_mapping(refresh=True)

    init_proxy_system(ProxySettings(enabled=False))

    assert calls, "fetch should be invoked"
    assert calls[0][2] == 10.0, "should use default 10s timeout"


@pytest.mark.asyncio
async def test_symbol_mapping_timeout_prevents_hang(monkeypatch):
    """Test that timeout is enforced and prevents indefinite hangs"""
    from aiohttp import ClientError

    async def slow_fetch(url, proxy_url, timeout, headers=None):
        # Simulate timeout by raising the appropriate error
        raise asyncio.TimeoutError("Request timed out")

    monkeypatch.setattr("cryptofeed.exchange._fetch_json_via_proxy", slow_fetch)
    settings = ProxySettings(
        enabled=True,
        default=ConnectionProxies(http=ProxyConfig(url="http://proxy:8080")),
    )
    init_proxy_system(settings)
    monkeypatch.setenv("CF_SYMBOL_FETCH_TIMEOUT", "0.1")

    # Should timeout and raise, not hang
    with pytest.raises(Exception):  # asyncio.TimeoutError or other exception from slow endpoint
        _DummyExchange.symbol_mapping(refresh=True)

    init_proxy_system(ProxySettings(enabled=False))


@pytest.mark.asyncio
async def test_symbol_mapping_works_without_proxy(monkeypatch):
    """Test that symbol mapping works in direct mode (no proxy)"""
    calls = []

    async def fake_fetch(url, proxy_url, timeout, headers=None):
        calls.append((url, proxy_url, timeout))
        return {
            "symbols": [
                {
                    "symbol": "BTCUSDT",
                    "status": "TRADING",
                    "contractStatus": "TRADING",
                    "baseAsset": "BTC",
                    "quoteAsset": "USDT",
                    "filters": [{"tickSize": "0.1"}],
                }
            ]
        }

    monkeypatch.setattr("cryptofeed.exchange._fetch_json_via_proxy", fake_fetch)

    # Initialize proxy system as disabled
    init_proxy_system(ProxySettings(enabled=False))

    # Trigger symbol mapping - should work without proxy
    _DummyExchange.symbol_mapping(refresh=True)

    assert calls, "fetch should be invoked"
    assert calls[0][1] is None, "proxy_url should be None in direct mode"
    assert calls[0][2] == 10.0, "should still have timeout in direct mode"


@pytest.mark.asyncio
async def test_binance_symbol_mapping_with_proxy(monkeypatch):
    """Test that Binance specifically uses proxy for symbol bootstrap"""
    from cryptofeed.exchanges.binance import Binance

    calls = []

    async def fake_fetch(url, proxy_url, timeout, headers=None):
        calls.append((url, proxy_url, timeout))
        # Return minimal valid Binance exchangeInfo response
        return {
            "symbols": [
                {
                    "symbol": "BTCUSDT",
                    "status": "TRADING",
                    "baseAsset": "BTC",
                    "quoteAsset": "USDT",
                    "filters": [{"tickSize": "0.01", "filterType": "PRICE_FILTER"}],
                }
            ]
        }

    monkeypatch.setattr("cryptofeed.exchange._fetch_json_via_proxy", fake_fetch)
    settings = ProxySettings(
        enabled=True,
        exchanges={
            "binance": ConnectionProxies(
                http=ProxyConfig(url="http://binance-proxy:8080")
            )
        },
    )
    init_proxy_system(settings)

    # Trigger Binance symbol mapping
    Binance.symbol_mapping(refresh=True)

    init_proxy_system(ProxySettings(enabled=False))

    assert calls, "Binance exchangeInfo fetch should be invoked"
    assert calls[0][1] == "http://binance-proxy:8080", "should use Binance-specific proxy"
    assert "api.binance.com" in calls[0][0], "should hit Binance API"


@pytest.mark.asyncio
async def test_binance_futures_symbol_mapping_with_proxy(monkeypatch):
    """Test that Binance Futures uses proxy for symbol bootstrap"""
    from cryptofeed.exchanges.binance_futures import BinanceFutures

    calls = []

    async def fake_fetch(url, proxy_url, timeout, headers=None):
        calls.append((url, proxy_url, timeout))
        # Return minimal valid Binance Futures response
        return {
            "symbols": [
                {
                    "symbol": "BTCUSDT",
                    "status": "TRADING",
                    "contractStatus": "TRADING",
                    "contractType": "PERPETUAL",
                    "baseAsset": "BTC",
                    "quoteAsset": "USDT",
                    "filters": [{"tickSize": "0.01", "filterType": "PRICE_FILTER"}],
                }
            ]
        }

    monkeypatch.setattr("cryptofeed.exchange._fetch_json_via_proxy", fake_fetch)
    settings = ProxySettings(
        enabled=True,
        exchanges={
            "binance_futures": ConnectionProxies(
                http=ProxyConfig(url="http://binance-futures-proxy:8080")
            )
        },
    )
    init_proxy_system(settings)

    # Trigger Binance Futures symbol mapping
    BinanceFutures.symbol_mapping(refresh=True)

    init_proxy_system(ProxySettings(enabled=False))

    assert calls, "Binance Futures exchangeInfo fetch should be invoked"
    assert calls[0][1] == "http://binance-futures-proxy:8080", "should use Binance Futures-specific proxy"
    assert "fapi.binance.com" in calls[0][0], "should hit Binance Futures API"
