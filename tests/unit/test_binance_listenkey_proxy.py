import pytest
import asyncio

from cryptofeed.exchanges.binance import Binance
from cryptofeed.exchanges.binance_futures import BinanceFutures
from cryptofeed.proxy import (
    ConnectionProxies,
    ProxyConfig,
    ProxySettings,
    init_proxy_system,
)
from cryptofeed.symbols import Symbols


def test_binance_generate_token_uses_proxy(monkeypatch):
    """Test that Binance spot listen-key generation uses configured proxy"""
    calls = []

    async def fake_request(method, url, headers, proxy_url, timeout, data=None):
        calls.append((method, proxy_url, url, timeout))
        return {"listenKey": "abc"}

    monkeypatch.setattr(
        "cryptofeed.exchanges.binance._http_request_with_proxy", fake_request
    )
    monkeypatch.setattr("cryptofeed.exchanges.binance.Binance.symbol_mapping", classmethod(lambda cls, refresh=False, headers=None: {}))

    settings = ProxySettings(
        enabled=True,
        exchanges={"binance": ConnectionProxies(http=ProxyConfig(url="socks5://p:1080"))},
    )
    init_proxy_system(settings)
    monkeypatch.setenv("CF_LISTEN_KEY_TIMEOUT", "7")

    Symbols.set("BINANCE", {"BTC-USDT": "BTCUSDT"}, {"tick_size": {}, "instrument_type": {}})
    b = Binance(symbols=[], channels=[])
    key = b._generate_token()

    init_proxy_system(ProxySettings(enabled=False))

    assert key == "abc"
    assert calls and calls[0][1] == "socks5://p:1080"
    assert calls[0][3] == 7.0


def test_binance_generate_token_direct_mode(monkeypatch):
    """Test that Binance spot listen-key generation works in direct mode (no proxy)"""
    calls = []

    async def fake_request(method, url, headers, proxy_url, timeout, data=None):
        calls.append((method, proxy_url, url, timeout))
        return {"listenKey": "direct-key"}

    monkeypatch.setattr(
        "cryptofeed.exchanges.binance._http_request_with_proxy", fake_request
    )
    monkeypatch.setattr("cryptofeed.exchanges.binance.Binance.symbol_mapping", classmethod(lambda cls, refresh=False, headers=None: {}))

    # Initialize with disabled proxy system
    init_proxy_system(ProxySettings(enabled=False))
    monkeypatch.setenv("CF_LISTEN_KEY_TIMEOUT", "10")

    Symbols.set("BINANCE", {"BTC-USDT": "BTCUSDT"}, {"tick_size": {}, "instrument_type": {}})
    b = Binance(symbols=[], channels=[])
    key = b._generate_token()

    assert key == "direct-key"
    assert calls and calls[0][1] is None  # No proxy URL
    assert calls[0][3] == 10.0


@pytest.mark.asyncio
async def test_binance_refresh_token_uses_proxy(monkeypatch):
    """Test that Binance spot listen-key refresh uses configured proxy"""
    calls = []
    sleep_count = [0]

    async def fake_request(method, url, headers, proxy_url, timeout, data=None):
        calls.append((method, proxy_url, url, timeout))
        return {}

    async def fake_sleep(seconds):
        # Mock sleep and cancel after first call to avoid infinite loop
        sleep_count[0] += 1
        if sleep_count[0] > 1:
            raise asyncio.CancelledError()

    monkeypatch.setattr(
        "cryptofeed.exchanges.binance._http_request_with_proxy", fake_request
    )
    monkeypatch.setattr("cryptofeed.exchanges.binance.sleep", fake_sleep)
    monkeypatch.setattr("cryptofeed.exchanges.binance.Binance.symbol_mapping", classmethod(lambda cls, refresh=False, headers=None: {}))

    settings = ProxySettings(
        enabled=True,
        exchanges={"binance": ConnectionProxies(http=ProxyConfig(url="http://proxy.example.com:8080"))},
    )
    init_proxy_system(settings)
    monkeypatch.setenv("CF_LISTEN_KEY_TIMEOUT", "5")

    Symbols.set("BINANCE", {"BTC-USDT": "BTCUSDT"}, {"tick_size": {}, "instrument_type": {}})
    b = Binance(symbols=[], channels=[])
    b._auth_token = "existing-token"

    # Run refresh - it will cancel itself after one cycle via our fake_sleep
    try:
        await b._refresh_token()
    except asyncio.CancelledError:
        pass

    init_proxy_system(ProxySettings(enabled=False))

    # The refresh should have been called
    assert len(calls) >= 1
    assert calls[0][0] == "PUT"
    assert calls[0][1] == "http://proxy.example.com:8080"
    assert calls[0][3] == 5.0


def test_binance_futures_generate_token_uses_proxy(monkeypatch):
    """Test that Binance futures listen-key generation uses configured proxy"""
    calls = []

    async def fake_request(method, url, headers, proxy_url, timeout, data=None):
        calls.append((method, proxy_url, url, timeout))
        return {"listenKey": "futures-key"}

    monkeypatch.setattr(
        "cryptofeed.exchanges.binance._http_request_with_proxy", fake_request
    )
    monkeypatch.setattr("cryptofeed.exchanges.binance_futures.BinanceFutures.symbol_mapping", classmethod(lambda cls, refresh=False, headers=None: {}))

    settings = ProxySettings(
        enabled=True,
        exchanges={"binance_futures": ConnectionProxies(http=ProxyConfig(url="http://futures-proxy:3128"))},
    )
    init_proxy_system(settings)
    monkeypatch.setenv("CF_LISTEN_KEY_TIMEOUT", "8")

    Symbols.set("BINANCE_FUTURES", {"BTC-USDT-PERP": "BTCUSDT"}, {"tick_size": {}, "instrument_type": {}})
    bf = BinanceFutures(symbols=[], channels=[])
    key = bf._generate_token()

    init_proxy_system(ProxySettings(enabled=False))

    assert key == "futures-key"
    assert calls and calls[0][1] == "http://futures-proxy:3128"
    assert calls[0][3] == 8.0


@pytest.mark.asyncio
async def test_binance_futures_refresh_token_uses_proxy(monkeypatch):
    """Test that Binance futures listen-key refresh uses configured proxy"""
    calls = []
    sleep_count = [0]

    async def fake_request(method, url, headers, proxy_url, timeout, data=None):
        calls.append((method, proxy_url, url, timeout))
        return {}

    async def fake_sleep(seconds):
        # Mock sleep and cancel after first call to avoid infinite loop
        sleep_count[0] += 1
        if sleep_count[0] > 1:
            raise asyncio.CancelledError()

    monkeypatch.setattr(
        "cryptofeed.exchanges.binance._http_request_with_proxy", fake_request
    )
    monkeypatch.setattr("cryptofeed.exchanges.binance.sleep", fake_sleep)
    monkeypatch.setattr("cryptofeed.exchanges.binance_futures.BinanceFutures.symbol_mapping", classmethod(lambda cls, refresh=False, headers=None: {}))

    settings = ProxySettings(
        enabled=True,
        exchanges={"binance_futures": ConnectionProxies(http=ProxyConfig(url="socks5://futures-socks:1080"))},
    )
    init_proxy_system(settings)
    monkeypatch.setenv("CF_LISTEN_KEY_TIMEOUT", "12")

    Symbols.set("BINANCE_FUTURES", {"BTC-USDT-PERP": "BTCUSDT"}, {"tick_size": {}, "instrument_type": {}})
    bf = BinanceFutures(symbols=[], channels=[])
    bf._auth_token = "futures-token"

    # Run refresh - it will cancel itself after one cycle via our fake_sleep
    try:
        await bf._refresh_token()
    except asyncio.CancelledError:
        pass

    init_proxy_system(ProxySettings(enabled=False))

    assert len(calls) >= 1
    assert calls[0][0] == "PUT"
    assert calls[0][1] == "socks5://futures-socks:1080"
    assert calls[0][3] == 12.0


def test_binance_generate_token_timeout_default(monkeypatch):
    """Test that listen-key generation uses default timeout when not configured"""
    calls = []

    async def fake_request(method, url, headers, proxy_url, timeout, data=None):
        calls.append((method, proxy_url, url, timeout))
        return {"listenKey": "timeout-test"}

    monkeypatch.setattr(
        "cryptofeed.exchanges.binance._http_request_with_proxy", fake_request
    )
    monkeypatch.setattr("cryptofeed.exchanges.binance.Binance.symbol_mapping", classmethod(lambda cls, refresh=False, headers=None: {}))

    # Remove any CF_LISTEN_KEY_TIMEOUT env var to test default
    monkeypatch.delenv("CF_LISTEN_KEY_TIMEOUT", raising=False)

    init_proxy_system(ProxySettings(enabled=False))

    Symbols.set("BINANCE", {"BTC-USDT": "BTCUSDT"}, {"tick_size": {}, "instrument_type": {}})
    b = Binance(symbols=[], channels=[])
    key = b._generate_token()

    assert key == "timeout-test"
    # Default timeout should be 10.0 seconds based on ExchangeRuntimeSettings
    assert calls and calls[0][3] == 10.0
