import pytest

from cryptofeed.exchanges.binance import Binance
from cryptofeed.proxy import (
    ConnectionProxies,
    ProxyConfig,
    ProxySettings,
    init_proxy_system,
)
from cryptofeed.symbols import Symbols


def test_binance_generate_token_uses_proxy(monkeypatch):
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
