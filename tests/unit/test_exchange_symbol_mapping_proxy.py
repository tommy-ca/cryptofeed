import pytest

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
