
from cryptofeed.exchanges.okx import OKX
from cryptofeed.proxy import ProxySettings, ConnectionProxies, ProxyConfig, init_proxy_system
from cryptofeed.symbols import Symbols


def test_okx_get_server_time_uses_proxy_and_timeout(monkeypatch):
    calls = []

    async def fake_fetch(url, proxy_url, timeout, headers=None):
        calls.append((url, proxy_url, timeout))
        return {"data": [{"ts": "1700000000000"}]}

    monkeypatch.setattr("cryptofeed.exchanges.okx._fetch_json_via_proxy", fake_fetch)
    monkeypatch.setattr(OKX, "symbol_mapping", classmethod(lambda cls, refresh=False, headers=None: {}))
    monkeypatch.setenv("CRYPTOFEED_SYMBOL_FETCH_TIMEOUT", "5")

    settings = ProxySettings(
        enabled=True,
        exchanges={"okx": ConnectionProxies(http=ProxyConfig(url="http://okx-proxy:9000"))},
    )
    init_proxy_system(settings)

    Symbols.set("OKX", {"BTC-USDT": "BTC-USDT"}, {"tick_size": {}, "instrument_type": {}})
    okx = OKX(symbols=[], channels=[])
    ts = okx._get_server_time()

    init_proxy_system(ProxySettings(enabled=False))

    assert ts == "1700000000000"
    assert calls and calls[0][1] == "http://okx-proxy:9000"
    assert calls[0][2] == 5.0
