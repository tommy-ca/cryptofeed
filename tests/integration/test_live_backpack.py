"""Live Backpack connectivity checks via SOCKS proxies.

These tests are opt-in and rely on public Backpack endpoints. They exercise the
native REST and WebSocket clients to ensure SOCKS proxy routing works end to
end. Set `CRYPTOFEED_TEST_SOCKS_PROXY` before running and optionally override
symbols or timeouts via environment variables.
"""

import asyncio
import json
import os

import pytest

from cryptofeed.exchanges.backpack.config import BackpackConfig
from cryptofeed.exchanges.backpack.rest import BackpackRestClient
from cryptofeed.exchanges.backpack.ws import BackpackSubscription, BackpackWsSession
from cryptofeed.proxy import ConnectionProxies, ProxyConfig, ProxySettings, init_proxy_system


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        pytest.skip(f"Set {name} to run live Backpack proxy tests")
    return value


@pytest.mark.asyncio
@pytest.mark.live_proxy
@pytest.mark.live_backpack
async def test_backpack_rest_over_socks_proxy():
    proxy_url = _require_env("CRYPTOFEED_TEST_SOCKS_PROXY")
    timeout = float(os.getenv("CRYPTOFEED_TEST_BACKPACK_REST_TIMEOUT", "10"))

    settings = ProxySettings(
        enabled=True,
        default=ConnectionProxies(http=ProxyConfig(url=proxy_url)),
    )

    init_proxy_system(settings)

    client = BackpackRestClient(BackpackConfig())
    try:
        markets = await asyncio.wait_for(client.fetch_markets(), timeout=timeout)
    except asyncio.TimeoutError:
        pytest.skip("Backpack REST request timed out via proxy")
    finally:
        await client.close()
        init_proxy_system(ProxySettings(enabled=False))

    assert markets, "Backpack markets payload empty"


@pytest.mark.asyncio
@pytest.mark.live_proxy
@pytest.mark.live_backpack
async def test_backpack_trades_websocket_over_socks_proxy():
    proxy_url = _require_env("CRYPTOFEED_TEST_SOCKS_PROXY")
    symbol = os.getenv("CRYPTOFEED_TEST_BACKPACK_SYMBOL", "BTC-USDT")
    native_symbol = symbol.replace('-', '_')
    timeout = float(os.getenv("CRYPTOFEED_TEST_BACKPACK_WS_TIMEOUT", "15"))

    settings = ProxySettings(
        enabled=True,
        default=ConnectionProxies(websocket=ProxyConfig(url=proxy_url)),
    )

    init_proxy_system(settings)

    session = BackpackWsSession(BackpackConfig(), heartbeat_interval=0.0)
    try:
        await session.open()
        await session.subscribe([BackpackSubscription(channel="trades", symbols=[native_symbol])])
        try:
            message = await asyncio.wait_for(session.read(), timeout=timeout)
        except asyncio.TimeoutError:
            pytest.skip("Backpack websocket produced no trades within timeout")
        payload = json.loads(message)
        assert payload.get("stream") == f"trade.{native_symbol}"
        data = payload.get("data") or {}
        assert data.get("e") == "trade"
    finally:
        await session.close()
        init_proxy_system(ProxySettings(enabled=False))
