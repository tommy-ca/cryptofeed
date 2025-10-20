"""Live Binance connectivity checks via SOCKS proxies.

These tests are gated behind pytest markers and environment variables to
avoid running in CI pipelines without explicit opt-in. They verify that the
HTTP proxy system can reach Binance REST endpoints through a real SOCKS5
relay, acknowledging that Binance may geofence specific regions (HTTP 451).
"""

import asyncio
import json
import os

import pytest
from aiohttp import ClientResponseError

from cryptofeed.connection import HTTPAsyncConn
from cryptofeed.proxy import ConnectionProxies, ProxyConfig, ProxySettings, get_proxy_injector, init_proxy_system


BINANCE_TICKER_URL = "https://api.binance.com/api/v3/ticker/price"


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        pytest.skip(f"Set {name} to run live proxy connectivity tests")
    return value


@pytest.mark.asyncio
@pytest.mark.live_proxy
@pytest.mark.live_binance
async def test_binance_ticker_over_socks_proxy():
    proxy_url = _require_env("CRYPTOFEED_TEST_SOCKS_PROXY")
    symbol = os.getenv("CRYPTOFEED_TEST_BINANCE_SYMBOL", "BTCUSDT")

    settings = ProxySettings(
        enabled=True,
        default=ConnectionProxies(http=ProxyConfig(url=proxy_url)),
    )

    init_proxy_system(settings)

    conn = HTTPAsyncConn("binance-live", exchange_id="binance")
    try:
        try:
            payload = await conn.read(BINANCE_TICKER_URL, params={"symbol": symbol})
        except ClientResponseError as exc:
            if exc.status == 451:
                pytest.skip("Binance geofenced this proxy location (HTTP 451)")
            raise

        data = json.loads(payload)
        assert data.get("symbol") == symbol
        price = float(data.get("price", 0))
        assert price > 0
    finally:
        await conn.close()
        init_proxy_system(ProxySettings(enabled=False))


@pytest.mark.asyncio
@pytest.mark.live_proxy
@pytest.mark.live_binance
async def test_binance_trades_websocket_over_socks_proxy():
    proxy_url = _require_env("CRYPTOFEED_TEST_SOCKS_PROXY")
    stream = os.getenv("CRYPTOFEED_TEST_BINANCE_WS_STREAM", "btcusdt@trade")
    timeout = float(os.getenv("CRYPTOFEED_TEST_BINANCE_WS_TIMEOUT", "10"))
    ws_url = f"wss://stream.binance.com/ws/{stream}"

    settings = ProxySettings(
        enabled=True,
        default=ConnectionProxies(websocket=ProxyConfig(url=proxy_url)),
    )

    init_proxy_system(settings)
    injector = get_proxy_injector()

    try:
        try:
            websocket = await injector.create_websocket_connection(ws_url, "binance")
        except Exception as exc:  # handshake or geofence failures
            status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
            if status in (403, 451):
                pytest.skip(f"Binance geofenced this proxy location (HTTP {status})")
            if isinstance(exc, OSError):
                pytest.skip(f"WebSocket handshake failed via proxy: {exc}")
            raise

        try:
            message = await asyncio.wait_for(websocket.recv(), timeout=timeout)
        except asyncio.TimeoutError:
            pytest.skip("No message received from Binance within timeout")
        else:
            payload = json.loads(message)
            # Binance streams may wrap data or send trade event directly
            if "stream" in payload and "data" in payload:
                payload = payload["data"]
            assert payload.get("e") == "trade"
        finally:
            await websocket.close()
    finally:
        init_proxy_system(ProxySettings(enabled=False))
