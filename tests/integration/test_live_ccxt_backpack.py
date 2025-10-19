"""Live Backpack connectivity using ccxt + ccxt.pro over SOCKS proxies."""

import asyncio
import os

import ccxt
import ccxt.pro as ccxtpro
import pytest


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        pytest.skip(f"Set {name} to run live Backpack ccxt proxy tests")
    return value


def _apply_proxy(exchange, proxy_url: str) -> None:
    if proxy_url.startswith("socks"):
        if hasattr(exchange, 'socksProxy'):
            exchange.socksProxy = proxy_url
        if hasattr(exchange, 'wsSocksProxy'):
            exchange.wsSocksProxy = proxy_url
    else:
        if hasattr(exchange, 'httpProxy'):
            exchange.httpProxy = proxy_url
        if hasattr(exchange, 'httpsProxy'):
            exchange.httpsProxy = proxy_url
        if hasattr(exchange, 'wsProxy'):
            exchange.wsProxy = proxy_url


@pytest.mark.live_proxy
@pytest.mark.live_ccxt
def test_backpack_ccxt_rest_over_socks_proxy():
    proxy_url = _require_env("CRYPTOFEED_TEST_SOCKS_PROXY")
    symbol = os.getenv("CRYPTOFEED_TEST_BACKPACK_CCXT_SYMBOL", "BTC/USDC")

    exchange = ccxt.backpack({'enableRateLimit': True})
    exchange.timeout = int(float(os.getenv("CRYPTOFEED_TEST_BACKPACK_REST_TIMEOUT", "10")) * 1000)
    _apply_proxy(exchange, proxy_url)

    markets = exchange.load_markets()
    assert symbol in markets, f"Symbol {symbol} not found in Backpack markets"

    orderbook = exchange.fetch_order_book(symbol, limit=5)
    assert orderbook['bids'] or orderbook['asks'], "Empty order book"


@pytest.mark.asyncio
@pytest.mark.live_proxy
@pytest.mark.live_ccxt
async def test_backpack_ccxt_ws_over_socks_proxy():
    proxy_url = _require_env("CRYPTOFEED_TEST_SOCKS_PROXY")
    symbol = os.getenv("CRYPTOFEED_TEST_BACKPACK_CCXT_SYMBOL", "BTC/USDC")
    timeout = float(os.getenv("CRYPTOFEED_TEST_BACKPACK_WS_TIMEOUT", "20"))

    exchange = ccxtpro.backpack({'enableRateLimit': True})
    exchange.timeout = int(timeout * 1000)
    _apply_proxy(exchange, proxy_url)

    try:
        try:
            trades = await asyncio.wait_for(exchange.watch_trades(symbol), timeout=timeout)
        except asyncio.TimeoutError:
            pytest.skip("Backpack ccxt websocket produced no trades within timeout")
    finally:
        await exchange.close()

    assert trades, "No trades received"
    last_trade = trades[-1]
    assert last_trade['price'] > 0
