"""Live Hyperliquid connectivity tests using ccxt + ccxt.pro over SOCKS proxies."""

import asyncio
import os

import ccxt
import ccxt.pro as ccxtpro
import pytest


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        pytest.fail(f"Set {name} to run live Hyperliquid proxy tests")
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
def test_hyperliquid_ccxt_rest_over_socks_proxy():
    proxy_url = _require_env("CRYPTOFEED_TEST_SOCKS_PROXY")
    symbol = os.getenv("CRYPTOFEED_TEST_CCXT_SYMBOL", "BTC/USDC:USDC")

    exchange = ccxt.hyperliquid({'enableRateLimit': True})
    exchange.timeout = int(float(os.getenv("CRYPTOFEED_TEST_CCXT_REST_TIMEOUT", "15")) * 1000)
    _apply_proxy(exchange, proxy_url)

    markets = exchange.load_markets()
    assert symbol in markets, f"Symbol {symbol} not found in Hyperliquid markets"

    orderbook = exchange.fetch_order_book(symbol, limit=5)
    assert orderbook['bids'] or orderbook['asks'], "Empty order book"
    if hasattr(exchange, 'close'):
        exchange.close()


@pytest.mark.asyncio
@pytest.mark.live_proxy
@pytest.mark.live_ccxt
async def test_hyperliquid_ccxt_ws_over_socks_proxy():
    proxy_url = _require_env("CRYPTOFEED_TEST_SOCKS_PROXY")
    symbol = os.getenv("CRYPTOFEED_TEST_CCXT_SYMBOL", "BTC/USDC:USDC")
    timeout = float(os.getenv("CRYPTOFEED_TEST_CCXT_WS_TIMEOUT", "20"))

    exchange = ccxtpro.hyperliquid({'enableRateLimit': True})
    exchange.timeout = int(timeout * 1000)
    _apply_proxy(exchange, proxy_url)

    try:
        trades = await asyncio.wait_for(exchange.watch_trades(symbol), timeout=timeout)
    finally:
        await exchange.close()

    assert trades, "No trades received"
    last_trade = trades[-1]
    assert last_trade['price'] > 0
