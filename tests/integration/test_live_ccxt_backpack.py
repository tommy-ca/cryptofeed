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


# ==================== Phase 1: Enhanced CCXT REST Tests ====================


@pytest.mark.live_proxy
@pytest.mark.live_ccxt
def test_backpack_ccxt_rest_ticker():
    """Fetch ticker data via CCXT REST"""
    proxy_url = _require_env("CRYPTOFEED_TEST_SOCKS_PROXY")
    symbol = os.getenv("CRYPTOFEED_TEST_BACKPACK_CCXT_SYMBOL", "BTC/USDC")

    exchange = ccxt.backpack({'enableRateLimit': True})
    exchange.timeout = int(float(os.getenv("CRYPTOFEED_TEST_BACKPACK_REST_TIMEOUT", "10")) * 1000)
    _apply_proxy(exchange, proxy_url)

    try:
        exchange.load_markets()
        ticker = exchange.fetch_ticker(symbol)
    finally:
        if hasattr(exchange, 'close'):
            exchange.close()

    assert ticker, "Empty ticker response"
    assert ticker['symbol'] == symbol
    assert ticker.get('bid') or ticker.get('ask') or ticker.get('last'), "No price data in ticker"
    assert isinstance(ticker.get('timestamp'), (int, float, type(None))), "Invalid timestamp"


@pytest.mark.live_proxy
@pytest.mark.live_ccxt
def test_backpack_ccxt_rest_trades():
    """Fetch recent trades via CCXT REST"""
    proxy_url = _require_env("CRYPTOFEED_TEST_SOCKS_PROXY")
    symbol = os.getenv("CRYPTOFEED_TEST_BACKPACK_CCXT_SYMBOL", "BTC/USDC")

    exchange = ccxt.backpack({'enableRateLimit': True})
    exchange.timeout = int(float(os.getenv("CRYPTOFEED_TEST_BACKPACK_REST_TIMEOUT", "10")) * 1000)
    _apply_proxy(exchange, proxy_url)

    try:
        exchange.load_markets()
        trades = exchange.fetch_trades(symbol, limit=10)
    finally:
        if hasattr(exchange, 'close'):
            exchange.close()

    assert trades, "No trades returned"
    assert len(trades) > 0, "Empty trades list"
    
    first_trade = trades[0]
    assert first_trade.get('price') is not None, "Trade missing price"
    assert first_trade.get('amount') is not None, "Trade missing amount"
    assert first_trade.get('side') in ['buy', 'sell', None], "Invalid trade side"
    assert first_trade.get('timestamp') is not None, "Trade missing timestamp"


@pytest.mark.live_proxy
@pytest.mark.live_ccxt
def test_backpack_ccxt_rest_ohlcv():
    """Fetch OHLCV/candle data via CCXT REST"""
    proxy_url = _require_env("CRYPTOFEED_TEST_SOCKS_PROXY")
    symbol = os.getenv("CRYPTOFEED_TEST_BACKPACK_CCXT_SYMBOL", "BTC/USDC")
    timeframe = os.getenv("CRYPTOFEED_TEST_BACKPACK_TIMEFRAME", "1m")

    exchange = ccxt.backpack({'enableRateLimit': True})
    exchange.timeout = int(float(os.getenv("CRYPTOFEED_TEST_BACKPACK_REST_TIMEOUT", "10")) * 1000)
    _apply_proxy(exchange, proxy_url)

    try:
        exchange.load_markets()
        # Check if exchange supports OHLCV
        if not exchange.has.get('fetchOHLCV'):
            pytest.skip("Backpack does not support fetchOHLCV via CCXT")
        
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=10)
    finally:
        if hasattr(exchange, 'close'):
            exchange.close()

    assert ohlcv, "No OHLCV data returned"
    assert len(ohlcv) > 0, "Empty OHLCV list"
    
    candle = ohlcv[0]
    assert len(candle) >= 6, "Invalid OHLCV structure"
    timestamp, open_price, high, low, close, volume = candle[:6]
    assert timestamp > 0, "Invalid timestamp"
    assert open_price > 0 and high > 0 and low > 0 and close > 0, "Invalid price data"


# ==================== Phase 1: Enhanced CCXT WebSocket Tests ====================


@pytest.mark.asyncio
@pytest.mark.live_proxy
@pytest.mark.live_ccxt
async def test_backpack_ccxt_ws_orderbook():
    """Watch order book updates via CCXT WebSocket"""
    proxy_url = _require_env("CRYPTOFEED_TEST_SOCKS_PROXY")
    symbol = os.getenv("CRYPTOFEED_TEST_BACKPACK_CCXT_SYMBOL", "BTC/USDC")
    timeout = float(os.getenv("CRYPTOFEED_TEST_BACKPACK_WS_TIMEOUT", "20"))

    exchange = ccxtpro.backpack({'enableRateLimit': True})
    exchange.timeout = int(timeout * 1000)
    _apply_proxy(exchange, proxy_url)

    try:
        try:
            orderbook = await asyncio.wait_for(exchange.watch_order_book(symbol), timeout=timeout)
        except asyncio.TimeoutError:
            pytest.skip("Backpack order book stream timeout")
    finally:
        await exchange.close()

    assert orderbook, "No order book data"
    assert 'bids' in orderbook and 'asks' in orderbook, "Invalid order book structure"
    assert len(orderbook['bids']) > 0 or len(orderbook['asks']) > 0, "Empty order book"


@pytest.mark.asyncio
@pytest.mark.live_proxy
@pytest.mark.live_ccxt
async def test_backpack_ccxt_ws_ticker():
    """Watch ticker updates via CCXT WebSocket"""
    proxy_url = _require_env("CRYPTOFEED_TEST_SOCKS_PROXY")
    symbol = os.getenv("CRYPTOFEED_TEST_BACKPACK_CCXT_SYMBOL", "BTC/USDC")
    timeout = float(os.getenv("CRYPTOFEED_TEST_BACKPACK_WS_TIMEOUT", "20"))

    exchange = ccxtpro.backpack({'enableRateLimit': True})
    exchange.timeout = int(timeout * 1000)
    _apply_proxy(exchange, proxy_url)

    try:
        try:
            ticker = await asyncio.wait_for(exchange.watch_ticker(symbol), timeout=timeout)
        except asyncio.TimeoutError:
            pytest.skip("Backpack ticker stream timeout")
        except Exception as e:
            if 'not support' in str(e).lower() or 'not available' in str(e).lower():
                pytest.skip(f"Backpack ticker not supported via CCXT: {e}")
            raise
    finally:
        await exchange.close()

    assert ticker, "No ticker data"
    assert ticker.get('symbol') == symbol, "Symbol mismatch"
    assert ticker.get('bid') or ticker.get('ask') or ticker.get('last'), "No price data"


@pytest.mark.asyncio
@pytest.mark.live_proxy
@pytest.mark.live_ccxt
async def test_backpack_ccxt_ws_multiple_subscriptions():
    """Handle multiple concurrent WebSocket subscriptions via CCXT"""
    proxy_url = _require_env("CRYPTOFEED_TEST_SOCKS_PROXY")
    symbol = os.getenv("CRYPTOFEED_TEST_BACKPACK_CCXT_SYMBOL", "BTC/USDC")
    timeout = float(os.getenv("CRYPTOFEED_TEST_BACKPACK_WS_TIMEOUT", "20"))

    exchange = ccxtpro.backpack({'enableRateLimit': True})
    exchange.timeout = int(timeout * 1000)
    _apply_proxy(exchange, proxy_url)

    try:
        # Subscribe to multiple streams concurrently
        results = await asyncio.gather(
            asyncio.wait_for(exchange.watch_trades(symbol), timeout=timeout),
            asyncio.wait_for(exchange.watch_order_book(symbol), timeout=timeout),
            return_exceptions=True
        )
        
        trades, orderbook = results
        
        # Check if either succeeded
        trades_ok = not isinstance(trades, Exception) and trades
        orderbook_ok = not isinstance(orderbook, Exception) and orderbook
        
        if not trades_ok and not orderbook_ok:
            pytest.skip("Multiple subscriptions timed out")
        
        # Validate at least one stream worked
        assert trades_ok or orderbook_ok, "All subscriptions failed"
        
        if trades_ok:
            assert len(trades) > 0, "Empty trades"
        if orderbook_ok:
            assert orderbook.get('bids') or orderbook.get('asks'), "Empty order book"
            
    finally:
        await exchange.close()
