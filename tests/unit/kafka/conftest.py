from __future__ import annotations

from decimal import Decimal
from unittest.mock import Mock

import pytest

from cryptofeed.types import Candle, Ticker, Trade

# Shared Kafka test helpers
from tests.helpers.kafka_env import get_bootstrap_servers


@pytest.fixture
def bootstrap_servers():
    """Resolve bootstrap servers from env/compose defaults for tests."""
    return get_bootstrap_servers()


@pytest.fixture
def trade_message():
    """Create a sample Trade message for testing."""
    return Trade(
        exchange="coinbase",
        symbol="BTC-USD",
        side="buy",
        amount=Decimal("0.25"),
        price=Decimal("68000.10"),
        timestamp=1700000000.0,
        id="trade-1",
        type="spot",
        raw=None,
    )


@pytest.fixture
def trade_binance():
    """Create a Binance trade message."""
    return Trade(
        exchange="binance",
        symbol="BTC-USDT",
        side="sell",
        amount=Decimal("0.5"),
        price=Decimal("68001.00"),
        timestamp=1700000001.0,
        id="trade-binance-1",
        type="spot",
        raw=None,
    )


@pytest.fixture
def ticker_message():
    """Create a sample Ticker message for testing."""
    return Ticker(
        exchange="kraken",
        symbol="ETH-USD",
        bid=Decimal("3500.00"),
        ask=Decimal("3501.00"),
        timestamp=1700000002.0,
        raw=None,
    )


@pytest.fixture
def candle_message():
    """Create a sample Candle message for testing."""
    return Candle(
        exchange="bitmex",
        symbol="XBT-USD",
        start=1700000000.0,
        stop=1700003600.0,
        interval="1h",
        trades=100,
        open=Decimal("68000.00"),
        high=Decimal("68500.00"),
        low=Decimal("67800.00"),
        close=Decimal("68100.00"),
        volume=Decimal("50.5"),
        closed=True,
        timestamp=1700003600.0,
        raw=None,
    )


@pytest.fixture
def orderbook_message():
    """Create a sample OrderBook (L2Book) message for testing."""
    msg = Mock()
    msg.exchange = "dydx"
    msg.symbol = "ETH-USD-PERP"
    msg.timestamp = 1700000003.0
    return msg


@pytest.fixture
def liquidation_message():
    """Create a sample Liquidation message for testing."""
    msg = Mock()
    msg.exchange = "dydx"
    msg.symbol = "BTC-USD-PERP"
    msg.timestamp = 1700000004.0
    return msg


@pytest.fixture
def funding_message():
    """Create a sample Funding message for testing."""
    msg = Mock()
    msg.exchange = "binance-futures"
    msg.symbol = "BTC-USDT-PERP"
    msg.timestamp = 1700000005.0
    return msg


@pytest.fixture
def index_message():
    """Create a sample Index message for testing."""
    msg = Mock()
    msg.exchange = "index-provider"
    msg.symbol = "BTC-USD"
    msg.timestamp = 1700000006.0
    return msg


@pytest.fixture
def openinterest_message():
    """Create a sample OpenInterest message for testing."""
    msg = Mock()
    msg.exchange = "okex"
    msg.symbol = "BTC-USDT"
    msg.timestamp = 1700000007.0
    return msg
