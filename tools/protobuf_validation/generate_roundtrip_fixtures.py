"""Generate golden protobuf fixtures for serialization roundtrip tests."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from cryptofeed.backends.protobuf_helpers import serialize_to_protobuf


FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "protobuf_roundtrip"


@dataclass
class Trade:
    exchange: str
    symbol: str
    side: str
    price: Decimal
    amount: Decimal
    timestamp: float
    id: str
    type: str


@dataclass
class Ticker:
    exchange: str
    symbol: str
    bid: Decimal
    ask: Decimal
    timestamp: float


@dataclass
class Candle:
    exchange: str
    symbol: str
    start: float
    stop: float
    interval: str
    trades: int
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    closed: bool
    timestamp: float


@dataclass
class Funding:
    exchange: str
    symbol: str
    mark_price: Decimal
    rate: Decimal
    predicted_rate: Decimal
    next_funding_time: float
    timestamp: float


@dataclass
class OrderBook:
    exchange: str
    symbol: str
    bids: OrderedDict
    asks: OrderedDict
    timestamp: float
    sequence_number: int
    checksum: str


@dataclass
class Liquidation:
    exchange: str
    symbol: str
    side: str
    quantity: Decimal
    price: Decimal
    id: str
    status: str
    timestamp: float


@dataclass
class OpenInterest:
    exchange: str
    symbol: str
    open_interest: Decimal
    timestamp: float


@dataclass
class Index:
    exchange: str
    symbol: str
    price: Decimal
    timestamp: float


@dataclass
class Balance:
    exchange: str
    currency: str
    balance: Decimal
    reserved: Decimal


@dataclass
class Position:
    exchange: str
    symbol: str
    position: Decimal
    entry_price: Decimal
    side: str
    unrealised_pnl: Decimal
    timestamp: float


@dataclass
class Fill:
    exchange: str
    symbol: str
    side: str
    amount: Decimal
    price: Decimal
    fee: Decimal
    liquidity: str
    id: str
    order_id: str
    type: str
    account: str
    timestamp: float


@dataclass
class OrderInfo:
    exchange: str
    symbol: str
    id: str
    client_order_id: str
    side: str
    status: str
    type: str
    price: Decimal
    amount: Decimal
    remaining: Decimal
    account: str
    timestamp: float


@dataclass
class Order:
    exchange: str
    symbol: str
    client_order_id: str
    side: str
    type: str
    price: Decimal
    amount: Decimal
    account: str
    timestamp: float


@dataclass
class Transaction:
    exchange: str
    currency: str
    type: str
    status: str
    amount: Decimal
    timestamp: float


SAMPLES = (
    ("trade", Trade("coinbase", "BTC-USD", "buy", Decimal("68000.123456"), Decimal("0.25"), 1_700_000_000.123456, "trade-42", "spot")),
    ("ticker", Ticker("kraken", "ETH-USD", Decimal("3125.0001"), Decimal("3126.0002"), 1_700_000_500.5)),
    ("candle", Candle("binance", "BTC-USDT", 1_700_000_000.0, 1_700_000_060.0, "1m", 120, Decimal("68000"), Decimal("68100"), Decimal("67950"), Decimal("68050"), Decimal("125.123"), True, 1_700_000_060.0)),
    ("funding", Funding("bitmex", "XBT-USD", Decimal("68010.1"), Decimal("0.00025"), Decimal("0.00030"), 1_700_000_800.0, 1_700_000_100.0)),
    ("orderbook", OrderBook(
        "coinbase",
        "BTC-USD",
        OrderedDict(((Decimal("67999.99"), Decimal("1.0")), (Decimal("67998.50"), Decimal("0.75")))),
        OrderedDict(((Decimal("68000.25"), Decimal("1.25")), (Decimal("68001.00"), Decimal("0.5")))),
        1_700_000_010.0,
        123456789,
        "abc123",
    )),
    ("liquidation", Liquidation("bybit", "BTC-USDT", "sell", Decimal("2.5"), Decimal("67900.0"), "liq-1", "filled", 1_700_000_200.0)),
    ("openinterest", OpenInterest("deribit", "BTC-PERP", Decimal("1200.1234"), 1_700_000_300.0)),
    ("index", Index("coinalyze", "BTC-USD", Decimal("68020.5"), 1_700_000_400.0)),
    ("balance", Balance("binance", "USDT", Decimal("5000.5"), Decimal("250.25"))),
    ("position", Position("okx", "BTC-USDT", Decimal("1.25"), Decimal("67000"), "long", Decimal("500"), 1_700_000_500.0)),
    ("fill", Fill("kraken", "ETH-USD", "buy", Decimal("1.75"), Decimal("3120.42"), Decimal("1.2"), "maker", "fill-123", "order-123", "limit", "primary", 1_700_000_600.0)),
    ("orderinfo", OrderInfo("kraken", "ETH-USD", "order-1", "client-1", "buy", "open", "limit", Decimal("3100.1"), Decimal("2.0"), Decimal("0.5"), "default", 1_700_000_700.0)),
    ("order", Order("bybit", "BTC-USDT", "client-xyz", "sell", "market", Decimal("67800"), Decimal("0.75"), "primary", 1_700_000_800.0)),
    ("transaction", Transaction("coinbase", "USD", "deposit", "completed", Decimal("1000.00"), 1_700_000_900.0)),
)


def main() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    for name, sample in SAMPLES:
        payload = serialize_to_protobuf(sample)
        path = FIXTURE_DIR / f"{name}.bin"
        path.write_bytes(payload)
        print(f"wrote {path} ({len(payload)} bytes)")


if __name__ == "__main__":
    main()
