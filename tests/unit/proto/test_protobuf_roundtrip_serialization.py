"""Roundtrip tests for protobuf serialization helpers."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import pytest

from cryptofeed.backends.protobuf_helpers import serialize_to_protobuf
from cryptofeed.proto_bindings import (
    balance_pb2,
    candle_pb2,
    fill_pb2,
    funding_pb2,
    index_price_pb2,
    liquidation_pb2,
    open_interest_pb2,
    order_book_pb2,
    order_info_pb2,
    order_pb2,
    position_pb2,
    ticker_pb2,
    trade_pb2,
    transaction_pb2,
)


FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "protobuf_roundtrip"


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


def _fixture_bytes(name: str) -> bytes:
    path = FIXTURE_DIR / f"{name}.bin"
    return path.read_bytes()


def _expected_timestamp(sec: float) -> int:
    return int(sec * 1_000_000)


ROUNDTRIP_CASES = [
    (
        "Trade",
        Trade(
            exchange="coinbase",
            symbol="BTC-USD",
            side="buy",
            price=Decimal("68000.123456"),
            amount=Decimal("0.25"),
            timestamp=1_700_000_000.123456,
            id="trade-42",
            type="spot",
        ),
        trade_pb2.Trade,
        {
            "exchange": "coinbase",
            "symbol": "BTC-USD",
            "price": "68000.123456",
            "amount": "0.25",
            "timestamp": _expected_timestamp(1_700_000_000.123456),
            "trade_type": "spot",
        },
    ),
    (
        "Ticker",
        Ticker(
            exchange="kraken",
            symbol="ETH-USD",
            bid=Decimal("3125.0001"),
            ask=Decimal("3126.0002"),
            timestamp=1_700_000_500.5,
        ),
        ticker_pb2.Ticker,
        {
            "bid": "3125.0001",
            "ask": "3126.0002",
            "timestamp": _expected_timestamp(1_700_000_500.5),
        },
    ),
    (
        "Candle",
        Candle(
            exchange="binance",
            symbol="BTC-USDT",
            start=1_700_000_000.0,
            stop=1_700_000_060.0,
            interval="1m",
            trades=120,
            open=Decimal("68000"),
            high=Decimal("68100"),
            low=Decimal("67950"),
            close=Decimal("68050"),
            volume=Decimal("125.123"),
            closed=True,
            timestamp=1_700_000_060.0,
        ),
        candle_pb2.Candle,
        {
            "open": "68000",
            "close": "68050",
            "volume": "125.123",
            "trades": 120,
            "timestamp": _expected_timestamp(1_700_000_060.0),
        },
    ),
    (
        "Funding",
        Funding(
            exchange="bitmex",
            symbol="XBT-USD",
            mark_price=Decimal("68010.1"),
            rate=Decimal("0.00025"),
            predicted_rate=Decimal("0.00030"),
            next_funding_time=1_700_000_800.0,
            timestamp=1_700_000_100.0,
        ),
        funding_pb2.Funding,
        {
            "mark_price": "68010.1",
            "rate": "0.00025",
            "predicted_rate": "0.00030",
            "next_funding_time": _expected_timestamp(1_700_000_800.0),
        },
    ),
    (
        "OrderBook",
        OrderBook(
            exchange="coinbase",
            symbol="BTC-USD",
            bids=OrderedDict(
                [
                    (Decimal("67999.99"), Decimal("1.0")),
                    (Decimal("67998.50"), Decimal("0.75")),
                ]
            ),
            asks=OrderedDict(
                [
                    (Decimal("68000.25"), Decimal("1.25")),
                    (Decimal("68001.00"), Decimal("0.5")),
                ]
            ),
            timestamp=1_700_000_010.0,
            sequence_number=123456789,
            checksum="abc123",
        ),
        order_book_pb2.Level2Book,
        {
            "bids": [
                ("67999.99", "1.0"),
                ("67998.50", "0.75"),
            ],
            "asks": [
                ("68000.25", "1.25"),
                ("68001.00", "0.5"),
            ],
            "timestamp": _expected_timestamp(1_700_000_010.0),
            "sequence": 123456789,
            "checksum": "abc123",
        },
    ),
    (
        "Liquidation",
        Liquidation(
            exchange="bybit",
            symbol="BTC-USDT",
            side="sell",
            quantity=Decimal("2.5"),
            price=Decimal("67900.0"),
            id="liq-1",
            status="filled",
            timestamp=1_700_000_200.0,
        ),
        liquidation_pb2.Liquidation,
        {
            "quantity": "2.5",
            "price": "67900.0",
            "timestamp": _expected_timestamp(1_700_000_200.0),
        },
    ),
    (
        "OpenInterest",
        OpenInterest(
            exchange="deribit",
            symbol="BTC-PERP",
            open_interest=Decimal("1200.1234"),
            timestamp=1_700_000_300.0,
        ),
        open_interest_pb2.OpenInterest,
        {
            "open_interest": "1200.1234",
            "timestamp": _expected_timestamp(1_700_000_300.0),
        },
    ),
    (
        "Index",
        Index(
            exchange="coinalyze",
            symbol="BTC-USD",
            price=Decimal("68020.5"),
            timestamp=1_700_000_400.0,
        ),
        index_price_pb2.IndexPrice,
        {
            "price": "68020.5",
            "timestamp": _expected_timestamp(1_700_000_400.0),
        },
    ),
    (
        "Balance",
        Balance(
            exchange="binance",
            currency="USDT",
            balance=Decimal("5000.5"),
            reserved=Decimal("250.25"),
        ),
        balance_pb2.Balance,
        {
            "currency": "USDT",
            "balance": "5000.5",
            "reserved": "250.25",
        },
    ),
    (
        "Position",
        Position(
            exchange="okx",
            symbol="BTC-USDT",
            position=Decimal("1.25"),
            entry_price=Decimal("67000"),
            side="long",
            unrealised_pnl=Decimal("500"),
            timestamp=1_700_000_500.0,
        ),
        position_pb2.Position,
        {
            "position": "1.25",
            "entry_price": "67000",
            "timestamp": _expected_timestamp(1_700_000_500.0),
        },
    ),
    (
        "Fill",
        Fill(
            exchange="kraken",
            symbol="ETH-USD",
            side="buy",
            amount=Decimal("1.75"),
            price=Decimal("3120.42"),
            fee=Decimal("1.2"),
            liquidity="maker",
            id="fill-123",
            order_id="order-123",
            type="limit",
            account="primary",
            timestamp=1_700_000_600.0,
        ),
        fill_pb2.Fill,
        {
            "amount": "1.75",
            "price": "3120.42",
            "fee": "1.2",
            "timestamp": _expected_timestamp(1_700_000_600.0),
        },
    ),
    (
        "OrderInfo",
        OrderInfo(
            exchange="kraken",
            symbol="ETH-USD",
            id="order-1",
            client_order_id="client-1",
            side="buy",
            status="open",
            type="limit",
            price=Decimal("3100.1"),
            amount=Decimal("2.0"),
            remaining=Decimal("0.5"),
            account="default",
            timestamp=1_700_000_700.0,
        ),
        order_info_pb2.OrderInfo,
        {
            "price": "3100.1",
            "remaining": "0.5",
            "timestamp": _expected_timestamp(1_700_000_700.0),
        },
    ),
    (
        "Order",
        Order(
            exchange="bybit",
            symbol="BTC-USDT",
            client_order_id="client-xyz",
            side="sell",
            type="market",
            price=Decimal("67800"),
            amount=Decimal("0.75"),
            account="primary",
            timestamp=1_700_000_800.0,
        ),
        order_pb2.Order,
        {
            "price": "67800",
            "amount": "0.75",
            "timestamp": _expected_timestamp(1_700_000_800.0),
        },
    ),
    (
        "Transaction",
        Transaction(
            exchange="coinbase",
            currency="USD",
            type="deposit",
            status="completed",
            amount=Decimal("1000.00"),
            timestamp=1_700_000_900.0,
        ),
        transaction_pb2.Transaction,
        {
            "currency": "USD",
            "amount": "1000.00",
            "timestamp": _expected_timestamp(1_700_000_900.0),
        },
    ),
]


@pytest.mark.parametrize("name, obj, proto_cls, expectations", ROUNDTRIP_CASES)
def test_protobuf_roundtrip_matches_fixture(name, obj, proto_cls, expectations):
    payload = serialize_to_protobuf(obj)
    assert payload == _fixture_bytes(name.lower())

    message = proto_cls()
    message.ParseFromString(payload)

    for field, expected in expectations.items():
        value = getattr(message, field)
        if isinstance(expected, list):
            assert len(value) == len(expected)
            for idx, (exp_price, exp_amount) in enumerate(expected):
                level = value[idx]
                assert level.price == exp_price
                assert level.quantity == exp_amount
        else:
            assert value == expected
