"""Regression matrix for protobuf serialization across supported data types."""

from decimal import Decimal

import pytest

from cryptofeed.backends.protobuf.serialization import serialize_to_protobuf, _SCHEMA_CLASS_MAP


def _make(cls_name, **fields):
    return type(cls_name, (), fields)()


OBJECTS = {
    "Trade": lambda: _make(
        "Trade",
        exchange="ex",
        symbol="BTC-USD",
        side="buy",
        price=Decimal("100"),
        amount=Decimal("1"),
        timestamp=1700000000.0,
        id="t1",
        type="spot",
    ),
    "Ticker": lambda: _make(
        "Ticker",
        exchange="ex",
        symbol="ETH-USD",
        bid=Decimal("10"),
        ask=Decimal("11"),
        timestamp=1700000000.0,
    ),
    "Candle": lambda: _make(
        "Candle",
        exchange="ex",
        symbol="BTC-USD",
        start=1700000000.0,
        stop=1700000600.0,
        interval="1m",
        trades=5,
        open=Decimal("100"),
        high=Decimal("101"),
        low=Decimal("99"),
        close=Decimal("100.5"),
        volume=Decimal("10"),
        closed=True,
        timestamp=1700000600.0,
    ),
    "Funding": lambda: _make(
        "Funding",
        exchange="ex",
        symbol="BTC-PERP",
        rate=Decimal("0.0001"),
        rate_daily=Decimal("0.001"),
        next_funding_rate=Decimal("0.0002"),
        timestamp=1700000000.0,
    ),
    "OrderBook": lambda: _make(
        "OrderBook",
        exchange="ex",
        symbol="BTC-USD",
        bids={Decimal("100"): Decimal("1")},
        asks={Decimal("101"): Decimal("1")},
        delta=False,
        timestamp=1700000000.0,
    ),
    "Liquidation": lambda: _make(
        "Liquidation",
        exchange="ex",
        symbol="BTC-USD",
        side="sell",
        price=Decimal("99"),
        amount=Decimal("0.5"),
        order_type="market",
        liquidation_type="partial",
        timestamp=1700000000.0,
    ),
    "OpenInterest": lambda: _make(
        "OpenInterest",
        exchange="ex",
        symbol="BTC-USD",
        open_interest=Decimal("1234"),
        timestamp=1700000000.0,
    ),
    "Index": lambda: _make(
        "Index",
        exchange="ex",
        symbol="BTC-USD",
        price=Decimal("100.5"),
        timestamp=1700000000.0,
    ),
    "Balance": lambda: _make(
        "Balance",
        exchange="ex",
        symbol="USD",
        currency="USD",
        balance=Decimal("1000"),
        reserved=Decimal("100"),
        timestamp=1700000000.0,
    ),
    "Position": lambda: _make(
        "Position",
        exchange="ex",
        symbol="BTC-USD",
        asset="BTC",
        qty=Decimal("0.1"),
        entry_price=Decimal("9000"),
        pnl=Decimal("10"),
        leverage=Decimal("3"),
        liquidation_price=Decimal("7000"),
        timestamp=1700000000.0,
    ),
    "Fill": lambda: _make(
        "Fill",
        exchange="ex",
        symbol="BTC-USD",
        side="buy",
        price=Decimal("100"),
        amount=Decimal("1.1"),
        order_id="oid",
        fee=Decimal("0.001"),
        liquidity="taker",
        timestamp=1700000000.0,
    ),
    "OrderInfo": lambda: _make(
        "OrderInfo",
        exchange="ex",
        symbol="BTC-USD",
        order_id="oid",
        status="open",
        order_type="limit",
        side="buy",
        price=Decimal("100"),
        amount=Decimal("1"),
        filled=Decimal("0.5"),
        timestamp=1700000000.0,
    ),
    "Order": lambda: _make(
        "Order",
        exchange="ex",
        symbol="BTC-USD",
        order_id="oid",
        status="open",
        order_type="limit",
        side="buy",
        price=Decimal("100"),
        amount=Decimal("1"),
        filled=Decimal("0.5"),
        timestamp=1700000000.0,
    ),
    "Transaction": lambda: _make(
        "Transaction",
        exchange="ex",
        symbol="USD",
        currency="USD",
        type="deposit",
        status="completed",
        amount=Decimal("50"),
        timestamp=1700000000.0,
    ),
}


@pytest.mark.parametrize("type_name", sorted(OBJECTS.keys()))
def test_serialize_all_types_roundtrip(type_name):
    obj = OBJECTS[type_name]()
    payload = serialize_to_protobuf(obj)
    assert isinstance(payload, (bytes, bytearray))

    schema_cls = _SCHEMA_CLASS_MAP[type_name]
    message = schema_cls()
    message.ParseFromString(payload)
    assert message.exchange == obj.exchange
    # symbol normalization when present
    if hasattr(message, "symbol"):
        assert message.symbol.replace("_", "-") == obj.symbol.replace("_", "-")
