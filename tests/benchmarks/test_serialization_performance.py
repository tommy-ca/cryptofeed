"""Performance regression checks for protobuf vs JSON serialization."""
from __future__ import annotations

import time
from decimal import Decimal

from cryptofeed.serializers import JSONSerializer, ProtobufSerializer
from cryptofeed.types import OrderBook, Trade


def _generate_trades(count: int) -> list[Trade]:
    return [
        Trade(
            exchange="coinbase",
            symbol=f"BTC-USD",
            side="buy" if index % 2 == 0 else "sell",
            amount=Decimal("1.0") + Decimal(index) / Decimal("1000"),
            price=Decimal("50000.0") + Decimal(index) / Decimal("10"),
            timestamp=1_700_000_000.0 + index * 0.0001,
        )
        for index in range(count)
    ]


def _generate_orderbooks(count: int) -> list[OrderBook]:
    books: list[OrderBook] = []
    for index in range(count):
        book = OrderBook("coinbase", "BTC-USD")
        for level in range(20):
            price = 50000 + level + index * 0.01
            size = 1 + level * 0.01
            book.book.bids[str(price)] = size
            book.book.asks[str(price + 50)] = size
        books.append(book)
    return books


def _time_serialization(serializer, payloads):
    start = time.perf_counter()
    for item in payloads:
        serializer.serialize(item)
    end = time.perf_counter()
    return (end - start) / len(payloads)


def test_trade_serialization_latency_budget():
    trades = _generate_trades(1000)
    json_serializer = JSONSerializer()
    proto_serializer = ProtobufSerializer()

    json_time = _time_serialization(json_serializer, trades)
    proto_time = _time_serialization(proto_serializer, trades)

    # Absolute latency budget per Requirement 8 (500 microseconds)
    assert proto_time <= 0.0005  # 500 microseconds per Requirement 8


def test_orderbook_serialization_latency_budget():
    books = _generate_orderbooks(200)
    json_serializer = JSONSerializer()
    proto_serializer = ProtobufSerializer()

    json_time = _time_serialization(json_serializer, books)
    proto_time = _time_serialization(proto_serializer, books)

    assert proto_time <= 0.002  # 2ms per Requirement 8
