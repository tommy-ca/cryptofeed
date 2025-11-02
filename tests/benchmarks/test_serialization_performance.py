"""Performance benchmarks for protobuf serialization helpers.

These benchmarks establish baseline latency and payload size
characteristics for the consolidated protobuf helpers introduced
in Spec 1 (protobuf-callback-serialization).
"""

from __future__ import annotations

import gzip
from copy import deepcopy
from decimal import Decimal
from typing import Iterable

import pytest

from cryptofeed.backends.protobuf_helpers import serialize_to_protobuf
from cryptofeed.json_utils import dumps_bytes
from cryptofeed.types import Candle, OrderBook, Trade


def _trade_sample(uid: int) -> Trade:
    """Create a representative trade object."""

    return Trade(
        exchange="COINBASE",
        symbol="BTC-USD",
        side="buy",
        amount=Decimal("0.75"),
        price=Decimal("68000.10"),
        timestamp=1700000000.0 + uid,
        id=f"t-{uid}",
        type="spot",
        raw=None,
    )


def _orderbook_sample() -> OrderBook:
    """Create an order book with multiple price levels."""

    bids = {
        Decimal("67995.5"): Decimal("1.2"),
        Decimal("67994.0"): Decimal("0.8"),
        Decimal("67992.5"): Decimal("2.4"),
    }
    asks = {
        Decimal("68004.5"): Decimal("1.1"),
        Decimal("68006.0"): Decimal("1.7"),
        Decimal("68008.0"): Decimal("0.9"),
    }

    book = OrderBook("COINBASE", "BTC-USD", bids=bids, asks=asks)
    book.timestamp = 1700000000.123456
    book.sequence_number = 42
    book.checksum = "abc123"
    return book


def _candle_sample() -> Candle:
    candle = Candle(
        exchange="BINANCE",
        symbol="ETH-USDT",
        start=1700000000.0,
        stop=1700000060.0,
        interval="1m",
        trades=128,
        open=Decimal("3600.10"),
        high=Decimal("3604.55"),
        low=Decimal("3598.75"),
        close=Decimal("3601.20"),
        volume=Decimal("152.3456"),
        closed=True,
        timestamp=1700000060.0,
    )
    return candle


def _payload_size_bytes(obj) -> int:
    return len(serialize_to_protobuf(obj))


def _json_size_bytes(obj) -> int:
    json_payload = obj.to_dict(numeric_type=str)
    return len(dumps_bytes(json_payload))


def _gzip_size(data: bytes) -> int:
    return len(gzip.compress(data))


@pytest.mark.benchmark(group="protobuf-latency")
def test_trade_serialization_latency(benchmark):
    sample = _trade_sample(1)
    payload = benchmark(serialize_to_protobuf, sample)
    assert isinstance(payload, bytes)
    assert payload  # non-empty payload


@pytest.mark.benchmark(group="protobuf-latency")
def test_orderbook_serialization_latency(benchmark):
    sample = _orderbook_sample()
    payload = benchmark(serialize_to_protobuf, sample)
    assert isinstance(payload, bytes)
    assert payload


@pytest.mark.benchmark(group="protobuf-throughput")
def test_trade_serialization_batch_throughput(benchmark):
    trades: list[Trade] = [_trade_sample(i) for i in range(100)]

    def _serialize_batch(batch: Iterable[Trade]) -> None:
        for trade in batch:
            serialize_to_protobuf(trade)

    benchmark(_serialize_batch, trades)


def test_trade_payload_smaller_than_json():
    sample = _trade_sample(999)
    proto_size = _payload_size_bytes(sample)
    json_size = _json_size_bytes(sample)

    assert proto_size < json_size


def test_orderbook_payload_compresses_better_than_json():
    sample = _orderbook_sample()

    proto_bytes = serialize_to_protobuf(sample)
    json_bytes = dumps_bytes(sample.to_dict(numeric_type=str))

    proto_compressed = _gzip_size(proto_bytes)
    json_compressed = _gzip_size(json_bytes)

    assert proto_compressed <= json_compressed


def test_candle_serialization_roundtrip_size_advantage():
    sample = _candle_sample()

    proto_size = _payload_size_bytes(sample)
    json_size = _json_size_bytes(sample)

    assert proto_size < json_size

