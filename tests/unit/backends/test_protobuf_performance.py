"""Lightweight performance check for protobuf serialization."""

from decimal import Decimal

from cryptofeed.backends.protobuf.serialization import serialize_to_protobuf


class Trade:
    exchange = "perf"
    symbol = "BTC-USD"
    side = "buy"
    price = Decimal("100")
    amount = Decimal("1")
    timestamp = 1700000000.0
    id = "t-perf"
    type = "spot"


def test_serialize_trade_benchmark(benchmark):
    trade = Trade()

    def _do():
        return serialize_to_protobuf(trade)

    payload = benchmark(_do)
    assert isinstance(payload, (bytes, bytearray))

