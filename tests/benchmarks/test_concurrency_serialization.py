"""Concurrency validation for protobuf serializer."""
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal

from cryptofeed.proto_bindings import trade_pb2
from cryptofeed.serializers import ProtobufSerializer
from cryptofeed.types import Trade


def _make_trade(idx: int) -> Trade:
    return Trade(
        exchange="binance",
        symbol=f"BTC-USDT-{idx % 3}",
        side="buy" if idx % 2 == 0 else "sell",
        amount=Decimal("0.1") + Decimal(idx) / Decimal("1000"),
        price=Decimal("50000.00") + Decimal(idx) / Decimal("10"),
        timestamp=1_700_000_000.0 + idx * 0.0001,
        id=f"trade-{idx}",
    )


def test_protobuf_serializer_thread_safety():
    serializer = ProtobufSerializer()
    trades = [_make_trade(idx) for idx in range(200)]

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(serializer.serialize, trade) for trade in trades]

    serialized_messages = [future.result() for future in futures]

    assert len(serialized_messages) == len(trades)

    for bytes_data, trade in zip(serialized_messages, trades):
        proto = trade_pb2.Trade()
        proto.ParseFromString(bytes_data)
        assert proto.exchange == trade.exchange
        assert proto.symbol == trade.symbol
        assert proto.price == str(trade.price)
        assert proto.amount == str(trade.amount)
        assert proto.timestamp == int(trade.timestamp * 1_000_000)
