"""Unit tests for Fill protobuf wrapper."""
from decimal import Decimal

import pytest

from cryptofeed.proto_bindings import fill_pb2, trade_side_pb2
from cryptofeed.proto_wrappers.fill import fill_to_proto
from cryptofeed.types import Fill


def make_fill(**overrides) -> Fill:
    """Helper to build Fill objects with sensible defaults."""

    params = {
        "exchange": overrides.get("exchange", "binance"),
        "symbol": overrides.get("symbol", "BTC-USDT"),
        "side": overrides.get("side", "buy"),
        "amount": overrides.get("amount", Decimal("1.0")),
        "price": overrides.get("price", Decimal("50000.00")),
        "fee": overrides.get("fee", Decimal("0.0005")),
        "id": overrides.get("id", "fill-1"),
        "order_id": overrides.get("order_id", "order-1"),
        "type": overrides.get("type", "limit"),
        "liquidity": overrides.get("liquidity", "maker"),
        "timestamp": overrides.get("timestamp", 1_700_000_000.123456),
        "account": overrides.get("account", None),
    }

    return Fill(
        params["exchange"],
        params["symbol"],
        params["side"],
        params["amount"],
        params["price"],
        params["fee"],
        params["id"],
        params["order_id"],
        params["type"],
        params["liquidity"],
        params["timestamp"],
        params["account"],
    )


def test_fill_to_proto_unknown_side_sets_unspecified():
    fill_obj = make_fill(side="hold", fee=None, liquidity="", account=None)

    proto = fill_to_proto(fill_obj)

    assert proto.side == trade_side_pb2.TRADE_SIDE_UNSPECIFIED
    # Optional string fields stay empty when source value falsy
    assert proto.liquidity == ""
    assert proto.account == ""
    # Fee remains unset when None
    assert proto.fee == ""


def test_fill_to_proto_optional_fields_and_account():
    fill_obj = make_fill(
        side="sell",
        liquidity="taker",
        account="margin",
        fee=Decimal("0.001"),
    )

    proto = fill_to_proto(fill_obj)

    assert proto.side == trade_side_pb2.TRADE_SIDE_SELL
    assert proto.liquidity == "taker"
    assert proto.account == "margin"
    assert proto.fee == "0.001"
    assert proto.fill_id == "fill-1"
    assert proto.order_id == "order-1"
    assert proto.type == "limit"


def test_fill_to_proto_timestamp_conversion_microseconds():
    ts_seconds = 1_700_000_000.987654
    fill_obj = make_fill(timestamp=ts_seconds)

    proto = fill_to_proto(fill_obj)

    expected_microseconds = int(ts_seconds * 1_000_000)
    assert proto.timestamp == expected_microseconds
    assert proto.price == "50000.00"
    assert proto.amount == "1.0"
