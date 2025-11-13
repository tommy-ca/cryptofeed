"""Tests for protobuf serialization error handling."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import pytest

import cryptofeed.backends.protobuf_helpers as helpers
from cryptofeed.backends.protobuf_helpers import serialize_to_protobuf, trade_to_proto
from cryptofeed.exceptions import ProtobufEncodeError, SerializationError


@dataclass
class UnsupportedType:
    exchange: str = "coinbase"
    symbol: str = "BTC-USD"


@dataclass
class FaultyTrade:
    exchange: str = "coinbase"
    symbol: str = "BTC-USD"
    side: str = "buy"
    amount: Decimal = Decimal("0.5")
    timestamp: float = 1_700_000_000.0
    id: str = "faulty"
    type: str = "spot"

    class _FailingPrice:
        def __str__(self):  # pragma: no cover - executed through conversion
            raise RuntimeError("price conversion failed")

    price: object = _FailingPrice()


class CustomProto:
    """Test double that mimics protobuf message interface."""

    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail

    def SerializeToString(self):
        if self.should_fail:
            raise TypeError("serialize failure")
        return b"custom"


class CustomToProtoObject:
    def __init__(self, result):
        self._result = result

    def to_proto(self):
        return self._result


def test_serialize_to_protobuf_unsupported_type_raises_serialization_error():
    with pytest.raises(SerializationError) as exc:
        serialize_to_protobuf(UnsupportedType())

    assert "UnsupportedType" in str(exc.value)


def test_serialize_to_protobuf_converter_exception_wrapped():
    with pytest.raises(ProtobufEncodeError) as exc:
        serialize_to_protobuf(FaultyTrade())

    assert "converter" in str(exc.value).lower()
    assert exc.value.__cause__ is not None


def test_serialize_to_protobuf_to_proto_returns_non_message():
    obj = CustomToProtoObject(result="not-a-message")

    with pytest.raises(ProtobufEncodeError) as exc:
        serialize_to_protobuf(obj)

    assert "expected protobuf Message" in str(exc.value)


def test_serialize_to_protobuf_to_proto_bubbles_exceptions():
    class RaisingToProto:
        def to_proto(self):  # pragma: no cover - executed during serialization
            raise ValueError("boom")

    with pytest.raises(ProtobufEncodeError) as exc:
        serialize_to_protobuf(RaisingToProto())

    assert "to_proto()" in str(exc.value)
    assert exc.value.__cause__ is not None


def test_serialize_to_protobuf_to_proto_serialization_failure():
    obj = CustomToProtoObject(result=CustomProto(should_fail=True))

    with pytest.raises(ProtobufEncodeError) as exc:
        serialize_to_protobuf(obj)

    assert "SerializeToString" in str(exc.value)
    assert exc.value.__cause__ is not None


@pytest.fixture(autouse=True)
def register_faulty_trade_converter():
    helpers._CONVERTER_MAP['FaultyTrade'] = trade_to_proto
    yield
    helpers._CONVERTER_MAP.pop('FaultyTrade', None)
