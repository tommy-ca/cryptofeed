from decimal import Decimal

import pytest

from cryptofeed.backends.protobuf.serialization import serialize_to_protobuf
from cryptofeed.backends.protobuf.validation import SchemaValidator
from cryptofeed.exceptions import ProtobufEncodeError
from cryptofeed.proto_bindings import trade_pb2


class Trade:
    exchange = "bitfinex"
    symbol = "BTC-USD"
    side = "buy"
    price = Decimal("100.5")
    amount = Decimal("2.0")
    timestamp = 1234567890.0
    id = None
    type = None


class ProtoWrapper:
    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail

    def to_proto(self):
        msg = trade_pb2.Trade()
        msg.exchange = "wrapper"
        if self.should_fail:
            raise RuntimeError("boom")
        return msg


def test_serialize_to_protobuf_converter_roundtrip():
    payload = serialize_to_protobuf(Trade())
    trade = trade_pb2.Trade()
    trade.ParseFromString(payload)
    assert trade.exchange == "bitfinex"
    assert trade.symbol == "BTC-USD"
    assert trade.amount == "2.0"


def test_serialize_to_protobuf_to_proto_roundtrip():
    payload = serialize_to_protobuf(ProtoWrapper())
    trade = trade_pb2.Trade()
    trade.ParseFromString(payload)
    assert trade.exchange == "wrapper"


def test_schema_validator_rejects_missing_fields():
    class FakeDescriptor:
        name = "Fake"
        full_name = "fake.Fake"

    class FakeMessage:
        DESCRIPTOR = FakeDescriptor

        def IsInitialized(self):
            return False

        def FindInitializationErrors(self):
            return ["field_a"]

    validator = SchemaValidator()
    with pytest.raises(ProtobufEncodeError):
        validator.validate(FakeMessage())
