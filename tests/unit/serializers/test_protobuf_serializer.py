'''
Copyright (C) 2017-2025 Bryant Moscon - bmoscon@gmail.com

Please see the LICENSE file for the terms and conditions
associated with this software.

Tests for ProtobufSerializer
'''
import pytest
from google.protobuf.message import Message
from cryptofeed.serializers.protobuf import ProtobufSerializer
from cryptofeed.exceptions import SerializationError, ProtobufEncodeError
from cryptofeed.proto_bindings import trade_pb2, trade_side_pb2


class MockTradeWithProto:
    """Mock Trade object with to_proto() method."""
    def __init__(self):
        self.symbol = 'BTC-USD'
        self.price = '50000'
        self.amount = '1.5'
        self.exchange = 'coinbase'
        self.timestamp = 1700000000123000
    
    def to_proto(self):
        trade = trade_pb2.Trade()
        trade.symbol = self.symbol
        trade.price = self.price
        trade.amount = self.amount
        trade.exchange = self.exchange
        trade.timestamp = self.timestamp
        trade.side = trade_side_pb2.TRADE_SIDE_BUY
        return trade


class MockTradeWithoutProto:
    """Mock object without to_proto() method."""
    def __init__(self):
        self.symbol = 'BTC-USD'


class MockTradeInvalidReturn:
    """Mock object with to_proto() that returns invalid type."""
    def to_proto(self):
        return "not a protobuf message"


def test_protobuf_serializer_basic():
    """ProtobufSerializer handles objects with to_proto()."""
    trade = MockTradeWithProto()
    serializer = ProtobufSerializer()
    result = serializer.serialize(trade)
    
    assert isinstance(result, bytes)
    assert len(result) > 0
    
    # Verify deserializable
    restored = trade_pb2.Trade()
    restored.ParseFromString(result)
    assert restored.symbol == 'BTC-USD'
    assert restored.price == '50000'
    assert restored.exchange == 'coinbase'


def test_protobuf_serializer_missing_to_proto():
    """ProtobufSerializer raises error for missing to_proto()."""
    obj = MockTradeWithoutProto()
    serializer = ProtobufSerializer()
    
    with pytest.raises(SerializationError, match="missing to_proto"):
        serializer.serialize(obj)


def test_protobuf_serializer_content_type():
    """ProtobufSerializer returns correct MIME type."""
    serializer = ProtobufSerializer()
    assert serializer.content_type() == 'application/x-protobuf'


def test_protobuf_serializer_invalid_return():
    """ProtobufSerializer handles invalid to_proto() return."""
    obj = MockTradeInvalidReturn()
    serializer = ProtobufSerializer()
    
    with pytest.raises(ProtobufEncodeError, match="expected protobuf Message"):
        serializer.serialize(obj)


def test_protobuf_serializer_roundtrip():
    """ProtobufSerializer preserves all data in roundtrip."""
    trade = MockTradeWithProto()
    serializer = ProtobufSerializer()
    
    # Serialize
    bytes_data = serializer.serialize(trade)
    
    # Deserialize
    restored = trade_pb2.Trade()
    restored.ParseFromString(bytes_data)
    
    # Verify all fields
    assert restored.symbol == trade.symbol
    assert restored.price == trade.price
    assert restored.amount == trade.amount
    assert restored.exchange == trade.exchange
    assert restored.timestamp == trade.timestamp


def test_protobuf_serializer_type_safety():
    """ProtobufSerializer verifies return type is protobuf Message."""
    obj = MockTradeWithProto()
    serializer = ProtobufSerializer()
    
    # Get the proto message first
    proto_msg = obj.to_proto()
    
    # Verify it's a Message
    assert isinstance(proto_msg, Message)
    
    # Serialization should succeed
    result = serializer.serialize(obj)
    assert isinstance(result, bytes)


def test_protobuf_serializer_error_message_context():
    """ProtobufSerializer error messages include type context."""
    class CustomType:
        pass
    
    serializer = ProtobufSerializer()
    
    with pytest.raises(SerializationError) as exc_info:
        serializer.serialize(CustomType())
    
    # Error message should include type name
    assert "CustomType" in str(exc_info.value)
    assert "to_proto" in str(exc_info.value)
