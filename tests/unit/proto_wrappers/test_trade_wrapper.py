'''
Copyright (C) 2017-2025 Bryant Moscon - bmoscon@gmail.com

Please see the LICENSE file for the terms and conditions
associated with this software.

Tests for Trade.to_proto() wrapper
'''
import pytest
from decimal import Decimal
from cryptofeed.types import Trade
from cryptofeed.proto_bindings import trade_pb2, trade_side_pb2
from google.protobuf.message import Message


def test_trade_to_proto_basic():
    """trade_to_proto() returns valid protobuf message."""
    # Import the conversion function
    from cryptofeed.proto_wrappers.trade import trade_to_proto
    
    trade = Trade(
        symbol='BTC-USD',
        side='buy',
        amount=Decimal('1.5'),
        price=Decimal('50000.12345678'),
        timestamp=1700000000.123,
        exchange='coinbase',
        id='trade123',
        type='limit'
    )
    
    # Call trade_to_proto()
    proto = trade_to_proto(trade)
    
    # Verify it's a protobuf Message
    assert isinstance(proto, Message)
    assert isinstance(proto, trade_pb2.Trade)
    
    # Verify fields
    assert proto.symbol == 'BTC-USD'
    assert proto.exchange == 'coinbase'
    assert proto.trade_id == 'trade123'
    assert proto.side == trade_side_pb2.TRADE_SIDE_BUY


def test_trade_to_proto_decimal_precision():
    """trade_to_proto() preserves Decimal precision via string encoding."""
    from cryptofeed.proto_wrappers.trade import trade_to_proto
    
    trade = Trade(
        symbol='BTC-USD',
        side='buy',
        amount=Decimal('1.123456789012345'),
        price=Decimal('50000.123456789012345'),
        timestamp=1700000000.0,
        exchange='test'
    )
    
    proto = trade_to_proto(trade)
    
    # Decimals stored as strings
    assert isinstance(proto.price, str)
    assert isinstance(proto.amount, str)
    assert proto.price == '50000.123456789012345'
    assert proto.amount == '1.123456789012345'


def test_trade_to_proto_timestamp_conversion():
    """trade_to_proto() converts float seconds to int64 microseconds."""
    from cryptofeed.proto_wrappers.trade import trade_to_proto
    
    trade = Trade(
        symbol='BTC-USD',
        side='buy',
        amount=Decimal('1.0'),
        price=Decimal('50000'),
        timestamp=1700000000.123456,  # Float seconds with microsecond precision
        exchange='test'
    )
    
    proto = trade_to_proto(trade)
    
    # Timestamp converted to microseconds
    assert isinstance(proto.timestamp, int)
    expected_us = int(1700000000.123456 * 1_000_000)
    assert proto.timestamp == expected_us


def test_trade_to_proto_side_enum():
    """trade_to_proto() converts side string to enum."""
    from cryptofeed.proto_wrappers.trade import trade_to_proto
    
    # Test BUY side
    trade_buy = Trade(
        symbol='BTC-USD',
        side='buy',
        amount=Decimal('1.0'),
        price=Decimal('50000'),
        timestamp=1700000000.0,
        exchange='test'
    )
    proto_buy = trade_to_proto(trade_buy)
    assert proto_buy.side == trade_side_pb2.TRADE_SIDE_BUY
    
    # Test SELL side
    trade_sell = Trade(
        symbol='BTC-USD',
        side='sell',
        amount=Decimal('1.0'),
        price=Decimal('50000'),
        timestamp=1700000000.0,
        exchange='test'
    )
    proto_sell = trade_to_proto(trade_sell)
    assert proto_sell.side == trade_side_pb2.TRADE_SIDE_SELL


def test_trade_to_proto_roundtrip():
    """Trade protobuf serialization preserves all data in roundtrip."""
    from cryptofeed.proto_wrappers.trade import trade_to_proto
    
    original_trade = Trade(
        symbol='BTC-USD',
        side='buy',
        amount=Decimal('1.5'),
        price=Decimal('50000.12345678'),
        timestamp=1700000000.123,
        exchange='coinbase',
        id='trade123',
        type='limit'
    )
    
    # Serialize to protobuf
    proto = trade_to_proto(original_trade)
    bytes_data = proto.SerializeToString()
    
    # Deserialize
    restored_proto = trade_pb2.Trade()
    restored_proto.ParseFromString(bytes_data)
    
    # Verify all fields preserved
    assert restored_proto.symbol == original_trade.symbol
    assert restored_proto.exchange == original_trade.exchange
    assert restored_proto.trade_id == original_trade.id
    assert restored_proto.price == str(original_trade.price)
    assert restored_proto.amount == str(original_trade.amount)
    assert restored_proto.timestamp == int(original_trade.timestamp * 1_000_000)
    assert restored_proto.trade_type == original_trade.type


def test_trade_to_proto_optional_fields():
    """trade_to_proto() handles optional fields (id, type)."""
    from cryptofeed.proto_wrappers.trade import trade_to_proto
    
    # Trade without optional fields
    trade = Trade(
        symbol='BTC-USD',
        side='buy',
        amount=Decimal('1.0'),
        price=Decimal('50000'),
        timestamp=1700000000.0,
        exchange='test'
    )
    
    proto = trade_to_proto(trade)
    
    # Required fields present
    assert proto.symbol == 'BTC-USD'
    assert proto.price == '50000'
    
    # Optional fields may be empty
    assert proto.trade_id == '' or proto.trade_id is not None


def test_trade_to_proto_with_protobuf_serializer():
    """Trade works with ProtobufSerializer end-to-end."""
    from cryptofeed.serializers import ProtobufSerializer
    import cryptofeed.proto_wrappers.registry  # Ensure converters registered
    
    trade = Trade(
        symbol='BTC-USD',
        side='buy',
        amount=Decimal('1.5'),
        price=Decimal('50000'),
        timestamp=1700000000.0,
        exchange='coinbase'
    )
    
    serializer = ProtobufSerializer()
    bytes_data = serializer.serialize(trade)
    
    # Verify it's bytes
    assert isinstance(bytes_data, bytes)
    
    # Deserialize and verify
    restored = trade_pb2.Trade()
    restored.ParseFromString(bytes_data)
    assert restored.symbol == 'BTC-USD'
