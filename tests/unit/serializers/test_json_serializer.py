'''
Copyright (C) 2017-2025 Bryant Moscon - bmoscon@gmail.com

Please see the LICENSE file for the terms and conditions
associated with this software.

Tests for JSONSerializer
'''
import json
import pytest
from decimal import Decimal
from cryptofeed.serializers.json import JSONSerializer
from cryptofeed.exceptions import SerializationError


class MockTrade:
    """Mock Trade object for testing."""
    def __init__(self):
        self.symbol = 'BTC-USD'
        self.price = Decimal('50000.12345678')
        self.amount = Decimal('1.5')
        self.timestamp = 1700000000.123
        self.side = 'buy'
        self.exchange = 'coinbase'
    
    def to_dict(self):
        return {
            'symbol': self.symbol,
            'price': str(self.price),
            'amount': str(self.amount),
            'timestamp': self.timestamp,
            'side': self.side,
            'exchange': self.exchange
        }


def test_json_serializer_basic():
    """JSONSerializer correctly serializes objects with to_dict()."""
    trade = MockTrade()
    serializer = JSONSerializer()
    result = serializer.serialize(trade)
    
    assert isinstance(result, bytes)
    obj = json.loads(result)
    assert obj['symbol'] == 'BTC-USD'
    assert obj['price'] == '50000.12345678'
    assert obj['amount'] == '1.5'
    assert obj['exchange'] == 'coinbase'


def test_json_serializer_decimal_precision():
    """Decimal precision preserved via string encoding."""
    trade = MockTrade()
    trade.price = Decimal('123.456789012345')
    
    serializer = JSONSerializer()
    result = serializer.serialize(trade)
    obj = json.loads(result)
    
    # Decimal preserved as string
    assert obj['price'] == '123.456789012345'


def test_json_serializer_content_type():
    """JSONSerializer returns correct MIME type."""
    serializer = JSONSerializer()
    assert serializer.content_type() == 'application/json'


def test_json_serializer_missing_to_dict():
    """JSONSerializer raises error for objects without to_dict()."""
    class BadObject:
        pass
    
    serializer = JSONSerializer()
    
    with pytest.raises(SerializationError, match="missing to_dict"):
        serializer.serialize(BadObject())


def test_json_serializer_roundtrip():
    """JSON serialization can be deserialized back."""
    trade = MockTrade()
    serializer = JSONSerializer()
    
    # Serialize
    bytes_data = serializer.serialize(trade)
    
    # Deserialize
    restored = json.loads(bytes_data)
    
    # Verify
    assert restored['symbol'] == trade.symbol
    assert restored['price'] == str(trade.price)
    assert restored['timestamp'] == trade.timestamp


def test_json_serializer_with_none_values():
    """JSONSerializer handles None values correctly."""
    class TradeWithNone:
        def to_dict(self):
            return {
                'symbol': 'BTC-USD',
                'price': None,
                'amount': '1.5'
            }
    
    serializer = JSONSerializer()
    result = serializer.serialize(TradeWithNone())
    obj = json.loads(result)
    
    assert obj['price'] is None
    assert obj['amount'] == '1.5'


def test_json_serializer_utf8_encoding():
    """JSONSerializer produces UTF-8 encoded bytes."""
    trade = MockTrade()
    serializer = JSONSerializer()
    result = serializer.serialize(trade)
    
    # Verify it's bytes
    assert isinstance(result, bytes)
    
    # Verify it's valid UTF-8
    decoded = result.decode('utf-8')
    assert isinstance(decoded, str)
    
    # Verify it's valid JSON
    obj = json.loads(decoded)
    assert obj['symbol'] == 'BTC-USD'
