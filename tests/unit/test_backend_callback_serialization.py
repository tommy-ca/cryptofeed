'''
Copyright (C) 2017-2025 Bryant Moscon - bmoscon@gmail.com

Please see the LICENSE file for the terms and conditions
associated with this software.

Tests for BackendCallback serialization integration
'''
import pytest
import json
from decimal import Decimal
from cryptofeed.backends.backend import BackendCallback
from cryptofeed.serializers import JSONSerializer
from cryptofeed.exceptions import SerializationError


class MockDataType:
    """Mock data type for testing."""
    def __init__(self):
        self.symbol = 'BTC-USD'
        self.price = Decimal('50000')
        self.timestamp = 1700000000.0
        self.exchange = 'coinbase'
    
    def to_dict(self, numeric_type=None, none_to=None):
        return {
            'symbol': self.symbol,
            'price': str(self.price),
            'timestamp': self.timestamp,
            'exchange': self.exchange
        }


class TestBackendCallback(BackendCallback):
    """Test implementation of BackendCallback."""
    def __init__(self, serialization_format='json', **kwargs):
        self.serialization_format = serialization_format
        self.numeric_type = kwargs.get('numeric_type', str)
        self.none_to = kwargs.get('none_to', None)
        self.written_data = []
        
        # Get serializer
        self.serializer = self._get_serializer(serialization_format)
    
    def _get_serializer(self, format_name):
        """Factory method for serializer selection."""
        if format_name == 'json':
            return JSONSerializer()
        elif format_name == 'protobuf':
            # Will be implemented in later task
            raise NotImplementedError("Protobuf serializer not yet implemented")
        else:
            raise ValueError(
                f"Invalid serialization format '{format_name}'. "
                f"Valid formats: json, protobuf"
            )
    
    async def write(self, data):
        """Store data for test verification."""
        self.written_data.append(data)


def test_backend_callback_defaults_to_json():
    """BackendCallback defaults to JSON serialization format."""
    callback = TestBackendCallback()
    assert callback.serialization_format == 'json'
    assert isinstance(callback.serializer, JSONSerializer)


def test_backend_callback_accepts_format_parameter():
    """BackendCallback accepts serialization_format parameter."""
    callback = TestBackendCallback(serialization_format='json')
    assert callback.serialization_format == 'json'
    assert isinstance(callback.serializer, JSONSerializer)


def test_backend_callback_rejects_invalid_format():
    """BackendCallback raises ValueError for invalid format."""
    with pytest.raises(ValueError, match="Invalid serialization format 'unknown'"):
        TestBackendCallback(serialization_format='unknown')


@pytest.mark.asyncio
async def test_backend_callback_json_serialization():
    """BackendCallback uses JSONSerializer when format='json'."""
    callback = TestBackendCallback(serialization_format='json')
    data_type = MockDataType()
    
    # Call the callback (simulates feed event)
    await callback(data_type, receipt_timestamp=1700000001.0)
    
    # Verify data was written
    assert len(callback.written_data) == 1
    
    # Data should be dict (existing behavior for backward compat)
    written = callback.written_data[0]
    assert isinstance(written, dict)
    assert written['symbol'] == 'BTC-USD'
    assert 'receipt_timestamp' in written


@pytest.mark.asyncio
async def test_backend_callback_preserves_timestamps():
    """BackendCallback adds receipt_timestamp to data."""
    callback = TestBackendCallback(serialization_format='json')
    data_type = MockDataType()
    receipt_ts = 1700000001.123
    
    await callback(data_type, receipt_timestamp=receipt_ts)
    
    written = callback.written_data[0]
    assert written['receipt_timestamp'] == receipt_ts
    assert written['timestamp'] == 1700000000.0


@pytest.mark.asyncio
async def test_backend_callback_handles_missing_timestamp():
    """BackendCallback sets timestamp if not present."""
    callback = TestBackendCallback(serialization_format='json')
    
    class DataTypeNoTimestamp:
        timestamp = None
        exchange = 'test'
        
        def to_dict(self, numeric_type=None, none_to=None):
            return {'exchange': self.exchange}
    
    data_type = DataTypeNoTimestamp()
    receipt_ts = 1700000001.123
    
    await callback(data_type, receipt_timestamp=receipt_ts)
    
    written = callback.written_data[0]
    assert written['timestamp'] == receipt_ts
    assert written['receipt_timestamp'] == receipt_ts


def test_backend_callback_serializer_selection_logic():
    """Serializer factory method selects correct serializer."""
    callback = TestBackendCallback(serialization_format='json')
    
    # JSON format should return JSONSerializer
    serializer = callback._get_serializer('json')
    assert isinstance(serializer, JSONSerializer)
    
    # Invalid format should raise ValueError
    with pytest.raises(ValueError):
        callback._get_serializer('invalid')
    
    # Protobuf not yet implemented should raise NotImplementedError
    with pytest.raises(NotImplementedError, match="Protobuf serializer not yet implemented"):
        callback._get_serializer('protobuf')
