'''
Copyright (C) 2017-2025 Bryant Moscon - bmoscon@gmail.com

Please see the LICENSE file for the terms and conditions
associated with this software.

Tests for Serialization Exception Classes
'''
import pytest
from cryptofeed.exceptions import (
    CryptofeedSerializationException,
    SerializationError,
    ProtobufEncodeError
)


def test_cryptofeed_serialization_exception_is_base():
    """CryptofeedSerializationException is the base exception."""
    assert issubclass(SerializationError, CryptofeedSerializationException)
    assert issubclass(ProtobufEncodeError, CryptofeedSerializationException)


def test_serialization_error_creation():
    """SerializationError can be created with message."""
    error = SerializationError("Test error message")
    assert str(error) == "Test error message"
    assert isinstance(error, Exception)
    assert isinstance(error, CryptofeedSerializationException)


def test_protobuf_encode_error_creation():
    """ProtobufEncodeError can be created with message."""
    error = ProtobufEncodeError("Protobuf encoding failed")
    assert str(error) == "Protobuf encoding failed"
    assert isinstance(error, Exception)
    assert isinstance(error, CryptofeedSerializationException)


def test_exception_with_cause():
    """Exceptions can be chained with 'from' clause."""
    original = ValueError("Original error")
    
    try:
        try:
            raise original
        except ValueError as e:
            raise SerializationError("Serialization failed") from e
    except SerializationError as se:
        assert se.__cause__ is original
        assert str(se) == "Serialization failed"


def test_exception_inheritance_hierarchy():
    """Verify exception hierarchy allows catching at different levels."""
    error = ProtobufEncodeError("Test")
    
    # Can catch as specific type
    with pytest.raises(ProtobufEncodeError):
        raise error
    
    # Can catch as base serialization exception
    with pytest.raises(CryptofeedSerializationException):
        raise ProtobufEncodeError("Test")
    
    # Can catch as generic Exception
    with pytest.raises(Exception):
        raise ProtobufEncodeError("Test")


def test_serialization_error_with_type_context():
    """SerializationError includes type name in message."""
    type_name = "Trade"
    error = SerializationError(f"{type_name} missing to_proto() method")
    assert "Trade" in str(error)
    assert "to_proto" in str(error)


def test_protobuf_encode_error_with_schema_context():
    """ProtobufEncodeError includes schema information."""
    schema_name = "cryptofeed.normalized.v1.Trade"
    error = ProtobufEncodeError(f"Failed to encode {schema_name}")
    assert "cryptofeed.normalized.v1.Trade" in str(error)
