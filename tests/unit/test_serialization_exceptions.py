"""
Unit tests for serialization exception classes.

Tests verify exception hierarchy, error messages, and exception chain preservation
according to requirements R4.5 (Exception Handling and Error Management).
"""
import pytest


class TestCryptofeedSerializationException:
    """Test base exception class for serialization errors."""
    
    def test_base_exception_exists(self):
        """CryptofeedSerializationException base class can be imported."""
        from cryptofeed.exceptions import CryptofeedSerializationException
        
        assert issubclass(CryptofeedSerializationException, Exception)
    
    def test_base_exception_catchable(self):
        """CryptofeedSerializationException can catch subclass exceptions."""
        from cryptofeed.exceptions import (
            CryptofeedSerializationException,
            SerializationError
        )
        
        with pytest.raises(CryptofeedSerializationException):
            raise SerializationError("test error")


class TestSerializationError:
    """Test SerializationError exception class."""
    
    def test_serialization_error_basic(self):
        """SerializationError can be raised with basic message."""
        from cryptofeed.exceptions import SerializationError
        
        with pytest.raises(SerializationError) as exc_info:
            raise SerializationError("test error")
        
        assert "test error" in str(exc_info.value)
    
    def test_serialization_error_with_data_type(self):
        """SerializationError includes data type in message."""
        from cryptofeed.exceptions import SerializationError
        
        error = SerializationError(
            "missing to_proto() method",
            data_type="Trade"
        )
        
        assert "Trade" in str(error)
        assert "missing to_proto()" in str(error)
        assert error.data_type == "Trade"
    
    def test_serialization_error_missing_method_message(self):
        """SerializationError formats missing method message correctly."""
        from cryptofeed.exceptions import SerializationError
        
        type_name = "OrderBook"
        error = SerializationError(
            f"{type_name} missing to_proto() method. "
            f"Ensure all data types implement to_proto().",
            data_type=type_name
        )
        
        error_str = str(error)
        assert "OrderBook" in error_str
        assert "missing to_proto()" in error_str
        assert "Ensure all data types implement to_proto()" in error_str


class TestProtobufEncodeError:
    """Test ProtobufEncodeError exception class."""
    
    def test_protobuf_encode_error_basic(self):
        """ProtobufEncodeError can be raised with basic message."""
        from cryptofeed.exceptions import ProtobufEncodeError
        
        with pytest.raises(ProtobufEncodeError) as exc_info:
            raise ProtobufEncodeError("encoding failed")
        
        assert "encoding failed" in str(exc_info.value)
    
    def test_protobuf_encode_error_with_context(self):
        """ProtobufEncodeError includes full context in message."""
        from cryptofeed.exceptions import ProtobufEncodeError
        
        error = ProtobufEncodeError(
            "encoding failed",
            data_type="Trade",
            schema_name="trade_pb2.Trade",
            schema_version="v0.1.0"
        )
        
        error_str = str(error)
        assert "encoding failed" in error_str
        assert "data_type=Trade" in error_str
        assert "schema=trade_pb2.Trade" in error_str
        assert "version=v0.1.0" in error_str
        
        # Verify attributes set
        assert error.data_type == "Trade"
        assert error.schema_name == "trade_pb2.Trade"
        assert error.schema_version == "v0.1.0"
    
    def test_protobuf_encode_error_invalid_return_type(self):
        """ProtobufEncodeError formats invalid return type message."""
        from cryptofeed.exceptions import ProtobufEncodeError
        
        returned_type = str
        error = ProtobufEncodeError(
            f"to_proto() returned {returned_type}, expected protobuf Message",
            data_type="Trade"
        )
        
        error_str = str(error)
        assert "to_proto() returned" in error_str
        assert "expected protobuf Message" in error_str


class TestExceptionChainPreservation:
    """Test exception chain preservation with 'from' clause."""
    
    def test_exception_chain_with_from_clause(self):
        """Exceptions preserve original cause with 'from' clause."""
        from cryptofeed.exceptions import SerializationError
        
        original = ValueError("original error")
        
        try:
            try:
                raise original
            except ValueError as e:
                raise SerializationError("wrapped error") from e
        except SerializationError as e:
            assert e.__cause__ is original
            # When using 'from', __suppress_context__ is True (correct Python behavior)
            assert e.__suppress_context__ is True
            assert "wrapped error" in str(e)
    
    def test_protobuf_encode_error_chain(self):
        """ProtobufEncodeError preserves exception chain."""
        from cryptofeed.exceptions import ProtobufEncodeError
        
        # Simulate protobuf encoding error
        original = TypeError("protobuf field type mismatch")
        
        try:
            try:
                raise original
            except TypeError as e:
                raise ProtobufEncodeError(
                    "Protobuf encoding failed",
                    data_type="Trade"
                ) from e
        except ProtobufEncodeError as e:
            assert e.__cause__ is original
            # When using 'from', __suppress_context__ is True (correct Python behavior)
            assert e.__suppress_context__ is True
            assert "Protobuf encoding failed" in str(e)


class TestExceptionInheritance:
    """Test exception class inheritance hierarchy."""
    
    def test_serialization_error_inheritance(self):
        """SerializationError inherits from CryptofeedSerializationException."""
        from cryptofeed.exceptions import (
            CryptofeedSerializationException,
            SerializationError
        )
        
        assert issubclass(SerializationError, CryptofeedSerializationException)
        assert issubclass(SerializationError, Exception)
    
    def test_protobuf_encode_error_inheritance(self):
        """ProtobufEncodeError inherits from CryptofeedSerializationException."""
        from cryptofeed.exceptions import (
            CryptofeedSerializationException,
            ProtobufEncodeError
        )
        
        assert issubclass(ProtobufEncodeError, CryptofeedSerializationException)
        assert issubclass(ProtobufEncodeError, Exception)
    
    def test_catch_all_serialization_exceptions(self):
        """CryptofeedSerializationException catches all serialization errors."""
        from cryptofeed.exceptions import (
            CryptofeedSerializationException,
            SerializationError,
            ProtobufEncodeError
        )
        
        # Test SerializationError
        with pytest.raises(CryptofeedSerializationException):
            raise SerializationError("test")
        
        # Test ProtobufEncodeError
        with pytest.raises(CryptofeedSerializationException):
            raise ProtobufEncodeError("test")
