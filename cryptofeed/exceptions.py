'''
Copyright (C) 2017-2025 Bryant Moscon - bmoscon@gmail.com

Please see the LICENSE file for the terms and conditions
associated with this software.
'''


class MissingSequenceNumber(Exception):
    pass


class MissingMessage(Exception):
    pass


class UnsupportedSymbol(Exception):
    pass


class UnsupportedDataFeed(Exception):
    pass


class UnsupportedTradingOption(Exception):
    pass


class UnsupportedType(Exception):
    pass


class ExhaustedRetries(Exception):
    pass


class BidAskOverlapping(Exception):
    pass


class BadChecksum(Exception):
    pass


class RestResponseError(Exception):
    pass


class ConnectionClosed(Exception):
    pass


class UnexpectedMessage(Exception):
    pass


# Serialization Exceptions


class CryptofeedSerializationException(Exception):
    """Base exception for all serialization-related errors.
    
    Allows catching all serialization errors with a single except clause.
    """
    pass


class SerializationError(CryptofeedSerializationException):
    """Raised when serialization fails due to missing methods or invalid data.
    
    Attributes:
        data_type (str): Name of the data type that failed serialization
    """
    
    def __init__(self, message: str, data_type: str = None):
        """Initialize SerializationError with context.
        
        Args:
            message: Error message describing the serialization failure
            data_type: Optional data type name for context
        """
        self.data_type = data_type
        if data_type:
            message = f"{data_type}: {message}"
        super().__init__(message)


class ProtobufEncodeError(CryptofeedSerializationException):
    """Raised when protobuf encoding fails.
    
    Attributes:
        data_type (str): Name of the data type being encoded
        schema_name (str): Protobuf schema name (e.g., 'trade_pb2.Trade')
        schema_version (str): Schema version (e.g., 'v0.1.0')
    """
    
    def __init__(
        self,
        message: str,
        data_type: str = None,
        schema_name: str = None,
        schema_version: str = None
    ):
        """Initialize ProtobufEncodeError with full context.
        
        Args:
            message: Error message describing the encoding failure
            data_type: Optional data type name
            schema_name: Optional protobuf schema name
            schema_version: Optional schema version
        """
        self.data_type = data_type
        self.schema_name = schema_name
        self.schema_version = schema_version
        
        # Build context string
        details = []
        if data_type:
            details.append(f"data_type={data_type}")
        if schema_name:
            details.append(f"schema={schema_name}")
        if schema_version:
            details.append(f"version={schema_version}")
        
        if details:
            message = f"{message} ({', '.join(details)})"
        
        super().__init__(message)
