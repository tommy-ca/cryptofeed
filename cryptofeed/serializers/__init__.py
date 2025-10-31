"""
Cryptofeed Serialization Framework

Provides pluggable serialization formats for backend callbacks.
Supports JSON (default, backward compatible) and Protobuf (efficient binary format).

Usage:
    from cryptofeed.serializers import Serializer, JSONSerializer, ProtobufSerializer
    
    # Use JSON serializer (default)
    json_serializer = JSONSerializer()
    json_bytes = json_serializer.serialize(trade_data)
    
    # Use Protobuf serializer
    protobuf_serializer = ProtobufSerializer()
    proto_bytes = protobuf_serializer.serialize(trade_data)
"""
from .base import Serializer

__all__ = ['Serializer']
