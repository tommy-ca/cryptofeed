"""
High-level entry points for protobuf serialization helpers.
"""

from .serialization import serialize_to_protobuf, get_converter

__all__ = [
    "serialize_to_protobuf",
    "get_converter",
]
