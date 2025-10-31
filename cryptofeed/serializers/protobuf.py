'''
Copyright (C) 2017-2025 Bryant Moscon - bmoscon@gmail.com

Please see the LICENSE file for the terms and conditions
associated with this software.

Protobuf serializer for binary format.
'''
from typing import Any
from google.protobuf.message import Message
from cryptofeed.serializers.base import Serializer
from cryptofeed.exceptions import SerializationError, ProtobufEncodeError


class ProtobufSerializer(Serializer):
    """
    Serialize cryptofeed data types to protobuf binary format.

    Invokes to_proto() method on data objects to obtain protobuf messages,
    then serializes to binary bytes via SerializeToString().

    Design Principles:
    - Single Responsibility: Handles only protobuf serialization
    - Liskov Substitution: Drop-in replacement for Serializer
    - Type Safety: Verifies to_proto() returns protobuf Message
    - Error Handling: Clear error messages with type context
    """

    def serialize(self, obj: Any) -> bytes:
        """
        Convert object to protobuf bytes.

        Handles both pure Python types (with to_proto() method) and
        C extension types (using registered converters).

        Args:
            obj: Data object to serialize

        Returns:
            bytes: Binary protobuf message

        Raises:
            SerializationError: If conversion fails
            ProtobufEncodeError: If protobuf encoding fails
        """
        try:
            # Use registry to handle both Python and C extension types
            from cryptofeed.proto_wrappers.registry import convert_to_proto

            proto_msg = convert_to_proto(obj)

            if not isinstance(proto_msg, Message):
                raise ProtobufEncodeError(
                    f"to_proto() returned {type(proto_msg).__name__}, "
                    f"expected protobuf Message"
                )

            return proto_msg.SerializeToString()

        except ProtobufEncodeError:
            # Re-raise our own exceptions
            raise
        except AttributeError as e:
            raise SerializationError(
                f"{type(obj).__name__} missing to_proto() method. "
                f"Ensure all data types implement to_proto(). Error: {e}"
            ) from e
        except Exception as e:
            raise SerializationError(
                f"Serialization error for {type(obj).__name__}: {e}"
            ) from e

    def content_type(self) -> str:
        """Return MIME type for protobuf."""
        return 'application/x-protobuf'
