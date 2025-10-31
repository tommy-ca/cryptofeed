'''
Copyright (C) 2017-2025 Bryant Moscon - bmoscon@gmail.com

Please see the LICENSE file for the terms and conditions
associated with this software.

Abstract base class for serializers.
'''
from abc import ABC, abstractmethod
from typing import Any


class Serializer(ABC):
    """
    Abstract base class for data serialization.

    All serializers must implement:
    - serialize(obj) -> bytes: Convert object to serialized bytes
    - content_type() -> str: Return MIME type for the format

    Design Principles:
    - Single Responsibility: Defines serialization contract only
    - Open/Closed: Open for extension (subclass), closed for modification
    - Liskov Substitution: All serializers substitutable
    - Interface Segregation: Minimal 2-method interface
    """

    @abstractmethod
    def serialize(self, obj: Any) -> bytes:
        """
        Convert object to serialized bytes.

        Args:
            obj: Data object to serialize (Trade, OrderBook, etc.)

        Returns:
            bytes: Serialized representation

        Raises:
            SerializationError: If serialization fails
        """
        pass

    @abstractmethod
    def content_type(self) -> str:
        """
        Return MIME type for this serialization format.

        Returns:
            str: MIME type (e.g., 'application/json', 'application/x-protobuf')
        """
        pass
