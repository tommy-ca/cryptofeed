'''
Copyright (C) 2017-2025 Bryant Moscon - bmoscon@gmail.com

Please see the LICENSE file for the terms and conditions
associated with this software.

JSON serializer for backward compatibility.
'''
import json
from typing import Any
from cryptofeed.serializers.base import Serializer
from cryptofeed.exceptions import SerializationError


class JSONSerializer(Serializer):
    """
    Serialize cryptofeed data types to JSON format.

    Uses existing to_dict() methods for backward compatibility.
    Preserves Decimal precision by relying on to_dict() string conversion.

    Design Principles:
    - Single Responsibility: Handles only JSON serialization
    - Liskov Substitution: Drop-in replacement for Serializer
    - KISS: Simple to_dict() → json.dumps() pipeline
    - DRY: Reuses existing to_dict() methods
    """

    def serialize(self, obj: Any) -> bytes:
        """
        Convert object to JSON bytes.

        Args:
            obj: Data object with to_dict() method

        Returns:
            bytes: UTF-8 encoded JSON

        Raises:
            SerializationError: If to_dict() missing or JSON encoding fails
        """
        if not hasattr(obj, 'to_dict'):
            raise SerializationError(
                f"{type(obj).__name__} missing to_dict() method. "
                f"Ensure all data types implement to_dict()."
            )

        try:
            data_dict = obj.to_dict()
            json_str = json.dumps(data_dict, default=str)
            return json_str.encode('utf-8')
        except (TypeError, ValueError) as e:
            raise SerializationError(f"JSON encoding failed: {e}") from e

    def content_type(self) -> str:
        """Return MIME type for JSON."""
        return 'application/json'
