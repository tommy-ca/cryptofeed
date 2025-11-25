"""
Schema validation utilities for protobuf payloads.
"""

from __future__ import annotations

from google.protobuf.message import Message

from cryptofeed.exceptions import ProtobufEncodeError
from cryptofeed.proto_bindings import SCHEMA_VERSION as DEFAULT_SCHEMA_VERSION


class SchemaValidator:
    """
    Lightweight validator that ensures protobuf messages are initialized and
    aligned with the expected schema version.
    """

    def __init__(self, expected_version: str | None = None) -> None:
        self._expected_version = expected_version or DEFAULT_SCHEMA_VERSION

    def validate(self, proto_msg: Message, *, schema_version: str | None = None) -> None:
        """
        Validate that required fields are populated and schema version matches.
        """

        version = schema_version or self._expected_version
        if version != self._expected_version:
            raise ProtobufEncodeError(
                "Schema version mismatch",
                schema_version=version,
            )

        if not hasattr(proto_msg, "IsInitialized"):
            # Allow custom proto-like objects used in tests/mocks.
            return

        if not proto_msg.IsInitialized():
            missing_fields = proto_msg.FindInitializationErrors()
            missing_detail = f": missing {', '.join(missing_fields)}" if missing_fields else ""
            raise ProtobufEncodeError(
                f"Missing required fields{missing_detail}",
                schema_version=version,
                data_type=proto_msg.DESCRIPTOR.name if proto_msg.DESCRIPTOR else None,
                schema_name=proto_msg.DESCRIPTOR.full_name if proto_msg.DESCRIPTOR else None,
            )


__all__ = ["SchemaValidator"]
