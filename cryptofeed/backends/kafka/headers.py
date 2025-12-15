"""
Compatibility header helpers for Kafka backend (Phase 2 shim).

These thin wrappers delegate to the inlined header builder in
`cryptofeed.backends.kafka.callback` while preserving the public API
expected by existing tests and legacy callers.
"""

from __future__ import annotations

from typing import Any, List, Optional
from datetime import datetime, timezone

from .normalization import normalize_exchange, normalize_symbol
from .callback import _build_headers


class MessageHeaders:
    """Build mandatory (and optional) headers for a message."""

    @staticmethod
    def build(message: Any, data_type: str, content_type: str) -> list[tuple[bytes, bytes]]:
        def _enc(val: Any) -> bytes:
            if isinstance(val, bytes):
                return val
            return str(val).encode("utf-8")

        exchange = normalize_exchange(getattr(message, "exchange", None))
        symbol = normalize_symbol(getattr(message, "symbol", None))

        return [
            (b"content-type", _enc(content_type)),
            (b"exchange", _enc(exchange)),
            (b"symbol", _enc(symbol)),
            (b"data_type", _enc(data_type)),
        ]


class OptionalHeaders:
    """Build optional headers only (schema_version, producer_version, timestamp_generated, cf.serialization_format)."""

    @staticmethod
    def build(
        schema_version: str = "v1",
        producer_version: Optional[str] = None,
        timestamp_generated: Optional[str] = None,
        serialization_format: str = "json",
        include_serialization_format: bool = True,
    ) -> list[tuple[bytes, bytes]]:
        def _enc(val: Any) -> bytes:
            if isinstance(val, bytes):
                return val
            return str(val).encode("utf-8")

        producer_version = producer_version or "2.4.1"
        if timestamp_generated is None:
            timestamp_generated = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        headers: list[tuple[bytes, bytes]] = [
            (b"schema_version", _enc(schema_version)),
            (b"producer_version", _enc(producer_version)),
            (b"timestamp_generated", _enc(timestamp_generated)),
        ]

        if include_serialization_format:
            headers.append((b"cf.serialization_format", _enc(serialization_format)))

        return headers


class HeaderEnricher:
    """Legacy enricher facade that forwards to the unified header builder."""

    def __init__(
        self,
        content_type: str = "application/x-protobuf",
        schema_version: str = "v1",
        producer_version: Optional[str] = None,
        timestamp_generated: Optional[str] = None,
        serialization_format: str = "json",
        include_serialization_header: bool = True,
    ) -> None:
        self.content_type = content_type
        self.schema_version = schema_version
        self.producer_version = producer_version
        self.timestamp_generated = timestamp_generated
        self.serialization_format = serialization_format
        self._include_serialization_header = include_serialization_header

    def build(self, message: Any, data_type: str) -> list[tuple[bytes, bytes]]:
        mandatory = MessageHeaders.build(message, data_type, self.content_type)
        optional = OptionalHeaders.build(
            schema_version=self.schema_version,
            producer_version=self.producer_version,
            timestamp_generated=self.timestamp_generated,
            serialization_format=self.serialization_format,
            include_serialization_format=self._include_serialization_header,
        )
        return mandatory + optional


__all__ = ["MessageHeaders", "OptionalHeaders", "HeaderEnricher"]
