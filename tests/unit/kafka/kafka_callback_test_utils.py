from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import Mock



# Import KafkaCallback and related components used across KafkaCallback tests


@dataclass
class _RecordedMessage:
    """Record of a produced message for verification."""

    topic: str
    key: Optional[bytes]
    value: bytes
    headers: List[tuple[bytes, bytes]]


class _StubProducer:
    """In-memory producer for testing without Kafka brokers."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connected = False
        self.messages: List[_RecordedMessage] = []
        self.poll_count = 0

    def list_topics(self, timeout: Optional[float] = None):
        self.connected = True
        return {"topics": {}}

    def produce(
        self,
        topic: str,
        value: bytes,
        key: Optional[bytes] = None,
        headers: Optional[List[tuple[bytes, bytes]]] = None,
        on_delivery=None,
    ):
        """Record produced message and optionally call delivery callback."""
        headers = headers or []
        self.messages.append(
            _RecordedMessage(topic=topic, key=key, value=value, headers=headers)
        )
        if on_delivery:
            # Simulate successful delivery
            msg = Mock()
            msg.topic.return_value = topic
            msg.partition.return_value = 0
            msg.offset.return_value = len(self.messages) - 1
            on_delivery(None, msg)

    def poll(self, timeout: float):
        self.poll_count += 1
        return 0

    def flush(self, timeout: Optional[float] = None):
        return 0


def _producer_factory(cls):
    """Create a producer factory function."""

    def _factory(config):
        return cls(config)

    return _factory


# Import from actual modules (not headers.py or partitioner.py since they're inlined into callback.py)
from cryptofeed.backends.kafka.callback import (
    KafkaCallback,
    _build_headers,
    Partitioner,
    PartitionerFactory,
    SymbolPartitioner,
    CompositePartitioner,
    ExchangePartitioner,
    RoundRobinPartitioner,
)
from cryptofeed.backends.kafka.config import (
    KafkaConfig,
    KafkaPartitionConfig,
    KafkaTopicConfig,
    KafkaProducerConfig,
)
from cryptofeed.backends.kafka.backend import TopicManager


# Compatibility shims for old header classes (now inlined into callback.py)
class MessageHeaders:
    """Compatibility shim for MessageHeaders (now inlined)."""

    @staticmethod
    def build(message: Any, data_type: str, content_type: str) -> list[tuple[bytes, bytes]]:
        """Build mandatory headers only (first 4 headers from _build_headers)."""
        from cryptofeed.backends.kafka.normalization import normalize_exchange, normalize_symbol

        exchange = normalize_exchange(getattr(message, "exchange", None))
        symbol = normalize_symbol(getattr(message, "symbol", None))

        def _enc(val: Any) -> bytes:
            if isinstance(val, bytes):
                return val
            return str(val).encode("utf-8")

        return [
            (b"content-type", _enc(content_type)),
            (b"exchange", _enc(exchange)),
            (b"symbol", _enc(symbol)),
            (b"data_type", _enc(data_type)),
        ]


class OptionalHeaders:
    """Compatibility shim for OptionalHeaders (now inlined)."""

    @staticmethod
    def build(
        schema_version: str = "v1",
        producer_version: Optional[str] = None,
        timestamp_generated: Optional[str] = None,
        serialization_format: str = "json",
        include_serialization_format: bool = True,
    ) -> list[tuple[bytes, bytes]]:
        """Build optional headers."""
        from datetime import datetime, timezone

        def _enc(val: Any) -> bytes:
            if isinstance(val, bytes):
                return val
            return str(val).encode("utf-8")

        producer_version = producer_version or "2.4.1"
        if timestamp_generated is None:
            iso_str = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            timestamp_generated = iso_str

        headers = [
            (b"schema_version", _enc(schema_version)),
            (b"producer_version", _enc(producer_version)),
            (b"timestamp_generated", _enc(timestamp_generated)),
        ]

        if include_serialization_format:
            headers.append((b"cf.serialization_format", _enc(serialization_format)))

        return headers


class HeaderEnricher:
    """Compatibility shim for HeaderEnricher (now inlined)."""

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
        """Build complete set of headers using inlined function."""
        return _build_headers(message, data_type, self.content_type, self.schema_version)
